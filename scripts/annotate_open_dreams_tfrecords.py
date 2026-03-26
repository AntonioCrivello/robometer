#!/usr/bin/env python3
"""Annotate Open Dreams TFRecord trajectories with Robometer progress/success scores.

Reads trajectory TFRecords from the Open Dreams dataset format, runs Robometer
inference to compute per-frame task progress [0,1] and success probability [0,1],
and writes annotation TFRecords in the Open Dreams annotation format.

Usage:
    # Single command — auto-detects GPUs
    uv run python scripts/annotate_open_dreams_tfrecords.py \
        --model-path robometer/Robometer-4B \
        --task "Push the T-shaped block to the goal position" \
        --metadata-file /path/to/metadata.json \
        --tfrecord-dir /path/to/tfrecords/ \
        --output-dir /path/to/robometer_annotations/

Troubleshooting:
    Blackwell GPUs (RTX 6000): If you see
    ``RuntimeError: CUDA driver error: invalid argument`` on torch.prod or
    torch.special.entr, clear the CUDA kernel cache:
        rm -rf ~/.nv/ComputeCache
    Then re-run. This is a known issue. See:
    https://github.com/pytorch/pytorch/issues/156010
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from pathlib import Path

import tqdm

import numpy as np
import torch
import torch.multiprocessing as mp

logger = logging.getLogger(__name__)


# TFRecord Feature Helpers

# Lazy import TensorFlow to avoid slow startup when just checking --help
_tf = None


def _get_tf():
    global _tf
    if _tf is None:
        import tensorflow as tf

        # Suppress TF GPU allocation — we use PyTorch for GPU
        tf.config.set_visible_devices([], "GPU")
        _tf = tf
    return _tf


def _int64_feature(value: int):
    tf = _get_tf()
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value: float):
    tf = _get_tf()
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


# Observation Fingerprinting (numpy port of JAX version)

def cheap_fingerprint32(img: np.ndarray) -> int:
    """Compute a fast 32-bit fingerprint of an image for verification.

    Numpy port of the JAX version in open_dreams/scripts/annotate_rewards.py.
    Samples a fixed 4x4 pixel grid and mixes with MurmurHash3-style scrambling.
    Expects input shape [H, W, C] with uint8 dtype.
    """
    x = img
    if x.dtype != np.uint8:
        x = np.clip(x, 0.0, 255.0).astype(np.uint8)

    h, w = x.shape[0], x.shape[1]
    ys = np.array([0, h // 3, 2 * h // 3, h - 1], dtype=np.int32)
    xs = np.array([0, w // 3, 2 * w // 3, w - 1], dtype=np.int32)
    patch = x[ys[:, None], xs[None, :], ...].reshape(-1).astype(np.uint32)

    # MurmurHash3 finalizer-style mixing (32-bit)
    acc = np.uint32(0x9E3779B9)
    acc = acc ^ np.bitwise_xor.reduce(patch * np.uint32(0x1B873593))
    acc = (acc ^ (acc >> np.uint32(16))) * np.uint32(0x85EBCA6B)
    acc = (acc ^ (acc >> np.uint32(13))) * np.uint32(0xC2B2AE35)
    acc = acc ^ (acc >> np.uint32(16))
    return int(acc)


def cheap_fingerprint32_batch(imgs: np.ndarray) -> list[int]:
    """Vectorized fingerprint for a batch of images.

    Args:
        imgs: uint8 numpy array of shape (T, H, W, C).

    Returns:
        List of T int fingerprints, identical to calling cheap_fingerprint32
        on each frame individually.
    """
    if imgs.dtype != np.uint8:
        imgs = np.clip(imgs, 0.0, 255.0).astype(np.uint8)

    T, H, W = imgs.shape[0], imgs.shape[1], imgs.shape[2]
    ys = np.array([0, H // 3, 2 * H // 3, H - 1], dtype=np.int32)
    xs = np.array([0, W // 3, 2 * W // 3, W - 1], dtype=np.int32)
    # Extract patches for all frames at once: (T, 4, 4, C) -> (T, 48)
    patches = imgs[:, ys[:, None], xs[None, :], :].reshape(T, -1).astype(np.uint32)

    acc = np.full(T, 0x9E3779B9, dtype=np.uint32)
    acc ^= np.bitwise_xor.reduce(patches * np.uint32(0x1B873593), axis=1)
    acc = (acc ^ (acc >> np.uint32(16))) * np.uint32(0x85EBCA6B)
    acc = (acc ^ (acc >> np.uint32(13))) * np.uint32(0xC2B2AE35)
    acc ^= acc >> np.uint32(16)
    return acc.tolist()


# =============================================================================
# Welford Statistics (O(1) Memory)
# =============================================================================


class WelfordStats:
    """Welford's online algorithm for mean and variance. O(1) memory."""

    def __init__(self):
        self.count: int = 0
        self.mean: float = 0.0
        self.M2: float = 0.0

    def update(self, x: float) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        return self.M2 / self.count if self.count > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance) if self.count > 1 else 1.0

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std, "count": self.count}

    @staticmethod
    def merge(stats_list: list[dict]) -> dict:
        """Merge multiple Welford stats dicts into one (parallel reduction)."""
        total_count = sum(s["count"] for s in stats_list)
        if total_count == 0:
            return {"mean": 0.0, "std": 1.0, "count": 0}

        # Combined mean
        combined_mean = sum(s["mean"] * s["count"] for s in stats_list) / total_count

        # Combined M2 (variance * count) via parallel Welford
        combined_M2 = 0.0
        for s in stats_list:
            if s["count"] > 1:
                combined_M2 += (s["std"] ** 2) * s["count"]
            delta = s["mean"] - combined_mean
            combined_M2 += delta * delta * s["count"]

        combined_std = (
            math.sqrt(combined_M2 / total_count) if total_count > 1 else 1.0
        )
        return {"mean": combined_mean, "std": combined_std, "count": total_count}


# =============================================================================
# Dataset Metadata Loading
# =============================================================================


@dataclass
class DatasetInfo:
    """Parsed dataset metadata."""

    file_paths: list[str]
    episode_lengths: list[int]
    encodings: list[str]
    observation_shapes: list[tuple[int, ...]]
    num_trajectories: int
    metadata: dict


def load_dataset_info(
    metadata_file: str,
    tfrecord_dir: str,
) -> DatasetInfo:
    """Load dataset metadata from metadata.json and trajectories_metadata.csv."""
    metadata_path = Path(metadata_file)
    tfrecord_path = Path(tfrecord_dir)

    with open(metadata_path) as f:
        metadata = json.load(f)

    csv_path = metadata_path.parent / "trajectories_metadata.csv"
    file_paths = []
    episode_lengths = []
    encodings = []
    observation_shapes = []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"]
            fpath = tfrecord_path / fname
            file_paths.append(str(fpath))
            episode_lengths.append(int(row["episode_length"]))
            encodings.append(row.get("encoding", "jpeg"))
            # Parse observation shape: "HxWxC" or "(H, W, C)"
            shape_str = row.get("observation_shape", "144x144x3")
            if "x" in shape_str:
                obs_shape = tuple(int(d) for d in shape_str.split("x"))
            else:
                import ast

                obs_shape = tuple(ast.literal_eval(shape_str))
            observation_shapes.append(obs_shape)

    return DatasetInfo(
        file_paths=file_paths,
        episode_lengths=episode_lengths,
        encodings=encodings,
        observation_shapes=observation_shapes,
        num_trajectories=len(file_paths),
        metadata=metadata,
    )


# =============================================================================
# TFRecord Reading
# =============================================================================


def read_trajectory_frames(
    tfrecord_path: str,
    num_frames: int,
    encoding: str = "jpeg",
    frame_step: int = 1,
) -> tuple[np.ndarray, list[int]]:
    """Read observation frames from a trajectory TFRecord.

    Args:
        tfrecord_path: Path to TFRecord file.
        num_frames: Number of frames in the trajectory.
        encoding: Image encoding format.
        frame_step: Subsample every Nth frame (1 = all frames).

    Returns:
        (frames, original_indices) where frames is uint8 array (T, H, W, C)
        and original_indices maps subsampled position to original frame index.
    """
    tf = _get_tf()

    feature_spec = {
        "observation": tf.io.FixedLenFeature([], tf.string),
    }

    frames = []
    original_indices = []
    ds = tf.data.TFRecordDataset([tfrecord_path])
    for frame_idx, raw_record in enumerate(ds.take(num_frames)):
        if frame_idx % frame_step != 0:
            continue

        parsed = tf.io.parse_single_example(raw_record, feature_spec)
        img_bytes = parsed["observation"]

        if encoding in ("jpeg", "jpg"):
            img = tf.io.decode_jpeg(img_bytes, channels=3)
        elif encoding == "png":
            img = tf.io.decode_png(img_bytes, channels=3)
        else:
            img = tf.io.decode_raw(img_bytes, tf.uint8)

        frames.append(img.numpy())
        original_indices.append(frame_idx)

    if not frames:
        return np.empty((0, 0, 0, 0), dtype=np.uint8), []

    return np.stack(frames, axis=0).astype(np.uint8), original_indices


# =============================================================================
# Annotation TFRecord Writing
# =============================================================================


def write_annotation_tfrecord(
    output_path: str,
    progress: np.ndarray,
    success: np.ndarray,
    fingerprints: list[int],
    original_indices: list[int] | None = None,
) -> None:
    """Write per-frame Robometer annotations to a TFRecord file."""
    tf = _get_tf()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    num_entries = len(original_indices) if original_indices else len(progress)
    with tf.io.TFRecordWriter(output_path) as writer:
        for t in range(num_entries):
            idx = original_indices[t] if original_indices else t
            feature = {
                "frame_idx": _int64_feature(idx),
                "robometer_progress": _float_feature(float(progress[t])),
                "robometer_success": _float_feature(
                    float(success[t]) if t < len(success) else 0.0
                ),
                "obs_fingerprint": _int64_feature(
                    fingerprints[t] if t < len(fingerprints) else 0
                ),
            }
            example = tf.train.Example(
                features=tf.train.Features(feature=feature)
            )
            writer.write(example.SerializeToString())


# =============================================================================
# Robometer Inference
# =============================================================================


def load_robometer(
    model_path: str,
    device: torch.device,
) -> tuple:
    """Load Robometer model, tokenizer, processor, and batch collator.

    Returns (exp_config, tokenizer, processor, reward_model, batch_collator).
    """
    from robometer.utils.save import load_model_from_hf
    from robometer.utils.setup_utils import setup_batch_collator

    exp_config, tokenizer, processor, reward_model = load_model_from_hf(
        model_path=model_path,
        device=device,
    )
    reward_model.eval()
    batch_collator = setup_batch_collator(
        processor, tokenizer, exp_config, is_eval=True
    )
    return exp_config, tokenizer, processor, reward_model, batch_collator


def run_robometer_inference(
    frames: np.ndarray,
    task: str,
    reward_model,
    tokenizer,
    exp_config,
    batch_collator,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Robometer inference on a single trajectory's frames.

    Args:
        frames: uint8 numpy array (T, H, W, C)
        task: Natural language task instruction

    Returns:
        (progress_array, success_array) — both float32 numpy arrays of shape (T,)
    """
    from robometer.data.dataset_types import ProgressSample, Trajectory
    from robometer.evals.eval_server import compute_batch_outputs

    T = int(frames.shape[0])
    traj = Trajectory(
        frames=frames,
        frames_shape=tuple(frames.shape),
        task=task,
        id="0",
        metadata={"subsequence_length": T},
        video_embeddings=None,
    )
    progress_sample = ProgressSample(trajectory=traj, sample_type="progress")
    batch = batch_collator([progress_sample])

    progress_inputs = batch["progress_inputs"]
    for key, value in progress_inputs.items():
        if hasattr(value, "to"):
            progress_inputs[key] = value.to(device)

    loss_config = getattr(exp_config, "loss", None)
    is_discrete = (
        getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete"
        if loss_config
        else False
    )
    num_bins = getattr(loss_config, "progress_discrete_bins", None) or getattr(
        exp_config.model, "progress_discrete_bins", 10
    )

    with torch.no_grad():
        results = compute_batch_outputs(
            reward_model,
            tokenizer,
            progress_inputs,
            sample_type="progress",
            is_discrete_mode=is_discrete,
            num_bins=num_bins,
        )

    # Extract progress predictions
    progress_pred = results.get("progress_pred", [])
    progress_array = (
        np.array(progress_pred[0], dtype=np.float32)
        if progress_pred and len(progress_pred) > 0
        else np.zeros(T, dtype=np.float32)
    )

    # Extract success predictions
    outputs_success = results.get("outputs_success", {})
    success_probs = (
        outputs_success.get("success_probs", []) if outputs_success else []
    )
    success_array = (
        np.array(success_probs[0], dtype=np.float32)
        if success_probs and len(success_probs) > 0
        else np.zeros(T, dtype=np.float32)
    )

    return progress_array, success_array


def run_robometer_inference_batched(
    frames_list: list[np.ndarray],
    task: str,
    reward_model,
    tokenizer,
    exp_config,
    batch_collator,
    device: torch.device,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run Robometer inference on multiple trajectories in a single forward pass.

    Args:
        frames_list: List of uint8 numpy arrays, each (T_i, H, W, C)
        task: Natural language task instruction (shared across all trajectories)

    Returns:
        List of (progress_array, success_array) tuples, one per trajectory.
    """
    from robometer.data.dataset_types import ProgressSample, Trajectory
    from robometer.evals.eval_server import compute_batch_outputs

    num_trajs = len(frames_list)
    traj_lengths = [int(f.shape[0]) for f in frames_list]

    progress_samples = []
    for i, frames in enumerate(frames_list):
        T = traj_lengths[i]
        traj = Trajectory(
            frames=frames,
            frames_shape=tuple(frames.shape),
            task=task,
            id=str(i),
            metadata={"subsequence_length": T},
            video_embeddings=None,
        )
        progress_samples.append(
            ProgressSample(trajectory=traj, sample_type="progress")
        )

    batch = batch_collator(progress_samples)

    progress_inputs = batch["progress_inputs"]
    for key, value in progress_inputs.items():
        if hasattr(value, "to"):
            progress_inputs[key] = value.to(device)

    loss_config = getattr(exp_config, "loss", None)
    is_discrete = (
        getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete"
        if loss_config
        else False
    )
    num_bins = getattr(loss_config, "progress_discrete_bins", None) or getattr(
        exp_config.model, "progress_discrete_bins", 10
    )

    with torch.no_grad():
        results = compute_batch_outputs(
            reward_model,
            tokenizer,
            progress_inputs,
            sample_type="progress",
            is_discrete_mode=is_discrete,
            num_bins=num_bins,
        )

    # Unpack per-trajectory results
    progress_pred = results.get("progress_pred", [])
    outputs_success = results.get("outputs_success", {})
    success_probs = (
        outputs_success.get("success_probs", []) if outputs_success else []
    )

    output = []
    for i in range(num_trajs):
        T = traj_lengths[i]

        if progress_pred and i < len(progress_pred) and progress_pred[i]:
            prog = np.array(progress_pred[i], dtype=np.float32)
        else:
            prog = np.zeros(T, dtype=np.float32)

        if success_probs and i < len(success_probs) and success_probs[i]:
            succ = np.array(success_probs[i], dtype=np.float32)
        else:
            succ = np.zeros(T, dtype=np.float32)

        output.append((prog, succ))

    return output


# =============================================================================
# Pipelined Batch Preparation & Post-Processing
# =============================================================================


@dataclass
class PreparedBatch:
    """Result of CPU-side batch preparation (TFRecord reading + collation)."""
    batch_frames: list[tuple[np.ndarray, list[int]]]  # (frames, orig_indices) per traj
    batch_valid_indices: list[int]  # dataset indices that were successfully read
    progress_inputs: dict  # collated tensors ready for .to(device)
    is_discrete: bool
    num_bins: int
    traj_lengths: list[int]
    batch_total_frames: int


def prepare_batch(
    batch_indices: list[int],
    dataset_info: DatasetInfo,
    task: str,
    frame_step: int,
    batch_collator,
    exp_config,
    rank: int,
) -> PreparedBatch | None:
    """CPU-only: read TFRecords + collate. Safe to run in a background thread.

    Returns None if no valid trajectories in this batch.
    """
    from robometer.data.dataset_types import ProgressSample, Trajectory

    # Read frames — parallel across trajectories in this batch
    def _read_one(idx):
        tfrecord_path = dataset_info.file_paths[idx]
        num_frames = dataset_info.episode_lengths[idx]
        encoding = dataset_info.encodings[idx]
        if not os.path.exists(tfrecord_path):
            logger.warning(f"[Worker {rank}] Missing file: {tfrecord_path}, skipping")
            return idx, None, None
        frames, orig_indices = read_trajectory_frames(
            tfrecord_path, num_frames, encoding, frame_step=frame_step,
        )
        if frames.shape[0] == 0:
            logger.warning(f"[Worker {rank}] Empty trajectory: {tfrecord_path}")
            return idx, None, None
        return idx, frames, orig_indices

    # Use threads to read multiple TFRecords concurrently
    read_workers = min(4, len(batch_indices))
    batch_frames = []
    batch_valid_indices = []
    if read_workers > 1:
        with ThreadPoolExecutor(max_workers=read_workers) as read_pool:
            futures = [read_pool.submit(_read_one, idx) for idx in batch_indices]
            # Collect in submission order to maintain ordering
            for future in futures:
                idx, frames, orig_indices = future.result()
                if frames is not None:
                    batch_frames.append((frames, orig_indices))
                    batch_valid_indices.append(idx)
    else:
        for idx in batch_indices:
            idx, frames, orig_indices = _read_one(idx)
            if frames is not None:
                batch_frames.append((frames, orig_indices))
                batch_valid_indices.append(idx)

    if not batch_frames:
        return None

    # Build ProgressSample objects and collate
    frames_list = [f for f, _ in batch_frames]
    traj_lengths = [int(f.shape[0]) for f in frames_list]
    progress_samples = []
    for i, frames in enumerate(frames_list):
        T = traj_lengths[i]
        traj = Trajectory(
            frames=frames,
            frames_shape=tuple(frames.shape),
            task=task,
            id=str(i),
            metadata={"subsequence_length": T},
            video_embeddings=None,
        )
        progress_samples.append(
            ProgressSample(trajectory=traj, sample_type="progress")
        )

    batch = batch_collator(progress_samples)
    progress_inputs = batch["progress_inputs"]

    # Precompute loss config
    loss_config = getattr(exp_config, "loss", None)
    is_discrete = (
        getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete"
        if loss_config
        else False
    )
    num_bins = getattr(loss_config, "progress_discrete_bins", None) or getattr(
        exp_config.model, "progress_discrete_bins", 10
    )

    batch_total_frames = sum(traj_lengths)

    return PreparedBatch(
        batch_frames=batch_frames,
        batch_valid_indices=batch_valid_indices,
        progress_inputs=progress_inputs,
        is_discrete=is_discrete,
        num_bins=num_bins,
        traj_lengths=traj_lengths,
        batch_total_frames=batch_total_frames,
    )


def _gpu_inference_single(
    progress_inputs: dict,
    device: torch.device,
    reward_model,
    tokenizer,
    is_discrete: bool,
    num_bins: int,
    traj_lengths: list[int],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run a single GPU forward pass and unpack results."""
    from robometer.evals.eval_server import compute_batch_outputs

    for key, value in progress_inputs.items():
        if hasattr(value, "to"):
            progress_inputs[key] = value.to(device, non_blocking=True)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        results = compute_batch_outputs(
            reward_model,
            tokenizer,
            progress_inputs,
            sample_type="progress",
            is_discrete_mode=is_discrete,
            num_bins=num_bins,
        )

    # Unpack per-trajectory results
    progress_pred = results.get("progress_pred", [])
    outputs_success = results.get("outputs_success", {})
    success_probs = (
        outputs_success.get("success_probs", []) if outputs_success else []
    )

    output = []
    for i in range(len(traj_lengths)):
        T = traj_lengths[i]
        if progress_pred and i < len(progress_pred) and progress_pred[i]:
            prog = np.array(progress_pred[i], dtype=np.float32)
        else:
            prog = np.zeros(T, dtype=np.float32)
        if success_probs and i < len(success_probs) and success_probs[i]:
            succ = np.array(success_probs[i], dtype=np.float32)
        else:
            succ = np.zeros(T, dtype=np.float32)
        output.append((prog, succ))

    return output


def gpu_inference(
    prepared: PreparedBatch,
    device: torch.device,
    reward_model,
    tokenizer,
    batch_collator,
    task: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """GPU inference with automatic OOM retry at half batch size."""
    try:
        return _gpu_inference_single(
            prepared.progress_inputs, device, reward_model, tokenizer,
            prepared.is_discrete, prepared.num_bins, prepared.traj_lengths,
        )
    except torch.cuda.OutOfMemoryError:
        n = len(prepared.batch_frames)
        logger.warning(
            f"OOM with {n} trajectories "
            f"(max {max(prepared.traj_lengths)} frames). "
            f"Retrying in two halves..."
        )
        torch.cuda.empty_cache()

        # Re-collate and run each half separately
        from robometer.data.dataset_types import ProgressSample, Trajectory

        all_results = []
        mid = n // 2
        for sub_frames_list in [prepared.batch_frames[:mid], prepared.batch_frames[mid:]]:
            if not sub_frames_list:
                continue
            frames_only = [f for f, _ in sub_frames_list]
            sub_lengths = [int(f.shape[0]) for f in frames_only]
            samples = []
            for i, frames in enumerate(frames_only):
                traj = Trajectory(
                    frames=frames,
                    frames_shape=tuple(frames.shape),
                    task=task,
                    id=str(i),
                    metadata={"subsequence_length": sub_lengths[i]},
                    video_embeddings=None,
                )
                samples.append(ProgressSample(trajectory=traj, sample_type="progress"))
            batch = batch_collator(samples)
            sub_inputs = batch["progress_inputs"]
            sub_results = _gpu_inference_single(
                sub_inputs, device, reward_model, tokenizer,
                prepared.is_discrete, prepared.num_bins, sub_lengths,
            )
            all_results.extend(sub_results)
            torch.cuda.empty_cache()

        return all_results


def post_process_and_write(
    results: list[tuple[np.ndarray, np.ndarray]],
    prepared: PreparedBatch,
    output_dir: Path,
    dataset_info: DatasetInfo,
    stats: dict[str, WelfordStats],
) -> None:
    """CPU-only: compute fingerprints, write TFRecords, update stats.

    Safe to run in a background thread. Note: stats updates are NOT thread-safe
    on their own, but we ensure only one write future runs at a time by draining
    the previous write before submitting the next.
    """
    for i, idx in enumerate(prepared.batch_valid_indices):
        progress, success = results[i]
        frames, orig_indices = prepared.batch_frames[i]

        # Vectorized fingerprint computation
        fingerprints = cheap_fingerprint32_batch(frames)

        # Write annotation TFRecord
        tfrecord_path = dataset_info.file_paths[idx]
        original_name = os.path.basename(tfrecord_path)
        output_path = str(output_dir / original_name)
        write_annotation_tfrecord(
            output_path, progress, success, fingerprints, orig_indices,
        )

        # Update Welford stats
        for t in range(len(progress)):
            stats["robometer_progress"].update(float(progress[t]))
        for t in range(len(success)):
            stats["robometer_success"].update(float(success[t]))


# =============================================================================
# Worker Function (one per GPU)
# =============================================================================


def build_batch_slices(
    dataset_info: DatasetInfo,
    batch_size: int,
    frame_step: int,
) -> list[list[int]]:
    """Build adaptive batch slices from all trajectories.

    Sorts trajectories by episode length descending (longest first) and packs
    them into batches using a frame budget rather than a fixed trajectory count.
    This prevents OOM on batches of long trajectories while still packing short
    trajectories densely.

    Called once in the main process; the resulting slices are distributed to
    workers via a shared mp.Queue for work-stealing load balancing.
    """
    all_indices = list(range(dataset_info.num_trajectories))
    # Sort descending — longest first when GPU memory is cleanest
    all_indices.sort(key=lambda i: dataset_info.episode_lengths[i], reverse=True)

    # Derive frame budget from batch_size * median episode length
    sorted_lens = sorted(dataset_info.episode_lengths)
    median_len = sorted_lens[len(sorted_lens) // 2] if sorted_lens else 34
    frame_budget = batch_size * max(median_len, 1)

    batch_slices = []
    current_batch: list[int] = []
    current_frames = 0
    for idx in all_indices:
        ep_len = dataset_info.episode_lengths[idx]
        effective_len = max(1, (ep_len + frame_step - 1) // frame_step)
        if current_batch and current_frames + effective_len > frame_budget:
            batch_slices.append(current_batch)
            current_batch = []
            current_frames = 0
        current_batch.append(idx)
        current_frames += effective_len
    if current_batch:
        batch_slices.append(current_batch)

    return batch_slices


def worker_fn(
    rank: int,
    num_workers: int,
    args: argparse.Namespace,
    dataset_info: DatasetInfo,
    work_queue: mp.Queue | None = None,
    total_batches: int = 0,
) -> None:
    """Process batches of trajectories on a single GPU.

    Uses a shared work queue (inspired by MultiGPUEvalServer's GPU pool pattern)
    for work-stealing load balancing: whichever GPU finishes first grabs the next
    batch. If work_queue is None, falls back to static sharding.

    Within each worker, uses a double-buffered prefetch pipeline:
      - Background thread prepares the NEXT batch (TFRecord I/O + collation)
      - Main thread runs GPU inference on the CURRENT batch
      - Previous batch's post-processing (fingerprints + TFRecord writes)
        is drained before submitting new post-processing to avoid stats races.
    """
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    frame_step = getattr(args, "frame_step", 1)
    logger.info(f"[Worker {rank}/{num_workers}] Starting on {device}")

    # Load Robometer model on this GPU
    exp_config, tokenizer, _processor, reward_model, batch_collator = load_robometer(
        args.model_path, device
    )

    # Initialize Welford stats
    stats = {
        "robometer_progress": WelfordStats(),
        "robometer_success": WelfordStats(),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_processed = 0
    total_frames = 0
    total_infer_time = 0.0
    t_start = time.time()

    # --- Helper to get next batch from the shared queue ---
    def next_batch_indices() -> list[int] | None:
        """Pull next batch from shared queue; returns None when exhausted."""
        if work_queue is None:
            return None
        try:
            return work_queue.get_nowait()
        except Exception:
            return None

    # --- Prefetch pipeline ---
    prefetch_executor = ThreadPoolExecutor(max_workers=1)
    write_executor = ThreadPoolExecutor(max_workers=1)
    write_future: Future | None = None

    # Grab first batch and kick off preparation
    first_indices = next_batch_indices()
    if first_indices is None:
        logger.info(f"[Worker {rank}] No batches to process")
        return

    prefetch_future = prefetch_executor.submit(
        prepare_batch,
        first_indices, dataset_info, args.task, frame_step,
        batch_collator, exp_config, rank,
    )

    batches_done = 0
    while True:
        # Wait for current batch's CPU prep
        t_prep_start = time.time()
        prepared = prefetch_future.result()
        t_prep = time.time() - t_prep_start

        # Grab next batch from queue and kick off prefetch immediately
        next_indices = next_batch_indices()
        if next_indices is not None:
            prefetch_future = prefetch_executor.submit(
                prepare_batch,
                next_indices, dataset_info, args.task, frame_step,
                batch_collator, exp_config, rank,
            )

        if prepared is not None:
            # Drain previous write before starting new one (stats thread safety)
            if write_future is not None:
                write_future.result()

            # GPU inference
            t_infer_start = time.time()
            results = gpu_inference(
                prepared, device, reward_model, tokenizer,
                batch_collator, args.task,
            )
            t_infer = time.time() - t_infer_start

            # Submit post-processing to background thread
            write_future = write_executor.submit(
                post_process_and_write,
                results, prepared, output_dir, dataset_info, stats,
            )

            # Update counters
            num_processed += len(prepared.batch_valid_indices)
            total_frames += prepared.batch_total_frames
            total_infer_time += t_infer
            batches_done += 1
            elapsed = time.time() - t_start
            avg_time_per_traj = elapsed / num_processed if num_processed > 0 else 0
            avg_infer_per_traj = total_infer_time / num_processed if num_processed > 0 else 0
            fps = total_frames / total_infer_time if total_infer_time > 0 else 0
            logger.info(
                f"[Worker {rank}] {num_processed} trajs done "
                f"(batch {batches_done}/{total_batches}, "
                f"{avg_time_per_traj:.2f}s/traj, {avg_infer_per_traj:.2f}s infer/traj, "
                f"{fps:.0f} frames/s) | "
                f"prep_wait={t_prep:.2f}s infer={t_infer:.2f}s"
            )

        # If no more batches queued, we're done
        if next_indices is None:
            break

    # Drain final write
    if write_future is not None:
        write_future.result()

    prefetch_executor.shutdown(wait=True)
    write_executor.shutdown(wait=True)

    # Write per-shard stats and timing
    shard_stats_path = output_dir / f"_stats_shard_{rank}.json"
    elapsed = time.time() - t_start
    shard_data = {
        "annotation_stats": {k: v.to_dict() for k, v in stats.items()},
        "timing": {
            "num_trajectories": num_processed,
            "total_frames": total_frames,
            "elapsed_s": elapsed,
            "total_infer_s": total_infer_time,
        },
    }
    with open(shard_stats_path, "w") as f:
        json.dump(shard_data, f, indent=2)


# =============================================================================
# Stats Merging
# =============================================================================


def merge_shard_stats(
    output_dir: str,
    args: argparse.Namespace,
    dataset_info: DatasetInfo,
) -> None:
    """Merge per-shard Welford stats into final metadata.json."""
    output_path = Path(output_dir)

    # Collect all shard stats files
    shard_files = sorted(output_path.glob("_stats_shard_*.json"))
    if not shard_files:
        logger.error("No shard stats files found!")
        return

    # Load all shard stats and timing
    all_stats: dict[str, list[dict]] = {}
    all_timing: list[dict] = []
    for sf in shard_files:
        with open(sf) as f:
            shard = json.load(f)
        # Support both old format (flat stats) and new format (nested)
        ann_stats = shard.get("annotation_stats", shard)
        for field_name, field_stats in ann_stats.items():
            if field_name not in all_stats:
                all_stats[field_name] = []
            all_stats[field_name].append(field_stats)
        if "timing" in shard:
            all_timing.append(shard["timing"])

    # Merge via parallel Welford
    merged_stats = {}
    for field_name, stats_list in all_stats.items():
        merged_stats[field_name] = WelfordStats.merge(stats_list)

    # Write final metadata.json
    metadata = {
        "original_dataset": {
            "name": Path(args.metadata_file).parent.name,
            "num_trajectories": dataset_info.num_trajectories,
            "metadata_file": args.metadata_file,
            "tfrecord_dir": args.tfrecord_dir,
        },
        "annotation_model": {
            "name": "robometer_4b",
            "model_path": args.model_path,
            "type": "robometer",
            "task_instruction": args.task,
        },
        "annotation_fields": ["robometer_progress", "robometer_success"],
        "normalization_statistics": merged_stats,
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Final metadata written to {metadata_path}")
    for field, s in merged_stats.items():
        logger.info(
            f"  {field}: mean={s['mean']:.4f}, std={s['std']:.4f}, "
            f"count={s['count']}"
        )

    # Print timing summary
    if all_timing:
        total_trajs = sum(t["num_trajectories"] for t in all_timing)
        total_frames = sum(t["total_frames"] for t in all_timing)
        wall_clock = max(t["elapsed_s"] for t in all_timing)
        total_infer = sum(t["total_infer_s"] for t in all_timing)
        avg_per_traj = wall_clock / total_trajs if total_trajs > 0 else 0
        avg_infer_per_traj = total_infer / total_trajs if total_trajs > 0 else 0
        fps = total_frames / wall_clock if wall_clock > 0 else 0
        logger.info(
            f"\n{'='*60}\n"
            f"  ANNOTATION TIMING SUMMARY\n"
            f"  Trajectories: {total_trajs}\n"
            f"  Total frames: {total_frames}\n"
            f"  Wall clock:   {wall_clock:.1f}s\n"
            f"  Avg per traj: {avg_per_traj:.2f}s (total), "
            f"{avg_infer_per_traj:.2f}s (infer only)\n"
            f"  Throughput:   {fps:.0f} frames/s\n"
            f"  Workers:      {len(all_timing)}\n"
            f"{'='*60}"
        )

    # Clean up shard stats files
    for sf in shard_files:
        sf.unlink()
    logger.info(f"Cleaned up {len(shard_files)} shard stats files")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate Open Dreams TFRecord trajectories with Robometer "
        "progress and success scores.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="HuggingFace model id or local checkpoint path "
        "(e.g., robometer/Robometer-4B)",
    )
    parser.add_argument(
        "--task",
        required=True,
        help='Task instruction '
        '(e.g., "Push the T-shaped block to the goal position")',
    )
    parser.add_argument(
        "--metadata-file",
        required=True,
        help="Path to Open Dreams dataset metadata.json",
    )
    parser.add_argument(
        "--tfrecord-dir",
        required=True,
        help="Directory containing trajectory TFRecord files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for annotation TFRecords and metadata.json",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: auto-detect all available)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of trajectories per Robometer forward pass (default: 32). "
        "Higher values improve GPU utilization but use more memory.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Annotate every Nth frame (default: 1 = all frames). "
        "Use 2 for ~2x speedup, 3 for ~3x speedup.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Log progress every N trajectories per worker (default: 50)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load dataset info
    dataset_info = load_dataset_info(args.metadata_file, args.tfrecord_dir)
    logger.info(
        f"Dataset: {dataset_info.num_trajectories} trajectories, "
        f"total frames: {sum(dataset_info.episode_lengths)}"
    )

    # Determine number of GPUs
    if args.num_gpus is not None:
        num_gpus = args.num_gpus
    elif torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = 1  # CPU fallback

    logger.info(f"Using {num_gpus} GPU(s) for annotation")

    # Build batch slices centrally — adaptive sizing based on frame budget
    batch_size = getattr(args, "batch_size", 32)
    frame_step = getattr(args, "frame_step", 1)
    batch_slices = build_batch_slices(dataset_info, batch_size, frame_step)
    logger.info(
        f"Built {len(batch_slices)} batch slices "
        f"(adaptive frame budget = {batch_size} * median_len)"
    )

    if num_gpus <= 1:
        # Single-process mode — feed batches via mp.Queue (same code path)
        work_queue = mp.Queue()
        for batch in batch_slices:
            work_queue.put(batch)
        worker_fn(0, 1, args, dataset_info, work_queue, len(batch_slices))
    else:
        # Multi-GPU: shared work queue for work-stealing load balancing
        # (inspired by MultiGPUEvalServer's GPU pool pattern)
        mp.set_start_method("spawn", force=True)
        work_queue = mp.Queue()
        for batch in batch_slices:
            work_queue.put(batch)
        mp.spawn(
            worker_fn,
            args=(num_gpus, args, dataset_info, work_queue, len(batch_slices)),
            nprocs=num_gpus,
            join=True,
        )

    # Merge per-shard stats into final metadata.json
    merge_shard_stats(args.output_dir, args, dataset_info)
    logger.info("Annotation complete!")


if __name__ == "__main__":
    main()

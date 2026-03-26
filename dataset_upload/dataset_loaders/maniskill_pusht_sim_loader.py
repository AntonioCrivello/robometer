#!/usr/bin/env python3
"""
Maniskill Push-T sim dataset loader for Robometer model training.

Reads Open Dreams TFRecord trajectories (one episode per file) and produces
trajectory dicts compatible with generate_hf_dataset.py.
"""

import csv
import json
import os
from pathlib import Path

import numpy as np
import tqdm

from dataset_upload.helpers import generate_unique_id

# Lazy TF import — avoid slow startup and GPU allocation
_tf = None


def _get_tf():
    global _tf
    if _tf is None:
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
        _tf = tf
    return _tf


class TFRecordFrameLoader:
    """Pickle-able lazy loader that reads frames from an Open Dreams TFRecord.

    generate_hf_dataset.py calls this to get frames for MP4 conversion.
    Must be pickle-able (no TF objects stored as instance vars).
    """

    def __init__(self, tfrecord_path: str, encoding: str = "jpeg") -> None:
        self.tfrecord_path = tfrecord_path
        self.encoding = encoding

    def __call__(self) -> np.ndarray:
        """Load all frames from the TFRecord.

        Returns:
            np.ndarray of shape (T, H, W, 3), dtype uint8
        """
        tf = _get_tf()
        feature_spec = {"observation": tf.io.FixedLenFeature([], tf.string)}

        frames = []
        ds = tf.data.TFRecordDataset([self.tfrecord_path])
        for raw_record in ds:
            parsed = tf.io.parse_single_example(raw_record, feature_spec)
            img_bytes = parsed["observation"]

            if self.encoding in ("jpeg", "jpg"):
                img = tf.io.decode_jpeg(img_bytes, channels=3)
            elif self.encoding == "png":
                img = tf.io.decode_png(img_bytes, channels=3)
            else:
                img = tf.io.decode_raw(img_bytes, tf.uint8)

            frames.append(img.numpy())

        if not frames:
            return np.empty((0, 0, 0, 3), dtype=np.uint8)

        return np.stack(frames, axis=0).astype(np.uint8)


def _read_trajectory_rewards(tfrecord_path: str) -> tuple[float, bool]:
    """Read reward and done fields from a TFRecord to get total_reward and success.

    Returns:
        (total_reward, success) where success is True if any done==1 with
        the trajectory completing.
    """
    tf = _get_tf()
    feature_spec = {
        "reward": tf.io.FixedLenFeature([], tf.float32),
        "done": tf.io.FixedLenFeature([], tf.int64),
    }

    total_reward = 0.0
    success = False
    ds = tf.data.TFRecordDataset([tfrecord_path])
    for raw_record in ds:
        parsed = tf.io.parse_single_example(raw_record, feature_spec)
        total_reward += float(parsed["reward"].numpy())
        if parsed["done"].numpy() == 1:
            success = True

    return total_reward, success


def load_maniskill_pusht_sim_dataset(
    dataset_path: str,
    dataset_name: str,
) -> dict[str, list[dict]]:
    """Load Maniskill Push-T Sim dataset from Open Dreams TFRecords.

    Args:
        dataset_path: Root directory containing *.tfrecord files + metadata.json
        dataset_name: Dataset name (e.g. 'maniskill_pusht_train')

    Returns:
        {task_instruction: [trajectory_dicts]} for generate_hf_dataset.py
    """
    root = Path(dataset_path)
    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {root}")

    # Read metadata for encoding and task
    encoding = "png"
    task = "Push the T block to the goal position"
    metadata_path = root / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        encoding = metadata.get("encoding", encoding)
    
    # Try to get task from CSV if available
    csv_path = metadata_path.parent / "trajectories_metadata.csv"
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            first_row = next(reader, None)
            if first_row and "language_instruction" in first_row:
                task = first_row["language_instruction"]

    print(f"Loading Maniskill Push-T from: {root}")
    print(f"  Encoding: {encoding}, Task: {task}")

    tfrecords = sorted(root.glob("trajectory_*.tfrecord"))
    print(f"  Total trajectories: {len(tfrecords)}")

    # --- Pass 1: Read rewards to get min/max for normalization ---
    print("Pass 1: Reading rewards...")
    reward_info = []  # (path, total_reward, success)
    for path in tqdm.tqdm(tfrecords, desc="Reading rewards"):
        total_reward, success = _read_trajectory_rewards(str(path))
        reward_info.append((path, total_reward, success))

    all_rewards = [r for _, r, _ in reward_info]
    reward_min = min(all_rewards)
    reward_max = max(all_rewards)
    reward_range = reward_max - reward_min if reward_max > reward_min else 1.0
    print(f"  Reward range: [{reward_min:.3f}, {reward_max:.3f}]")
    print(f"  Successful: {sum(1 for _, _, s in reward_info if s)}/{len(reward_info)}")

    # --- Pass 2: Build trajectory dicts ---
    print("Pass 2: Building trajectory dicts...")
    trajectories = []
    for path, total_reward, success in reward_info:
        partial_success = (total_reward - reward_min) / reward_range

        traj = {
            "id": generate_unique_id(),
            "task": task,
            "frames": TFRecordFrameLoader(str(path), encoding),
            "is_robot": True,
            "quality_label": "successful" if success else "failure",
            "partial_success": float(partial_success),
            "data_source": "maniskill_pusht",
            "preference_group_id": None,
            "preference_rank": None,
        }
        trajectories.append(traj)

    task_data = {task: trajectories}
    print(f"  Built {len(trajectories)} trajectory dicts under task: '{task}'")
    return task_data

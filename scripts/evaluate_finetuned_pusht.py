#!/usr/bin/env python3
"""Evaluate finetuned Robometer on PushT TFRecord trajectories.

Identical to evaluate_robometer_zero_shot.py but runs inference live
using a finetuned (or base) model instead of reading pre-computed annotations.

Usage:
    uv run python scripts/evaluate_finetuned_pusht.py \
        --model-path /scratch/gpfs/FISAC/ac8755/robometer/logs/robometer4b_lora_pusht \
        --metadata-file /scratch/gpfs/FISAC/ac8755/open_dreams/pusht_eval_200/metadata.json \
        --tfrecord-dir /scratch/gpfs/FISAC/ac8755/open_dreams/pusht_eval_200 \
        --output-dir outputs/eval_finetuned/ \
        --index 0
"""

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


# Lazy TF import
_tf = None


def _get_tf():
    global _tf
    if _tf is None:
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
        _tf = tf
    return _tf


# =============================================================================
# Dataset Loading (from evaluate_robometer_zero_shot.py)
# =============================================================================


@dataclass
class DatasetInfo:
    file_paths: list[str]
    episode_lengths: list[int]
    encodings: list[str]
    observation_shapes: list[tuple[int, ...]]
    num_trajectories: int
    metadata: dict


def load_dataset_info(metadata_file: str, tfrecord_dir: str) -> DatasetInfo:
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


def read_trajectory(tfrecord_path: str, num_frames: int, encoding: str = "png") -> dict:
    """Read observations, rewards, and dones from a trajectory TFRecord."""
    tf = _get_tf()

    feature_spec = {
        "observation": tf.io.FixedLenFeature([], tf.string),
        "reward": tf.io.FixedLenFeature([], tf.float32),
        "done": tf.io.FixedLenFeature([], tf.int64),
    }

    frames = []
    rewards = []
    dones = []
    ds = tf.data.TFRecordDataset([tfrecord_path])
    for raw_record in ds.take(num_frames):
        parsed = tf.io.parse_single_example(raw_record, feature_spec)

        if encoding in ("jpeg", "jpg"):
            img = tf.io.decode_jpeg(parsed["observation"], channels=3)
        else:
            img = tf.io.decode_png(parsed["observation"], channels=3)
        frames.append(img.numpy())
        rewards.append(parsed["reward"].numpy())
        dones.append(parsed["done"].numpy())

    return {
        "frames": np.stack(frames).astype(np.uint8),
        "rewards": np.stack(rewards).astype(np.float32),
        "dones": np.stack(dones).astype(np.int64),
    }


# =============================================================================
# Model Loading & Inference (from annotate_open_dreams_tfrecords.py)
# =============================================================================


def load_robometer(model_path: str, device: torch.device):
    """Load Robometer model from HF or local checkpoint."""
    from robometer.utils.save import load_model_from_hf
    from robometer.utils.setup_utils import setup_batch_collator

    exp_config, tokenizer, processor, reward_model = load_model_from_hf(
        model_path=model_path, device=device,
    )
    reward_model.eval()
    batch_collator = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)
    return exp_config, tokenizer, processor, reward_model, batch_collator


def run_inference(
    frames: np.ndarray,
    task: str,
    reward_model,
    tokenizer,
    exp_config,
    batch_collator,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Robometer inference on frames. Returns (progress, success) arrays."""
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
    batch = batch_collator([ProgressSample(trajectory=traj, sample_type="progress")])

    progress_inputs = batch["progress_inputs"]
    for key, value in progress_inputs.items():
        if hasattr(value, "to"):
            progress_inputs[key] = value.to(device)

    loss_config = getattr(exp_config, "loss", None)
    is_discrete = (
        getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete"
        if loss_config else False
    )
    num_bins = getattr(loss_config, "progress_discrete_bins", None) or getattr(
        exp_config.model, "progress_discrete_bins", 10
    )

    with torch.no_grad():
        results = compute_batch_outputs(
            reward_model, tokenizer, progress_inputs,
            sample_type="progress", is_discrete_mode=is_discrete, num_bins=num_bins,
        )

    progress_pred = results.get("progress_pred", [])
    progress = (
        np.array(progress_pred[0], dtype=np.float32)
        if progress_pred and len(progress_pred) > 0
        else np.zeros(T, dtype=np.float32)
    )

    outputs_success = results.get("outputs_success", {})
    success_probs = outputs_success.get("success_probs", []) if outputs_success else []
    success = (
        np.array(success_probs[0], dtype=np.float32)
        if success_probs and len(success_probs) > 0
        else np.zeros(T, dtype=np.float32)
    )

    return progress, success


# =============================================================================
# Metrics
# =============================================================================


def normalize_dense_reward(rewards: np.ndarray) -> np.ndarray:
    """Normalize per-step dense rewards to [0, 1].

    ManiSkill PushT: reward is exactly 1.0 on success, [0, ~0.5] otherwise.
    """
    MAX_REWARD = 1.0
    return np.clip(rewards / MAX_REWARD, 0.0, 1.0)


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def monotonicity_score(progress: np.ndarray) -> float:
    if len(progress) < 2:
        return 1.0
    diffs = np.diff(progress)
    return float(np.mean(diffs >= -1e-6))


# =============================================================================
# Visualization
# =============================================================================


def render_debug_video(
    frames: np.ndarray,
    progress: np.ndarray,
    cum_reward: np.ndarray,
    output_path: str,
    title: str = "",
    fps: int = 10,
) -> None:
    """Render side-by-side video: observation (left) + growing reward plot (right)."""
    import imageio
    from PIL import Image

    T = min(len(frames), len(progress), len(cum_reward))
    obs_h, obs_w = frames.shape[1], frames.shape[2]

    writer = imageio.get_writer(output_path, fps=fps)

    for t in range(T):
        obs_img = frames[t]

        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        ax.plot(range(t + 1), progress[:t + 1], "b-", label="RBM Progress", linewidth=2)
        ax.plot(range(t + 1), cum_reward[:t + 1], "r--", label="Dense Reward (norm.)", linewidth=1.5)
        ax.plot(t, progress[t], "bo", markersize=6)
        ax.plot(t, cum_reward[t], "rs", markersize=5)
        ax.set_xlim(0, T)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Frame")
        ax.set_title(f"{title} — Frame {t}/{T}")
        if t == 0:
            ax.legend(fontsize=8)

        fig.canvas.draw()
        plot_img = np.array(fig.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)

        plot_pil = Image.fromarray(plot_img)
        plot_pil = plot_pil.resize(
            (int(obs_h * plot_img.shape[1] / plot_img.shape[0]), obs_h)
        )
        plot_resized = np.array(plot_pil)

        combined = np.concatenate([obs_img, plot_resized], axis=1)
        writer.append_data(combined)

    writer.close()
    print(f"Debug video saved to {output_path} ({T} frames, {fps} fps)")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate finetuned Robometer on PushT TFRecord trajectories.",
    )
    parser.add_argument("--model-path", required=True,
                        help="Path to finetuned checkpoint or HF model id (e.g., robometer/Robometer-4B)")
    parser.add_argument("--metadata-file", required=True,
                        help="Path to Open Dreams dataset metadata.json")
    parser.add_argument("--tfrecord-dir", required=True,
                        help="Directory containing trajectory TFRecord files")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for plots and videos")
    parser.add_argument("--index", type=int, default=None,
                        help="Single trajectory index to evaluate")
    parser.add_argument("--indices", type=int, nargs="+", default=None,
                        help="Multiple trajectory indices to evaluate with per-trajectory "
                             "plots+videos (e.g. --indices 0 1 2 5 10). Single model load.")
    parser.add_argument("--task", default=None,
                        help="Task instruction (default: read from CSV)")
    parser.add_argument("--num-trajs", type=int, default=None,
                        help="Max trajectories to evaluate (default: all)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset info
    info = load_dataset_info(args.metadata_file, args.tfrecord_dir)

    # Get task from CSV or CLI
    task = args.task
    if task is None:
        csv_path = Path(args.metadata_file).parent / "trajectories_metadata.csv"
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            first_row = next(reader, None)
            if first_row and "language_instruction" in first_row:
                task = first_row["language_instruction"]
        if task is None:
            task = "Push the T block to the goal position"

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    print(f"Model: {args.model_path}")
    exp_config, tokenizer, processor, reward_model, batch_collator = load_robometer(
        args.model_path, device,
    )

    if args.indices is not None:
        # Multi-index mode: per-trajectory plots+videos, single model load
        print(f"Evaluating {len(args.indices)} trajectories: {args.indices}")
        for idx in args.indices:
            eval_single(
                info, idx, task, reward_model, tokenizer, exp_config,
                batch_collator, device, output_dir,
            )
    elif args.index is not None:
        # Single trajectory mode
        eval_single(
            info, args.index, task, reward_model, tokenizer, exp_config,
            batch_collator, device, output_dir,
        )
    else:
        # All trajectories mode
        eval_all(
            info, task, reward_model, tokenizer, exp_config,
            batch_collator, device, output_dir,
            num_trajs=args.num_trajs,
        )


def eval_single(info, idx, task, reward_model, tokenizer, exp_config,
                batch_collator, device, output_dir):
    """Evaluate a single trajectory with static plot + debug video."""
    traj_path = info.file_paths[idx]
    num_frames = info.episode_lengths[idx]
    encoding = info.encodings[idx]

    print(f"Evaluating: {os.path.basename(traj_path)} ({num_frames} frames)")

    traj_data = read_trajectory(traj_path, num_frames, encoding)
    frames = traj_data["frames"]
    rewards = traj_data["rewards"]

    progress, success = run_inference(
        frames, task, reward_model, tokenizer, exp_config, batch_collator, device,
    )

    dense_reward = normalize_dense_reward(rewards)
    corr = pearson_correlation(progress, dense_reward)
    mono = monotonicity_score(progress)

    print(f"  Pearson: {corr:.3f}, Monotonicity: {mono:.3f}")

    # Static plot
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(progress))
    ax.plot(x, progress, "b-", label="RBM Progress", linewidth=2)
    ax.plot(x, dense_reward, "r--", label="Dense Reward (norm.)", linewidth=1.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Frame")
    ax.set_title(f"Pearson r = {corr:.3f}, Monotonicity = {mono:.3f}")
    ax.legend()
    plot_path = output_dir / f"pearson_{idx}.png"
    fig.savefig(str(plot_path), dpi=200)
    plt.close(fig)
    print(f"Plot saved to {plot_path}")

    # Debug video
    video_path = output_dir / f"debug_video_{idx}.mp4"
    render_debug_video(
        frames, progress, dense_reward,
        output_path=str(video_path),
        title=f"r={corr:.2f}",
    )


def eval_all(info, task, reward_model, tokenizer, exp_config,
             batch_collator, device, output_dir, num_trajs=None):
    """Evaluate all trajectories. Produce scatter plot, histogram, and summary."""
    import time

    n = num_trajs if num_trajs else info.num_trajectories
    n = min(n, info.num_trajectories)

    print(f"Evaluating {n} trajectories...")

    correlations = []
    monotonicities = []
    gt_total_rewards = []
    pred_final_progress = []
    t_start = time.time()

    for idx in range(n):
        traj_path = info.file_paths[idx]
        num_frames = info.episode_lengths[idx]
        encoding = info.encodings[idx]

        if not os.path.exists(traj_path):
            print(f"  [{idx}] Missing: {traj_path}, skipping")
            continue

        traj_data = read_trajectory(traj_path, num_frames, encoding)
        frames = traj_data["frames"]
        rewards = traj_data["rewards"]

        progress, success = run_inference(
            frames, task, reward_model, tokenizer, exp_config, batch_collator, device,
        )

        dense_reward = normalize_dense_reward(rewards)
        corr = pearson_correlation(progress, dense_reward)
        mono = monotonicity_score(progress)

        correlations.append(corr)
        monotonicities.append(mono)
        gt_total_rewards.append(float(dense_reward[-1]))
        pred_final_progress.append(float(progress[-1]) if len(progress) > 0 else 0.0)

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t_start
            print(f"  [{idx+1}/{n}] avg Pearson={np.mean(correlations):.3f}, "
                  f"{elapsed:.0f}s elapsed")

    correlations = np.array(correlations)
    monotonicities = np.array(monotonicities)
    gt_total_rewards = np.array(gt_total_rewards)
    pred_final_progress = np.array(pred_final_progress)

    elapsed = time.time() - t_start

    # Print summary
    print(f"\n{'='*60}")
    print(f"  EVALUATION SUMMARY ({len(correlations)} trajectories, {elapsed:.0f}s)")
    print(f"  Pearson correlation:  mean={correlations.mean():.3f}, "
          f"median={np.median(correlations):.3f}, std={correlations.std():.3f}")
    print(f"  Monotonicity:         mean={monotonicities.mean():.3f}, "
          f"median={np.median(monotonicities):.3f}, std={monotonicities.std():.3f}")
    print(f"{'='*60}")

    # Scatter plot: ground truth final reward (normalized) vs predicted final progress
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(gt_total_rewards, pred_final_progress, alpha=0.5, s=20)
    ax.set_xlabel("Ground Truth Final Reward (norm.)")
    ax.set_ylabel("Predicted Final Progress")
    ax.set_title(f"Final Reward vs Progress (n={len(correlations)})\n"
                 f"Pearson r={pearson_correlation(gt_total_rewards, pred_final_progress):.3f}")
    # Add diagonal reference
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.set_ylim(-0.05, 1.05)
    scatter_path = output_dir / "scatter_final_reward_vs_progress.png"
    fig.savefig(str(scatter_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Scatter plot saved to {scatter_path}")

    # Histogram of per-trajectory Pearson correlations
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(correlations, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(correlations.mean(), color="r", linestyle="--",
               label=f"Mean={correlations.mean():.3f}")
    ax.axvline(np.median(correlations), color="g", linestyle="--",
               label=f"Median={np.median(correlations):.3f}")
    ax.set_xlabel("Pearson Correlation")
    ax.set_ylabel("Count")
    ax.set_title(f"Per-Trajectory Pearson Correlation (n={len(correlations)})")
    ax.legend()
    hist_path = output_dir / "histogram_pearson.png"
    fig.savefig(str(hist_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Histogram saved to {hist_path}")

    # Save summary JSON
    summary = {
        "model_path": str(output_dir),
        "num_trajectories": len(correlations),
        "elapsed_s": elapsed,
        "pearson": {
            "mean": float(correlations.mean()),
            "median": float(np.median(correlations)),
            "std": float(correlations.std()),
            "min": float(correlations.min()),
            "max": float(correlations.max()),
        },
        "monotonicity": {
            "mean": float(monotonicities.mean()),
            "median": float(np.median(monotonicities)),
            "std": float(monotonicities.std()),
        },
        "per_trajectory": [
            {"index": i, "pearson": float(correlations[i]),
             "monotonicity": float(monotonicities[i]),
             "final_reward_norm": float(gt_total_rewards[i]),
             "final_progress": float(pred_final_progress[i])}
            for i in range(len(correlations))
        ],
    }
    summary_path = output_dir / "summary_all.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()

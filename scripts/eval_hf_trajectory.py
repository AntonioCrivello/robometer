#!/usr/bin/env python3
"""Evaluate Robometer on a single trajectory from a HuggingFace dataset.

Loads one trajectory from a HuggingFace dataset, runs Robometer inference,
and produces:
  - Static plot: per-frame progress + success curves
  - Debug video: observation (left) + growing progress curve (right)

Usage:
    uv run python scripts/eval_hf_trajectory.py \
        --model-path robometer/Robometer-4B \
        --dataset teetone/RoboReward \
        --split train \
        --index 0 \
        --output-dir outputs/eval_roboreward/
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import cv2


def load_hf_trajectory(
    dataset_name: str,
    split: str = "train",
    subset: str | None = None,
    index: int = 0,
) -> dict:
    """Load a single trajectory from a HuggingFace dataset.

    Returns dict with 'frames' (uint8 ndarray), 'task', 'quality_label',
    'partial_success', and any other metadata.
    """
    from datasets import load_dataset

    kwargs = {"split": split}
    if subset:
        kwargs["name"] = subset

    ds = load_dataset(dataset_name, streaming=True, **kwargs)

    # Skip to the desired index
    sample = None
    for i, s in enumerate(ds):
        if i == index:
            sample = s
            break

    if sample is None:
        raise ValueError(f"Could not find sample at index {index}")

    # Extract frames from video field
    frames_field = sample.get("frames") or sample.get("video")
    if frames_field is None:
        raise ValueError(f"No 'frames' or 'video' field in sample. Keys: {list(sample.keys())}")

    # frames_field could be a path, bytes, or dict with 'path'/'bytes'
    if isinstance(frames_field, dict):
        video_bytes = frames_field.get("bytes")
        video_path = frames_field.get("path")
    elif isinstance(frames_field, (str, Path)):
        video_path = str(frames_field)
        video_bytes = None
    elif isinstance(frames_field, bytes):
        video_bytes = frames_field
        video_path = None
    else:
        raise ValueError(f"Unexpected frames field type: {type(frames_field)}")

    if video_bytes:
        # Write to temp file for decord/imageio to read
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            video_path = f.name

    cap = cv2.VideoCapture(video_path)
    frame_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frame_list:
        raise RuntimeError(f"Could not extract frames from {video_path}")
    frames = np.array(frame_list, dtype=np.uint8)

    # Clean up temp file
    if video_bytes and os.path.exists(video_path):
        os.unlink(video_path)

    return {
        "frames": frames,
        "task": sample.get("task", "Unknown task"),
        "quality_label": sample.get("quality_label", "unknown"),
        "partial_success": sample.get("partial_success", None),
        "data_source": sample.get("data_source", dataset_name),
        "id": sample.get("id", str(index)),
    }


def run_inference(
    model_path: str,
    frames: np.ndarray,
    task: str,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Robometer inference on frames. Returns (progress, success) arrays."""
    from robometer.data.dataset_types import ProgressSample, Trajectory
    from robometer.evals.eval_server import compute_batch_outputs
    from robometer.utils.save import load_model_from_hf
    from robometer.utils.setup_utils import setup_batch_collator

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_config, tokenizer, processor, reward_model = load_model_from_hf(
        model_path=model_path, device=device,
    )
    reward_model.eval()
    batch_collator = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)

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


def render_debug_video(
    frames: np.ndarray,
    progress: np.ndarray,
    success: np.ndarray,
    output_path: str,
    title: str = "",
    fps: int = 5,
) -> None:
    """Render side-by-side video: observation (left) + growing reward plot (right)."""
    from PIL import Image

    T = min(len(frames), len(progress))
    obs_h, obs_w = frames.shape[1], frames.shape[2]

    writer = imageio.get_writer(output_path, fps=fps)

    for t in range(T):
        obs_img = frames[t]

        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        ax.plot(range(t + 1), progress[:t + 1], "b-", label="Progress", linewidth=2)
        ax.plot(range(t + 1), success[:t + 1], "g--", label="Success", linewidth=1.5)
        ax.plot(t, progress[t], "bo", markersize=6)
        ax.plot(t, success[t], "gs", markersize=5)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Robometer on a single HuggingFace dataset trajectory.",
    )
    parser.add_argument("--model-path", required=True, help="Robometer model path (HF id or local)")
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset name (e.g., teetone/RoboReward)")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--subset", default=None, help="Dataset subset/config name")
    parser.add_argument("--index", type=int, default=0, help="Trajectory index (default: 0)")
    parser.add_argument("--output-dir", required=True, help="Output directory for plots and video")
    parser.add_argument("--fps", type=int, default=5, help="Debug video FPS (default: 5)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load trajectory from HuggingFace
    print(f"Loading trajectory {args.index} from {args.dataset} ({args.split})...")
    traj = load_hf_trajectory(
        args.dataset, split=args.split, subset=args.subset, index=args.index,
    )
    frames = traj["frames"]
    task = traj["task"]
    print(f"  Task: {task}")
    print(f"  Frames: {frames.shape}")
    print(f"  Quality: {traj['quality_label']}, partial_success: {traj['partial_success']}")

    # Run Robometer inference
    print(f"Running Robometer inference ({args.model_path})...")
    progress, success = run_inference(args.model_path, frames, task)
    print(f"  Progress range: [{progress.min():.3f}, {progress.max():.3f}]")
    print(f"  Success range:  [{success.min():.3f}, {success.max():.3f}]")

    # Save arrays
    np.save(str(output_dir / "progress.npy"), progress)
    np.save(str(output_dir / "success.npy"), success)

    # Static plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    x = np.arange(len(progress))
    ax1.plot(x, progress, "b-", linewidth=2)
    ax1.set_ylabel("Progress")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title(f"{task}\nQuality: {traj['quality_label']}, "
                  f"partial_success: {traj['partial_success']}")

    ax2.plot(x, success, "g-", linewidth=2)
    ax2.set_ylabel("Success Prob")
    ax2.set_xlabel("Frame")
    ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plot_path = output_dir / "progress_success.png"
    fig.savefig(str(plot_path), dpi=200)
    plt.close(fig)
    print(f"Static plot saved to {plot_path}")

    # Debug video
    video_path = output_dir / "debug_video.mp4"
    render_debug_video(
        frames, progress, success,
        output_path=str(video_path),
        title=task[:40],
        fps=args.fps,
    )

    # Summary
    summary = {
        "dataset": args.dataset,
        "index": args.index,
        "task": task,
        "quality_label": traj["quality_label"],
        "partial_success": traj["partial_success"],
        "num_frames": int(frames.shape[0]),
        "progress_min": float(progress.min()),
        "progress_max": float(progress.max()),
        "progress_mean": float(progress.mean()),
        "success_mean": float(success.mean()),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

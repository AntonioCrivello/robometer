#!/usr/bin/env python3
"""Compare zero-shot vs finetuned Robometer on video trajectories.

Loads each model once, runs inference on discovered videos, and produces
per-video comparison plots showing progress + success curves side by side.

Usage (RoboFAC):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/compare_zeroshot_vs_finetuned.py \
        --base-model robometer/Robometer-4B \
        --finetuned-model /path/to/logs/robometer4b_lora_robofac/robometer4b_lora_robofac \
        --video-dir /path/to/RoboFAC-dataset/realworld_data \
        --output-dir outputs/compare_robofac/ \
        --num-videos 10

Usage (single videos):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/compare_zeroshot_vs_finetuned.py \
        --base-model robometer/Robometer-4B \
        --finetuned-model /path/to/checkpoint \
        --videos /path/to/video1.mp4 /path/to/video2.mp4 \
        --tasks "Insert the cylinder" "Pick up the cube" \
        --output-dir outputs/compare/
"""

from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from robometer.data.dataset_types import ProgressSample, Trajectory
from robometer.evals.eval_server import compute_batch_outputs
from robometer.evals.eval_viz_utils import extract_frames
from robometer.utils.save import load_model_from_hf
from robometer.utils.setup_utils import setup_batch_collator


# Task name mapping from RoboFAC loader (for auto-discovery mode)
TASK_NAME_TO_DESCRIPTION: dict[str, str] = {
    "SpinStack": "Pick up the cube on the spinning disc and stack it on another cube on the disc.",
    "SpinPullStack": "Pull out the cube on the spinning disc and stack it on another cube on the disc.",
    "MicrowaveTask": "Put the spoon on the table into the cup. Open the door of microwave, put the cup into the microwave and close the door.",
    "SafeTask": "Put the gold bar into the safe, close the door of the safe and rotate the cross knob on the door to lock it.",
    "ToolsTask": "Choose the correct (L-shaped) tools, grasp it to pull the correct (2-pins) charger and plug it.",
    "UprightStask": "Upright the peg and stack it on the cube.",
    "PegInsetionSide": "Insert the peg into the hole on the side of the block.",
    "PullCubeTool": "Grasp the L-shaped tool and pull the cube by it.",
    "PlugCharger": "Grasp the charger and plug it into the receptacle.",
    "InsertCylinder": "Upright the cylinder and insert it into the middle hole on the shelf.",
    "PlaceCube": "Pick up the cube and place it into the box.",
    "LiftPegUpright": "Lift the peg and upright it.",
    "PickCube": "Pick the cube to the target position.",
    "PullCube": "Pull the cube to the red and white target.",
    "PushCube": "Push the cube to the red and white target.",
    "StackCube": "Pick up the cube and stack it on another cube.",
}


def _snake_to_camel(snake: str) -> str:
    return "".join(word.capitalize() for word in snake.split("_") if word)


def _folder_to_task(folder_name: str) -> str:
    """Convert RoboFAC folder name to task description."""
    name = folder_name.replace("so100_", "").replace("_error", "").strip("_")
    task_key = _snake_to_camel(name)
    return TASK_NAME_TO_DESCRIPTION.get(task_key, name.replace("_", " ").title())


def discover_videos(
    video_dir: Path, num_videos: int | None = None
) -> list[tuple[Path, str]]:
    """Discover videos from a RoboFAC-style directory structure.

    Expects: video_dir/<task_folder>/videos/*.mp4
    Returns: list of (video_path, task_description)
    """
    videos: list[tuple[Path, str]] = []
    for task_dir in sorted(video_dir.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith("."):
            continue
        task_desc = _folder_to_task(task_dir.name)
        for mp4 in sorted(task_dir.rglob("*.mp4")):
            videos.append((mp4, task_desc))

    if num_videos and len(videos) > num_videos:
        # Sample evenly across tasks
        step = max(1, len(videos) // num_videos)
        videos = videos[::step][:num_videos]

    return videos


def load_model(model_path: str, device: torch.device):
    """Load model, processor, collator. Returns tuple for inference."""
    exp_config, tokenizer, processor, reward_model = load_model_from_hf(
        model_path=model_path, device=device,
    )
    reward_model.eval()
    batch_collator = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)

    loss_config = getattr(exp_config, "loss", None)
    is_discrete = (
        getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete"
        if loss_config else False
    )
    num_bins = (
        getattr(loss_config, "progress_discrete_bins", None)
        or getattr(exp_config.model, "progress_discrete_bins", 10)
    )

    return reward_model, tokenizer, batch_collator, is_discrete, num_bins


def run_inference(
    frames: np.ndarray,
    task: str,
    reward_model,
    tokenizer,
    batch_collator,
    is_discrete: bool,
    num_bins: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on frames. Returns (progress, success) arrays."""
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


def run_all_videos(
    model_path: str,
    videos: list[tuple[Path, str]],
    device: torch.device,
    fps: float = 1.0,
    max_frames: int = 512,
) -> list[dict]:
    """Load model once, run inference on all videos, return results."""
    print(f"\nLoading model: {model_path}")
    reward_model, tokenizer, batch_collator, is_discrete, num_bins = load_model(
        model_path, device
    )

    results = []
    for i, (video_path, task) in enumerate(videos):
        print(f"  [{i+1}/{len(videos)}] {video_path.name} — {task[:50]}")
        try:
            frames = extract_frames(str(video_path), fps=fps, max_frames=max_frames)
            if frames is None or frames.size == 0:
                print(f"    Skipping: could not extract frames")
                continue
            if frames.dtype != np.uint8:
                frames = np.clip(frames, 0, 255).astype(np.uint8)

            progress, success = run_inference(
                frames, task, reward_model, tokenizer, batch_collator,
                is_discrete, num_bins, device,
            )
            results.append({
                "video_path": str(video_path),
                "task": task,
                "num_frames": int(frames.shape[0]),
                "progress": progress,
                "success": success,
            })
        except Exception as e:
            print(f"    Error: {e}")
            continue

    # Unload model
    del reward_model, tokenizer, batch_collator
    torch.cuda.empty_cache()
    gc.collect()

    return results


def plot_comparison(
    base_result: dict,
    finetuned_result: dict,
    output_path: Path,
) -> None:
    """Plot side-by-side progress + success curves for base vs finetuned."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, result, label in [
        (axes[0], base_result, "Zero-shot"),
        (axes[1], finetuned_result, "Finetuned"),
    ]:
        T = result["num_frames"]
        x = np.arange(T)
        progress = result["progress"][:T]
        success = result["success"][:T]

        ax.plot(x, progress, "b-", linewidth=2, label="Progress")
        if success.size == progress.size:
            ax.plot(x, success, "r--", linewidth=1.5, alpha=0.7, label="Success prob")
        ax.set_xlim(0, T)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Frame")
        ax.set_title(label)
        ax.legend(fontsize=8)

    task_short = base_result["task"][:60]
    video_name = Path(base_result["video_path"]).stem
    fig.suptitle(f"{video_name}\n{task_short}", fontsize=10)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare zero-shot vs finetuned Robometer on video trajectories."
    )
    parser.add_argument("--base-model", required=True,
                        help="Base model path (e.g. robometer/Robometer-4B)")
    parser.add_argument("--finetuned-model", required=True,
                        help="Finetuned model checkpoint path")
    parser.add_argument("--video-dir", default=None,
                        help="Directory with RoboFAC-style task folders (auto-discovers videos)")
    parser.add_argument("--videos", nargs="+", default=None,
                        help="Explicit list of video paths")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Task descriptions (one per video, required with --videos)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-videos", type=int, default=10,
                        help="Max videos to evaluate (default: 10)")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="FPS for frame extraction (default: 1.0)")
    parser.add_argument("--max-frames", type=int, default=512,
                        help="Max frames per video (default: 512)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover or use explicit videos
    if args.videos:
        if args.tasks and len(args.tasks) != len(args.videos):
            parser.error("--tasks must have same length as --videos")
        tasks = args.tasks or ["Perform the task"] * len(args.videos)
        videos = [(Path(v), t) for v, t in zip(args.videos, tasks)]
    elif args.video_dir:
        videos = discover_videos(Path(args.video_dir), args.num_videos)
    else:
        parser.error("Provide either --video-dir or --videos")

    print(f"Found {len(videos)} videos to evaluate")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run base model on all videos, then unload
    base_results = run_all_videos(
        args.base_model, videos, device, args.fps, args.max_frames,
    )

    # Run finetuned model on all videos, then unload
    finetuned_results = run_all_videos(
        args.finetuned_model, videos, device, args.fps, args.max_frames,
    )

    # Match results by video path and plot
    finetuned_by_path = {r["video_path"]: r for r in finetuned_results}
    summary = []

    for base_r in base_results:
        vpath = base_r["video_path"]
        ft_r = finetuned_by_path.get(vpath)
        if ft_r is None:
            continue

        video_name = Path(vpath).stem
        plot_path = output_dir / f"compare_{video_name}.png"
        plot_comparison(base_r, ft_r, plot_path)
        print(f"  Saved: {plot_path}")

        summary.append({
            "video": vpath,
            "task": base_r["task"],
            "num_frames": base_r["num_frames"],
            "base_progress_mean": float(np.mean(base_r["progress"])),
            "base_progress_final": float(base_r["progress"][-1]) if len(base_r["progress"]) > 0 else None,
            "finetuned_progress_mean": float(np.mean(ft_r["progress"])),
            "finetuned_progress_final": float(ft_r["progress"][-1]) if len(ft_r["progress"]) > 0 else None,
        })

    # Save summary
    summary_path = output_dir / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    print(f"Compared {len(summary)} videos")


if __name__ == "__main__":
    main()

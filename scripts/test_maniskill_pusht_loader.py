#!/usr/bin/env python3
"""Quick test to verify the ManiSkill Push-T loader works before running full conversion.

Usage:
    uv run python scripts/test_maniskill_pusht_loader.py \
        --dataset-path /scratch/gpfs/FISAC/ac8755/open_dreams/pusht_eval_200
"""

import argparse
import time

from dataset_upload.dataset_loaders.maniskill_pusht_sim_loader import load_maniskill_pusht_sim_dataset
from dataset_upload.helpers import flatten_task_data


def main():
    parser = argparse.ArgumentParser(description="Test ManiSkill Push-T loader")
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to directory with metadata.json + trajectory_*.tfrecord files",
    )
    parser.add_argument(
        "--num-test-frames",
        type=int,
        default=3,
        help="Number of trajectories to test frame loading on (default: 3)",
    )
    args = parser.parse_args()

    # Test loader
    print("=" * 60)
    print("  TESTING MANISKILL PUSH-T LOADER")
    print("=" * 60)

    t_start = time.time()
    task_data = load_maniskill_pusht_sim_dataset(args.dataset_path, "maniskill_pusht_sim")
    trajectories = flatten_task_data(task_data)
    t_load = time.time() - t_start

    print(f"\nLoaded {len(trajectories)} trajectories in {t_load:.1f}s")
    print(f"Tasks: {list(task_data.keys())}")

    # Check structure
    sample = trajectories[0]
    print(f"\nSample trajectory keys: {sorted(sample.keys())}")
    print(f"  task: {sample['task']}")
    print(f"  quality_label: {sample['quality_label']}")
    print(f"  partial_success: {sample['partial_success']:.4f}")
    print(f"  data_source: {sample['data_source']}")
    print(f"  is_robot: {sample['is_robot']}")
    print(f"  frames type: {type(sample['frames']).__name__}")

    # Distribution of partial_success
    successes = [t["partial_success"] for t in trajectories]
    num_successful = sum(1 for t in trajectories if t["quality_label"] == "successful")
    print(f"\nPartial success distribution:")
    print(f"  min:  {min(successes):.4f}")
    print(f"  max:  {max(successes):.4f}")
    print(f"  mean: {sum(successes)/len(successes):.4f}")
    print(f"  successful: {num_successful}/{len(trajectories)}")

    # Test lazy frame loading on a few trajectories
    print(f"\nTesting frame loading on {args.num_test_frames} trajectories...")
    for i in range(min(args.num_test_frames, len(trajectories))):
        t = time.time()
        frames = trajectories[i]["frames"]()
        elapsed = time.time() - t
        print(
            f"  [{i}] shape={frames.shape}, dtype={frames.dtype}, "
            f"partial_success={trajectories[i]['partial_success']:.3f}, "
            f"quality={trajectories[i]['quality_label']}, "
            f"load_time={elapsed:.2f}s"
        )

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()

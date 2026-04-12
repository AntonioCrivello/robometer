#!/usr/bin/env python3
"""Phase 1: Become One with the Data.

Inspects raw TFRecords, generated HF dataset, preprocessed cache, and live
batch collation. Run incrementally as you move through the data pipeline.

Usage:
    # 1.1: Inspect raw TFRecords (before generate_hf_dataset)
    uv run python scripts/debug_phase1_inspect_data.py raw \
        --tfrecord-dir /scratch/gpfs/FISAC/ac8755/open_dreams/maniskill_pusht_wide_recovery_5k \
        --num-trajectories 32

    # 1.2: Inspect generated HF dataset (after generate_hf_dataset)
    uv run python scripts/debug_phase1_inspect_data.py hf \
        --dataset-dir /scratch/gpfs/FISAC/ac8755/robometer/robometer_data/maniskill_pusht_sim_32_overfit/maniskill_pusht_sim_overfit/default

    # 1.3: Inspect preprocessed cache (after preprocess_datasets)
    uv run python scripts/debug_phase1_inspect_data.py preprocessed \
        --cache-dir /scratch/gpfs/FISAC/ac8755/robometer/processed_datasets \
        --dataset-path "/scratch/gpfs/FISAC/ac8755/robometer/robometer_data/maniskill_pusht_sim_32_overfit/maniskill_pusht_sim_overfit/default"

    # 1.4: Inspect live training batches (after preprocessing, before training)
    export ROBOMETER_PROCESSED_DATASETS_PATH=/scratch/gpfs/FISAC/ac8755/robometer/processed_datasets
    CUDA_VISIBLE_DEVICES="" uv run python scripts/debug_phase1_inspect_data.py batch \
        --config-name train_maniskill_pusht_lora_overfit \
        --num-batches 3
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def banner(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def verdict(ok: bool, msg: str) -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {msg}")


# ─────────────────────────────────────────────────────────────────────
# 1.1  Raw TFRecords
# ─────────────────────────────────────────────────────────────────────
def inspect_raw(args):
    """Inspect raw TFRecords: per-frame rewards, frame quality, success split."""
    banner("PHASE 1.1 — Inspect Raw TFRecords")

    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")

    root = Path(args.tfrecord_dir)
    tfrecords = sorted(root.glob("trajectory_*.tfrecord"))
    print(f"Found {len(tfrecords)} TFRecord files in {root}")

    if len(tfrecords) == 0:
        print("ERROR: No trajectory_*.tfrecord files found!")
        sys.exit(1)

    # Only inspect up to --num-trajectories (first N from sorted order,
    # matching the generate_hf_dataset trajectory_offset=0 behaviour)
    n_total = min(args.num_trajectories, len(tfrecords))
    tfrecords = tfrecords[:n_total]
    print(f"Inspecting first {n_total} trajectories (matching overfit subset)\n")

    # Read metadata
    metadata_path = root / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        print(f"Metadata: {json.dumps(metadata, indent=2)}")
    else:
        print("WARNING: No metadata.json found")

    # Per-trajectory stats
    all_last_rewards = []
    all_num_frames = []
    all_per_frame_rewards = []  # list of arrays, one per trajectory
    quality_counts = {"successful": 0, "failure": 0}
    short_trajectories = []  # < 3 frames

    feature_spec = {
        "reward": tf.io.FixedLenFeature([], tf.float32),
        "observation": tf.io.FixedLenFeature([], tf.string),
    }

    for i, path in enumerate(tfrecords):
        rewards = []
        first_frame = None
        last_frame = None

        ds = tf.data.TFRecordDataset([str(path)])
        for raw_record in ds:
            parsed = tf.io.parse_single_example(raw_record, feature_spec)
            rewards.append(float(parsed["reward"].numpy()))

            # Decode frames for detailed inspection
            if i < 5 or len(rewards) == 1:
                img = tf.io.decode_image(parsed["observation"], channels=3).numpy()
                if first_frame is None:
                    first_frame = img
                last_frame = img

        num_frames = len(rewards)
        last_reward = rewards[-1] if rewards else 0.0
        rewards_arr = np.array(rewards)

        all_last_rewards.append(last_reward)
        all_num_frames.append(num_frames)
        all_per_frame_rewards.append(rewards_arr)

        if num_frames < 3:
            short_trajectories.append((i, path.name, num_frames))

        success = last_reward >= 0.3
        quality_counts["successful" if success else "failure"] += 1

        # Detailed output for first 5 trajectories
        if i < 5:
            print(f"  Trajectory {i:3d}: {path.name}")
            print(f"    Frames: {num_frames}")
            print(f"    Reward range: [{rewards_arr.min():.4f}, {rewards_arr.max():.4f}]")
            print(f"    Quality: {'successful' if success else 'failure'}")
            print(f"    partial_success would be: {np.clip(last_reward, 0.0, 1.0):.4f}"
                  f" (None for successful)" if success else
                  f"    partial_success would be: {np.clip(last_reward, 0.0, 1.0):.4f}")

            # Sampled reward curve
            sample_idx = np.linspace(0, num_frames - 1, min(10, num_frames), dtype=int)
            curve_str = " -> ".join(f"{rewards_arr[j]:.3f}" for j in sample_idx)
            print(f"    Reward curve ({len(sample_idx)} pts): {curve_str}")

            # ASCII sparkline
            if num_frames > 1:
                width = 40
                bins_idx = np.linspace(0, num_frames - 1, width, dtype=int)
                binned = rewards_arr[bins_idx]
                r_min, r_max = rewards_arr.min(), rewards_arr.max()
                blocks = " ▁▂▃▄▅▆▇█"
                if r_max - r_min < 1e-6:
                    sparkline = blocks[-1] * width
                else:
                    sparkline = "".join(
                        blocks[int((v - r_min) / (r_max - r_min) * 8)]
                        for v in binned
                    )
                print(f"    Sparkline: |{sparkline}| [{r_min:.3f}..{r_max:.3f}]")

            # Frame pixel sanity
            if first_frame is not None:
                print(f"    Frame shape: {first_frame.shape}, dtype: {first_frame.dtype}")
                print(f"    First frame pixels: [{first_frame.min()}, {first_frame.max()}]")
                is_black = first_frame.max() < 5
                is_white = first_frame.min() > 250
                is_constant = first_frame.std() < 1.0
                if is_black:
                    print(f"    *** ALL-BLACK FRAME — CORRUPTED ***")
                elif is_white:
                    print(f"    *** ALL-WHITE FRAME — SUSPICIOUS ***")
                elif is_constant:
                    print(f"    *** CONSTANT FRAME (std<1) — SUSPICIOUS ***")
            print()

    # ── Summary ──
    banner("RAW DATA SUMMARY")
    rewards_arr = np.array(all_last_rewards)
    frames_arr = np.array(all_num_frames)

    print(f"Trajectories inspected: {n_total}")
    print(f"Quality split: successful={quality_counts['successful']}, "
          f"failure={quality_counts['failure']}")

    print(f"\nFrame counts: min={frames_arr.min()}, max={frames_arr.max()}, "
          f"mean={frames_arr.mean():.1f}, median={np.median(frames_arr):.0f}")

    print(f"\nLast-frame reward: min={rewards_arr.min():.4f}, max={rewards_arr.max():.4f}, "
          f"mean={rewards_arr.mean():.4f}, std={rewards_arr.std():.4f}")

    # Histogram
    bins = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]
    counts, _ = np.histogram(rewards_arr, bins=bins)
    print(f"\n  Last-reward histogram:")
    for j in range(len(counts)):
        bar = "#" * min(counts[j] * 40 // max(n_total, 1), 40)
        print(f"    [{bins[j]:5.1f}, {bins[j+1]:5.1f}): {counts[j]:4d}  {bar}")

    # ── Checks ──
    banner("CHECKS FOR TRAINING READINESS")

    # Check 1: Both quality labels present (required for preference sampling)
    verdict(
        quality_counts["successful"] > 0 and quality_counts["failure"] > 0,
        f"Both quality labels present: {quality_counts['successful']} successful, "
        f"{quality_counts['failure']} failure. "
        f"Preference sampling needs BOTH."
    )

    # Check 2: Enough of each for preference pairing
    min_per_class = min(quality_counts["successful"], quality_counts["failure"])
    verdict(
        min_per_class >= 2,
        f"Minority class has {min_per_class} trajectories "
        f"(need >=2 for diverse preference pairs)."
    )

    # Check 3: No trajectories with < 3 frames (subsampling requires >= 3)
    verdict(
        len(short_trajectories) == 0,
        f"All trajectories have >=3 frames. "
        f"{'Short: ' + str(short_trajectories) if short_trajectories else ''}"
    )

    # Check 4: partial_success gap for preference pairing
    # The pref sampler has a partial_success_threshold (default 0.2) —
    # chosen and rejected must differ by at least this much
    failure_ps = [np.clip(r, 0.0, 1.0) for r, q in
                  zip(all_last_rewards, [last_reward >= 0.3 for last_reward in all_last_rewards])
                  if not q]
    success_ps = [np.clip(r, 0.0, 1.0) for r, q in
                  zip(all_last_rewards, [last_reward >= 0.3 for last_reward in all_last_rewards])
                  if q]
    if failure_ps and success_ps:
        min_gap = min(s for s in success_ps) - max(f for f in failure_ps)
        verdict(
            min_gap >= 0.2,
            f"Partial success gap between best failure ({max(failure_ps):.3f}) "
            f"and worst success ({min(success_ps):.3f}): {min_gap:.3f} "
            f"(config partial_success_threshold=0.2)."
        )

    # Check 5: Reward values not degenerate
    all_rewards_flat = np.concatenate(all_per_frame_rewards)
    verdict(
        all_rewards_flat.std() > 0.01,
        f"Per-frame reward std={all_rewards_flat.std():.4f}. "
        f"If near zero, all frames have same reward (useless signal)."
    )

    # Check 6: No NaN/Inf in rewards
    n_nan = np.isnan(all_rewards_flat).sum()
    n_inf = np.isinf(all_rewards_flat).sum()
    verdict(
        n_nan == 0 and n_inf == 0,
        f"NaN rewards: {n_nan}, Inf rewards: {n_inf}."
    )


# ─────────────────────────────────────────────────────────────────────
# 1.2  HF Dataset
# ─────────────────────────────────────────────────────────────────────
def inspect_hf(args):
    """Inspect the generated HuggingFace dataset."""
    banner("PHASE 1.2 — Inspect Generated HF Dataset")

    from datasets import Dataset

    dataset_dir = args.dataset_dir
    print(f"Loading from: {dataset_dir}")

    ds = Dataset.load_from_disk(dataset_dir)
    print(f"Dataset: {ds}")
    print(f"Num rows: {len(ds)}")
    print(f"Features: {list(ds.features.keys())}")
    print(f"Feature types: {ds.features}")

    quality_labels = ds["quality_label"]
    quality_counts = Counter(quality_labels)
    print(f"\nQuality label distribution: {dict(quality_counts)}")

    # partial_success distribution
    partial_success = ds["partial_success"]
    valid_ps = [p for p in partial_success if p is not None]
    none_ps = len(partial_success) - len(valid_ps)
    print(f"\nPartial success: {len(valid_ps)} with values, {none_ps} None")
    if valid_ps:
        ps_arr = np.array(valid_ps)
        print(f"  min={ps_arr.min():.4f}, max={ps_arr.max():.4f}, mean={ps_arr.mean():.4f}")

    # data_source
    data_sources = ds["data_source"]
    source_counts = Counter(data_sources)
    print(f"\nData sources: {dict(source_counts)}")

    # task distribution
    tasks = ds["task"]
    task_counts = Counter(tasks)
    print(f"Tasks: {dict(task_counts)}")

    # Inspect a few entries + video file checks
    print(f"\nSample entries:")
    missing_videos = 0
    for i in range(min(5, len(ds))):
        row = ds[i]
        print(f"  [{i}] quality={row['quality_label']}, "
              f"partial_success={row['partial_success']}, "
              f"data_source={row['data_source']}")

        if isinstance(row["frames"], str):
            video_path = row["frames"]
            if not os.path.isabs(video_path):
                video_path = os.path.join(os.path.dirname(dataset_dir), video_path)
            exists = os.path.exists(video_path)
            if exists:
                size = os.path.getsize(video_path)
                print(f"       video: {row['frames']} ({size:,} bytes)")
            else:
                print(f"       video: {row['frames']} — MISSING!")
                missing_videos += 1

    # Check ALL video files exist
    print(f"\nChecking all video files...")
    for i in range(len(ds)):
        row = ds[i]
        if isinstance(row["frames"], str):
            video_path = row["frames"]
            if not os.path.isabs(video_path):
                video_path = os.path.join(os.path.dirname(dataset_dir), video_path)
            if not os.path.exists(video_path):
                missing_videos += 1

    banner("CHECKS")

    verdict(
        quality_counts.get("successful", 0) > 0 and quality_counts.get("failure", 0) > 0,
        f"Both quality labels: successful={quality_counts.get('successful', 0)}, "
        f"failure={quality_counts.get('failure', 0)}."
    )

    # partial_success: successful trajectories should have None,
    # failure trajectories should have a float
    successful_with_ps = sum(
        1 for q, p in zip(quality_labels, partial_success)
        if q == "successful" and p is not None
    )
    failure_without_ps = sum(
        1 for q, p in zip(quality_labels, partial_success)
        if q == "failure" and p is None
    )
    verdict(
        successful_with_ps == 0,
        f"Successful trajectories with partial_success!=None: {successful_with_ps} "
        f"(should be 0 — loader sets None for successful)."
    )
    verdict(
        failure_without_ps == 0,
        f"Failure trajectories with partial_success=None: {failure_without_ps} "
        f"(should be 0 — failures need partial_success)."
    )

    verdict(
        missing_videos == 0,
        f"Missing video files: {missing_videos}/{len(ds)}."
    )

    # Verify data_source is 'maniskill_pusht' (matches dataset_success_cutoff.txt)
    expected_source = "maniskill_pusht"
    all_correct = all(s == expected_source for s in data_sources)
    verdict(
        all_correct,
        f"All data_source='{expected_source}': {all_correct}. "
        f"Must match dataset_success_cutoff.txt entry."
    )

    # Single task check
    verdict(
        len(task_counts) == 1,
        f"Single task: {len(task_counts)} unique tasks. "
        f"preference_strategy_ratio should disable different_task."
    )


# ─────────────────────────────────────────────────────────────────────
# 1.3  Preprocessed cache
# ─────────────────────────────────────────────────────────────────────
def inspect_preprocessed(args):
    """Inspect the preprocessed dataset cache."""
    banner("PHASE 1.3 — Inspect Preprocessed Cache")

    cache_dir = args.cache_dir
    dataset_path = args.dataset_path

    # Compute cache key (same logic as preprocess_datasets.py)
    cache_key = dataset_path.replace("/", "_")
    if cache_key.startswith("_"):
        cache_key = cache_key[1:]
    full_cache_path = os.path.join(cache_dir, cache_key)

    print(f"Cache dir: {cache_dir}")
    print(f"Dataset path: {dataset_path}")
    print(f"Expected cache key: {cache_key}")
    print(f"Full cache path: {full_cache_path}")
    print(f"Exists: {os.path.exists(full_cache_path)}")

    if not os.path.exists(full_cache_path):
        if os.path.exists(cache_dir):
            contents = os.listdir(cache_dir)
            print(f"\nContents of {cache_dir}:")
            for c in sorted(contents):
                print(f"  {c}")
        print("\n*** Cache not found! Run preprocessing first. ***")
        return

    # List cache contents
    print(f"\nCache contents:")
    for item in sorted(os.listdir(full_cache_path)):
        item_path = os.path.join(full_cache_path, item)
        if os.path.isdir(item_path):
            sub_items = os.listdir(item_path)
            print(f"  {item}/ ({len(sub_items)} files)")
        else:
            size = os.path.getsize(item_path)
            print(f"  {item} ({size:,} bytes)")

    # Load index mappings
    index_path = os.path.join(full_cache_path, "index_mappings.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index_mappings = json.load(f)

        banner("INDEX MAPPINGS")
        for key, value in index_mappings.items():
            if isinstance(value, dict):
                print(f"  {key}: {len(value)} entries")
                for k, v in list(value.items())[:5]:
                    count = len(v) if isinstance(v, list) else v
                    print(f"    '{k}': {count} items")
            elif isinstance(value, list):
                print(f"  {key}: {len(value)} items")

        # Check: quality_indices has both successful and failure
        qi = index_mappings.get("quality_indices", {})
        banner("CHECKS")
        verdict(
            "successful" in qi and len(qi.get("successful", [])) > 0,
            f"quality_indices has 'successful': {len(qi.get('successful', []))} trajectories."
        )
        verdict(
            "failure" in qi and len(qi.get("failure", [])) > 0,
            f"quality_indices has 'failure': {len(qi.get('failure', []))} trajectories."
        )

        # Check: optimal_by_task exists and has entries for our task
        obt = index_mappings.get("optimal_by_task", {})
        verdict(
            len(obt) > 0,
            f"optimal_by_task has {len(obt)} tasks with optimal trajectories. "
            f"Needed for suboptimal_same_task preference strategy."
        )
        for task, indices in obt.items():
            print(f"    task='{task[:60]}': {len(indices)} optimal trajectories")

        # Check: suboptimal/failure trajectories exist for this task
        sobt = index_mappings.get("suboptimal_by_task", {})
        if sobt:
            for task, indices in sobt.items():
                print(f"    suboptimal_by_task '{task[:60]}': {len(indices)} trajectories")

    # Load and inspect processed frames
    try:
        from datasets import Dataset
        frames_dir = os.path.join(full_cache_path, "frames")
        if os.path.exists(frames_dir):
            ds = Dataset.load_from_disk(frames_dir)
            print(f"\nProcessed dataset: {len(ds)} rows, columns: {ds.column_names}")

            # Spot-check frame loading from npz
            for i in range(min(3, len(ds))):
                row = ds[i]
                npz_path = row.get("frames") or row.get("npz_path")
                if npz_path and isinstance(npz_path, str) and npz_path.endswith(".npz"):
                    if os.path.exists(npz_path):
                        data = np.load(npz_path)
                        arr = data[list(data.keys())[0]]
                        print(f"  [{i}] npz: shape={arr.shape}, dtype={arr.dtype}, "
                              f"pixel_range=[{arr.min()}, {arr.max()}]")
                    else:
                        print(f"  [{i}] npz: {npz_path} — MISSING!")
                else:
                    print(f"  [{i}] frames field: {type(npz_path).__name__}")
    except Exception as e:
        print(f"\nCould not load processed dataset: {e}")


# ─────────────────────────────────────────────────────────────────────
# 1.4  Live training batches
# ─────────────────────────────────────────────────────────────────────
def inspect_batch(args):
    """Instantiate the real RBMDataset + collator and inspect actual training batches."""
    banner("PHASE 1.4 — Inspect Live Training Batches")

    import torch

    # Use Hydra to load config (same as train.py)
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "robometer", "configs")

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=args.config_name)

    from robometer.configs.experiment_configs import ExperimentConfig
    from robometer.utils.hydra_utils import convert_hydra_to_dataclass
    from robometer.utils.setup_utils import (
        setup_model_and_processor,
        setup_dataset,
        setup_batch_collator,
        update_cfg_with_pretrained_ckpt,
    )

    exp_cfg = convert_hydra_to_dataclass(cfg, ExperimentConfig)

    # Load processor only (no model weights needed for data inspection)
    print("Loading processor (no model weights)...")
    checkpoint = exp_cfg.training.load_from_checkpoint
    update_cfg_with_pretrained_ckpt(exp_cfg, checkpoint)

    tokenizer, processor, _ = setup_model_and_processor(
        exp_cfg.model, hf_model_id=checkpoint or "", peft_config=exp_cfg.peft,
        load_model=False,
    )

    print("Setting up dataset and collator...")
    batch_collator = setup_batch_collator(processor, tokenizer, exp_cfg, is_eval=False)
    train_dataset = setup_dataset(exp_cfg.data)
    print(f"Dataset length: {len(train_dataset)}")

    batch_size = exp_cfg.training.per_device_train_batch_size

    for batch_idx in range(args.num_batches):
        banner(f"BATCH {batch_idx}")

        # Sample items and collate
        try:
            samples = [train_dataset[batch_idx * batch_size + j] for j in range(batch_size)]
            batch = batch_collator(samples)
        except Exception as e:
            print(f"  *** COLLATION FAILED: {e} ***")
            import traceback
            traceback.print_exc()
            continue

        # Inspect batch structure
        print(f"  Top-level keys: {sorted(batch.keys())}")
        print(f"  num_preferences: {batch.get('num_preferences', 0)}")
        print(f"  num_progress: {batch.get('num_progress', 0)}")

        # Preference inputs
        pref = batch.get("preference_inputs", {})
        if pref:
            print(f"\n  Preference inputs:")
            for k, v in sorted(pref.items()):
                if isinstance(v, torch.Tensor):
                    print(f"    {k}: shape={list(v.shape)}, dtype={v.dtype}")
                elif isinstance(v, (list, np.ndarray)):
                    print(f"    {k}: len={len(v)}, type={type(v).__name__}")
                else:
                    print(f"    {k}: {v}")

            # Check preference labels
            pref_labels = pref.get("preference_labels")
            if pref_labels is not None:
                print(f"\n    preference_labels: {pref_labels.tolist()}")

            # Check target_progress_A values
            tp_a = pref.get("target_progress_A")
            tp_a_mask = pref.get("target_progress_A_mask")
            if tp_a is not None:
                print(f"\n    target_progress_A: shape={list(tp_a.shape)}")
                for j in range(min(3, tp_a.shape[0])):
                    vals = tp_a[j].tolist()
                    # Show only first 8 values
                    vals_str = ", ".join(f"{v:.3f}" for v in vals[:8])
                    mask_val = tp_a_mask[j].item() if tp_a_mask is not None else "?"
                    print(f"      [{j}] mask={mask_val} progress=[{vals_str}...]")

            # Check success labels
            sl_a = pref.get("success_labels_A")
            if sl_a is not None:
                for j in range(min(3, sl_a.shape[0])):
                    vals = sl_a[j].tolist()
                    vals_str = ", ".join(f"{v:.0f}" for v in vals[:8])
                    print(f"      [{j}] success_labels=[{vals_str}...]")

            # Check data gen strategies
            strat = pref.get("rejected_data_gen_strategy")
            if strat is not None:
                print(f"\n    rejected_data_gen_strategy: {strat}")

        # Progress inputs
        prog = batch.get("progress_inputs", {})
        if prog:
            print(f"\n  Progress inputs:")
            for k, v in sorted(prog.items()):
                if isinstance(v, torch.Tensor):
                    print(f"    {k}: shape={list(v.shape)}, dtype={v.dtype}")

        print()

    banner("BATCH INSPECTION CHECKS")
    verdict(
        batch.get("num_preferences", 0) > 0 or batch.get("num_progress", 0) > 0,
        "Batches contain training signal (num_preferences > 0 or num_progress > 0)."
    )
    if pref:
        verdict(
            pref.get("target_progress_A_mask") is not None and pref["target_progress_A_mask"].sum() > 0,
            "At least some preference samples have unmasked progress "
            "(target_progress_A_mask has nonzero entries)."
        )


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Inspect data at each pipeline stage"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1.1: Raw TFRecords
    p_raw = subparsers.add_parser("raw", help="Inspect raw TFRecords")
    p_raw.add_argument("--tfrecord-dir", required=True)
    p_raw.add_argument("--num-trajectories", type=int, default=32,
                        help="Number of trajectories to inspect (default=32, matching overfit subset)")

    # 1.2: HF Dataset
    p_hf = subparsers.add_parser("hf", help="Inspect generated HF dataset")
    p_hf.add_argument("--dataset-dir", required=True)

    # 1.3: Preprocessed cache
    p_pp = subparsers.add_parser("preprocessed", help="Inspect preprocessed cache")
    p_pp.add_argument("--cache-dir", required=True)
    p_pp.add_argument("--dataset-path", required=True,
                       help="Original dataset path (used to compute cache key)")

    # 1.4: Live training batches
    p_batch = subparsers.add_parser("batch", help="Inspect live collated training batches")
    p_batch.add_argument("--config-name", required=True,
                          help="Hydra config name (e.g. train_maniskill_pusht_lora_overfit)")
    p_batch.add_argument("--num-batches", type=int, default=3)

    args = parser.parse_args()

    if args.command == "raw":
        inspect_raw(args)
    elif args.command == "hf":
        inspect_hf(args)
    elif args.command == "preprocessed":
        inspect_preprocessed(args)
    elif args.command == "batch":
        inspect_batch(args)


if __name__ == "__main__":
    main()

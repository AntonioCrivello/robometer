# Finetuning Robometer on Custom TFRecord Datasets

End-to-end guide for taking Open Dreams TFRecord trajectories, converting them to Robometer format, LoRA finetuning Robometer-4B, and using the finetuned model for annotation.

## Prerequisites

- Robometer repo cloned with `uv` environment set up
- GPU with ≥48GB VRAM (tested on RTX 6000 Pro 96GB)
- TFRecord dataset with per-frame `observation` (encoded image), `reward` (float32), `done` (int64)
- `metadata.json` and `trajectories_metadata.csv` alongside the TFRecords

## Pipeline Overview

```
TFRecords → Loader → generate_hf_dataset.py → Preprocess → LoRA Finetune → Annotate/Evaluate
```

---

## Step 1: Dataset Loader

**File:** `dataset_upload/dataset_loaders/maniskill_pusht_sim_loader.py`

The loader provides two things:
- `TFRecordFrameLoader`: Pickle-able callable that lazily reads frames from a TFRecord. Returns `(T, H, W, 3)` uint8 numpy array. Must import TensorFlow lazily inside `__call__`, not at module level.
- `load_maniskill_pusht_sim_dataset(dataset_path, dataset_name)`: Scans for TFRecords, reads rewards to compute `partial_success` (normalized to [0,1]), returns `{task: [trajectory_dicts]}`.

**Important:** Use `from tqdm import tqdm`, not `import tqdm` (the module isn't callable).

**Register in `generate_hf_dataset.py`:** Add an `elif` block in the `main()` dispatcher:
```python
elif "maniskill_pusht" in cfg.dataset.dataset_name.lower():
    from dataset_upload.dataset_loaders.maniskill_pusht_sim_loader import load_maniskill_pusht_sim_dataset
    task_data = load_maniskill_pusht_sim_dataset(cfg.dataset.dataset_path, dataset_name=cfg.dataset.dataset_name)
    trajectories = flatten_task_data(task_data)
```

**Test the loader before proceeding:**
```bash
uv run python scripts/test_maniskill_pusht_loader.py \
    --dataset-path /path/to/your/tfrecords
```

## Step 2: Data Gen Config

**File:** `dataset_upload/configs/data_gen_configs/maniskill_pusht_sim.yaml`

```yaml
dataset:
  dataset_path: /path/to/your/tfrecords  # directory with metadata.json + trajectory_*.tfrecord
  dataset_name: maniskill_pusht_sim      # must contain "maniskill_pusht" to match dispatcher

output:
  output_dir: /path/to/robometer_data/maniskill_pusht_sim  # where MP4s + HF dataset get saved
  max_trajectories: -1
  max_frames: 64
  use_video: true
  fps: 10
  shortest_edge_size: 144  # match your source image size to avoid upscaling
  num_workers: 2

hub:
  push_to_hub: true  # push to HuggingFace Hub
  hub_repo_id: maniskill_pusht_sim_rfm
```

**Run conversion:**
```bash
export HF_TOKEN=your_token
export HF_USERNAME=your_username
uv run python dataset_upload/generate_hf_dataset.py \
    --config_path dataset_upload/configs/data_gen_configs/maniskill_pusht_sim.yaml
```

This reads TFRecords, converts frames to MP4 videos, computes language embeddings, and saves an HF dataset. If `push_to_hub: true`, it also uploads to HuggingFace.

## Step 3: Register Dataset

Three files to edit:

1. **`robometer/data/datasets/name_mapping.py`** — add to `DS_SHORT_NAME_MAPPING`:
   ```python
   "maniskill_pusht_sim_rfm_maniskill_pusht_sim": "maniskill_pusht_sim",
   ```
   Key format: `{hub_repo_id}_{dataset_name}`

2. **`robometer/data/dataset_category.py`** — add `"maniskill_pusht_sim"` to `ALL_DATASOURCES` list

3. **`robometer/data/dataset_success_cutoff.txt`** — add:
   ```
   maniskill_pusht_sim,0.95
   ```

## Step 4: Preprocess

**File:** `robometer/configs/preprocess_maniskill_pusht.yaml`

```yaml
train_datasets:
  - "YourUsername/maniskill_pusht_sim_rfm"  # HF repo path
train_subsets:
  - ["maniskill_pusht_sim"]
eval_datasets:
  - "YourUsername/maniskill_pusht_sim_rfm"
eval_subsets:
  - ["maniskill_pusht_sim"]
cache_dir: /path/to/processed_datasets
max_frames_for_preprocessing: 64
video_frame_sampling: "uniform"
num_proc: 1
num_threads: 4
force_reprocess: false
precompute_embeddings: false
```

**Symlink required:** The preprocessor resolves video paths as `{ROBOMETER_DATASET_PATH}/{repo_name}/{frames_field}`. If your local output directory name doesn't match the HF repo name, create a symlink:
```bash
cd /path/to/robometer_data
ln -s maniskill_pusht_sim maniskill_pusht_sim_rfm
```

**Run preprocessing:**
```bash
export ROBOMETER_DATASET_PATH=/path/to/robometer_data
export ROBOMETER_PROCESSED_DATASETS_PATH=/path/to/processed_datasets
uv run python -m robometer.data.scripts.preprocess_datasets \
    --config robometer/configs/preprocess_maniskill_pusht.yaml
```

**Note on local datasets:** If using local paths instead of HF paths in `train_datasets`, the preprocessor calls `load_dataset()` which fails on `save_to_disk` format. Use HF paths for preprocessing.

## Step 5: LoRA Finetune

**File:** `robometer/configs/train_maniskill_pusht_lora.yaml`

```yaml
defaults:
  - config
  - _self_

model:
  base_model_id: Qwen/Qwen3-VL-4B-Instruct
  use_peft: true
  use_unsloth: false  # CRITICAL: unsloth crashes on multi-GPU and some single-GPU configs

data:
  train_datasets:
    - "YourUsername/maniskill_pusht_sim_rfm/maniskill_pusht_sim"
  eval_datasets:
    - "YourUsername/maniskill_pusht_sim_rfm/maniskill_pusht_sim"

training:
  load_from_checkpoint: "robometer/Robometer-4B"
  per_device_train_batch_size: 4    # increase if GPU memory allows
  learning_rate: 1.0e-5             # LoRA is sensitive — 1e-5 to 2e-5
  warmup_ratio: 0.1
  weight_decay: 0.01
  max_steps: 50
  save_steps: 10
  gradient_checkpointing: false
  output_dir: /path/to/logs/robometer4b_lora_pusht  # weights + config saved here
  exp_name: robometer4b_lora_pusht
  overwrite_output_dir: true
  do_eval: true
  eval_steps: 200
  custom_eval_steps: 0

custom_eval:
  eval_types: []  # disable policy_ranking etc. unless you've preprocessed those datasets

logging:
  log_to: [wandb]
  wandb_project: robometer-pusht
  wandb_entity: your-wandb-entity  # NOT "clvr" — that's the Robometer team's entity
```

**Run training (single GPU):**
```bash
export ROBOMETER_PROCESSED_DATASETS_PATH=/path/to/processed_datasets
CUDA_VISIBLE_DEVICES=0 uv run python train.py --config-name=train_maniskill_pusht_lora
```

**Data config must match cache key:** `train_datasets` value gets transformed to cache key via `path.replace("/", "_")`. The result must match the directory name in `processed_datasets/`. For example: `YourUsername/maniskill_pusht_sim_rfm/maniskill_pusht_sim` → `YourUsername_maniskill_pusht_sim_rfm_maniskill_pusht_sim`.

## Step 6: Evaluate

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/evaluate_finetuned_pusht.py \
    --model-path /path/to/logs/robometer4b_lora_pusht \
    --metadata-file /path/to/tfrecords/metadata.json \
    --tfrecord-dir /path/to/tfrecords \
    --output-dir outputs/eval_finetuned/ \
    --index 0
```

Compare against zero-shot baseline:
```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/evaluate_finetuned_pusht.py \
    --model-path robometer/Robometer-4B \
    --metadata-file /path/to/tfrecords/metadata.json \
    --tfrecord-dir /path/to/tfrecords \
    --output-dir outputs/eval_base/ \
    --index 0
```

## Step 7: Annotate with Finetuned Model

Use the same annotation script with the finetuned checkpoint:
```bash
uv run python scripts/annotate_open_dreams_tfrecords.py \
    --model-path /path/to/logs/robometer4b_lora_pusht \
    --task "Push the T block to the goal position" \
    --metadata-file /path/to/tfrecords/metadata.json \
    --tfrecord-dir /path/to/tfrecords \
    --output-dir /path/to/annotations/ \
    --batch-size 16
```

---

## Known Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| `unsloth_zoo setStorage RuntimeError` | Unsloth patches `torch.utils.checkpoint` globally at import time, crashes with DataParallel | Set `model.use_unsloth: false` in config |
| FSDP `uniform dtype` error | LoRA adapters are float32, base model is bfloat16 | Use single GPU — FSDP/multi-GPU with LoRA doesn't work |
| `ROBOMETER_DATASET_PATH not set` | Preprocessor needs to know where MP4 videos live | `export ROBOMETER_DATASET_PATH=/path/to/robometer_data` |
| `ROBOMETER_PROCESSED_DATASETS_PATH not set` | Trainer needs the preprocessed cache | `export ROBOMETER_PROCESSED_DATASETS_PATH=/path/to/processed_datasets` |
| `config.yaml not found` | Model weights and config saved in different directories | Set `output_dir` to include the experiment name so everything saves together |
| `No configured datasets available in cache` | `train_datasets` value doesn't match cache directory name | Use full HF path with subset: `Username/repo/subset` |
| Symlink needed for video paths | Preprocessor constructs path as `{DATASET_PATH}/{repo_name}/...` but local folder has different name | `ln -s actual_name repo_name` |
| W&B 403 permission denied | Default entity is `clvr` (Robometer team) | Set `logging.wandb_entity: your-entity` |
| `torchcodec` FFmpeg errors | `torchcodec` tries to load system FFmpeg at import time | `uv pip uninstall torchcodec` |
| Blackwell CUDA errors | `torch.prod` broken on RTX 6000 Pro Blackwell | `rm -rf ~/.nv/ComputeCache` |
| All predictions same value after finetuning | Overfitting / learning rate too high | Use `learning_rate: 1e-5`, fewer steps, more data |
| `load_dataset` vs `load_from_disk` | Local datasets saved with `save_to_disk` can't be loaded with `load_dataset` | Use HF paths instead of local paths for preprocessing |

## Finetuning Tips

- **200 trajectories is very small** — expect limited improvement over zero-shot. The base model was trained on millions of diverse trajectories.
- **Learning rate 1e-5 to 2e-5** for LoRA. Higher rates cause collapse (all predictions converge to the mean).
- **Watch eval loss** — if train loss drops but eval loss rises, you're overfitting.
- **Batch size** — with 96GB VRAM and gradient_checkpointing=false, batch_size=8 uses ~60GB peak. Don't go above 16 without monitoring.
- **Steps** — for 200 trajectories at batch_size=4, one epoch = 50 steps. More than 2-3 epochs risks overfitting.
- **For production** — use the full 38K+ trajectory dataset, 1000-5000 steps, with eval every 200 steps.

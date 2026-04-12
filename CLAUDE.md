# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workflow Orchestration

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just rebuilding
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY connection from the user: update 'tasks/lessons.md' with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5 Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management
1. **Plan First**: Write plan to 'tasks/todo.md' with chechable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to 'tasks/todo.md'
6. **Capture Lessons**: Update 'tasks/lessons.md' after corrections

## Code Implementation Protocol
1. **ALWAYS Explain Before Implementing**:
   - Before writing ANY new file or significant code block, EXPLAIN the design first
   - Include: architecture, key methods, data structures, algorithms, tradeoffs
   - Wait for user confirmation before proceeding
   - This applies to: new files, new classes, complex functions, algorithms
2. **Explain Edits Before Making Them**:
   - Before editing existing code, describe what changes will be made and why
   - Show before/after snippets for complex changes
   - For simple edits (typos, formatting), can proceed directly
3. **Collaborative Implementation**:
   - User may want to make edits together in IDE
   - Default to explanation-first workflow unless explicitly told to just implement

## Core Principles
- **Simplicity**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.


Search for this information in a structured way. As you gather data, develop several competing hypotheses. Track your confidence levels in your progress notes to improve calibration. Regularly self-critique your approach and plan. Update a hypothesis tree or research notes file to persist information and provide transparency. Break down this complex research task systematically.

Please write a high-quality, general-purpose solution using the standard tools available. Do not create helper scripts or workarounds to accomplish the task more efficiently. Implement a solution that works correctly for all valid inputs, not just the test cases. Do not hard-code values or create solutions that only work for specific test inputs. Instead, implement the actual logic that solves the problem generally.

Focus on understanding the problem requirements and implementing the correct algorithm. Tests are there to verify correctness, not to define the solution. Provide a principled implementation that follows best practices and software design principles.

If the task is unreasonable or infeasible, or if any of the tests are incorrect, please inform me rather than working around them. The solution should be robust, maintainable, and extendable.


## Project Overview

Open Dreams is a JAX/Flax NNX research codebase for training world models—neural networks that learn to predict how environments evolve over time based on actions. The codebase implements diffusion-based world models using flow matching for stable, high-quality video prediction.

**Key Technologies:**
- **JAX/Flax NNX**: Core ML framework with automatic differentiation and JIT compilation
- **Hydra**: Configuration management with composable YAML configs
- **TensorFlow**: Data loading pipeline (TFRecords)
- **Orbax**: Checkpoint management
- **Weights & Biases**: Experiment tracking

## Code Quality Standards

### Logging vs Print Statements

**Rule:** Never use `print()` in model code or library functions. Only use in user-facing scripts.

```python
# ✅ GOOD - Model code uses logging
import logging
logger = logging.getLogger(__name__)

class LatentDiffusionWorldModel(BaseWorldModel):
    def _load_finetuned_vae(self, config, rngs):
        logger.info(f"Loading VAE from {config.vae_checkpoint_path}")
        # ... load logic
        logger.info(f"VAE loaded successfully from step {step}")

# ✅ GOOD - CLI scripts can use print for user output
def main(config):
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

# ❌ BAD - Model code using print
class WorldModel:
    def inference(self, inputs):
        print(f"Running inference with shape {inputs.shape}")  # NO!
```

**Debug prints:** Remove ALL debug print statements before committing. Use logging with `logger.debug()` if needed for development.

### Function Length Guidelines

**Philosophy:** Functions can start long while prototyping, but should be refactored to <100 lines during review passes.

**The "One Screen" Heuristic:** A function should fit on one screen so developers (and Claude!) can see the entire function at once without scrolling. This dramatically improves understandability and enables both humans and LLMs to reason about the complete logic in one context window.

**Examples:**

```python
# ❌ Needs refactoring - 154-line main function
def main(config):
    # 154 lines of initialization, loading, setup...
    # Too much to reason about at once

# ✅ GOOD - Refactored into focused functions
def main(config):
    devices = setup_devices(config)
    dataloader = create_dataloader(config, devices)
    model = load_model(config, devices)
    run_training(model, dataloader, config)

def setup_devices(config):
    # Device initialization logic (20 lines)
    ...

def create_dataloader(config, devices):
    # Dataloader setup (30 lines)
    ...
```

**For long `__init__` methods:** Extract helper methods:
```python
def __init__(self, config, rngs):
    super().__init__()
    self._init_config(config)
    self._init_vae(config, rngs)
    self._init_encoder_decoder(config, rngs)
    self._init_embeddings(config, rngs)
```

### Type Annotations

**Required:** All public functions must have complete type annotations.

```python
# ✅ GOOD
def encode_latents(self, images: Array) -> Array:
    """Encode images to latents."""
    ...

# ✅ GOOD - Explicit Any for dynamic types
def log_eval_metrics(
    self, eval_metrics: dict[str, Array], config: DictConfig
) -> dict[str, Any]:
    ...

# ❌ BAD - Missing return type
def log_device_memory(prefix: str):
    ...

# ✅ GOOD - Add return type
def log_device_memory(prefix: str) -> None:
    ...
```

### Style Compliance

**Reference:** [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

**Key rules:**
- **Line length:** 80 characters (Black formatter may extend to 88, but aim for 80)
- **Imports:** Grouped (stdlib, third-party, first-party) with blank lines between
- **Trailing commas:** Use in multi-line function signatures and collections
- **Docstrings:** Google format with Args/Returns/Raises sections

**Run before committing:**
```bash
pixi run fmt  # Black formatter
# TODO: Add pylint/pytype once configured
```

### Error Messages

**Guideline:** Error messages must be precise, greppable, and actionable.

```python
# ❌ BAD - Vague
if not path:
    raise ValueError("Path must be specified")

# ✅ GOOD - Specific and actionable
if not path:
    raise ValueError(
        f"vae_checkpoint_path must be specified in config. "
        f"Got config keys: {list(config.keys())}"
    )

# ✅ GOOD - Include available options
if step is None:
    available = mngr.all_steps() if hasattr(mngr, 'all_steps') else []
    raise ValueError(
        f"No checkpoint found in {checkpoint_dir}. "
        f"Available steps: {available if available else 'none'}"
    )
```

## Development Commands

### Environment Setup
```bash
# Install dependencies with pixi (recommended)
pixi install

# Activate environment
pixi shell

# For TPU environment
pixi shell -e tpu
```

### Code Formatting
```bash
# Format code with black
pixi run fmt

# Check formatting
black --check src/ scripts/
```

### Running Commands

**Rule:** Always use `pixi run` to run everything. This ensures reproducibility and correct environment.

```bash
# ✅ CORRECT - Always use pixi run
pixi run python scripts/train.py --config-name=train_latent_world_model_local

# ❌ WRONG - Never use bare python
python scripts/train.py --config-name=train_latent_world_model_local
```

### Training
```bash
# Train latent diffusion world model (recommended, local GPU)
pixi run python scripts/train.py --config-name=train_latent_world_model_local

# Train pixel diffusion world model (local GPU)
pixi run python scripts/train.py --config-name=train_world_model_local

# Finetune VAE on your domain (local GPU)
pixi run python scripts/train.py --config-name=train_stable_vae_local

# TPU training - use deploy script to run on TPU pod
# Deploys code to all workers and runs in tmux sessions
WANDB_API_KEY=your_key ./scripts/deploy_to_tpu.sh scripts/train.py --config-name=train_latent_world_model_tpu

# Monitor training on worker 0
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=0
tmux attach -t training-YYYYMMDD-HHMMSS

# Override config values
pixi run python scripts/train.py \
    --config-name=train_latent_world_model_local \
    training.batch_size_per_replica=2 \
    training.num_train_steps=100000 \
    model.model_dim=256

# Submit TPU training via job queue (pikes us-east1)
# Override dataset to useast1 variant using: dataset@data.dataset=<useast1_config_name>
# Override VAE and checkpoint_dir to gs://prism-us-east1-data/...
pixi run python -m infra.job_queue.cli submit \
    -r pikes-32 \
    -n my-job-name \
    -e "scripts/train.py --config-name=train_softmax_imitation_policy_tpu \
        dataset@data.dataset=20251208_pomdp_ppo_50m_early_stop_300k_gcs_cached_sharded_useast1 \
        model.vae_checkpoint_path=gs://prism-us-east1-data/checkpoint_zoo/stable_vae/StableVAE_20251211_pomdp_ppo_early_stop_300k \
        training.checkpoint_dir=gs://prism-us-east1-data/checkpoints/softmax_imitation_policy"
```

### Inference & Visualization
```bash
# Interactive world model visualization (arrow keys to control)
pixi run python scripts/interactive_inference.py

# Visualize training data batches
pixi run python scripts/visualize_batch.py
```

### Testing & Debugging
```bash
# Test model forward pass and shapes
pixi run python scripts/test_model_shapes.py

# Test latent diffusion model API
pixi run python scripts/test_latent_diffusion_api.py

# Benchmark data loading performance
pixi run python scripts/test_dataloader_performance.py

# Test multi-host JAX setup (distributed training)
pixi run python scripts/test_jax_distributed.py

# Debug parameter freezing
pixi run python scripts/check_params.py
```

### Checkpoint Management
```bash
# Extract inference-only checkpoint (removes optimizer state, much smaller)
pixi run python scripts/extract_inference_checkpoint.py \
    --checkpoint_dir=checkpoints/LatentDiffusion_20251118_120000 \
    --step=50000 \
    --output_dir=checkpoints/inference/

# Extract VAE from VAE training checkpoint (removes optimizer)
pixi run python scripts/extract_inference_checkpoint.py \
    --config-name=train_stable_vae_local \
    checkpoint_dir=checkpoints/StableVAE_20251116_201840 \
    checkpoint_step=50000 \
    output_dir=checkpoints/vae_inference/

# Verify VAE checkpoint works with real data
pixi run python scripts/verify_vae_checkpoint.py \
    --checkpoint_dir=checkpoints/vae_inference/ \
    --step=0

# Extract from TPU checkpoint on local machine (override mesh shape)
# TPU configs expect 32 devices, so override to run on single GPU/CPU
pixi run python scripts/extract_inference_checkpoint.py \
    --config-name=train_latent_world_model_simple_mazes_tpu \
    +checkpoint_dir=gs://bucket/path/to/training/checkpoint \
    +checkpoint_step=95000 \
    +output_dir=gs://bucket/path/to/output/checkpoint \
    "model.mesh_shape=[1]" \
    "model.mesh_axis_names=['data']"
```

## Debugging Tactics

### General Principle

**Ground debugging in experimentation, avoid guessing.** Use empirical measurements (profiling, timing, memory stats) rather than assumptions about performance or behavior.

### Debugging Strategies

**1. Print Debugging for Interactive Experimentation**

Useful during initial development to understand data flow:

```python
# During development - OK for exploration
def prototype_function(inputs):
    print(f"Input shape: {inputs.shape}")
    result = complex_transform(inputs)
    print(f"Output shape: {result.shape}")
    return result
```

**Before committing:** Either remove or convert to logging:

```python
# After development - production ready
import logging
logger = logging.getLogger(__name__)

def prototype_function(inputs):
    logger.debug(f"Input shape: {inputs.shape}")
    result = complex_transform(inputs)
    logger.debug(f"Output shape: {result.shape}")
    return result
```

**2. Latency Testing**

Measure per-operation timing for performance bottlenecks:

```python
import time

# During development/profiling
start = time.perf_counter()
output = expensive_operation(input)
elapsed = time.perf_counter() - start
logger.info(f"Operation took {elapsed:.3f}s")

# Production: Keep behind verbose flag
if config.get("profile_latency", False):
    start = time.perf_counter()
    output = expensive_operation(input)
    logger.info(f"Operation took {time.perf_counter() - start:.3f}s")
```

**3. Memory Profiling**

Track device memory usage to debug OOM errors:

```python
def log_device_memory(prefix: str, verbose: bool = False) -> None:
    """Log device memory stats if verbose mode enabled."""
    if not verbose:
        return

    device = jax.devices()[0]
    if hasattr(device, 'memory_stats'):
        stats = device.memory_stats()
        used = stats.get("bytes_in_use", 0) / 1e9
        total = stats.get("bytes_limit", 0) / 1e9
        logger.info(f"[{prefix}] Memory: {used:.2f}GB / {total:.2f}GB")

# Usage - behind verbose flag
verbose = config.get("verbose_memory_logging", False)
log_device_memory("After model init", verbose=verbose)
```

**4. JAX Profiling**

Use JAX's built-in profiling for detailed performance analysis:

```python
# For detailed profiling sessions
with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    # Code to profile
    output = model(batch, rngs)
```

### Debugging Checklist Before Committing

- [ ] Remove all debug `print()` statements
- [ ] Convert essential prints to `logger.debug()`
- [ ] Put profiling/timing code behind verbose flags
- [ ] Remove temporary breakpoints or inspection code
- [ ] Ensure logging uses appropriate levels (DEBUG, INFO, WARNING, ERROR)

## Architecture Overview

### Model Hierarchy

The codebase uses an abstract API pattern for model-agnostic training:

```
ModelTrainerAPI (abstract)
    ├─ BaseWorldModel (abstract)
    │   ├─ LatentDiffusionWorldModel (concrete, RECOMMENDED)
    │   └─ PixelDiffusionWorldModel (concrete, not recommended at small scale)
    └─ BaseAutoencoder (abstract)
        ├─ StableVAE (concrete, finetunable Stable Diffusion VAE)
        └─ ResNetAutoencoder (concrete)
```

**Key Interface (ModelTrainerAPI):**
- `__call__(inputs, rngs)`: Training forward pass
- `inference(inputs, rngs)`: Inference forward pass (default: calls `__call__`)
- `loss(inputs, outputs)`: Compute loss and metrics
- `compute_eval_metrics(inputs, outputs, loss_metrics)`: Evaluation metrics (JIT-compiled, on device)
- `log_eval_metrics(eval_metrics, config)`: Prepare metrics for W&B logging (host, process 0)
- `get_partition_spec()`: Return parameter sharding specs for distributed training

### Recommended Model: Latent Diffusion World Model

**Location:** `src/open_dreams/models/latent_diffusion_world_model.py`

The latent diffusion model is **strongly recommended** for both quality and efficiency at typical model scales (<500M parameters):

**Architecture Flow:**
1. **VAE Encoder** (frozen): Compress 144×144 images → 18×18×4 latents (8x spatial downsampling)
2. **Latent Patchify**: 2×2 patches on latents → 9×9 patch grid (81 tokens per frame)
3. **Encoder Transformer**: 12-layer ViT encodes current frame latent patches
4. **Action + Time Conditioning**: Action embeddings + sinusoidal time embeddings as conditioning tokens
5. **Decoder Transformer**: 12-layer cross-attention transformer decodes noisy next frame latent patches
6. **Unpatchify**: Reconstruct full latent representation (18×18×4)
7. **VAE Decoder** (frozen): Decode latents → 144×144×3 RGB images

**Why Latent Space Wins at Small Scale:**
- VAE decoder is robust to noise from its training → insufficiently-parameterized diffusion models can predict "close enough" latents that decode to high-quality images
- Pixel diffusion requires exact RGB prediction (0-255), which needs much higher model capacity
- 10x faster training, 64x less memory than pixel-space models
- **Better quality** at typical model scales (empirically validated)

**Key Files:**
- Model: `src/open_dreams/models/latent_diffusion_world_model.py`
- VAE: `src/open_dreams/models/stable_vae.py`
- Transformer: `src/open_dreams/models/transformer.py` (shared encoder/decoder blocks)
- Config: `configs/model/latent_diffusion_world_model.yaml`

### Data Pipeline Architecture

**Key Insight:** The data pipeline is pure TensorFlow ops (no Python overhead) with file-level sharding for distributed training.

**Data Flow:**
1. **TFRecord Storage**: One episode per file with observations, actions, rewards, dones
2. **File-Level Sharding**: Each process loads from disjoint set of TFRecord files
3. **Greedy Bin Packing**: Variable-length episodes packed into fixed-length sequences
4. **Block Causal Masking**: Bidirectional attention within episodes, causal between episodes
5. **JAX Global Arrays**: Convert TF tensors to sharded JAX arrays via `jax.make_array_from_process_local_data`

**Key Files:**
- `src/open_dreams/data/distributed_loader.py`: Mesh-aware distributed dataloader
- `src/open_dreams/data/tf_native_dataloader.py`: Pure TF trajectory dataset with bin packing
- `src/open_dreams/data/tf_masking.py`: Block causal attention mask generation

**Batch Size Semantics:**
- Config specifies `batch_size_per_replica` (what each replica sees)
- Number of replicas determined by `data` axis length in mesh
- Global batch = `batch_size_per_replica` × (data axis length)
- Each process loads: `batch_size_per_replica` × (replicas per process)

Example with 32 devices, 8 processes, pure data parallel:
- `batch_size_per_replica=8`
- Mesh: `shape=[32], axis=['data']` → 32 replicas
- Each process: 8 × 4 = 32 examples (4 local replicas per process)
- Global batch: 8 × 32 = 256 examples

### Training Loop Architecture

**Location:** `scripts/train.py`

**Key Design Patterns:**
1. **Multi-Host Coordination**: Uses `jax.experimental.multihost_utils` for synchronization
2. **Mesh-Based Sharding**: All arrays are global sharded arrays via `jax.make_array_from_process_local_data`
3. **Stateless Dataloading**: Deterministic sampling with explicit RNG seeds (no iterator state to checkpoint)
4. **Split GraphDef/State**: Flax NNX pattern separates static graph from mutable state for efficient checkpointing
5. **Process 0 Logging**: Only process 0 logs to W&B and saves checkpoints

**Training Step Flow:**
1. Get sharded batch from dataloader
2. Update training RNG key
3. `train_step(graphdef, state, optimizer_state, batch, train_key)` - JIT-compiled
   - Merge graphdef + state → model
   - Forward pass: `model(batch, rngs)`
   - Compute loss: `model.loss(batch, outputs)`
   - Compute gradients via `jax.grad`
   - Update optimizer state
   - Split model → (graphdef, state)
4. Log metrics (process 0 only)
5. Periodic evaluation and checkpointing

**Checkpoint Structure:**
```
checkpoints/{model_name}_{timestamp}/
├── step_1000/
│   ├── default/           # Model state (parameters)
│   ├── optimizer/         # Optimizer state (Adam moments)
│   └── metadata/          # Training metadata (step, config)
```

## Configuration System

**Hydra Composition Pattern:**

Configs are composed from modular YAML files in `configs/`:

```
configs/
├── train_latent_world_model_local.yaml      # Main training config (composes others)
├── train_latent_world_model_tpu.yaml        # TPU variant
├── inference_latent_world_model.yaml        # Inference config
├── model/
│   ├── latent_diffusion_world_model.yaml    # Model architecture hyperparams
│   ├── diffusion_world_model.yaml           # Pixel diffusion variant
│   └── stable_vae.yaml                      # VAE config
└── data/
    ├── dataloader_default.yaml              # Data loading config
    └── dataloader_tpu.yaml                  # TPU data loading variant
```

**Composition Example:**
```yaml
# train_latent_world_model_local.yaml
defaults:
  - data: dataloader_default
  - model: latent_diffusion_world_model
  - _self_

training:
  mesh_shape: [1]              # Single device
  batch_size_per_replica: 2
  # ... other training hyperparams
```

### Config Composition Patterns

**Principle:** Avoid duplication through Hydra composition.

**Example - Reducing duplication between local and TPU configs:**

```yaml
# configs/train_base.yaml (shared settings)
defaults:
  - data: dataloader_default
  - model: latent_diffusion_world_model

training:
  batch_size_per_replica: 2
  num_train_steps: 100000
  learning_rate: 1e-4
  # ... other shared settings

# configs/train_latent_world_model_local.yaml
defaults:
  - train_base
  - _self_

training:
  mesh_shape: [1]  # Only override what's different

# configs/train_latent_world_model_tpu.yaml
defaults:
  - train_base
  - override data: dataloader_tpu  # TPU-specific loader
  - _self_

training:
  mesh_shape: [256]  # Only override what's different
```

### Checkpoint Zoo Management

**Principle:** Centralize canonical checkpoint references to avoid fragmentation.

**Pattern:**
```yaml
# configs/checkpoint_zoo.yaml
vae:
  stable_vae_finetuned_fourrooms:
    path: "gs://open-dreams/checkpoints/StableVAE_20251116_201840_final"
    description: "StableVAE finetuned on FourRooms dataset"

world_models:
  latent_diffusion_tiny_fourrooms:
    path: "gs://open-dreams/checkpoints/LatentDiffusion_20251118_120000"
    step: 50000
    description: "Tiny latent diffusion model trained on FourRooms"

# Then reference in other configs:
# configs/model/latent_diffusion_world_model.yaml
defaults:
  - /checkpoint_zoo@vae_checkpoint: vae.stable_vae_finetuned_fourrooms

vae_checkpoint_path: ${vae_checkpoint.path}
```

**Benefits:**
- Single source of truth for important checkpoints
- Easy to identify which checkpoints to keep
- Clear deprecation path for old checkpoints

### Dataset Configuration

**Principle:** Separate dataset source from dataloader config for composability.

**Current issue:** Configs hardcode paths like `/home/sbateman/datasets` (not reproducible).

**Proposed pattern:**
```yaml
# configs/dataset/four_rooms_ppo.yaml
name: "four_rooms_ppo"
source: "local"
local_path: "/home/sbateman/datasets/four_rooms_ppo_rollouts"

# configs/dataset/four_rooms_ppo_gcs.yaml
name: "four_rooms_ppo"
source: "gcs"
gcs_path: "gs://open-dreams/datasets/four_rooms_ppo_rollouts"
cache_locally: true
cache_dir: "${oc.env:HOME}/.open_dreams/datasets"

# configs/data/dataloader_default.yaml
defaults:
  - /dataset: four_rooms_ppo_gcs  # Default to GCS with caching

batch_size: ???  # Set by training config
# ... dataloader settings (no dataset path)

# Then in training config:
defaults:
  - data: dataloader_default
  - override dataset: four_rooms_ppo_gcs  # Or local for dev
```

**Benefits:**
- Reproducible across machines
- Easy to switch between local/GCS
- Reduces egress costs through caching
- Supports multiple datasets cleanly

## Testing & Script Organization

### Test File Organization

**Rule:** Test files belong in `tests/` directory, not `scripts/`.

```
# ✅ GOOD structure
tests/
  test_latent_diffusion_api.py
  test_vae_loading.py
  test_dataloader_performance.py

scripts/
  train.py
  interactive_inference.py
  extract_inference_checkpoint.py

# ❌ BAD - Tests mixed with scripts
scripts/
  train.py
  test_latent_diffusion_api.py  # Should be in tests/
  interactive_inference.py
```

**Script categories:**
- `scripts/train.py` - Main training entrypoint
- `scripts/interactive_*.py` - User-facing interactive tools
- `scripts/extract_*.py` - Checkpoint utilities
- `scripts/visualize_*.py` - Data visualization tools
- `tests/test_*.py` - Unit and integration tests

## Important Code Patterns

### Adding a New Model

1. **Inherit from appropriate base class:**
   - World models: inherit from `BaseWorldModel`
   - Autoencoders: inherit from `BaseAutoencoder`

2. **Implement required methods:**
   ```python
   from open_dreams.models.base_world_model import BaseWorldModel

   class MyWorldModel(BaseWorldModel):
       def __call__(self, inputs, rngs):
           # Training forward pass
           pass

       def inference(self, inputs, rngs):
           # Inference forward pass (can differ from training)
           pass

       def loss(self, inputs, outputs):
           # Return (total_loss, metrics_dict)
           pass

       def compute_eval_metrics(self, inputs, outputs, loss_metrics):
           # Compute metrics on device (JIT-compiled)
           pass

       def log_eval_metrics(self, eval_metrics, config):
           # Prepare metrics for W&B (host-side, process 0)
           pass

       def get_partition_spec(self):
           # Return partition specs for distributed training
           pass
   ```

3. **Create config files:**
   - Model config: `configs/model/my_model.yaml`
   - Training config: `configs/train_my_model.yaml` (compose with model config)

4. **Test:**
   ```bash
   python scripts/train.py --config-name=train_my_model
   ```

### API Design Conventions

**Principle:** Function names should clearly indicate their behavior.

**Examples from this codebase:**

```python
# ❌ AMBIGUOUS - What kind of inference?
def inference(self, inputs, rngs):
    # Actually does teacher forcing with ground truth next frames
    ...

# ✅ CLEAR - Indicates teacher forcing
def teacher_forcing_inference(self, inputs, rngs):
    """Inference using teacher forcing (requires ground truth next frames)."""
    ...

# ✅ CLEAR - Indicates autoregressive generation
def autoregressive_inference(self, inputs, rngs):
    """Autoregressive generation (no ground truth needed)."""
    ...
```

**Common patterns:**
- `encode_*` / `decode_*` - Clearly paired operations
- `_load_*` - Loading from disk/checkpoint (private helper)
- `compute_*` - Pure computation (JIT-friendly)
- `log_*` - Side effects (logging, I/O)
- `*_step` - Single iteration of a loop

### Model Surgery vs Runtime Loading

**Anti-pattern:** Loading pretrained weights at model construction time.

```python
# ❌ DISCOURAGED - Loads HuggingFace weights every time
class LatentDiffusionWorldModel(BaseWorldModel):
    def __init__(self, config, rngs):
        # Calls HuggingFace API to load pretrained VAE
        self.vae = self._load_pretrained_vae(config, rngs)
```

**Better pattern:** One-time checkpoint conversion + config-based model surgery.

```python
# Step 1: One-time conversion script
# scripts/convert_huggingface_vae_to_orbax.py
def main():
    vae, params = FlaxAutoencoderKL.from_pretrained("pcuenq/sd-vae-ft-mse-flax")
    # Save as Orbax checkpoint
    save_checkpoint(params, "checkpoints/stable_vae_pretrained/")

# Step 2: Model references converted checkpoint
class LatentDiffusionWorldModel(BaseWorldModel):
    def __init__(self, config, rngs):
        # Load from our own Orbax checkpoint (no HuggingFace API)
        self.vae = self._load_vae_from_checkpoint(config.vae_checkpoint_path)
```

**Benefits:**
- No HuggingFace API dependency at runtime
- Consistent checkpoint format
- Easier to swap VAE architectures
- Clear provenance of pretrained weights

**TODO:** Create conversion scripts and remove HuggingFace loading from model code.

### Parameter Freezing Pattern

**Key Insight:** Use Flax NNX's `split` API to separate trainable vs frozen parameters.

Example from `LatentDiffusionWorldModel`:
```python
# Freeze VAE parameters
_, vae_frozen_state = nnx.split(self.vae, nnx.Param)

# Get trainable parameters (excludes VAE)
trainable_graphdef, trainable_state = nnx.split(self, nnx.Param, ...)

# Merge back for inference (includes frozen VAE)
model = nnx.merge(full_graphdef, trainable_state, vae_frozen_state)
```

**Debug freezing:** `scripts/check_params.py` prints trainable vs frozen parameter counts.

### Distributed Training Pattern

**Core Principle:** All data arrays are global sharded arrays constructed via `jax.make_array_from_process_local_data`.

```python
# Create mesh (example: pure data parallel with 32 devices)
mesh = Mesh(jax.devices(), axis_names=["data"])

# Each process has local data
local_batch = next(local_dataloader)  # Shape: [local_batch_size, ...]

# Construct global sharded array
global_batch = jax.make_array_from_process_local_data(
    sharding=NamedSharding(mesh, P("data")),
    local_data=local_batch
)
# global_batch shape: [global_batch_size, ...] sharded across devices
```

**Key Functions:**
- `jax.make_array_from_process_local_data`: Construct global array from per-process data
- `multihost_utils.process_allgather`: Gather arrays from all processes to all processes
- `multihost_utils.sync_global_devices`: Synchronize all processes

## Infrastructure & Cost Optimization

### GCS Egress Cost Management

**Problem:** Repeated data downloads from GCS incur egress charges.

**Example Solution - Local Caching:**

This is ONE approach - you may find other creative solutions based on your specific constraints.

```yaml
# configs/dataset/four_rooms_ppo_gcs.yaml
source: "gcs"
gcs_path: "gs://open-dreams/datasets/four_rooms_ppo_rollouts.tar.gz"
cache_locally: true
cache_dir: "${oc.env:HOME}/.open_dreams/datasets"
verify_checksum: true  # Ensure cache is valid

# Dataloader checks cache before downloading
# If cache exists and valid: use local copy
# If cache missing/invalid: download from GCS, extract, cache
```

**Other potential approaches to consider:**
- Regional GCS buckets (free egress within same region for TPU training)
- Requester pays buckets for shared datasets
- Streaming with aggressive prefetching
- Compression formats optimized for your access patterns
- Dataset sharding strategies

**Best practices:**
- Document expected egress costs in dataset README
- Provide checksums/integrity verification for cached data
- For TPU training: use GCS directly (no egress within same region)
- For local development: evaluate caching, streaming, or regional mirrors

## Model Quality Recommendations

**For typical model scales (<500M parameters):**

1. **Use LatentDiffusionWorldModel** (not PixelDiffusionWorldModel)
   - Better quality at small scale due to VAE robustness
   - 10x faster training, 64x less memory
   - Config: `configs/model/latent_diffusion_world_model.yaml`

2. **Memory-efficient settings:**
   - `dtype: bfloat16` for activations (2x memory reduction)
   - `param_dtype: float32` for parameters (stability)
   - Gradient accumulation if needed: `training.gradient_accumulation: 2`

3. **Typical hyperparameters:**
   - `model_dim: 512` (or 256 for smaller models)
   - `encoder_layers: 12`, `decoder_layers: 12`
   - `num_heads: 8`
   - `num_inference_steps: 128` (can reduce to 64 for faster inference)
   - `learning_rate: 1e-4` (scale up for larger batches)

## Common Gotchas

### Module Docstrings vs ABOUTME Comments

**Current pattern:** Files start with two `# ABOUTME:` comments.

**Recommended pattern:** Use comprehensive module docstrings instead.

```python
# ❌ Current pattern (being deprecated)
# ABOUTME: Test that LatentDiffusionWorldModel API matches PixelDiffusionWorldModel
"""Test that LatentDiffusionWorldModel API matches PixelDiffusionWorldModel."""

# ✅ Recommended pattern
"""Test LatentDiffusionWorldModel API consistency.

Verifies that LatentDiffusionWorldModel implements the ModelTrainerAPI
interface correctly and produces expected output shapes and dtypes.

This test ensures that the latent diffusion model can be used as a drop-in
replacement for the pixel diffusion model in the training pipeline.

Typical usage:
    python tests/test_latent_diffusion_api.py
"""
```

**Transition plan:**
- New files: Use comprehensive docstrings (no ABOUTME)
- Existing files: Gradually migrate ABOUTME → docstrings during refactors
- No need for mass migration, handle organically over time

**Rationale:** Docstrings are standard Python, show up in `help()`, and allow for richer documentation than single-line comments.

### Checkpoint Loading
- For inference, use `checkpoint_step=null` to load the latest checkpoint
- Extract inference-only checkpoints to remove optimizer state (100MB vs 1GB+)
- VAE checkpoints are stored in world model checkpoints and can be extracted separately

### TPU Training
- File-level sharding ensures each process reads disjoint data
- All processes must call checkpointing code (orbax requirement), but only process 0 writes
- Use `multihost_utils.sync_global_devices` before checkpointing to ensure synchronization

### Data Format
- Observations must be uint8 [0, 255] (not normalized floats)
- TFRecords store one episode per file
- Metadata JSON must match actual data dimensions

### Autoregressive Inference
- Latent models have `autoregressive_inference` method for multi-step rollouts
- Current frame observations are VAE-encoded → predicted → VAE-decoded for next step
- Uses teacher forcing during training (ground truth), autoregressive during inference

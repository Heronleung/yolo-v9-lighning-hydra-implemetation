# YOLOv9 Lightning + Hydra

PyTorch Lightning and Hydra training template built around the original YOLOv9 implementation. The project supports configurable optimizer, model-size, and training-branch presets while reusing the upstream model and loss code.

## Environment setup

Use `uv` so every command runs with the locked project environment.

```bash
git fetch origin
git switch feature/configurable-model-variants
git pull --ff-only origin feature/configurable-model-variants
uv sync
```

## Main entrypoints

| Command | Purpose |
|---|---|
| `uv run python src/train.py` | Train or resume a model |
| `uv run python src/inference.py` | Run image inference |
| `uv run python src/reparameterize.py` | Convert a supported dual-branch checkpoint for deployment |
| `uv run python val.py` | Run standalone checkpoint evaluation |

## Configuration selection

The default selections are stored in `configs/train.yaml`:

```yaml
defaults:
  - _self_
  - data: taco_yolo_300
  - model: dual/c
  - trainer: single_gpu
  - optimizer: sgd
  - scheduler: linear
  - callbacks: ema
  - paths: default
```

The selected model preset automatically composes its compatible single-, dual-, or triple-branch loss. A separate `loss=` override is not required.

### Optimizers

| Hydra selector | PyTorch optimizer |
|---|---|
| `optimizer=sgd` | SGD |
| `optimizer=adam` | Adam |
| `optimizer=adamw` | AdamW |
| `optimizer=lion` | LION |

All four use the original YOLO parameter grouping for decay weights, normalization weights, and biases.

### Models

| Branch | Supported sizes | Example |
|---|---|---|
| Single branch (`DDetect`) | T, S, M, C, E | `model=single/s` |
| Dual branch (`DualDDetect`) | T, S, M, C, E | `model=dual/e` |
| Triple branch (`TripleDDetect`) | C/CF only | `model=triple/c` |

The official upstream repository only provides the triple-branch C/CF architecture. Unsupported triple T/S/M/E variants are intentionally not exposed.

## Training examples

```bash
# Default: dual-branch YOLOv9-C with SGD
uv run python src/train.py

# Single-branch GELAN-S with Adam
uv run python src/train.py model=single/s optimizer=adam

# Dual-branch YOLOv9-E with AdamW
uv run python src/train.py model=dual/e optimizer=adamw

# Triple-branch YOLOv9-CF with LION
uv run python src/train.py model=triple/c optimizer=lion

# Override ordinary run settings
uv run python src/train.py model=dual/m optimizer=sgd epochs=100 batch_size=4

# Multi-GPU training
uv run python src/train.py model=dual/c trainer=ddp trainer.devices=4

# Resume a Lightning checkpoint
uv run python src/train.py resume=checkpoints/ckpts_0/last.ckpt overwrite=true
```

`batch_size` is per GPU. Under DDP, total batch size is `batch_size × trainer.devices`.

# Test commands

Run the following tests in order. The configuration checks are quick and do not start training. The smoke tests require the configured dataset and a working PyTorch device.

## 1. Verify the default composed configuration

```bash
uv run python src/train.py --cfg job
```

Confirm that the output contains:

```yaml
model:
  branch: dual
  size: c
  cfg: models/detect/yolov9-c.yaml
optimizer:
  name: SGD
loss:
  _target_: utils.loss_tal_dual.ComputeLoss
```

## 2. Compose all 11 supported model presets

This verifies model paths and automatic loss selection without starting training.

```bash
for model in single/t single/s single/m single/c single/e \
             dual/t dual/s dual/m dual/c dual/e \
             triple/c; do
  echo "Testing model config: $model"
  uv run python src/train.py model="$model" --cfg job >/dev/null || exit 1
done

echo "PASS: all model configurations composed"
```

Expected loss mapping:

| Model branch | Loss target |
|---|---|
| `single/*` | `utils.loss_tal.ComputeLoss` |
| `dual/*` | `utils.loss_tal_dual.ComputeLoss` |
| `triple/c` | `utils.loss_tal_triple.ComputeLoss` |

## 3. Compose all four optimizer presets

```bash
for optimizer in sgd adam adamw lion; do
  echo "Testing optimizer config: $optimizer"
  uv run python src/train.py optimizer="$optimizer" --cfg job >/dev/null || exit 1
done

echo "PASS: all optimizer configurations composed"
```

## 4. Smoke-test each branch type

Each command runs one training batch and one validation batch. A batch size of one reduces GPU-memory usage.

```bash
uv run python src/train.py model=single/t optimizer=sgd \
  +trainer.fast_dev_run=true batch_size=1 data.workers=0

uv run python src/train.py model=dual/t optimizer=sgd \
  +trainer.fast_dev_run=true batch_size=1 data.workers=0

uv run python src/train.py model=triple/c optimizer=sgd \
  +trainer.fast_dev_run=true batch_size=1 data.workers=0
```

## 5. Smoke-test every optimizer

The smallest dual model is used to keep these tests relatively quick.

```bash
uv run python src/train.py model=dual/t optimizer=sgd \
  +trainer.fast_dev_run=true batch_size=1 data.workers=0

uv run python src/train.py model=dual/t optimizer=adam \
  +trainer.fast_dev_run=true batch_size=1 data.workers=0

uv run python src/train.py model=dual/t optimizer=adamw \
  +trainer.fast_dev_run=true batch_size=1 data.workers=0

uv run python src/train.py model=dual/t optimizer=lion \
  +trainer.fast_dev_run=true batch_size=1 data.workers=0
```

## 6. Optional complete architecture smoke-test

This executes every supported architecture. The E and triple-C models require considerably more GPU memory and time.

```bash
for model in single/t single/s single/m single/c single/e \
             dual/t dual/s dual/m dual/c dual/e \
             triple/c; do
  echo "Running smoke test: $model"
  uv run python src/train.py model="$model" optimizer=sgd \
    +trainer.fast_dev_run=true batch_size=1 data.workers=0 || exit 1
done

echo "PASS: complete architecture matrix"
```

## Successful smoke-test criteria

A successful test should:

- Build the selected upstream architecture without a missing-file exception.
- Print the intended optimizer and parameter groups.
- Print finite `train/box`, `train/cls`, and `train/dfl` values.
- Complete one validation batch.
- Exit without model-output, loss-shape, or Hydra-composition errors.

If an E or triple-C model runs out of memory, reduce `imgsz`, keep `batch_size=1`, or test it on a larger GPU:

```bash
uv run python src/train.py model=triple/c optimizer=sgd imgsz=320 \
  +trainer.fast_dev_run=true batch_size=1 data.workers=0
```

## Inference

```bash
# Folder of images
uv run python src/inference.py \
  weights=checkpoints/ckpts_0/best.pt \
  source=data/taco_yolo_300/valid/images

# Single image
uv run python src/inference.py \
  weights=checkpoints/ckpts_0/best.pt \
  source=test.jpg conf_thres=0.4

# Save YOLO labels and confidence values
uv run python src/inference.py \
  weights=checkpoints/ckpts_0/best.pt \
  source=data/taco_yolo_300/valid/images \
  save_txt=true save_conf=true
```

## Checkpoints

Training writes versioned folders such as:

```text
checkpoints/ckpts_0/
├── last.pt
├── best.pt
└── last.ckpt
```

Use `last.ckpt` for training resume. Use `last.pt` or `best.pt` for evaluation and inference.

## Common issues

- Quote list overrides such as `'freeze=[10]'` in the shell.
- Keep `trainer.accumulate_grad_batches=1`; YOLO warmup accumulation is handled manually.
- Reduce `batch_size` or `imgsz` if a large architecture runs out of memory.
- Use `+trainer.fast_dev_run=true`; the `+` is required because the field is appended dynamically.
- If Hydra cannot find a model, use the nested selector format such as `model=dual/c`, not the previous `model=yolov9_c` name.
- Do not manually combine a model with an incompatible `loss=` override.

## Upstream references

- [YOLOv9 paper](https://arxiv.org/abs/2402.13616)
- [Original YOLOv9 repository](https://github.com/WongKinYiu/yolov9)
- [Official YOLOv9 checkpoints](https://github.com/WongKinYiu/yolov9/releases/tag/v0.1)

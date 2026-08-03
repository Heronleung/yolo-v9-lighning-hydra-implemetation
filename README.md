# yolo-v9
=======
# YOLOv9 Lightning + Hydra Training and Inference Usage

This README is the practical reference for the Hydra + Lightning YOLOv9 project. It explains how the config files fit together, how to train, resume, evaluate, convert checkpoints, run inference, and debug common issues.

The project uses Hydra for configuration and PyTorch Lightning for training orchestration.

---

## 1. Entrypoints

| Script | Use for |
| --- | --- |
| `src/train.py` | Main training entrypoint. Use it for training, resume, pretrained fine-tune, layer freezing, DDP, and YOLOv9-compatible checkpoint writing. |
| `src/inference.py` | Main inference entrypoint for images, folders, and glob patterns. |
| `src/reparameterize.py` | Convert a trained dual-branch checkpoint (`DualDDetect`) into the deploy-ready single-branch `yolov9-c-converted` model. |

Use `uv run` for all commands so the project virtual environment is used consistently.

---

## 2. Config layout

```
configs/
├── train.yaml
├── inference.yaml
├── data/taco_yolo_300.yaml
├── model/yolov9_c.yaml
├── model/gelan_c.yaml
├── trainer/single_gpu.yaml
├── trainer/ddp.yaml
├── optimizer/sgd.yaml
├── scheduler/linear.yaml
├── loss/yolov9_tal_dual.yaml
├── loss/yolov9_tal.yaml
├── callbacks/ema.yaml
├── paths/default.yaml
└── experiment/              # reusable named presets for runs we repeat often
    ├── finetune_taco.yaml
    ├── smoke.yaml
    ├── ddp_taco.yaml
    ├── gelan_taco.yaml
    └── overnight_cool.yaml
```

| File / folder | Owns |
| --- | --- |
| `configs/train.yaml` | Run-level training settings: seed, epochs, image size, batch size, task name, close-mosaic timing, weights, resume, checkpoint policy, `single_cls`, `multi_scale`, `freeze`, `noval`, and `nosave`. |
| `configs/data/*.yaml` | Dataset identity and paths (root, train/val image dirs, `nc`, class names) plus augmentation hyperparameters (from `hyp.scratch-high.yaml`) and loader settings (`cache`, `workers`, `rect_train`, `rect_val`). |
| `configs/model/*.yaml` | Model architecture selection (`models/detect/yolov9-c.yaml` or `gelan-c.yaml`), input channels, class count, and anchors. |
| `configs/trainer/*.yaml` | Lightning runtime: accelerator, devices, strategy, precision, sync-batchnorm, max epochs. |
| `configs/optimizer/sgd.yaml` | `lr0`, momentum, weight decay, warmup values (smart_optimizer 3-group setup). |
| `configs/scheduler/linear.yaml` | Learning-rate schedule settings: linear default with `cos_lr`, `flat_cos_lr`, and `fixed_lr` alternatives. |
| `configs/loss/*.yaml` | ComputeLoss target and gains (`box` / `cls` / `dfl`), `reg_max`, TAL top-k, label smoothing. Dual vs single-branch loss selected here. |
| `configs/callbacks/ema.yaml` | EMA decay/tau, checkpoint monitor + `save_period`, close-mosaic epoch, early-stopping patience. |
| `configs/paths/default.yaml` | Root / data / log / output dirs. |
| `configs/inference.yaml` | Inference inputs, thresholds, device selection, output controls, and visualization options. |
| `configs/experiment/*.yaml` | Named presets (`# @package _global_`) bundling the overrides we usually retype. Invoked with `+experiment=<name>` — the `+` appends the group after all others, so a preset can override any group; explicit CLI overrides still beat the preset. |

### Source-of-truth rule

```
train.yaml        = run-level training settings and optional behavior flags
data/*.yaml       = dataset identity, paths, augmentation
model/*.yaml      = network architecture selection
trainer/*.yaml    = hardware / Lightning runtime
optimizer/, scheduler/, loss/, callbacks/ = one concern per group
```

---

## 3. Main training config

`configs/train.yaml` composes the tree and keeps run-level knobs in one place:

```yaml
defaults:
  - _self_
  - data: taco_yolo_300
  - model: yolov9_c
  - trainer: single_gpu
  - optimizer: sgd
  - scheduler: linear
  - loss: yolov9_tal_dual
  - callbacks: ema
  - paths: default

seed: 0
epochs: 300
imgsz: 640
batch_size: 8        # PER GPU under DDP
task_name: yolov9c_taco
close_mosaic: 15
weights: null        # optional pretrained start, e.g. yolov9-c.pt

ckpt_prefix: ckpts             # versioned per run -> ckpts_0, ckpts_1, ...
checkpoint_save_dir: checkpoints
overwrite: false               # true: reuse/replace ckpts_0
resume: null                   # path to a Lightning last.ckpt

# Optional training behavior
single_cls: false    # train all annotations as one class
multi_scale: false   # randomly resize training batches
freeze: [0]          # [10] freezes layers 0-9; [0] freezes none
noval: false         # validate only on the final epoch when true
nosave: false        # write YOLO checkpoints only on the final epoch when true
```

- `batch_size` is **per GPU**. Under DDP, total batch size equals `batch_size × trainer.devices`.
- `resume` takes priority over `weights`: the Lightning checkpoint already contains model, optimizer, scheduler, EMA, and best-fitness state.
- `freeze=[10]` freezes layers 0–9; a multi-element list such as `freeze=[3,5,7]` freezes only those exact indices.
- Precision defaults to `bf16-mixed` because fp16 can NaN when training yolov9-c from scratch.

---

## 4. Data config

`configs/data/taco_yolo_300.yaml` (TACO, `nc=18`):

```yaml
path: /home/heron/projects/yolov9-main/data/taco_yolo_300
train: train/images      # 300 images
val: valid/images        # 195 images
test: null
nc: 18
names: [Aluminium foil, Bottle cap, Bottle, ...]  # 18 classes, order = label IDs

# augmentation (from hyp.scratch-high.yaml)
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
translate: 0.1
scale: 0.9
fliplr: 0.5
mosaic: 1.0
mixup: 0.15
copy_paste: 0.3
cache: false
workers: 8
rect_val: true
rect_train: false        # disables shuffle when rectangular batching is on
```

Rules:

- `names` order defines the class label IDs; `nc` must match `len(names)`.
- `model.nc` resolves from `data.nc` via `${data.nc}` — change the dataset config only.
- The data pipeline returns `(imgs, targets, paths, shapes)` batches with `[n, 6]` targets. Training enables augmentation, while validation uses rectangular batching.

To add a new dataset, copy `taco_yolo_300.yaml`, update paths / `nc` / `names`, then run with `data=<new_name>`.

---

## 5. Training commands and `train.py` argument reference

### Normal training

```bash
uv run python src/train.py
uv run python src/train.py epochs=100 batch_size=4
```

### Experiment presets

`configs/experiment/` stores the settings we usually retype as named presets. The `experiment` group is **not** declared in `train.yaml`'s `defaults` list, so a preset must be invoked with the append prefix `+experiment=<name>` , barely `experiment=<name>` fails with `Could not override 'experiment'. No match in the defaults list.` The appended preset composes after every config group, so it can override any of them; explicit CLI overrides still win over the preset

```bash
# quick correctness check (epochs=1, batch_size=4) — verified end-to-end 2026-07-21:
uv run python src/train.py +experiment=smoke
# multi-GPU (trainer=ddp, per-GPU batch_size=4):
uv run python src/train.py +experiment=ddp_taco trainer.devices=4
# single-branch gelan-c (model=gelan_c, loss=yolov9_tal, weights=gelan-c.pt):
uv run python src/train.py +experiment=gelan_taco
# preset + ad-hoc override (CLI always wins):
uv run python src/train.py +experiment=finetune_taco epochs=100
```

### Writing a new experiment preset

Create `configs/experiment/<name>.yaml` and run it with `+experiment=<name>`:

```yaml
# @package _global_
# ^ REQUIRED first line: lifts every key in this file to the config root.

# (Optional) swap whole config groups with override directives:
defaults:
  - override /trainer: ddp
  - override /model: gelan_c

# Override individual keys at their real location in the composed config:
task_name: my_run        # top-level keys live directly in train.yaml
epochs: 50
batch_size: 4

optimizer:               # nested keys use the config-group name as prefix
  lr0: 0.001

data:
  workers: 4
```

Rules:

1. `# @package _global_` must be the first line. Without it, every key lands under a useless `experiment:` node and `override /group:` errors and prevents Hydra from placing the preset under an unintended `experiment:` node.
2. A preset may only override keys that already exist in the composed config (Hydra struct mode); adding brand-new keys fails.
3. Swap whole groups (`trainer`, `model`, `loss`, …) only via `override /group: value` entries in the preset's own `defaults` list; plain key overrides handle everything else.
4. Invoke with `+experiment=<name>` — the `+` is required because the group is not declared in `train.yaml`.
5. Verify a new preset before relying on it: `uv run python src/train.py +experiment=<name> --cfg job` prints the resolved config without training.

### Smoke tests before a long run

```bash
# 1-epoch smoke:
uv run python src/train.py epochs=1
# fast_dev_run (1 train + 1 val batch):
uv run python src/train.py +trainer.fast_dev_run=true
```

### Pretrained start (fine-tune)

```bash
uv run python src/train.py weights=yolov9-c.pt epochs=50
```

If the checkpoint has a different class count, only intersecting weights transfer (EMA preferred); the 18-class head layers are skipped and trained from scratch.

### Complete `train.py` Hydra argument reference

Hydra uses two different override types:

| Override type | Example | Where it comes from | Purpose |
| --- | --- | --- | --- |
| Config group selector | `model=gelan_c` | The default selection is declared in `configs/train.yaml`; the value chooses a file from `configs/model/`. | Replaces the whole selected config group. |
| Nested field override | `model.nc=18` | The field is defined inside the selected file, such as `configs/model/yolov9_c.yaml`. | Changes one value inside the selected group. |
| Root field override | `epochs=100` | The field is defined directly in `configs/train.yaml`. | Changes a run-level training value. |

Therefore, `model`, `trainer`, `data`, `optimizer`, `scheduler`, `loss`, `callbacks`, and `paths` are selectors declared by the `train.yaml` defaults list. Their nested fields come from the selected files in the corresponding config folders.

### Config group selectors

These selectors choose an entire YAML file. Replace `CONFIG_NAME` with the filename without `.yaml`. For example, `trainer=CONFIG_NAME` becomes `trainer=ddp`.

| Selector syntax | Default command | Selected file |
| --- | --- | --- |
| `data=CONFIG_NAME` | `data=taco_yolo_300` | `configs/data/taco_yolo_300.yaml` |
| `model=CONFIG_NAME` | `model=yolov9_c` | `configs/model/yolov9_c.yaml` |
| `trainer=CONFIG_NAME` | `trainer=single_gpu` | `configs/trainer/single_gpu.yaml` |
| `optimizer=CONFIG_NAME` | `optimizer=sgd` | `configs/optimizer/sgd.yaml` |
| `scheduler=CONFIG_NAME` | `scheduler=linear` | `configs/scheduler/linear.yaml` |
| `loss=CONFIG_NAME` | `loss=yolov9_tal_dual` | `configs/loss/yolov9_tal_dual.yaml` |
| `callbacks=CONFIG_NAME` | `callbacks=ema` | `configs/callbacks/ema.yaml` |
| `paths=CONFIG_NAME` | `paths=default` | `configs/paths/default.yaml` |

### Multi-GPU DDP

```bash
uv run python src/train.py trainer=ddp
uv run python src/train.py trainer=ddp trainer.devices=4
```

DDP notes:

- The DataModule shards the training set, so `trainer.use_distributed_sampler` remains `false` to avoid adding a second sampler.
- Loss is scaled by world size; validation metrics are synchronized across ranks; only rank 0 writes `last.pt` and `best.pt`.
- `trainer=ddp` enables SyncBatchNorm by default.

### Single-branch training

```bash
uv run python src/train.py model=gelan_c loss=yolov9_tal
uv run python src/train.py model=gelan_c loss=yolov9_tal weights=gelan-c.pt epochs=1 batch_size=4
```

All single-branch behavior is selected purely by Hydra config groups — the dual-branch path is unchanged.

### Root fields — defined directly in `configs/train.yaml`

| Argument | Default | Purpose |
| --- | --- | --- |
| `seed` | `0` | Seeds Python, NumPy, PyTorch, and data workers for reproducible runs. |
| `epochs` | `300` | Sets the final number of training epochs. |
| `imgsz` | `640` | Sets the square training and validation image size. |
| `batch_size` | `8` | Sets samples per batch per GPU. |
| `task_name` | `yolov9c_taco` | Names the run or experiment in logs and saved metadata. |
| `close_mosaic` | `15` | Disables mosaic augmentation this many epochs before training ends. |
| `weights` | `null` | Loads pretrained YOLO weights for fine-tuning; ignored when `resume` is set. |
| `resume` | `null` | Restores the full training state from a Lightning `last.ckpt`. |
| `ckpt_prefix` | `ckpts` | Sets the base name of versioned checkpoint folders. |
| `checkpoint_save_dir` | `checkpoints` | Sets the parent directory for checkpoint folders. |
| `overwrite` | `false` | Reuses `ckpts_0` instead of creating the next versioned folder. |
| `single_cls` | `false` | Collapses all labels into one class for class-agnostic detection. |
| `multi_scale` | `false` | Randomly resizes each training batch from about 0.5× to 1.5× `imgsz`. |
| `freeze` | `[0]` | Freezes model layers; `[10]` freezes layers 0–9, while `[3,5,7]` freezes those exact indices. |
| `noval` | `false` | Skips intermediate validation and validates only the final epoch. |
| `nosave` | `false` | Delays YOLO `last.pt` and `best.pt` writing until the final epoch. |

### Data fields — `configs/data/*.yaml`

| Argument | Default | Purpose |
| --- | --- | --- |
| `data.path` | TACO root | Sets the dataset root directory. |
| `data.train` / `data.val` / `data.test` | Dataset-relative paths | Locate the train, validation, and optional test image sets. |
| `data.nc` | `18` | Sets the number of object classes. |
| `data.names` | 18 TACO names | Maps numeric class IDs to class names; order must match labels. |
| `data.hsv_h` / `hsv_s` / `hsv_v` | `0.015 / 0.7 / 0.4` | Control random hue, saturation, and brightness augmentation. |
| `data.degrees` | `0.0` | Sets the maximum random image rotation. |
| `data.translate` | `0.1` | Sets the maximum random horizontal and vertical translation fraction. |
| `data.scale` | `0.9` | Controls random image scaling during geometric augmentation. |
| `data.shear` | `0.0` | Sets the maximum random shear angle. |
| `data.perspective` | `0.0` | Sets the strength of random perspective transformation. |
| `data.flipud` / `data.fliplr` | `0.0 / 0.5` | Set vertical and horizontal flip probabilities. |
| `data.mosaic` | `1.0` | Sets the probability of combining four images into one training sample. |
| `data.mixup` | `0.15` | Sets the probability of blending two training images and labels. |
| `data.copy_paste` | `0.3` | Sets the probability of copy-paste object augmentation. |
| `data.cache` | `false` | Caches images to reduce disk I/O at the cost of RAM or storage. |
| `data.workers` | `8` | Sets DataLoader worker processes per rank. |
| `data.rect_train` / `data.rect_val` | `false / true` | Enable rectangular batching for training or validation; train shuffle is disabled when active. |

### Model fields — `configs/model/*.yaml`

| Argument | Default | Purpose |
| --- | --- | --- |
| `model.cfg` | `models/detect/yolov9-c.yaml` | Points to the upstream network architecture YAML. |
| `model.ch` | `3` | Sets input channels, normally RGB. |
| `model.nc` | `${data.nc}` | Links the detection head class count to the dataset. |
| `model.anchors` | `null` | Supplies anchors when required; YOLOv9-C is anchor-free. |

### Trainer fields — `configs/trainer/*.yaml`

| Argument | Default | Purpose |
| --- | --- | --- |
| `trainer.accelerator` | `gpu` | Selects GPU, CPU, or automatic accelerator handling. |
| `trainer.devices` | `1` | Sets the number or list of devices used by Lightning. |
| `trainer.strategy` | `auto` | Selects the distributed strategy, such as `ddp`. |
| `trainer.sync_batchnorm` | `false` | Synchronizes BatchNorm statistics across GPUs. |
| `trainer.use_distributed_sampler` | `false` | Keeps Lightning from adding another sampler because the DataModule already shards data. |
| `trainer.precision` | `bf16-mixed` | Sets numerical precision; BF16 avoids the observed FP16 NaN issue. |
| `trainer.max_epochs` | `${epochs}` | Passes the root epoch count into Lightning. |
| `trainer.accumulate_grad_batches` | `1` | Must stay at 1 because YOLO warmup accumulation is handled manually. |
| `trainer.gradient_clip_val` | `null` | Optionally clips gradient magnitude. |
| `+trainer.fast_dev_run` | Not set | Runs one train and one validation batch for a quick pipeline check. |

### Optimizer fields — `configs/optimizer/*.yaml`

| Argument | Default | Purpose |
| --- | --- | --- |
| `optimizer.name` | `SGD` | Field inside the selected optimizer config that chooses the optimizer implementation. |
| `optimizer.lr0` | `0.01` | Sets the initial learning rate. |
| `optimizer.momentum` | `0.937` | Controls SGD momentum. |
| `optimizer.weight_decay` | `0.0005` | Applies L2-style regularization to decay-weight parameter groups. |
| `optimizer.warmup_epochs` | `3.0` | Sets the duration of learning-rate and momentum warmup. |
| `optimizer.warmup_momentum` | `0.8` | Sets momentum at the start of warmup. |
| `optimizer.warmup_bias_lr` | `0.1` | Sets the initial warmup learning rate for bias parameters. |

### Scheduler fields — `configs/scheduler/*.yaml`

| Argument | Default | Purpose |
| --- | --- | --- |
| `scheduler.type` | `linear` | Field inside the selected scheduler config that identifies the base schedule. |
| `scheduler.lrf` | `0.01` | Sets the final learning rate as a fraction of `lr0`. |
| `scheduler.cos_lr` | `false` | Uses cosine one-cycle decay instead of linear decay. |
| `scheduler.flat_cos_lr` | `false` | Keeps LR flat initially, then applies cosine decay. |
| `scheduler.fixed_lr` | `false` | Keeps the learning-rate multiplier constant. |

### Loss fields — `configs/loss/*.yaml`

| Argument | Default | Purpose |
| --- | --- | --- |
| `loss._target_` | `utils.loss_tal_dual.ComputeLoss` | Field inside the selected loss config that names the concrete loss class. |
| `loss.box` / `loss.cls` / `loss.dfl` | `7.5 / 0.5 / 1.5` | Weight box-regression, classification, and distribution-focal loss. |
| `loss.reg_max` | `16` | Sets the number of discrete bins used by DFL box regression. |
| `loss.tal_topk` | `10` | Sets how many candidate anchors Task-Aligned Assignment considers per target. |
| `loss.label_smoothing` | `0.0` | Softens classification targets to reduce overconfidence. |

### Callback fields — `configs/callbacks/*.yaml`

| Argument | Default | Purpose |
| --- | --- | --- |
| `callbacks.ema.decay` / `tau` | `0.9999 / 2000` | Control exponential moving-average weight smoothing and its ramp. |
| `callbacks.checkpoint.monitor` / `mode` | `fitness / max` | Select the validation score and improvement direction for `best.pt`. |
| `callbacks.checkpoint.save_last` | `true` | Keeps the latest checkpoint state. |
| `callbacks.checkpoint.save_period` | `-1` | Writes `epoch{n}.pt` every N epochs; `-1` disables periodic files. |
| `callbacks.early_stopping.patience` | `100` | Stops after this many validation checks without fitness improvement. |

### Path fields — `configs/paths/*.yaml`

| Argument | Default | Purpose |
| --- | --- | --- |
| `paths.root_dir` / `data_dir` / `log_dir` / `output_dir` | Project-derived paths | Define common project, dataset, log, and Hydra output locations. |

---

## 6. Resume and checkpoint files

### Checkpoint folder versioning

Checkpoints go to `<checkpoint_save_dir>/ckpts_{n}` with an auto-incrementing suffix so runs never clobber each other:

```
checkpoints/ckpts_0
checkpoints/ckpts_1
checkpoints/ckpts_2
```

```bash
# reuse/replace ckpts_0 instead of adding a new index:
uv run python src/train.py overwrite=true
# save checkpoints under a chosen folder:
uv run python src/train.py checkpoint_save_dir=runs/exp1
```

### Files written

| File | Purpose |
| --- | --- |
| `last.pt` | YOLOv9 checkpoint containing the latest model, EMA, optimizer metadata, epoch, and fitness state; written after validation. |
| `best.pt` | Same format, written when val fitness improves. |
| `epoch{n}.pt` | Optional periodic checkpoints controlled by `callbacks.checkpoint.save_period`; `-1` disables them. |
| `last.ckpt` | Lightning-format twin of `last.pt`: carries optimizer / scheduler / loop state plus EMA and best-fitness via callback state. Used by `resume=`. |

### Resume an interrupted run

```bash
uv run python src/train.py resume=checkpoints/ckpts_0/last.ckpt overwrite=true
```

Resume notes:

- Resume uses the Lightning `last.ckpt`, not `last.pt`.
- `weights=` is ignored when `resume=` is set (the checkpoint already has trained weights).

---

## 7. Standalone checkpoint evaluation with `val.py`

Training already runs validation at the end of every validation epoch through the Lightning module. Use the root-level legacy `val.py` when you want to evaluate a saved YOLO-compatible `.pt` checkpoint independently, create validation plots, or export prediction files. This script is retained for standalone evaluation compatibility; it is not the main training entrypoint.

`val.py` accepts the `.pt` files written by `src/train.py`, such as `checkpoints/ckpts_0/best.pt` and `last.pt`. The dataset YAML and checkpoint must describe the same number and order of classes.

### Evaluate the TACO validation split

```bash
uv run python val.py \
  --data data/taco_yolo_300/taco_yolo_300.yaml \
  --weights checkpoints/ckpts_0/best.pt \
  --imgsz 640 \
  --batch-size 8 \
  --conf-thres 0.001 \
  --iou-thres 0.7 \
  --device 0 \
  --name taco_val
```

The command reports precision (P), recall (R), mAP@0.5, and mAP@0.5:0.95. It writes plots and any requested exports to `runs/val/taco_val` (or an incremented directory such as `taco_val2`). Keep `--conf-thres 0.001` for valid mAP measurement; a higher threshold intentionally filters detections and makes the reported mAP non-comparable.

### Useful standalone evaluation variants

```bash
# Print a metric row for each class:
uv run python val.py --data data/taco_yolo_300/taco_yolo_300.yaml \
  --weights checkpoints/ckpts_0/best.pt --device 0 --verbose

# Save normalized YOLO prediction labels with confidence values:
uv run python val.py --data data/taco_yolo_300/taco_yolo_300.yaml \
  --weights checkpoints/ckpts_0/best.pt --device 0 \
  --save-txt --save-conf --name taco_val_labels

# Save COCO-style prediction JSON; use this only with a dataset that has matching COCO annotations:
uv run python val.py --data data/coco.yaml --weights yolov9-c-converted.pt \
  --imgsz 640 --batch-size 32 --conf-thres 0.001 --iou-thres 0.7 \
  --device 0 --save-json --name yolov9_c_val
```

Important options:

- `--data`: dataset YAML, including `val`, `nc`, and `names`.
- `--weights`: path to the YOLO-compatible `.pt` checkpoint to evaluate.
- `--batch-size`: validation batch size; reduce it if GPU memory is insufficient.
- `--imgsz`: square evaluation image size; use the same size as the training run for comparable results.
- `--device`: GPU index such as `0`, a list such as `0,1`, or `cpu`.
- `--save-txt` and `--save-conf`: save normalized YOLO prediction files under `runs/val/<name>/labels/`.
- `--save-json`: writes a COCO-format prediction JSON and attempts COCO evaluation when the matching annotations are available.

Do not use `--save-hybrid` for a real model-quality report: it mixes ground-truth labels into the saved predictions and can inflate mAP. `val.py` is the standalone route only; for an end-to-end pipeline smoke check, use `uv run python src/train.py +trainer.fast_dev_run=true`.

---

## 8. Reparameterization (dual → single branch)

A trained yolov9-c checkpoint has a dual-branch `DualDDetect` head (the auxiliary/PGI branch only helps training). Convert it into the deploy-ready single-branch `yolov9-c-converted` model (≈half the params):

```bash
# convert a trained ckpt (EMA preferred):
uv run python src/reparameterize.py --weights checkpoints/ckpts_0/best.pt
# custom output path / model yaml / device:
uv run python src/reparameterize.py --weights best.pt --output best-converted.pt
# sanity check on the pretrained release ckpt (nc=80):
uv run python src/reparameterize.py --weights yolov9-c.pt
```

Behavior:

- Output defaults to `<weights>-converted.pt` next to the input.
- The script self-verifies before writing: the converted model output must match the dual model's lead branch (`pred[0][1]`) within `1e-4` on a random input.
- The saved checkpoint uses FP16 weights and excludes optimizer state for deployment.
- ONNX / TensorRT export is out of scope for now.

---

## 9. Inference commands

`weights` and `source` are mandatory — the run fails loudly if either is missing.

### Folder of images

```bash
uv run python src/inference.py \
  weights=checkpoints/ckpts_0/best.pt \
  source=data/taco_yolo_300/valid/images
```

### Single image, custom threshold

```bash
uv run python src/inference.py weights=best-converted.pt source=test.jpg conf_thres=0.4
```

### Save YOLO-format labels (+ confidence)

```bash
uv run python src/inference.py \
  weights=checkpoints/ckpts_0/best.pt \
  source=data/taco_yolo_300/valid/images \
  save_txt=true save_conf=true
```

Outputs land in `runs/detect/exp`, `exp2`, `exp3`, and so on, with annotated images and optional `labels/` text files.

Useful inference options:

- `conf_thres` / `iou_thres` / `max_det`: NMS settings (defaults 0.25 / 0.45 / 1000).
- `device`: `""` = auto, or `"0"`, `"0,1"`, `"cpu"`.
- `classes=[0,2]`: filter classes; `agnostic_nms=true` for class-agnostic NMS.
- `save_crop=true`: save cropped detections; `nosave=true`: skip annotated images.
- `backend_data=<yaml>`: provide backend dataset metadata; the key cannot be `data` because Hydra already uses that name for the data config group.
- `visualize=true`: write per-image feature-map directories.
- `update=true`: destructively strip optimizer state from the checkpoint after successful inference.
- `half=true`: fp16 inference.
- `project` / `name` / `exist_ok`: output layout control.

Both checkpoint families load with no extra flag: `attempt_load` prefers `ckpt["ema"]` over `ckpt["model"]`, and dual vs converted heads are auto-detected from the output structure.

Scope: images / folders / globs only — video, webcam, stream sources, and ONNX / TensorRT backends are deferred.

### Complete `inference.py` argument reference

Use `uv run python src/inference.py key=value ...`. `weights` and `source` are required.

| Argument | Default | Purpose |
| --- | --- | --- |
| `weights` | `null` — required | Path to a dual-branch or converted YOLO checkpoint. |
| `source` | `null` — required | Input image, image folder, or glob pattern. |
| `backend_data` | `null` | Dataset metadata YAML passed to the inference backend. |
| `data` | `taco_yolo_300` | Selects the class-name dataset config group. |
| `imgsz` | `640` | Sets the square inference image size, adjusted to the model stride. |
| `conf_thres` | `0.25` | Removes detections below this confidence score. |
| `iou_thres` | `0.45` | Sets the IoU threshold used by non-maximum suppression. |
| `max_det` | `1000` | Limits detections retained per image. |
| `device` | `""` | Selects automatic device, a GPU such as `0`, multiple GPUs, or `cpu`. |
| `view_img` | `false` | Displays annotated images in an OpenCV window. |
| `save_txt` | `false` | Saves detections as normalized YOLO-format label files. |
| `save_conf` | `false` | Adds confidence scores to saved label rows; requires `save_txt=true`. |
| `save_crop` | `false` | Saves each detected object as a cropped image grouped by class. |
| `nosave` | `false` | Prevents annotated images from being written. |
| `classes` | `null` | Filters output to selected class IDs, for example `[0,2]`. |
| `agnostic_nms` | `false` | Makes overlapping boxes suppress one another regardless of class. |
| `augment` | `false` | Enables augmented inference in the model forward pass. |
| `visualize` | `false` | Saves per-image feature-map visualization directories. |
| `update` | `false` | Destructively strips optimizer state from the source checkpoint after success. |
| `half` | `false` | Uses FP16 inference when supported. |
| `dnn` | `false` | Requests OpenCV DNN for supported non-PyTorch backends; currently outside the tested image/PT scope. |
| `vid_stride` | `1` | Processes every Nth video frame; video sources are currently deferred. |
| `project` | `runs/detect` | Sets the parent output directory. |
| `name` | `exp` | Sets the run subdirectory name. |
| `exist_ok` | `false` | Reuses the named output directory instead of creating `exp2`, `exp3`, and so on. |
| `line_thickness` | `3` | Sets the pixel width of drawn bounding boxes. |
| `hide_labels` | `false` | Draws boxes without class-name text. |
| `hide_conf` | `false` | Shows class names but omits confidence values. |

---

## 10. Before a Reakun

Before a real run:

```bash
# Compose the config, build the model, and run one train and validation batch:
uv run python src/train.py +trainer.fast_dev_run=true
```

After changing checkpoint conversion code:

```bash
# Verify dual-to-converted checkpoint mapping and output consistency:
uv run pytest tests/test_reparam_parity.py -v
```

A healthy training start should show:

- The fully-resolved config printed (and written to `train.log`) with `nc=18` and 18 class names.
- Finite `box` / `cls` / `dfl` training losses.
- Validation logging P / R / mAP50 / mAP50-95 and a `fitness` value.
- `last.pt` / `last.ckpt` appearing under the versioned `ckpts_{n}` folder after the first validation.

---

## 11. Common pitfalls

- If Hydra rejects `weights=...`, confirm that `weights: null` exists in `configs/train.yaml`.
- Quote list overrides in the shell: `'freeze=[10]'`, otherwise the brackets are eaten by the shell.
- `batch_size` is per GPU under `trainer=ddp` — divide the intended total by `trainer.devices`.
- If resume trains zero epochs, remember `epochs` is the final target epoch, not the number of extra epochs.
- Resume needs the Lightning `last.ckpt`; inference and deployment use `last.pt` or `best.pt`. These files are written to the same checkpoint folder.
- Keep `precision: bf16-mixed`; fp16 can produce NaN losses when training yolov9-c from scratch.
- `update=true` on inference mutates the source checkpoint (strips optimizer) — do not use it on a checkpoint you still want to resume from.
- If val metrics warn about divide-by-zero, at least one validation class has zero samples
- Experiment presets are invoked with `+experiment=<name>` — bare `experiment=<name>` fails with "Could not override 'experiment'. No match in the defaults list." Presets must start with `# @package _global_` and can only override keys that already exist (Hydra struct mode) — swap whole groups with `override /group:` entries inside the preset.

---

## 12. Upstream YOLOv9 references, benchmarks, and attribution

This project modernizes the YOLOv9 object-detection workflow with PyTorch Lightning and Hydra. It retains required upstream runtime components while providing a new training, validation, checkpointing, reparameterization, and inference interface. The project currently covers object detection only; upstream segmentation, panoptic-segmentation, and image-captioning workflows are outside this repository's documented scope.

### Upstream resources

- YOLOv9 paper — Learning What You Want to Learn Using Programmable Gradient Information
- Original YOLOv9 repository
- Official YOLOv9 release checkpoints

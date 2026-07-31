# YOLOv9 Hydra configuration

Select the optimizer and model preset in `configs/train.yaml`.

```yaml
defaults:
  - model: dual/c
  - optimizer: sgd
```

## Optimizers

| Hydra option | Optimizer |
|---|---|
| `optimizer: sgd` | SGD |
| `optimizer: adam` | Adam |
| `optimizer: adamw` | AdamW |
| `optimizer: lion` | LION |

## Model presets

| Branch | Sizes | Example |
|---|---|---|
| Single (GELAN / DDetect) | T, S, M, C, E | `model: single/s` |
| Dual (YOLOv9 / DualDDetect) | T, S, M, C, E | `model: dual/e` |
| Triple (YOLOv9-CF / TripleDDetect) | C only | `model: triple/c` |

Each model preset selects its compatible loss automatically. The upstream repository only provides the official triple-branch C/CF architecture; unsupported triple T/S/M/E variants are intentionally not exposed.

Command-line overrides also work:

```bash
python -m src.train model=single/t optimizer=adam
python -m src.train model=dual/e optimizer=adamw
python -m src.train model=triple/c optimizer=lion
```

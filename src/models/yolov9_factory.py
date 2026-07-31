"""src/models/yolov9_factory.py

M2 model factory. Rebuilds YOLOv9-C by WRAPPING the upstream models.yolo.Model
-- no reimplementation -- so the DualDDetect head, strides, and block structure
stay identical to the M0 baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch.nn as nn

# The upstream YOLOv9 code (models/yolo.py, models/common.py, utils/, ...) lives
# at the repo root. This factory sits at <repo>/src/models/. Add the repo root
# to sys.path so the TOP-LEVEL `models` package resolves to the upstream repo,
# not to this src/models subpackage. Import this factory as
# `src.models.yolov9_factory`, never as top-level `models.yolov9_factory`, or the
# two `models` names will clash.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.yolo import Model  # noqa: E402  upstream YOLOv9 model


def build_yolov9_model(
    cfg: str,
    ch: int = 3,
    nc: Optional[int] = None,
    anchors=None,
) -> nn.Module:
    """Build the original YOLOv9-C detection model (with the DualDDetect head).

    Args:
        cfg: model yaml path, e.g. "models/detect/yolov9-c.yaml". Resolved
             relative to the repo root when not absolute.
        ch: input channels (3 for RGB).
        nc: number of classes; overrides the yaml `nc` when provided.
        anchors: anchor override (None for the anchor-free yolov9-c).

    Returns:
        The upstream `Model` instance.
    """
    cfg_path = Path(cfg)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Model cfg not found: {cfg_path}")

    # Model.__init__ prints the build summary (layers / params / GFLOPs) via
    # self.info(); that printout is the parity target from the M0 baseline.
    model = Model(str(cfg_path), ch=ch, nc=nc, anchors=anchors)
    return model


def count_parameters(model: nn.Module) -> int:
    """Total parameter count (parity metric)."""
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # Quick manual parity check.
    m = build_yolov9_model("models/detect/yolov9-c.yaml", ch=3, nc=60, anchors=None)
    print(f"parameters: {count_parameters(m):,}")
    # Baseline target: 962 layers / 51,038,860 params / 239.1 GFLOPs
"""src/reparameterize.py

M6: convert a trained YOLOv9-C DUAL-branch checkpoint (DualDDetect; the
auxiliary/PGI branch only helps training) into the deploy-ready SINGLE-branch
yolov9-c-converted model (DDetect) -- the same operation as upstream
tools/reparameterization.ipynb (yolov9-c cell).

Key mapping (ported from the upstream notebook):
  converted model.{i}.*    <- dual model.{i+1}.*   for i < 22   (Silence at 0)
  converted model.22.cv2.* <- dual model.38.cv4.*  (lead box head)
  converted model.22.cv3.* <- dual model.38.cv5.*  (lead cls head)
  converted model.22.dfl.* <- dual model.38.dfl2.* (lead DFL)

Self-verifying: before anything is written, the converted model's inference
output must match the dual model's lead branch (detect_dual.py uses
pred[0][1]) on a random input.
"""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.yolo import Model  # noqa: E402  upstream DetectionModel

BOUNDARY = 22      # DDetect layer index in the converted graph
HEAD_OFFSET = 16   # 22 + 16 = 38 = DualDDetect layer index in yolov9-c


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def pick_cfg(cfg=None):
    """yolov9-c-converted.yaml if present, else gelan-c.yaml (same graph)."""
    if cfg:
        return resolve(cfg)
    for name in ("models/detect/yolov9-c-converted.yaml", "models/detect/gelan-c.yaml"):
        p = REPO_ROOT / name
        if p.exists():
            return p
    raise FileNotFoundError("No models/detect/yolov9-c-converted.yaml or gelan-c.yaml found")


def convert_state_dict(dual_sd, converted_model):
    """Build the converted state dict via the upstream notebook mapping.

    Returns (new_sd, skipped). `skipped` lists converted keys with no mapped
    source (e.g. non-parameter buffers such as stride, if this checkout
    registers them in the state dict).
    """
    new_sd, skipped = {}, []
    conv_sd = converted_model.state_dict()
    for k, v in conv_sd.items():
        idx = int(k.split(".")[1])
        if idx < BOUNDARY:
            kr = k.replace(f"model.{idx}.", f"model.{idx + 1}.", 1)
        elif f"model.{idx}.cv2." in k:
            kr = k.replace(f"model.{idx}.cv2.", f"model.{idx + HEAD_OFFSET}.cv4.", 1)
        elif f"model.{idx}.cv3." in k:
            kr = k.replace(f"model.{idx}.cv3.", f"model.{idx + HEAD_OFFSET}.cv5.", 1)
        elif f"model.{idx}.dfl." in k:
            kr = k.replace(f"model.{idx}.dfl.", f"model.{idx + HEAD_OFFSET}.dfl2.", 1)
        else:
            skipped.append(k)
            continue
        if kr not in dual_sd:
            raise KeyError(f"source key missing in dual ckpt: {kr} (for {k})")
        if dual_sd[kr].shape != v.shape:
            raise ValueError(
                f"shape mismatch {k}: {tuple(v.shape)} vs source {tuple(dual_sd[kr].shape)}"
            )
        new_sd[k] = dual_sd[kr].clone()
    return new_sd, skipped


def load_dual(weights, device="cpu"):
    """Load our YOLOv9-dict ckpt (EMA preferred) or an upstream release ckpt."""
    ckpt = torch.load(str(resolve(weights)), map_location=device, weights_only=False)
    return (ckpt.get("ema") or ckpt["model"]).float().to(device).eval()


def reparameterize(weights, cfg=None, output=None, device="cpu", imgsz=640, verify=True):
    dual = load_dual(weights, device)
    nc = int(getattr(dual, "nc", dual.model[-1].nc))
    cfg_path = pick_cfg(cfg)
    converted = Model(str(cfg_path), ch=3, nc=nc, anchors=3).float().to(device).eval()

    new_sd, skipped = convert_state_dict(dual.state_dict(), converted)
    converted.load_state_dict(new_sd, strict=False)
    print(f"[map] copied {len(new_sd)}/{len(converted.state_dict())} tensors"
          + (f", skipped (no source): {skipped}" if skipped else ""))

    converted.nc = nc
    converted.names = getattr(dual, "names", {i: str(i) for i in range(nc)})

    if verify:
        torch.manual_seed(0)
        x = torch.rand(1, 3, imgsz, imgsz, device=device)
        with torch.no_grad():
            lead = dual(x)[0][1]       # lead branch, as detect_dual.py (pred[0][1])
            single = converted(x)[0]
        diff = (lead - single).abs().max().item()
        print(f"[verify] max |dual lead - converted| = {diff:.3e}")
        assert diff < 1e-4, "converted output does not match the dual lead branch"

    wpath = resolve(weights)
    out = resolve(output) if output else wpath.with_name(wpath.stem + "-converted.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt_out = {  # same fields the upstream notebook writes
        "model": deepcopy(converted).half(),
        "optimizer": None, "best_fitness": None, "ema": None, "updates": None,
        "opt": None, "git": None, "date": datetime.now().isoformat(), "epoch": -1,
    }
    torch.save(ckpt_out, str(out))
    n_dual = sum(p.numel() for p in dual.parameters())
    n_conv = sum(p.numel() for p in converted.parameters())
    print(f"[done] params: dual {n_dual:,} -> converted {n_conv:,}")
    print(f"[done] wrote {out}")
    return out


def main():
    ap = argparse.ArgumentParser(
        description="YOLOv9-C dual -> single branch reparameterization")
    ap.add_argument("--weights", required=True,
                    help="trained dual ckpt (best.pt / last.pt / yolov9-c.pt)")
    ap.add_argument("--cfg", default=None, help="converted model yaml (default: auto-detect)")
    ap.add_argument("--output", default=None, help="output path (default: <weights>-converted.pt)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()
    reparameterize(a.weights, a.cfg, a.output, a.device, a.imgsz, verify=not a.no_verify)


if __name__ == "__main__":
    main()
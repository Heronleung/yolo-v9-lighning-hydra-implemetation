"""src/checkpoint.py

M4 YOLOv9 checkpoint compatibility layer.
- save_yolov9_ckpt(): writes the original YOLOv9 dict
  (epoch / best_fitness / model / ema / updates / optimizer / opt / date)
- load_pretrained(): loads a trusted local ckpt (weights_only=False),
  preferring EMA weights and intersecting with the current model
- YOLOv9Checkpoint callback: last.pt every validation, best.pt on improved
  val/fitness; best_fitness persists into the Lightning .ckpt for resume;
  save_period (epoch{n}.pt every n epochs) and nosave (final epoch only)
  mirror the upstream --save-period / --nosave flags
- strip(): strip_optimizer wrapper for deploy-ready weights
"""

from __future__ import annotations

import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch

try:
    import lightning.pytorch as pl
except ImportError:  # older install name
    import pytorch_lightning as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.torch_utils import de_parallel  # noqa: E402  upstream


def save_yolov9_ckpt(path, pl_module, ema=None, optimizer=None,
                     epoch=-1, best_fitness=0.0, opt=None):
    """Write the original YOLOv9 checkpoint dict (frozen contract from M1)."""
    ckpt = {
        "epoch": epoch,
        "best_fitness": best_fitness,
        "model": deepcopy(de_parallel(pl_module.model)).half(),
        "ema": deepcopy(ema.ema).half() if ema is not None else None,
        "updates": getattr(ema, "updates", 0) if ema is not None else 0,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "opt": opt or {},
        "git": None,
        "date": datetime.now().isoformat(),
    }
    torch.save(ckpt, str(path))


def load_pretrained(pl_module, weights_path, device="cpu"):
    """Load trusted local weights (e.g. yolov9-c.pt), preferring EMA."""
    from utils.general import intersect_dicts

    wpath = Path(weights_path)
    if not wpath.is_absolute():
        wpath = REPO_ROOT / wpath  # Hydra chdir's into outputs/, so anchor to repo root
    if not wpath.exists():
        raise FileNotFoundError(
            f"Pretrained weights not found: {wpath} -- download with: "
            "wget -P <repo_root> "
            "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt"
        )
    ckpt = torch.load(str(wpath), map_location=device, weights_only=False)
    src = (ckpt.get("ema") or ckpt["model"]).float().state_dict()
    csd = intersect_dicts(src, pl_module.model.state_dict())  # drop mismatched (nc head)
    pl_module.model.load_state_dict(csd, strict=False)
    print(f"Transferred {len(csd)}/{len(pl_module.model.state_dict())} items from {weights_path}")
    return ckpt


def strip(path):
    """Strip optimizer from a saved ckpt (deploy-ready, like end of train_dual.py)."""
    from utils.general import strip_optimizer
    strip_optimizer(str(path))


class YOLOv9Checkpoint(pl.Callback):
    """last.pt on every validation; best.pt when val/fitness improves."""

    def __init__(self, dirpath="ckpts", save_period=-1, nosave=False):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.save_period = int(save_period)  # upstream --save-period: epoch{n}.pt every n epochs (-1 = off)
        self.nosave = bool(nosave)           # upstream --nosave: write last/best only on the final epoch
        self.best_fitness = 0.0

    def state_dict(self):
        # Persisted inside the Lightning .ckpt so a resumed run keeps the
        # best.pt threshold (upstream --resume restores best_fitness too).
        return {"best_fitness": float(self.best_fitness)}

    def load_state_dict(self, state_dict):
        self.best_fitness = float(state_dict.get("best_fitness", 0.0))

    def _ema(self, trainer):
        for cb in trainer.callbacks:
            if cb.__class__.__name__ == "EMACallback":
                return cb.ema
        return None

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or trainer.fast_dev_run:
            return
        if not trainer.is_global_zero:
            return  # M5 (DDP): only rank 0 writes checkpoints, like train_dual.py
        fit = trainer.callback_metrics.get("val/fitness")
        fit = float(fit) if fit is not None else 0.0
        is_best = fit >= self.best_fitness
        self.best_fitness = max(self.best_fitness, fit)
        epoch = trainer.current_epoch
        final_epoch = (epoch + 1) >= int(trainer.max_epochs or 0)
        if self.nosave and not final_epoch:
            return  # upstream --nosave: keep tracking best_fitness, write only at the end
        self.dirpath.mkdir(parents=True, exist_ok=True)
        kwargs = dict(
            pl_module=pl_module,
            ema=self._ema(trainer),
            optimizer=trainer.optimizers[0] if trainer.optimizers else None,
            epoch=trainer.current_epoch,
            best_fitness=self.best_fitness,
        )
        save_yolov9_ckpt(self.dirpath / "last.pt", **kwargs)
        if is_best:
            save_yolov9_ckpt(self.dirpath / "best.pt", **kwargs)
        # Upstream --save-period: keep a full epoch{n}.pt every n epochs (epoch > 0).
        if self.save_period > 0 and epoch > 0 and epoch % self.save_period == 0:
            save_yolov9_ckpt(self.dirpath / f"epoch{epoch}.pt", **kwargs)
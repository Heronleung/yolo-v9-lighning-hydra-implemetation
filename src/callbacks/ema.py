"""src/callbacks/ema.py

M4 EMACallback: wraps the original utils.torch_utils.ModelEMA.
- M8: updates EMA weights ONLY on batches where the optimizer actually stepped
  (YOLOv9LitModule sets `stepped_this_batch` under manual optimization),
  matching train_dual.py which calls ema.update() inside the accumulate block
- swaps EMA weights IN for validation and back OUT afterwards, matching
  train_dual.py which validates with ema.ema
- resume: persists EMA weights + updates counter into the Lightning .ckpt
  (state_dict/load_state_dict), re-applied in on_fit_start after ModelEMA is
  rebuilt -- mirrors upstream --resume restoring ckpt["ema"] / ckpt["updates"]
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import lightning.pytorch as pl
except ImportError:  # older install name
    import pytorch_lightning as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.torch_utils import ModelEMA, de_parallel  # noqa: E402  upstream


class EMACallback(pl.Callback):
    def __init__(self, decay: float = 0.9999, tau: int = 2000):
        super().__init__()
        self.decay = decay
        self.tau = tau
        self.ema = None
        self._backup = None
        self._restored_state = None  # set by load_state_dict on resume

    def on_fit_start(self, trainer, pl_module):
        # Model is already on-device here, so the EMA copy lands on-device too.
        self.ema = ModelEMA(de_parallel(pl_module.model), decay=self.decay, tau=self.tau)
        if self._restored_state is not None:
            # Resume: Lightning calls load_state_dict BEFORE on_fit_start (while
            # the ModelEMA doesn't exist yet), so the actual restore happens here.
            ema_sd, updates = self._restored_state
            if ema_sd is not None:
                device = next(self.ema.ema.parameters()).device
                self.ema.ema.load_state_dict(
                    {k: v.to(device) for k, v in ema_sd.items()}
                )
                self.ema.updates = int(updates)
            self._restored_state = None

    def state_dict(self):
        # Persisted inside the Lightning .ckpt (saved by the ModelCheckpoint
        # twin in src/train.py) so resume continues the EMA exactly: weights
        # AND the updates counter that drives the decay ramp d(updates).
        if self.ema is None:
            return {"ema": None, "updates": 0}
        return {
            "ema": {k: v.detach().cpu() for k, v in self.ema.ema.state_dict().items()},
            "updates": int(self.ema.updates),
        }

    def load_state_dict(self, state_dict):
        self._restored_state = (state_dict.get("ema"), state_dict.get("updates", 0))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # M8: train_dual.py updates EMA only when the optimizer actually steps.
        # Fall back to every-batch updates if the module doesn't expose the flag.
        if self.ema is not None and getattr(pl_module, "stepped_this_batch", True):
            self.ema.update(de_parallel(pl_module.model))

    def on_validation_start(self, trainer, pl_module):
        if self.ema is None or trainer.sanity_checking:
            return
        # Copy non-parameter attrs, then swap EMA weights in (like train_dual.py).
        self.ema.update_attr(
            de_parallel(pl_module.model),
            include=["yaml", "nc", "hyp", "names", "stride", "class_weights"],
        )
        self._backup = {k: v.detach().clone()
                        for k, v in pl_module.model.state_dict().items()}
        pl_module.model.load_state_dict(self.ema.ema.state_dict(), strict=True)

    def on_validation_end(self, trainer, pl_module):
        if self._backup is not None:
            pl_module.model.load_state_dict(self._backup, strict=True)
            self._backup = None
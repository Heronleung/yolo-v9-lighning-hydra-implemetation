"""src/callbacks/close_mosaic.py

M4 CloseMosaicCallback: disables mosaic augmentation at
epoch == max_epochs - disable_at, matching train_dual.py's close_mosaic.
"""

from __future__ import annotations

try:
    import lightning.pytorch as pl
except ImportError:  # older install name
    import pytorch_lightning as pl


class CloseMosaicCallback(pl.Callback):
    def __init__(self, disable_at: int = 15):
        super().__init__()
        self.disable_at = int(disable_at)
        self._done = False

    def on_train_epoch_start(self, trainer, pl_module):
        if self._done:
            return
        if trainer.current_epoch == trainer.max_epochs - self.disable_at:
            ds = getattr(getattr(trainer, "datamodule", None), "train_dataset", None)
            if ds is not None and hasattr(ds, "mosaic"):
                print("Closing dataloader mosaic")
                ds.mosaic = False
                self._done = True
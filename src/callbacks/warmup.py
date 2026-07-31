"""src/callbacks/warmup.py

M4 WarmupCallback: reproduces the per-iteration warmup ramp from train_dual.py.
Over the first nw iterations (nw = max(round(warmup_epochs * nb), 100)):
  - each param group's lr is interpolated from its warm start (bias group
    starts at warmup_bias_lr, others at 0.0) up to initial_lr * lf(epoch)
  - momentum is interpolated from warmup_momentum to the target momentum
"""

from __future__ import annotations

import numpy as np

try:
    import lightning.pytorch as pl
except ImportError:  # older install name
    import pytorch_lightning as pl


class WarmupCallback(pl.Callback):
    def __init__(self, warmup_epochs=3.0, warmup_momentum=0.8,
                 warmup_bias_lr=0.1, momentum=0.937):
        super().__init__()
        self.warmup_epochs = float(warmup_epochs)
        self.warmup_momentum = float(warmup_momentum)
        self.warmup_bias_lr = float(warmup_bias_lr)
        self.momentum = float(momentum)
        self.nb = None
        self.nw = None

    def on_train_start(self, trainer, pl_module):
        self.nb = max(int(trainer.num_training_batches), 1)
        self.nw = max(round(self.warmup_epochs * self.nb), 100)  # as train_dual.py

    def _epoch_lr_factor(self, trainer):
        # Reuse the LambdaLR lambda so the warmup target matches the schedule.
        try:
            sched = trainer.lr_scheduler_configs[0].scheduler
            return float(sched.lr_lambdas[0](trainer.current_epoch))
        except Exception:
            return 1.0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        ni = batch_idx + self.nb * trainer.current_epoch  # integrated batches
        if self.nw is None or ni > self.nw:
            return
        optimizer = trainer.optimizers[0]
        factor = self._epoch_lr_factor(trainer)
        xi = [0, self.nw]
        for j, g in enumerate(optimizer.param_groups):
            # smart_optimizer group order: 0 = biases, 1 = decay weights, 2 = BN
            start = self.warmup_bias_lr if j == 0 else 0.0
            target = g.get("initial_lr", g["lr"]) * factor
            g["lr"] = float(np.interp(ni, xi, [start, target]))
            if "momentum" in g:
                g["momentum"] = float(
                    np.interp(ni, xi, [self.warmup_momentum, self.momentum])
                )
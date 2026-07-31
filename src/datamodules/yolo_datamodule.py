"""src/data/yolo_datamodule.py

M3 LightningDataModule wrapping the upstream YOLOv9 create_dataloader, so the
collate output and [n, 6] target format stay identical to train_dual.py.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Optional

try:
    import lightning.pytorch as pl
except ImportError:  # older install name
    import pytorch_lightning as pl

from omegaconf import DictConfig, OmegaConf

# Make the upstream YOLOv9 repo importable (utils.dataloaders lives at the root).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.dataloaders import create_dataloader  # noqa: E402  upstream YOLOv9

# Only pass kwargs this YOLOv9 version's create_dataloader actually accepts
# (e.g. `seed` exists on newer checkouts but not older ones).
_ACCEPTED_KWARGS = set(inspect.signature(create_dataloader).parameters)

# Keys create_dataloader reads off `hyp` for augmentation.
_HYP_KEYS = (
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
    "perspective", "flipud", "fliplr", "mosaic", "mixup", "copy_paste",
)


class YOLODataModule(pl.LightningDataModule):
    """Wraps create_dataloader for train (augment) and val (rect) splits."""

    def __init__(
        self,
        data_cfg,
        imgsz: int = 640,
        batch_size: int = 8,
        stride: int = 32,
        workers: int = 8,
        single_cls: bool = False,
        seed: int = 0,
    ):
        super().__init__()
        # Store a resolved plain dict so it survives hparam saving.
        self.data_cfg = (
            OmegaConf.to_container(data_cfg, resolve=True)
            if isinstance(data_cfg, DictConfig)
            else dict(data_cfg)
        )
        self.imgsz = int(imgsz)
        self.batch_size = int(batch_size)
        self.stride = int(stride)
        self.workers = int(workers)
        self.single_cls = bool(single_cls)
        self.seed = int(seed)

        self.train_dataset = None
        self.val_dataset = None
        self._train_loader = None
        self._val_loader = None

    # --- convenience -------------------------------------------------------
    @property
    def nc(self) -> int:
        return int(self.data_cfg["nc"])

    @property
    def names(self) -> list:
        return list(self.data_cfg.get("names", []) or [])

    def _split_path(self, split_key: str) -> str:
        return str(Path(self.data_cfg["path"]) / self.data_cfg[split_key])

    def _hyp(self) -> dict:
        return {k: self.data_cfg[k] for k in _HYP_KEYS if k in self.data_cfg}

    def _ddp_rank(self) -> int:
        # M5 (DDP): upstream create_dataloader builds a DistributedSampler when
        # rank != -1 -- the exact mechanism train_dual.py uses under DDP.
        # -1 = no sharding (single process).
        try:
            trainer = self.trainer
        except RuntimeError:
            trainer = None
        if trainer is not None and getattr(trainer, "world_size", 1) > 1:
            return int(trainer.global_rank)
        return -1

    # --- Lightning API -----------------------------------------------------
    def _make_loader(self, split_key, augment, rect, pad, shuffle, prefix, rank=-1):
        kwargs = dict(
            path=self._split_path(split_key),
            imgsz=self.imgsz,
            batch_size=self.batch_size,
            stride=self.stride,
            single_cls=self.single_cls,
            hyp=self._hyp(),
            augment=augment,
            cache=self.data_cfg.get("cache", False),
            rect=rect,
            pad=pad,
            workers=self.workers,
            shuffle=shuffle,
            prefix=prefix,
            rank=rank,
            seed=self.seed,
        )
        # Drop kwargs this create_dataloader version doesn't accept (e.g. seed).
        kwargs = {k: v for k, v in kwargs.items() if k in _ACCEPTED_KWARGS}
        return create_dataloader(**kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit") and self._train_loader is None:
            self._train_loader, self.train_dataset = self._make_loader(
                "train", augment=True,
                # Upstream --rect (train side): create_dataloader itself downgrades
                # shuffle to False with a warning when rect=True, like train_dual.py.
                rect=bool(self.data_cfg.get("rect_train", False)),
                pad=0.0, shuffle=True, prefix="train: ",
                rank=self._ddp_rank(),   # DDP: shard via the upstream DistributedSampler
            )
        if stage in (None, "fit", "validate") and self._val_loader is None:
            # val uses 2x batch in train_dual.py; keep 1x here for simplicity.
            # DDP note: val stays UNSHARDED (rank=-1) -- every rank evaluates the
            # full val set, so metrics are identical on all ranks and match the
            # single-GPU numbers exactly (needs trainer.use_distributed_sampler=false).
            self._val_loader, self.val_dataset = self._make_loader(
                "val", augment=False,
                rect=bool(self.data_cfg.get("rect_val", True)),
                pad=0.5, shuffle=False, prefix="val: ",
            )

    def train_dataloader(self):
        if self._train_loader is None:
            self.setup("fit")
        return self._train_loader

    def val_dataloader(self):
        if self._val_loader is None:
            self.setup("validate")
        return self._val_loader
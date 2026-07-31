"""src/train.py

Hydra entry point for the YOLOv9 -> Lightning + Hydra migration.

  1. Composes the config tree under configs/.
  2. Resolves and logs the fully-resolved config via the logging module so
     Hydra's job_logging also writes it to train.log (fails loudly on any
     unresolved ${...} interpolation).
  3. Wires the YOLODataModule + YOLOv9LitModule + pl.Trainer and runs fit()
     with the M4 callbacks: warmup, EMA, close-mosaic, early stopping, and
     YOLOv9-compatible checkpoints (last.pt / best.pt under a versioned
     per-run dir: ckpts_0, ckpts_1, ... anchored at cfg.checkpoint_save_dir when set,
     otherwise the repo root -- cfg.overwrite=true reuses ckpts_0).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure the repo root is importable so `src` is a package and the TOP-LEVEL
# `models` package resolves to the upstream YOLOv9 code (not src/models).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
from omegaconf import DictConfig, OmegaConf

try:
    import lightning.pytorch as pl
except ImportError:  # older install name
    import pytorch_lightning as pl

from src.datamodules.yolo_datamodule import YOLODataModule
from src.lit_module import YOLOv9LitModule
from src.callbacks.warmup import WarmupCallback
from src.callbacks.ema import EMACallback
from src.callbacks.close_mosaic import CloseMosaicCallback
from src.checkpoint import YOLOv9Checkpoint, load_pretrained

# Hydra's job_logging captures the logging module (NOT print), writing it to
# train.log / train_ddp_process_N.log as well as the console.
log = logging.getLogger(__name__)


def resolve_ckpt_dir(base: str, overwrite: bool, checkpoint_save_dir: str | None = None) -> Path:
    """Version the checkpoint dir so runs never clobber each other.

    checkpoint_save_dir=None (default): anchor at the repo root (old behavior).
    checkpoint_save_dir="runs/taco": checkpoints go under <repo>/runs/taco/.
    overwrite=False (default): use the first free suffix -- ckpts_0 for the
    initial run, then ckpts_1, ckpts_2, ... (every run keeps its weights).
    overwrite=True: always use ckpts_0, replacing whatever is inside.
    """
    root = Path(base)
    if not root.is_absolute():
        anchor = REPO_ROOT / checkpoint_save_dir if checkpoint_save_dir else REPO_ROOT
        root = anchor / root  # keep runs anchored to the repo root (or checkpoint_save_dir)
    if overwrite:
        return root.with_name(f"{root.name}_0")
    i = 0
    while root.with_name(f"{root.name}_{i}").exists():
        i += 1
    return root.with_name(f"{root.name}_{i}")


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    # 1. Show the fully-resolved config. resolve=True forces every ${...}
    #    interpolation to evaluate, so a misconfigured tree fails here loudly.
    #    Use logging (not print) so it also lands in train.log.
    log.info("=" * 80)
    log.info("Resolved configuration:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    log.info("=" * 80)

    # 2. Sanity-check the key values the rest of M2 depends on.
    nc = cfg.data.nc
    names = cfg.data.get("names", [])
    log.info(f"[data]  nc={nc}  names={len(names)} classes")
    log.info(f"[model] cfg={cfg.model.cfg}  ch={cfg.model.ch}  anchors={cfg.model.anchors}")
    log.info(f"[train] epochs={cfg.epochs}  imgsz={cfg.imgsz}  batch_size={cfg.batch_size}")

    assert nc and int(nc) > 0, "data.nc must be a positive integer (fill it in configs/data)."
    if len(names) not in (0, int(nc)):
        raise ValueError(f"len(names)={len(names)} does not match nc={nc}")

    # 3. Seed, then wire the DataModule + LightningModule + Trainer and fit.
    pl.seed_everything(int(cfg.seed), workers=True)

    datamodule = YOLODataModule(
        data_cfg=cfg.data,
        imgsz=int(cfg.imgsz),
        batch_size=int(cfg.batch_size),
        stride=32,                       # yolov9-c max stride
        workers=int(cfg.data.get("workers", 8)),
        single_cls=bool(cfg.get("single_cls", False)),  # upstream --single-cls
        seed=int(cfg.seed),
    )
    lit_module = YOLOv9LitModule(cfg)

    # Resume takes priority over pretrained weights: the Lightning checkpoint
    # already contains the trained weights, optimizer, scheduler, and callback
    # state, so load_pretrained would be redundant (and wrong) on resume.
    resume_path = cfg.get("resume") or None
    if resume_path:
        resume_path = Path(resume_path)
        if not resume_path.is_absolute():
            resume_path = REPO_ROOT / resume_path  # Hydra chdir's into outputs/
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        log.info(f"[resume] resuming training state from {resume_path}")

    # Optional pretrained start (true parity check):
    #   uv run python src/train.py +weights=yolov9-c.pt
    if cfg.get("weights") and not resume_path:
        load_pretrained(lit_module, cfg.weights)

    # Upstream --freeze: freeze=[10] freezes layers 0-9 (backbone); a multi-
    # element list like freeze=[3,5,7] freezes exactly those indices; the
    # default [0] freezes nothing (range(0) is empty) -- same rule as train_dual.py.
    freeze_cfg = [int(x) for x in (cfg.get("freeze") or [0])]
    freeze = [f"model.{x}." for x in (freeze_cfg if len(freeze_cfg) > 1 else range(freeze_cfg[0]))]
    for k, v in lit_module.model.named_parameters():
        v.requires_grad = True
        if any(x in k for x in freeze):
            log.info(f"[freeze] freezing {k}")
            v.requires_grad = False

    # Versioned checkpoint dir: ckpts_0, ckpts_1, ... (overwrite=true reuses ckpts_0).
    # cfg.checkpoint_save_dir=null keeps them at the repo root; e.g. checkpoint_save_dir=runs/taco
    # puts them under <repo>/runs/taco/.
    overwrite = bool(cfg.get("overwrite", False))
    checkpoint_save_dir = cfg.get("checkpoint_save_dir") or None  # empty string counts as None
    ckpt_name = resolve_ckpt_dir(str(cfg.get("ckpt_prefix", "ckpts")), overwrite, checkpoint_save_dir)
    log.info(f"[ckpt] checkpoints -> {ckpt_name}  (overwrite={overwrite})")

    callbacks = [
        WarmupCallback(
            warmup_epochs=float(cfg.optimizer.warmup_epochs),
            warmup_momentum=float(cfg.optimizer.warmup_momentum),
            warmup_bias_lr=float(cfg.optimizer.warmup_bias_lr),
            momentum=float(cfg.optimizer.momentum),
        ),
        EMACallback(decay=float(cfg.callbacks.ema.decay), tau=int(cfg.callbacks.ema.tau)),
        CloseMosaicCallback(disable_at=int(cfg.close_mosaic)),
        YOLOv9Checkpoint(
            dirpath=ckpt_name,  # last.pt / best.pt in YOLOv9 dict format
            save_period=int(cfg.callbacks.checkpoint.get("save_period", -1)),  # upstream --save-period
            nosave=bool(cfg.get("nosave", False)),  # upstream --nosave
        ),
        # Lightning-format twin of last.pt: last.ckpt carries model / optimizer /
        # LR-scheduler / loop state plus callback state (EMA weights + updates via
        # EMACallback.state_dict, best_fitness via YOLOv9Checkpoint.state_dict),
        # which is what trainer.fit(ckpt_path=...) consumes for resume=... .
        pl.callbacks.ModelCheckpoint(dirpath=str(ckpt_name), save_last=True, save_top_k=0),
    ]
    if cfg.callbacks.get("early_stopping"):
        callbacks.append(pl.callbacks.EarlyStopping(
            monitor="val/fitness", mode="max",
            patience=int(cfg.callbacks.early_stopping.patience),
        ))

    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    # Upstream --noval: validate only the final epoch. YOLOv9Checkpoint still
    # writes last.pt/best.pt on that final validation pass.
    if bool(cfg.get("noval", False)):
        trainer_kwargs["check_val_every_n_epoch"] = int(cfg.epochs)
    trainer = pl.Trainer(callbacks=callbacks, **trainer_kwargs)

    # 1-epoch smoke:   uv run python src/train.py epochs=1
    # fast_dev_run:    uv run python src/train.py +trainer.fast_dev_run=true
    # resume:          uv run python src/train.py resume=checkpoints/ckpts_0/last.ckpt
    trainer.fit(
        lit_module,
        datamodule=datamodule,
        ckpt_path=str(resume_path) if resume_path else None,
    )


if __name__ == "__main__":
    main()
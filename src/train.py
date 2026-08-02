"""Hydra entry point for YOLOv9 training with Lightning."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
from omegaconf import DictConfig, OmegaConf

try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl

from src.callbacks.close_mosaic import CloseMosaicCallback
from src.callbacks.ema import EMACallback
from src.callbacks.training_plots import TrainingPlotsCallback
from src.callbacks.warmup import WarmupCallback
from src.checkpoint import YOLOv9Checkpoint, load_pretrained
from src.datamodules.yolo_datamodule import YOLODataModule
from src.lit_module import YOLOv9LitModule

log = logging.getLogger(__name__)


def resolve_ckpt_dir(base: str, overwrite: bool, checkpoint_save_dir: str | None = None) -> Path:
    """Return a versioned checkpoint directory without overwriting prior runs."""
    root = Path(base)
    if not root.is_absolute():
        anchor = REPO_ROOT / checkpoint_save_dir if checkpoint_save_dir else REPO_ROOT
        root = anchor / root
    if overwrite:
        return root.with_name(f"{root.name}_0")
    i = 0
    while root.with_name(f"{root.name}_{i}").exists():
        i += 1
    return root.with_name(f"{root.name}_{i}")


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    log.info("=" * 80)
    log.info("Resolved configuration:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    log.info("=" * 80)

    nc = cfg.data.nc
    names = cfg.data.get("names", [])
    log.info(f"[data]  nc={nc}  names={len(names)} classes")
    log.info(f"[model] cfg={cfg.model.cfg}  ch={cfg.model.ch}  anchors={cfg.model.anchors}")
    log.info(f"[train] epochs={cfg.epochs}  imgsz={cfg.imgsz}  batch_size={cfg.batch_size}")

    assert nc and int(nc) > 0, "data.nc must be a positive integer (fill it in configs/data)."
    if len(names) not in (0, int(nc)):
        raise ValueError(f"len(names)={len(names)} does not match nc={nc}")

    pl.seed_everything(int(cfg.seed), workers=True)

    datamodule = YOLODataModule(
        data_cfg=cfg.data,
        imgsz=int(cfg.imgsz),
        batch_size=int(cfg.batch_size),
        stride=32,
        workers=int(cfg.data.get("workers", 8)),
        single_cls=bool(cfg.get("single_cls", False)),
        seed=int(cfg.seed),
    )
    lit_module = YOLOv9LitModule(cfg)

    resume_path = cfg.get("resume") or None
    if resume_path:
        resume_path = Path(resume_path)
        if not resume_path.is_absolute():
            resume_path = REPO_ROOT / resume_path
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        log.info(f"[resume] resuming training state from {resume_path}")

    if cfg.get("weights") and not resume_path:
        load_pretrained(lit_module, cfg.weights)

    freeze_cfg = [int(x) for x in (cfg.get("freeze") or [0])]
    freeze = [f"model.{x}." for x in (freeze_cfg if len(freeze_cfg) > 1 else range(freeze_cfg[0]))]
    for k, v in lit_module.model.named_parameters():
        v.requires_grad = True
        if any(x in k for x in freeze):
            log.info(f"[freeze] freezing {k}")
            v.requires_grad = False

    overwrite = bool(cfg.get("overwrite", False))
    checkpoint_save_dir = cfg.get("checkpoint_save_dir") or None
    ckpt_name = resolve_ckpt_dir(str(cfg.get("ckpt_prefix", "ckpts")), overwrite, checkpoint_save_dir)
    log.info(f"[ckpt] checkpoints and plots -> {ckpt_name}  (overwrite={overwrite})")

    plots_cfg = cfg.callbacks.get("plots", {})
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
            dirpath=ckpt_name,
            save_period=int(cfg.callbacks.checkpoint.get("save_period", -1)),
            nosave=bool(cfg.get("nosave", False)),
        ),
        TrainingPlotsCallback(
            dirpath=ckpt_name,
            enabled=bool(plots_cfg.get("enabled", True)),
            csv_name=str(plots_cfg.get("csv_name", "results.csv")),
            results_name=str(plots_cfg.get("results_name", "results.png")),
            loss_name=str(plots_cfg.get("loss_name", "loss_curves.png")),
        ),
        pl.callbacks.ModelCheckpoint(dirpath=str(ckpt_name), save_last=True, save_top_k=0),
    ]
    if cfg.callbacks.get("early_stopping"):
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor="val/fitness",
                mode="max",
                patience=int(cfg.callbacks.early_stopping.patience),
            )
        )

    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    if bool(cfg.get("noval", False)):
        trainer_kwargs["check_val_every_n_epoch"] = int(cfg.epochs)
    trainer = pl.Trainer(callbacks=callbacks, **trainer_kwargs)

    trainer.fit(
        lit_module,
        datamodule=datamodule,
        ckpt_path=str(resume_path) if resume_path else None,
    )


if __name__ == "__main__":
    main()

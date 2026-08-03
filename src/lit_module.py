"""LightningModule for YOLOv9 training and validation."""

from __future__ import annotations

import hashlib
import math
import os
import random
import sys
from importlib import import_module
from pathlib import Path

try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metrics.confusion_matrix import DetectionConfusionMatrix
from src.models.yolov9_factory import build_yolov9_model
from utils.general import non_max_suppression, scale_boxes, xywh2xyxy
from utils.metrics import ap_per_class
from utils.torch_utils import smart_optimizer
from val_dual import process_batch


class YOLOv9LitModule(pl.LightningModule):
    """YOLOv9 training, validation metrics, and confusion-matrix collection."""

    def __init__(self, cfg):
        super().__init__()
        self.automatic_optimization = False
        self.epochs = int(cfg.epochs)
        self.opt_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
        self.sched_cfg = OmegaConf.to_container(cfg.scheduler, resolve=True)
        self.nbs = 64
        self.batch_size = int(cfg.batch_size)
        self.warmup_epochs = float(self.opt_cfg.get("warmup_epochs", 3.0))
        self._nb = None
        self._nw = None
        self._last_opt_step = -1
        self.stepped_this_batch = False

        self.single_cls = bool(cfg.get("single_cls", False))
        self.nc = 1 if self.single_cls else int(cfg.data.nc)
        names = list(cfg.data.get("names", []) or [])
        if self.single_cls and len(names) != 1:
            names = ["item"]
        self.names = {i: n for i, n in enumerate(names)}
        self.hyp = {
            "box": float(cfg.loss.box), "cls": float(cfg.loss.cls), "dfl": float(cfg.loss.dfl),
            "cls_pw": 1.0, "obj_pw": 1.0, "fl_gamma": 0.0,
            "label_smoothing": float(cfg.loss.get("label_smoothing", 0.0)),
        }

        loss_target = str(cfg.loss.get("_target_", "utils.loss_tal_dual.ComputeLoss"))
        module_name, class_name = loss_target.rsplit(".", 1)
        self._loss_cls = getattr(import_module(module_name), class_name)
        self.model = build_yolov9_model(
            cfg=cfg.model.cfg, ch=cfg.model.ch, nc=self.nc, anchors=cfg.model.anchors
        )
        self.compute_loss = None
        self.multi_scale = bool(cfg.get("multi_scale", False))
        self.imgsz = int(cfg.imgsz)
        self.gs = max(int(self.model.stride.max()), 32)

        self.conf_thres = 0.001
        self.iou_thres = 0.6
        self.max_det = 300
        self._stats: list = []
        self.iouv = None
        self.niou = 10

        plots_cfg = cfg.callbacks.get("plots", {})
        confusion_cfg = plots_cfg.get("confusion_matrix", {})
        self.confusion_matrix = None
        if bool(confusion_cfg.get("enabled", True)):
            self.confusion_matrix = DetectionConfusionMatrix(
                nc=self.nc,
                names=self.names,
                conf=float(confusion_cfg.get("conf", 0.25)),
                iou_thres=float(confusion_cfg.get("iou_thres", 0.45)),
            )

    def setup(self, stage=None):
        self.model.nc = self.nc
        self.model.hyp = self.hyp
        self.model.names = self.names

    def _ensure_loss(self):
        if self.compute_loss is None:
            self.compute_loss = self._loss_cls(self.model)

    def on_train_start(self):
        self._nb = max(int(self.trainer.num_training_batches), 1)
        self._nw = max(round(self.warmup_epochs * self._nb), 100)
        self._last_opt_step = -1
        self._opt_steps = 0
        self.optimizers().zero_grad()

    def training_step(self, batch, batch_idx):
        self._ensure_loss()
        opt = self.optimizers()
        ni = batch_idx + self._nb * self.current_epoch
        accumulate = max(round(self.nbs / self.batch_size), 1)
        if ni <= self._nw:
            accumulate = max(1, int(np.interp(ni, [0, self._nw], [1, self.nbs / self.batch_size]).round()))

        imgs, targets, paths, _shapes = batch
        imgs = imgs.float() / 255.0
        if self.multi_scale:
            sz = random.randrange(int(self.imgsz * 0.5), int(self.imgsz * 1.5) + self.gs) // self.gs * self.gs
            sf = sz / max(imgs.shape[2:])
            if sf != 1:
                ns = [math.ceil(x * sf / self.gs) * self.gs for x in imgs.shape[2:]]
                imgs = torch.nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)

        preds = self.model(imgs)
        loss, loss_items = self.compute_loss(preds, targets.to(self.device))
        box, cls, dfl = loss_items
        self.log_dict(
            {"train/box": box, "train/cls": cls, "train/dfl": dfl},
            prog_bar=True, on_step=True, on_epoch=True, batch_size=imgs.shape[0],
        )
        if self.trainer is not None and self.trainer.world_size > 1:
            loss = loss * self.trainer.world_size
        if ni == 0 and os.environ.get("PARITY_DUMP_FIRST_BATCH"):
            self._dump_first_batch(imgs, targets, paths, loss_items)

        self.manual_backward(loss)
        self.stepped_this_batch = False
        if ni - self._last_opt_step >= accumulate:
            opt.step()
            opt.zero_grad()
            self._last_opt_step = ni
            self._opt_steps += 1
            self.stepped_this_batch = True

    def on_before_optimizer_step(self, optimizer):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

    def on_train_epoch_end(self):
        self.print(f"[m8] cumulative optimizer steps: {self._opt_steps}")
        self.lr_schedulers().step()

    def _dump_first_batch(self, imgs, targets, paths, loss_items):
        out = Path(os.environ.get("PARITY_DUMP_PATH", "runs/parity/new/first_batch.pt"))
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "imgs_shape": tuple(imgs.shape),
                "imgs_sum": imgs.detach().double().sum().item(),
                "imgs_sha1": hashlib.sha1(imgs.detach().cpu().float().numpy().tobytes()).hexdigest(),
                "targets": targets.detach().cpu(),
                "paths": list(paths),
                "loss_items": loss_items.detach().cpu(),
            }, out,
        )
        self.print(f"[parity] first-batch diagnostics written to {out}")

    def configure_optimizers(self):
        optimizer = smart_optimizer(
            self.model,
            name=self.opt_cfg.get("name", "SGD"),
            lr=self.opt_cfg["lr0"],
            momentum=self.opt_cfg["momentum"],
            decay=self.opt_cfg["weight_decay"],
        )
        lrf = self.sched_cfg["lrf"]
        if self.sched_cfg.get("cos_lr", False):
            from utils.general import one_cycle
            lf = one_cycle(1, lrf, self.epochs)
        elif self.sched_cfg.get("flat_cos_lr", False):
            from utils.general import one_flat_cycle
            lf = one_flat_cycle(1, lrf, self.epochs)
        elif self.sched_cfg.get("fixed_lr", False):
            lf = lambda x: 1.0
        else:
            lf = lambda x: (1 - x / self.epochs) * (1.0 - lrf) + lrf
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def on_validation_epoch_start(self):
        self._stats = []
        self.iouv = torch.linspace(0.5, 0.95, 10, device=self.device)
        self.niou = self.iouv.numel()
        if self.confusion_matrix is not None:
            self.confusion_matrix.reset()

    def validation_step(self, batch, batch_idx):
        im, targets, _paths, shapes = batch
        im = im.float() / 255.0
        _, _, height, width = im.shape
        preds, _train_out = self.model(im)
        targets = targets.to(self.device)
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=self.device)
        preds = non_max_suppression(
            preds, self.conf_thres, self.iou_thres,
            labels=[], multi_label=True, agnostic=self.single_cls, max_det=self.max_det,
        )

        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            if self.single_cls and labels.numel():
                labels[:, 0] = 0
            nl, npr = labels.shape[0], pred.shape[0]
            correct = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)

            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_boxes(im[si].shape[1:], tbox, shapes[si][0], shapes[si][1])
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)
            else:
                labelsn = torch.zeros((0, 5), device=self.device)

            if npr:
                if self.single_cls:
                    pred[:, 5] = 0
                predn = pred.clone()
                scale_boxes(im[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])
            else:
                predn = torch.zeros((0, 6), device=self.device)

            if self.confusion_matrix is not None:
                self.confusion_matrix.process_batch(predn, labelsn)

            if npr == 0:
                if nl:
                    self._stats.append((correct, torch.zeros(0, device=self.device),
                                        torch.zeros(0, device=self.device), labels[:, 0]))
                continue
            if nl:
                correct = process_batch(predn, labelsn, self.iouv)
            self._stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))

    def on_validation_epoch_end(self):
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self._stats)] if self._stats else []
        if len(stats) and stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, names=self.names)
            ap50, ap = ap[:, 0], ap.mean(1)
            mp, mr, map50, map_ = p.mean(), r.mean(), ap50.mean(), ap.mean()
        else:
            mp = mr = map50 = map_ = 0.0
        fitness = 0.1 * float(map50) + 0.9 * float(map_)
        self.log_dict(
            {"val/P": float(mp), "val/R": float(mr), "val/mAP50": float(map50),
             "val/mAP50-95": float(map_), "val/fitness": fitness},
            prog_bar=True, sync_dist=False,
        )
        n_preds = int(stats[1].shape[0]) if len(stats) else 0
        n_tp50 = int(stats[0][:, 0].sum()) if n_preds else 0
        self.print(
            f"[val] preds={n_preds} tp@iou0.5={n_tp50} "
            f"P={float(mp):.5f} R={float(mr):.5f} mAP50={float(map50):.5f} "
            f"mAP50-95={float(map_):.5f} fitness={fitness:.5f}"
        )

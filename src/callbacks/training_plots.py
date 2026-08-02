"""Write YOLO-style training metrics and curve images after every epoch."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl


class TrainingPlotsCallback(pl.Callback):
    """Persist CSV, training curves, result charts, and confusion heatmaps."""

    COLUMNS = (
        "epoch", "train/box_loss", "train/cls_loss", "train/dfl_loss",
        "metrics/precision", "metrics/recall", "metrics/mAP_0.5",
        "metrics/mAP_0.5:0.95", "val/fitness", "lr/pg0", "lr/pg1", "lr/pg2",
    )

    def __init__(self, dirpath="ckpts", enabled=True, csv_name="results.csv",
                 results_name="results.png", loss_name="loss_curves.png"):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.enabled = bool(enabled)
        self.csv_path = self.dirpath / csv_name
        self.results_path = self.dirpath / results_name
        self.loss_path = self.dirpath / loss_name
        self.rows: dict[int, dict[str, float | int | None]] = {}
        self._loaded = False

    @staticmethod
    def _number(value):
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "item"):
            value = value.item()
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    @staticmethod
    def _metric(metrics, *names):
        for name in names:
            if name in metrics:
                return TrainingPlotsCallback._number(metrics[name])
        return None

    def _load_existing(self):
        if self._loaded:
            return
        self._loaded = True
        if not self.csv_path.is_file():
            return
        with self.csv_path.open(newline="", encoding="utf-8") as f:
            for raw in csv.DictReader(f):
                try:
                    epoch = int(float(raw["epoch"]))
                except (KeyError, TypeError, ValueError):
                    continue
                row = {key: None for key in self.COLUMNS}
                row["epoch"] = epoch
                for key in self.COLUMNS[1:]:
                    row[key] = self._number(raw.get(key))
                self.rows[epoch] = row

    def _row(self, epoch):
        self._load_existing()
        return self.rows.setdefault(
            int(epoch), {key: int(epoch) if key == "epoch" else None for key in self.COLUMNS}
        )

    def _write_csv(self):
        self.dirpath.mkdir(parents=True, exist_ok=True)
        tmp = self.csv_path.with_suffix(self.csv_path.suffix + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writeheader()
            for epoch in sorted(self.rows):
                writer.writerow(self.rows[epoch])
        tmp.replace(self.csv_path)

    def _plot(self, path, specs, columns):
        self.dirpath.mkdir(parents=True, exist_ok=True)
        epochs = sorted(self.rows)
        nrows = math.ceil(len(specs) / columns)
        fig, axes = plt.subplots(nrows, columns, figsize=(5 * columns, 3.8 * nrows), squeeze=False)
        for ax, (title, key) in zip(axes.flat, specs):
            points = [(e, self.rows[e].get(key)) for e in epochs]
            points = [(e, v) for e, v in points if v is not None]
            if points:
                x, y = zip(*points)
                ax.plot(x, y, marker="o", linewidth=2, markersize=3)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)
        for ax in axes.flat[len(specs):]:
            ax.set_visible(False)
        fig.tight_layout()
        tmp = path.with_suffix(path.suffix + ".tmp")
        fig.savefig(tmp, format="png", dpi=160)
        plt.close(fig)
        tmp.replace(path)

    def _render(self):
        self._write_csv()
        self._plot(
            self.loss_path,
            [("Box Loss", "train/box_loss"), ("Classification Loss", "train/cls_loss"),
             ("DFL Loss", "train/dfl_loss")],
            columns=3,
        )
        self._plot(
            self.results_path,
            [("Box Loss", "train/box_loss"), ("Classification Loss", "train/cls_loss"),
             ("DFL Loss", "train/dfl_loss"), ("Precision", "metrics/precision"),
             ("Recall", "metrics/recall"), ("mAP@0.5", "metrics/mAP_0.5"),
             ("mAP@0.5:0.95", "metrics/mAP_0.5:0.95"), ("Fitness", "val/fitness")],
            columns=4,
        )

    def _active(self, trainer):
        return self.enabled and trainer.is_global_zero and not trainer.sanity_checking

    def on_train_epoch_end(self, trainer, pl_module):
        if not self._active(trainer):
            return
        metrics = trainer.callback_metrics
        row = self._row(trainer.current_epoch)
        row["train/box_loss"] = self._metric(metrics, "train/box_epoch", "train/box")
        row["train/cls_loss"] = self._metric(metrics, "train/cls_epoch", "train/cls")
        row["train/dfl_loss"] = self._metric(metrics, "train/dfl_epoch", "train/dfl")
        if trainer.optimizers:
            for i, group in enumerate(trainer.optimizers[0].param_groups[:3]):
                row[f"lr/pg{i}"] = self._number(group.get("lr"))
        self._render()

    def on_validation_end(self, trainer, pl_module):
        if not self._active(trainer):
            return
        metrics = trainer.callback_metrics
        row = self._row(trainer.current_epoch)
        row["metrics/precision"] = self._metric(metrics, "val/P")
        row["metrics/recall"] = self._metric(metrics, "val/R")
        row["metrics/mAP_0.5"] = self._metric(metrics, "val/mAP50")
        row["metrics/mAP_0.5:0.95"] = self._metric(metrics, "val/mAP50-95")
        row["val/fitness"] = self._metric(metrics, "val/fitness")
        self._render()
        confusion_matrix = getattr(pl_module, "confusion_matrix", None)
        if confusion_matrix is not None:
            confusion_matrix.save(self.dirpath)

    def on_fit_end(self, trainer, pl_module):
        if self._active(trainer) and self.rows:
            self._render()

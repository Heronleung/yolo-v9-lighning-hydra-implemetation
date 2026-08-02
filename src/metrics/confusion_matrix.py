"""Detection confusion matrix and YOLO-style heatmap rendering."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    if box1.numel() == 0 or box2.numel() == 0:
        return torch.zeros((len(box1), len(box2)), device=box1.device)
    a1, a2 = box1[:, None, :2], box1[:, None, 2:]
    b1, b2 = box2[None, :, :2], box2[None, :, 2:]
    inter = (torch.minimum(a2, b2) - torch.maximum(a1, b1)).clamp(0).prod(2)
    area1 = (a2 - a1).clamp(0).prod(2)
    area2 = (b2 - b1).clamp(0).prod(2)
    return inter / (area1 + area2 - inter + eps)


class DetectionConfusionMatrix:
    """Rows are predicted classes; columns are true classes; last index is background."""

    def __init__(self, nc: int, names=(), conf: float = 0.25, iou_thres: float = 0.45):
        self.nc = int(nc)
        self.names = list(names.values()) if isinstance(names, dict) else list(names)
        self.conf = float(conf)
        self.iou_thres = float(iou_thres)
        self.matrix = np.zeros((self.nc + 1, self.nc + 1), dtype=np.int64)

    def reset(self):
        self.matrix.fill(0)

    def process_batch(self, detections: torch.Tensor | None, labels: torch.Tensor):
        labels = labels.detach()
        true_classes = labels[:, 0].long() if labels.numel() else torch.empty(0, dtype=torch.long, device=labels.device)
        if detections is None or detections.numel() == 0:
            for true_class in true_classes.cpu().tolist():
                self.matrix[self.nc, true_class] += 1
            return

        detections = detections.detach()
        detections = detections[detections[:, 4] >= self.conf]
        if detections.numel() == 0:
            for true_class in true_classes.cpu().tolist():
                self.matrix[self.nc, true_class] += 1
            return

        pred_classes = detections[:, 5].long()
        if labels.numel() == 0:
            for pred_class in pred_classes.cpu().tolist():
                self.matrix[pred_class, self.nc] += 1
            return

        ious = box_iou(labels[:, 1:5], detections[:, :4])
        label_idx, detection_idx = torch.where(ious > self.iou_thres)
        if label_idx.numel():
            matches = torch.stack((label_idx, detection_idx, ious[label_idx, detection_idx]), 1).cpu().numpy()
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            matched_labels = matches[:, 0].astype(int)
            matched_detections = matches[:, 1].astype(int)
        else:
            matched_labels = np.empty(0, dtype=int)
            matched_detections = np.empty(0, dtype=int)

        true_cpu = true_classes.cpu().numpy()
        pred_cpu = pred_classes.cpu().numpy()
        for label_i, true_class in enumerate(true_cpu):
            found = np.where(matched_labels == label_i)[0]
            if len(found) == 1:
                self.matrix[pred_cpu[matched_detections[found[0]]], true_class] += 1
            else:
                self.matrix[self.nc, true_class] += 1
        for detection_i, pred_class in enumerate(pred_cpu):
            if detection_i not in matched_detections:
                self.matrix[pred_class, self.nc] += 1

    def _plot(self, path: Path, normalize: bool):
        values = self.matrix.astype(float)
        if normalize:
            values /= values.sum(0, keepdims=True) + 1e-9
        annotations = values.copy()
        annotations[annotations < (0.005 if normalize else 0.5)] = np.nan
        labels = self.names + ["background"] if len(self.names) == self.nc else "auto"
        fig, ax = plt.subplots(figsize=(13, 11), tight_layout=True)
        sns.heatmap(
            annotations,
            ax=ax,
            cmap="Blues",
            annot=self.nc < 30,
            fmt=".2f" if normalize else ".0f",
            square=True,
            vmin=0.0,
            xticklabels=labels,
            yticklabels=labels,
        )
        ax.set_xlabel("True class")
        ax.set_ylabel("Predicted class")
        ax.set_title("Normalized Confusion Matrix" if normalize else "Confusion Matrix")
        fig.savefig(path, dpi=250)
        plt.close(fig)

    def save(self, directory: str | Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._plot(directory / "confusion_matrix.png", normalize=False)
        self._plot(directory / "confusion_matrix_normalized.png", normalize=True)

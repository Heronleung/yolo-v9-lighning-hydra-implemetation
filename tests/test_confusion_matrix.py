import torch

from src.metrics.confusion_matrix import DetectionConfusionMatrix


def test_confusion_counts_and_heatmaps(tmp_path):
    matrix = DetectionConfusionMatrix(2, ["class0", "class1"], conf=0.25, iou_thres=0.45)
    labels = torch.tensor([
        [0, 0, 0, 10, 10],
        [1, 20, 20, 30, 30],
    ], dtype=torch.float32)
    detections = torch.tensor([
        [0, 0, 10, 10, 0.9, 0],
        [20, 20, 30, 30, 0.8, 0],
        [40, 40, 50, 50, 0.7, 1],
    ], dtype=torch.float32)

    matrix.process_batch(detections, labels)
    assert matrix.matrix.tolist() == [[1, 1, 0], [0, 0, 1], [0, 0, 0]]

    matrix.process_batch(None, torch.tensor([[1, 60, 60, 70, 70]], dtype=torch.float32))
    assert matrix.matrix[2, 1] == 1

    matrix.process_batch(
        torch.tensor([[0, 0, 5, 5, 0.9, 1]], dtype=torch.float32),
        torch.empty((0, 5)),
    )
    assert matrix.matrix[1, 2] == 2

    matrix.save(tmp_path)
    for name in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
        path = tmp_path / name
        assert path.is_file()
        assert path.stat().st_size > 0

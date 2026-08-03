import csv

from src.callbacks.training_plots import TrainingPlotsCallback


class DummyOptimizer:
    param_groups = [{"lr": 0.01}, {"lr": 0.02}, {"lr": 0.03}]


class DummyTrainer:
    is_global_zero = True
    sanity_checking = False
    current_epoch = 0
    optimizers = [DummyOptimizer()]
    callback_metrics = {
        "train/box_epoch": 5.0,
        "train/cls_epoch": 3.0,
        "train/dfl_epoch": 2.0,
    }


def test_epoch_ticks_are_integer_and_include_endpoints():
    assert TrainingPlotsCallback._epoch_ticks([]) == []
    assert TrainingPlotsCallback._epoch_ticks([0]) == [0]
    assert TrainingPlotsCallback._epoch_ticks(range(5)) == [0, 1, 2, 3, 4]

    ticks = TrainingPlotsCallback._epoch_ticks(range(300))
    assert ticks[0] == 0
    assert ticks[-1] == 299
    assert all(isinstance(tick, int) for tick in ticks)


def test_writes_consolidated_results_and_resumes(tmp_path):
    callback = TrainingPlotsCallback(tmp_path)
    trainer = DummyTrainer()

    callback.on_train_epoch_end(trainer, None)
    trainer.callback_metrics.update(
        {
            "val/P": 0.4,
            "val/R": 0.3,
            "val/mAP50": 0.2,
            "val/mAP50-95": 0.1,
            "val/fitness": 0.11,
        }
    )
    callback.on_validation_end(trainer, None)

    for name in ("results.csv", "results.png"):
        path = tmp_path / name
        assert path.is_file()
        assert path.stat().st_size > 0
    assert not (tmp_path / "loss_curves.png").exists()

    with (tmp_path / "results.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["train/box_loss"] == "5.0"
    assert rows[0]["metrics/mAP_0.5:0.95"] == "0.1"

    resumed = TrainingPlotsCallback(tmp_path)
    trainer.current_epoch = 1
    trainer.callback_metrics = {
        "train/box_epoch": 4.0,
        "train/cls_epoch": 2.5,
        "train/dfl_epoch": 1.8,
    }
    resumed.on_train_epoch_end(trainer, None)

    with (tmp_path / "results.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [row["epoch"] for row in rows] == ["0", "1"]

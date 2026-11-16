from pathlib import Path

import numpy as np
import pytest

from vitcifar.errors import ResultsError
from vitcifar.experiments.ablation import StepSummary
from vitcifar.experiments.ablation_plots import DataEfficiencyPlotter, LadderPlotter, SweepPlotter
from vitcifar.training import EpochRecord, RunResult

PNG_MAGIC = b"\x89PNG"


def _summary(name: str, finals: tuple[float, ...]) -> StepSummary:
    return StepSummary(name, name, finals, np.array([1, 2]), np.array([0.3, 0.5]), np.array([0.01, 0.02]), False)


def _result(acc: float, diverged: bool = False) -> RunResult:
    history = [EpochRecord(1, 2.0, 0.3, 1e-3, 1.0, 1.9, acc / 2), EpochRecord(2, 1.5, 0.5, 1e-3, 1.0, 1.4, acc)]
    return RunResult("r", "r", history, acc, acc, diverged=diverged)


def test_ladder_plots_write_png(tmp_path: Path) -> None:
    summaries = [_summary("resnet18_noaug", (0.8, 0.82)), _summary("vit_a", (0.5,)), _summary("vit_b", (0.6, 0.62))]
    plotter = LadderPlotter()
    assert plotter.plot_curves(summaries, tmp_path / "c.png").read_bytes()[:4] == PNG_MAGIC
    assert plotter.plot_ladder(summaries, tmp_path / "l.png").read_bytes()[:4] == PNG_MAGIC


def test_ladder_labels_show_gain_over_previous_vit_mean() -> None:
    summaries = [_summary("vit_a", (0.5,)), _summary("resnet18", (0.9,)), _summary("vit_b", (0.6, 0.62))]
    assert LadderPlotter.bar_labels(summaries) == ["50.0%", "90.0%", "61.0%\n(+11.0)"]


def test_sweep_plot_writes_png_with_a_diverged_run(tmp_path: Path) -> None:
    results = {(1e-3, 0): _result(0.6), (1e-3, 5): _result(0.7), (4e-3, 0): _result(0.1, True), (4e-3, 5): _result(0.65)}
    assert SweepPlotter().plot(results, tmp_path / "s.png").read_bytes()[:4] == PNG_MAGIC


def test_data_efficiency_plot_writes_png(tmp_path: Path) -> None:
    series = {"ResNet-18": [(5000, _result(0.7)), (50000, _result(0.95))], "ViT": [(5000, _result(0.4))]}
    assert DataEfficiencyPlotter().plot(series, tmp_path / "d.png").read_bytes()[:4] == PNG_MAGIC


def test_ladder_plotter_rejects_empty_input(tmp_path: Path) -> None:
    with pytest.raises(ResultsError):
        LadderPlotter().plot_ladder([], tmp_path / "x.png")

from pathlib import Path

import pytest
import torch

from vitcifar.data import Augmentation
from vitcifar.errors import CheckpointError, ConfigError
from vitcifar.experiments.ablation import (
    ABLATION_LADDER, AblationStep, DataEfficiencyPlanner, LadderCatalog, LadderPlanner, ModelKind, RunSpec,
    SeedAggregator, StepModelBuilder, SweepPlanner, VitCheckpoint,
)
from vitcifar.models.vit import VisionTransformer
from vitcifar.training import EpochRecord, OptimizerKind, ResultStore, RunResult

VIT_STEPS = [s for s in ABLATION_LADDER if s.model is ModelKind.VIT]
RECIPE_FIELDS = ("augmentation", "patch_size", "warmup_epochs", "label_smoothing", "mixing", "drop_path", "epoch_scale")
CATALOG = LadderCatalog()


def _spec(step: AblationStep, **overrides) -> RunSpec:
    return RunSpec("r", "R", step, 0, **overrides)


@pytest.mark.parametrize("before,after", list(zip(VIT_STEPS, VIT_STEPS[1:])), ids=lambda s: s.name)
def test_each_vit_step_changes_exactly_one_thing(before: AblationStep, after: AblationStep) -> None:
    assert len([f for f in RECIPE_FIELDS if getattr(before, f) != getattr(after, f)]) == 1


def test_ladder_starts_naive_and_ends_with_every_fix() -> None:
    first, last = VIT_STEPS[0], VIT_STEPS[-1]
    assert first.augmentation is Augmentation.NONE and first.warmup_epochs == 0
    assert last.mixing and last.drop_path > 0 and last.label_smoothing > 0 and last.epoch_scale > 1
    assert last.patch_size < first.patch_size


@pytest.mark.parametrize(
    "kwargs", [{"model": ModelKind.RESNET18, "patch_size": 4}, {"model": ModelKind.VIT}, {"model": ModelKind.RESNET18, "drop_path": 0.1}],
)
def test_step_rejects_settings_that_do_not_fit_the_model(kwargs: dict) -> None:
    with pytest.raises(ConfigError):
        AblationStep("bad", "bad", augmentation=Augmentation.NONE, **kwargs)


def test_catalog_selects_in_ladder_order() -> None:
    names = CATALOG.names()
    assert [s.name for s in CATALOG.select([names[3], names[0]])] == [names[0], names[3]]


def test_catalog_rejects_unknown_step() -> None:
    with pytest.raises(ConfigError):
        CATALOG.get("nope")


def test_ladder_planner_runs_seeds_in_outer_loop_and_checkpoints_first_seed_vits() -> None:
    specs = LadderPlanner((3, 4)).plan(CATALOG.select(["resnet18_noaug", "vit_p8_naive"]))
    assert [s.run_name for s in specs] == ["resnet18_noaug_s3", "vit_p8_naive_s3", "resnet18_noaug_s4", "vit_p8_naive_s4"]
    assert [s.save_checkpoint for s in specs] == [False, True, False, False]


def test_ladder_planner_needs_a_seed() -> None:
    with pytest.raises(ConfigError):
        LadderPlanner(())


def test_sweep_planner_pairs_each_lr_with_and_without_warmup() -> None:
    specs = SweepPlanner((1e-3, 2e-3), warmup_epochs=5, epochs=30).plan(CATALOG.get("vit_p4_aug"))
    assert [(s.lr, s.warmup_epochs) for s in specs] == [(1e-3, 0), (1e-3, 5), (2e-3, 0), (2e-3, 5)]
    assert {s.epochs for s in specs} == {30}


def test_data_planner_keeps_optimizer_steps_equal() -> None:
    specs = DataEfficiencyPlanner((0.25, 1.0), base_epochs=100).plan(CATALOG.select(["resnet18_aug"]))
    assert [(s.train_fraction, s.epochs) for s in specs] == [(0.25, 400), (1.0, 100)]


def test_data_planner_rejects_fraction_above_one() -> None:
    with pytest.raises(ConfigError):
        DataEfficiencyPlanner((1.5,), base_epochs=10)


def test_builder_uses_adamw_for_vit_and_sgd_for_resnet() -> None:
    builder = StepModelBuilder(base_epochs=20)
    assert builder.train_config(_spec(VIT_STEPS[0])).optimizer is OptimizerKind.ADAMW
    assert builder.train_config(_spec(ABLATION_LADDER[0])).optimizer is OptimizerKind.SGD


def test_builder_scales_epochs_and_applies_overrides() -> None:
    builder = StepModelBuilder(base_epochs=20)
    assert builder.train_config(_spec(CATALOG.get("vit_p4_long"))).epochs == 40
    config = builder.train_config(_spec(CATALOG.get("vit_p4_aug"), lr=4e-3, warmup_epochs=5, epochs=30))
    assert (config.lr, config.warmup_epochs, config.epochs) == (4e-3, 5, 30)


def test_builder_caps_warmup_for_short_runs() -> None:
    assert StepModelBuilder(base_epochs=2).train_config(_spec(CATALOG.get("vit_p4_aug_warmup"))).warmup_epochs == 1


def test_builder_adds_mixer_only_for_mixing_steps() -> None:
    assert StepModelBuilder.mixer(CATALOG.get("vit_p4_ls")) is None
    assert StepModelBuilder.mixer(CATALOG.get("vit_p4_mix")) is not None


@pytest.mark.parametrize("step", ABLATION_LADDER, ids=lambda s: s.name)
def test_every_step_builds_a_model_that_classifies_cifar(step: AblationStep) -> None:
    model = StepModelBuilder(base_epochs=1).model(step).eval()
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_checkpoint_round_trips(tiny_vit: VisionTransformer, tmp_path: Path) -> None:
    store = VitCheckpoint(tmp_path)
    store.save("x", tiny_vit)
    images = torch.randn(1, 3, 16, 16)
    assert torch.allclose(store.load("x", torch.device("cpu"))(images), tiny_vit(images))


def test_checkpoint_load_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError):
        VitCheckpoint(tmp_path).load("missing", torch.device("cpu"))


def _result(name: str, accs: list[float]) -> RunResult:
    history = [EpochRecord(i + 1, 1.0, 0.5, 1e-3, 1.0, 1.0, a) for i, a in enumerate(accs)]
    return RunResult(name, name, history, accs[-1], max(accs))


def test_aggregator_averages_seeds_over_shared_epochs(tmp_path: Path) -> None:
    store = ResultStore(tmp_path)
    store.save(_result("vit_p8_naive_s0", [0.4, 0.6]))
    store.save(_result("vit_p8_naive_s1", [0.6, 0.8]))
    (summary,) = SeedAggregator(store, (0, 1, 2)).summarize(CATALOG.select(["vit_p8_naive", "vit_p8_aug"]))
    assert summary.finals == (0.6, 0.8)
    assert summary.mean == pytest.approx(0.7)
    assert summary.curve_mean.tolist() == pytest.approx([0.5, 0.7])
    assert summary.std == pytest.approx(0.1414, abs=1e-4)

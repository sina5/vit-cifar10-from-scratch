"""Project 1: ViT versus ResNet on CIFAR-10, with the fixes that close the gap.

Three studies share one runner:

- the ladder: each ViT step adds one fix to the step before it, over several seeds;
- the learning-rate sweep: what happens without warmup as the learning rate grows;
- data efficiency: accuracy against training-set size for a ViT and a ResNet.

Every study turns into a list of `RunSpec` objects. The runner trains each spec
and skips any spec whose result file already exists, so a long run can resume.
"""

from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path

import numpy as np
import torch
from torch import nn

from vitcifar import constants as C
from vitcifar.data import Augmentation, BatchMixer, Cifar10Data
from vitcifar.errors import CheckpointError, ConfigError
from vitcifar.models.resnet import CifarResNet18Builder
from vitcifar.models.vit import ViTConfig, VisionTransformer
from vitcifar.training import OptimizerKind, ResultStore, RunResult, TrainConfig, Trainer


class ModelKind(Enum):
    """Model families in the ladder."""

    VIT = "vit"
    RESNET18 = "resnet18"


@dataclass(frozen=True)
class AblationStep:
    """One rung of the ladder: a model plus the training recipe it uses."""

    name: str
    label: str
    model: ModelKind
    augmentation: Augmentation
    patch_size: int | None = None
    warmup_epochs: int = 0
    label_smoothing: float = 0.0
    mixing: bool = False
    drop_path: float = 0.0
    epoch_scale: float = 1.0

    def __post_init__(self) -> None:
        if (self.model is ModelKind.VIT) != (self.patch_size is not None):
            raise ConfigError(f"{self.name}: patch_size is required for ViT steps and only for them")
        if self.model is ModelKind.RESNET18 and self.drop_path > 0.0:
            raise ConfigError(f"{self.name}: drop_path applies to ViT steps only")
        if self.epoch_scale <= 0.0:
            raise ConfigError(f"{self.name}: epoch_scale must be positive")


_STRONG = Augmentation.STRONG
_P4 = C.P1_SMALL_PATCH_SIZE
_WARM = C.P1_WARMUP_EPOCHS
_LS = C.P1_LABEL_SMOOTHING
_DP = C.P1_DROP_PATH

ABLATION_LADDER: tuple[AblationStep, ...] = (
    AblationStep("resnet18_noaug", "ResNet-18, no aug", ModelKind.RESNET18, Augmentation.NONE),
    AblationStep("vit_p8_naive", "ViT p8, no aug", ModelKind.VIT, Augmentation.NONE, C.P1_NAIVE_PATCH_SIZE),
    AblationStep("vit_p8_aug", "+ augmentation", ModelKind.VIT, _STRONG, C.P1_NAIVE_PATCH_SIZE),
    AblationStep("vit_p4_aug", "+ patch 4", ModelKind.VIT, _STRONG, _P4),
    AblationStep("vit_p4_aug_warmup", "+ warmup", ModelKind.VIT, _STRONG, _P4, _WARM),
    AblationStep("vit_p4_ls", "+ label smoothing", ModelKind.VIT, _STRONG, _P4, _WARM, _LS),
    AblationStep("vit_p4_mix", "+ MixUp/CutMix", ModelKind.VIT, _STRONG, _P4, _WARM, _LS, True),
    AblationStep("vit_p4_droppath", "+ stochastic depth", ModelKind.VIT, _STRONG, _P4, _WARM, _LS, True, _DP),
    AblationStep(
        C.P1_FINAL_VIT_STEP, "+ 2x epochs", ModelKind.VIT, _STRONG, _P4, _WARM, _LS, True, _DP,
        C.P1_LONG_SCHEDULE_SCALE,
    ),
    AblationStep("resnet18_aug", "ResNet-18, same aug", ModelKind.RESNET18, _STRONG),
)


class LadderCatalog:
    """Looks up ladder steps by name."""

    def __init__(self, steps: tuple[AblationStep, ...] = ABLATION_LADDER) -> None:
        self.steps = steps

    def names(self) -> list[str]:
        """Step names in ladder order."""
        return [s.name for s in self.steps]

    def get(self, name: str) -> AblationStep:
        """Return one step by name.

        Raises:
            ConfigError: if the name is unknown.
        """
        return self.select([name])[0]

    def select(self, names: list[str] | None) -> list[AblationStep]:
        """Return the named steps in ladder order, or every step when `names` is None.

        Raises:
            ConfigError: if a name is unknown.
        """
        if names is None:
            return list(self.steps)
        unknown = set(names) - set(self.names())
        if unknown:
            raise ConfigError(f"unknown steps {sorted(unknown)}; choose from {self.names()}")
        return [s for s in self.steps if s.name in names]


@dataclass(frozen=True)
class RunSpec:
    """One training run: a ladder step plus the overrides of a particular study.

    `None` overrides mean "use the step's own value".
    """

    run_name: str
    label: str
    step: AblationStep
    seed: int
    lr: float | None = None
    warmup_epochs: int | None = None
    epochs: int | None = None
    train_fraction: float = 1.0
    save_checkpoint: bool = False


class LadderPlanner:
    """Plans every ladder step for every seed. Checkpoints are kept for the first seed."""

    def __init__(self, seeds: tuple[int, ...]) -> None:
        if not seeds:
            raise ConfigError("need at least one seed")
        self.seeds = seeds

    @staticmethod
    def run_name(step_name: str, seed: int) -> str:
        """Result name of one step and seed."""
        return f"{step_name}_s{seed}"

    def plan(self, steps: list[AblationStep]) -> list[RunSpec]:
        """Seeds in the outer loop, so one complete seed finishes before the next starts."""
        return [
            RunSpec(self.run_name(step.name, seed), step.label, step, seed,
                    save_checkpoint=(seed == self.seeds[0] and step.model is ModelKind.VIT))
            for seed in self.seeds for step in steps
        ]


class SweepPlanner:
    """Plans a learning-rate sweep of one step, with and without warmup."""

    def __init__(self, lrs: tuple[float, ...], warmup_epochs: int, epochs: int, seed: int = C.DEFAULT_SEED):
        if not lrs or warmup_epochs <= 0 or warmup_epochs >= epochs:
            raise ConfigError("need learning rates and 0 < warmup_epochs < epochs")
        self.lrs = lrs
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.seed = seed

    @staticmethod
    def run_name(lr: float, warmup: int) -> str:
        """Result name of one sweep point."""
        return f"sweep_lr{lr:g}_w{warmup}"

    def plan(self, step: AblationStep) -> list[RunSpec]:
        """One run per (learning rate, warmup on or off)."""
        return [
            RunSpec(self.run_name(lr, warmup), f"lr {lr:g}, warmup {warmup}", step, self.seed,
                    lr=lr, warmup_epochs=warmup, epochs=self.epochs)
            for lr in self.lrs for warmup in (0, self.warmup_epochs)
        ]


class DataEfficiencyPlanner:
    """Plans runs on growing shares of the training set, at equal optimizer steps.

    A run on a fraction f trains for epochs / f epochs, so every run takes about
    the same number of optimizer steps. Smaller sets then do not lose simply
    because they saw fewer updates.
    """

    def __init__(self, fractions: tuple[float, ...], base_epochs: int, seed: int = C.DEFAULT_SEED) -> None:
        if not fractions or any(not 0.0 < f <= 1.0 for f in fractions):
            raise ConfigError("fractions must be in (0, 1]")
        self.fractions = fractions
        self.base_epochs = base_epochs
        self.seed = seed

    @staticmethod
    def run_name(step_name: str, fraction: float) -> str:
        """Result name of one step and fraction."""
        return f"data_{step_name}_f{fraction:g}"

    def plan(self, steps: list[AblationStep]) -> list[RunSpec]:
        """One run per (step, fraction)."""
        return [
            RunSpec(self.run_name(step.name, f), step.label, step, self.seed,
                    epochs=max(1, round(self.base_epochs * step.epoch_scale / f)), train_fraction=f)
            for step in steps for f in self.fractions
        ]


class StepModelBuilder:
    """Turns a run spec into a model and a training config."""

    def __init__(self, base_epochs: int) -> None:
        self.base_epochs = base_epochs

    def vit_config(self, step: AblationStep) -> ViTConfig:
        """The ViT shape for a ViT step."""
        return ViTConfig(
            image_size=C.CIFAR10_IMAGE_SIZE, patch_size=step.patch_size, num_classes=C.CIFAR10_NUM_CLASSES,
            dim=C.P1_VIT_DIM, depth=C.P1_VIT_DEPTH, num_heads=C.P1_VIT_HEADS, mlp_ratio=C.P1_VIT_MLP_RATIO,
            drop_path_rate=step.drop_path,
        )

    def model(self, step: AblationStep) -> nn.Module:
        """A freshly initialized model for the step."""
        if step.model is ModelKind.VIT:
            return VisionTransformer(self.vit_config(step))
        return CifarResNet18Builder(C.CIFAR10_NUM_CLASSES).build()

    def epochs(self, spec: RunSpec) -> int:
        """Epochs for a run: the explicit override, or the base scaled by the step."""
        return spec.epochs if spec.epochs is not None else max(1, round(self.base_epochs * spec.step.epoch_scale))

    def train_config(self, spec: RunSpec) -> TrainConfig:
        """AdamW for ViTs, SGD with momentum for ResNets, plus the spec's overrides."""
        step, epochs = spec.step, self.epochs(spec)
        warmup = step.warmup_epochs if spec.warmup_epochs is None else spec.warmup_epochs
        common = dict(epochs=epochs, warmup_epochs=min(warmup, epochs - 1), label_smoothing=step.label_smoothing)
        if step.model is ModelKind.VIT:
            return TrainConfig(
                lr=spec.lr or C.P1_VIT_LR, weight_decay=C.P1_VIT_WEIGHT_DECAY, optimizer=OptimizerKind.ADAMW,
                grad_clip_norm=C.P1_GRAD_CLIP_NORM, **common,
            )
        return TrainConfig(
            lr=spec.lr or C.P1_RESNET_LR, weight_decay=C.P1_RESNET_WEIGHT_DECAY, optimizer=OptimizerKind.SGD,
            **common,
        )

    @staticmethod
    def mixer(step: AblationStep) -> BatchMixer | None:
        """The MixUp/CutMix transform for steps that use it."""
        if not step.mixing:
            return None
        return BatchMixer(C.CIFAR10_NUM_CLASSES, C.P1_MIXUP_ALPHA, C.P1_CUTMIX_ALPHA)


class VitCheckpoint:
    """Saves and loads a trained from-scratch ViT together with its shape."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def path_for(self, name: str) -> Path:
        """Checkpoint path for a step name."""
        return self.folder / f"{name}{C.CHECKPOINT_SUFFIX}"

    def save(self, name: str, model: VisionTransformer) -> Path:
        """Write the weights and config; return the path."""
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.path_for(name)
        torch.save({"config": asdict(model.config), "state_dict": model.state_dict()}, path)
        return path

    def load(self, name: str, device: torch.device) -> VisionTransformer:
        """Rebuild the model from a checkpoint.

        Raises:
            CheckpointError: if the file is missing or has the wrong keys.
        """
        path = self.path_for(name)
        if not path.exists():
            raise CheckpointError(f"no checkpoint at {path}; train step '{name}' in Project 1 first")
        payload = torch.load(path, map_location=device, weights_only=True)
        if not {"config", "state_dict"} <= payload.keys():
            raise CheckpointError(f"{path} is not a ViT checkpoint written by VitCheckpoint")
        model = VisionTransformer(ViTConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"])
        return model.to(device).eval()


@dataclass(frozen=True)
class LadderSettings:
    """Run-wide settings shared by the Project 1 studies."""

    epochs: int = C.P1_EPOCHS
    batch_size: int = C.P1_BATCH_SIZE
    num_workers: int = C.DEFAULT_NUM_WORKERS
    subset: int | None = None
    seeds: tuple[int, ...] = C.P1_SEEDS

    def smoke(self) -> "LadderSettings":
        """A tiny version of these settings for a quick end-to-end check."""
        return replace(self, epochs=C.P1_SMOKE_EPOCHS, subset=C.P1_SMOKE_SUBSET, seeds=self.seeds[:1])


class AblationRunner:
    """Trains run specs, saves result JSON files, and saves ViT checkpoints on request."""

    def __init__(self, settings: LadderSettings, data_dir: Path, output_dir: Path, device: torch.device) -> None:
        self.settings = settings
        self.data_dir = data_dir
        self.store = ResultStore(output_dir)
        self.checkpoints = VitCheckpoint(output_dir)
        self.device = device
        self.builder = StepModelBuilder(settings.epochs)

    def run(self, specs: list[RunSpec], force: bool = False) -> list[RunResult]:
        """Train the specs in order, skipping finished ones unless `force`; return all results."""
        results = []
        for spec in specs:
            if not force and self.store.path_for(spec.run_name).exists():
                print(f"[{spec.run_name}] already done, skipping", flush=True)
                results.append(self.store.load(spec.run_name))
            else:
                results.append(self._run_one(spec))
        return results

    def _run_one(self, spec: RunSpec) -> RunResult:
        torch.manual_seed(spec.seed)
        data = Cifar10Data(
            self.data_dir, self.settings.batch_size, self.settings.num_workers, spec.step.augmentation,
            self.settings.subset, spec.train_fraction, spec.seed,
        ).load()
        model = self.builder.model(spec.step)
        trainer = Trainer(
            model, data.train_loader(), data.test_loader(), self.builder.train_config(spec), self.device,
            self.builder.mixer(spec.step),
        )
        result = trainer.fit(spec.run_name, spec.label)
        print(f"[{spec.run_name}] saved {self.store.save(result)}", flush=True)
        if spec.save_checkpoint and isinstance(model, VisionTransformer) and not result.diverged:
            self.checkpoints.save(spec.step.name, model)
        return result


@dataclass(frozen=True)
class StepSummary:
    """One ladder step over its seeds."""

    name: str
    label: str
    finals: tuple[float, ...]
    epochs: np.ndarray
    curve_mean: np.ndarray
    curve_std: np.ndarray
    any_diverged: bool

    @property
    def mean(self) -> float:
        """Mean final test accuracy."""
        return float(np.mean(self.finals))

    @property
    def std(self) -> float:
        """Sample standard deviation of the final test accuracy (0 for one seed)."""
        return float(np.std(self.finals, ddof=1)) if len(self.finals) > 1 else 0.0


class SeedAggregator:
    """Combines the per-seed results of each ladder step."""

    def __init__(self, store: ResultStore, seeds: tuple[int, ...]) -> None:
        self.store = store
        self.seeds = seeds

    def summarize(self, steps: list[AblationStep]) -> list[StepSummary]:
        """Summaries of the steps that have at least one finished seed, in ladder order."""
        summaries = []
        for step in steps:
            names = [LadderPlanner.run_name(step.name, s) for s in self.seeds]
            results = self.store.load_available(names)
            if results:
                summaries.append(self._summarize(step, results))
        return summaries

    @staticmethod
    def _summarize(step: AblationStep, results: list[RunResult]) -> StepSummary:
        curves = [{r.epoch: r.test_acc for r in res.history if r.test_acc is not None} for res in results]
        shared = sorted(set.intersection(*(set(c) for c in curves)))
        matrix = np.array([[c[e] for e in shared] for c in curves])
        return StepSummary(
            name=step.name, label=step.label,
            finals=tuple(r.final_test_acc or 0.0 for r in results),
            epochs=np.array(shared), curve_mean=matrix.mean(axis=0), curve_std=matrix.std(axis=0),
            any_diverged=any(r.diverged for r in results),
        )

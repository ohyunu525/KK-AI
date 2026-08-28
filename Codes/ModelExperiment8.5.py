from __future__ import annotations

"""Fair, resumable model experiments for the two-charge inverse problem.

The data semantics, normalization, split, loss targets, and non-identifiability
handling are inherited from ``NewLearning8.py``.  This file supplies the parts
that should vary in a model study: a model registry, explicit input-routing
policies, a common training loop, durable per-run state, and multi-seed reports.

Default research comparison
---------------------------
``g05_sign_only`` and ``g05_full_reconstruction`` have identical parameters
and identical initial states for a given seed.  The only architectural switch
is whether the masked G05 representation may enter the position, magnitude,
and relative-sign heads.  Both models use G05 for the identifiable global-sign
head.  With zero observed G05 points their forward outputs are identical.

Checkpoint selection
--------------------
Each training trajectory saves two complete, single-epoch model states:
``best_total.pt`` minimizes validation total loss, and ``best_structure.pt``
minimizes validation structure loss without global-sign loss in the selection
objective.  Both are evaluated on the same held-out test set only after
training.  Reports always distinguish the two ``checkpoint_selection`` values.
``latest.pt`` is the atomic resume authority, including both best snapshots.

Adding a model
--------------
Register a no-argument factory with ``@register_model``.  The returned module
must implement ``forward(g00, g05, g05_mask) -> ModelOutput``.  No data,
training, evaluation, checkpoint, or reporting code needs to change.
"""

import argparse
import csv
import hashlib
import itertools
import json
import os
import platform
import random
import sys
import tempfile
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    import NewLearning8 as physics
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "ModelExperiment8.5.py must remain beside Codes/NewLearning8.py so the "
        "validated project physics protocol can be imported."
    ) from error


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = physics.DEFAULT_DATA_PATH
DEFAULT_RESULTS_ROOT = PROJECT_DIR / "Results" / "model_experiments"
DEFAULT_CHECKPOINT_ROOT = PROJECT_DIR / "Models" / "model_experiments"
DEFAULT_EXPERIMENT_NAME = "g05_routing_dual_selection_v2"
DEFAULT_MODELS = ("g05_sign_only", "g05_full_reconstruction")
PROTOCOL_VERSION = "model-experiment-v2-dual-selection"
CHECKPOINT_SCHEMA_VERSION = 2
RESULT_SCHEMA_VERSION = 2
CHECKPOINT_SELECTIONS = ("total", "structure")
POSITION_MAE_NAMES = tuple(
    f"position_mae_{coordinate}" for coordinate in ("x1", "y1", "z1", "x2", "y2", "z2")
)
METRIC_NAMES = (
    "mean_position_mae",
    "position_error_1",
    "position_error_2",
    "mean_position_3d_error",
    "charge_mae_q1",
    "charge_mae_q2",
    "charge_magnitude_mae",
    "relative_sign_accuracy",
    "global_sign_bce",
    "global_sign_accuracy",
    "absolute_sign_accuracy",
    "signed_pair_accuracy",
    *POSITION_MAE_NAMES,
)
STRUCTURE_METRIC_NAMES = (
    "mean_position_mae",
    *POSITION_MAE_NAMES,
    "position_error_1",
    "position_error_2",
    "mean_position_3d_error",
    "charge_magnitude_mae",
    "relative_sign_accuracy",
)
LOWER_IS_BETTER = {
    "mean_position_mae",
    "position_error_1",
    "position_error_2",
    "mean_position_3d_error",
    "charge_mae_q1",
    "charge_mae_q2",
    "charge_magnitude_mae",
    "global_sign_bce",
    *POSITION_MAE_NAMES,
}


@dataclass(frozen=True)
class ModelOutput:
    position: torch.Tensor
    magnitude: torch.Tensor
    relative_sign_logit: torch.Tensor
    global_sign_logit: torch.Tensor


@dataclass(frozen=True)
class LossWeights:
    position: float = physics.POSITION_LOSS_WEIGHT
    magnitude: float = physics.MAGNITUDE_LOSS_WEIGHT
    relative_sign: float = physics.RELATIVE_SIGN_LOSS_WEIGHT
    global_sign: float = physics.GLOBAL_SIGN_LOSS_WEIGHT


@dataclass(frozen=True)
class BatchLoss:
    total: torch.Tensor
    structure: torch.Tensor
    position: torch.Tensor
    magnitude: torch.Tensor
    relative_sign: torch.Tensor
    global_sign: torch.Tensor | None


@dataclass(frozen=True)
class EpochLoss:
    total: float
    structure: float
    position: float
    magnitude: float
    relative_sign: float
    global_sign: float | None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str
    input_policy: Mapping[str, str]
    factory: Callable[[], nn.Module]


@dataclass(frozen=True)
class TrainingSettings:
    batch_size: int
    max_epochs: int
    learning_rate: float
    weight_decay: float
    loss_weights: LossWeights


MODEL_REGISTRY: dict[str, ModelSpec] = {}


def register_model(
    name: str,
    *,
    description: str,
    input_policy: Mapping[str, str],
) -> Callable[[Callable[[], nn.Module]], Callable[[], nn.Module]]:
    """Register a model factory without coupling it to the experiment loop."""

    if not name or any(character.isspace() for character in name):
        raise ValueError("Model names must be non-empty and contain no whitespace")

    def decorator(factory: Callable[[], nn.Module]) -> Callable[[], nn.Module]:
        if name in MODEL_REGISTRY:
            raise KeyError(f"Duplicate model registration: {name}")
        MODEL_REGISTRY[name] = ModelSpec(
            name=name,
            description=description,
            input_policy=dict(input_policy),
            factory=factory,
        )
        return factory

    return decorator


class RoutedChargeNet(nn.Module):
    """Capacity-matched network with an explicit G05-to-structure route.

    Module construction order preserves the existing separated model's common
    parameter initialization.  ``structure_context`` is present in both model
    variants, so parameter count and the full initialized state are identical.
    Its final projection starts at zero, making the variants' initial outputs
    identical even when G05 is present.
    """

    def __init__(self, *, allow_g05_for_structure: bool) -> None:
        super().__init__()
        self.allow_g05_for_structure = allow_g05_for_structure
        self.g00_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.g00_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
        )
        self.position_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
        )
        self.magnitude_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        self.relative_sign_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.g05_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.global_sign_head = nn.Sequential(
            nn.Linear(32 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # The route exists in both variants.  No bias and explicit observation
        # gating ensure G05=0 contributes exactly zero structural context.
        self.structure_context = nn.Sequential(
            nn.Linear(32 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 128, bias=False),
        )
        final_projection = self.structure_context[-1]
        assert isinstance(final_projection, nn.Linear)
        nn.init.zeros_(final_projection.weight)

    @staticmethod
    def _masked_g05_summary(
        point_features: torch.Tensor,
        g05_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = g05_mask.to(dtype=point_features.dtype)
        observed_count = mask.sum(dim=1).clamp_min(1.0)
        mean = (point_features * mask).sum(dim=1) / observed_count
        centered = point_features - mean.unsqueeze(1)
        variance = ((centered**2) * mask).sum(dim=1) / observed_count
        standard_deviation = torch.sqrt(variance.clamp_min(0.0) + 1e-6)

        masked_for_max = point_features.masked_fill(~g05_mask.bool(), -torch.inf)
        maximum = masked_for_max.max(dim=1).values
        has_observation = g05_mask.sum(dim=(1, 2)) > 0
        maximum = torch.where(
            has_observation[:, None],
            maximum,
            torch.zeros_like(maximum),
        )
        summary = torch.cat((mean, maximum, standard_deviation), dim=1)
        summary = summary * has_observation[:, None].to(summary.dtype)
        return summary, has_observation

    def forward(
        self,
        g00: torch.Tensor,
        g05: torch.Tensor,
        g05_mask: torch.Tensor,
    ) -> ModelOutput:
        if g05.shape[:2] != g05_mask.shape[:2] or g05_mask.shape[-1] != 1:
            raise ValueError(f"G05/mask shape mismatch: {g05.shape}, {g05_mask.shape}")

        structure_features = self.g00_encoder(self.g00_cnn(g00))
        point_features = self.g05_encoder(g05)
        g05_summary, has_observation = self._masked_g05_summary(
            point_features,
            g05_mask,
        )
        if self.allow_g05_for_structure:
            context = self.structure_context(g05_summary)
            context = context * has_observation[:, None].to(context.dtype)
            structure_features = structure_features + context

        return ModelOutput(
            position=self.position_head(structure_features),
            magnitude=F.softplus(self.magnitude_head(structure_features)),
            relative_sign_logit=self.relative_sign_head(structure_features).squeeze(1),
            global_sign_logit=self.global_sign_head(g05_summary).squeeze(1),
        )


@register_model(
    "g05_sign_only",
    description=(
        "G00 restores ordered positions, magnitudes, and relative sign; G05 is "
        "restricted to the q1-anchored global-sign prediction."
    ),
    input_policy={
        "position": "G00 only",
        "magnitude": "G00 only",
        "relative_sign": "G00 only",
        "global_sign": "G05 only",
    },
)
def build_g05_sign_only() -> nn.Module:
    return RoutedChargeNet(allow_g05_for_structure=False)


@register_model(
    "g05_full_reconstruction",
    description=(
        "G05 context is available to position, magnitude, and relative-sign "
        "restoration as well as the q1-anchored global-sign prediction."
    ),
    input_policy={
        "position": "G00 + masked G05",
        "magnitude": "G00 + masked G05",
        "relative_sign": "G00 + masked G05",
        "global_sign": "G05 only",
    },
)
def build_g05_full_reconstruction() -> nn.Module:
    return RoutedChargeNet(allow_g05_for_structure=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        json_ready(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def object_fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_with_retry(source: Path, destination: Path) -> None:
    """Atomically replace a file despite short-lived Windows file locks."""

    for attempt in range(8):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(0.05 * (2**attempt), 0.8))


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                json_ready(value),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        replace_with_retry(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows and not path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    if not rows:
        # Clear stale rows without changing an existing report's columns.
        with path.open("r", encoding="utf-8", newline="") as handle:
            fieldnames = next(csv.reader(handle), [])
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(
                {
                    key: "" if value is None else value
                    for key, value in json_ready(row).items()
                }
                for row in rows
            )
            handle.flush()
            os.fsync(handle.fileno())
        replace_with_retry(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        replace_with_retry(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_torch_save(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(dict(value), temporary_path)
        replace_with_retry(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_torch_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    # Checkpoints also contain CPU-only RNG state tensors.  Loading the whole
    # mapping onto CUDA makes Generator.set_state()/torch.set_rng_state() fail.
    # Model parameters are copied by load_state_dict, and optimizer tensors are
    # moved explicitly by optimizer_to_device after loading.
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint is not a mapping: {path}")
    return checkpoint


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def optimizer_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if isinstance(value, torch.Tensor):
                optimizer_state[key] = value.to(device)


def make_loader(
    dataset: TensorDataset,
    *,
    batch_size: int,
    device: torch.device,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def calculate_losses(
    output: ModelOutput,
    position_target: torch.Tensor,
    charge_target: torch.Tensor,
    g05_mask: torch.Tensor,
    weights: LossWeights,
) -> BatchLoss:
    position_loss = F.mse_loss(output.position, position_target)
    magnitude_loss = F.mse_loss(output.magnitude, torch.abs(charge_target))
    relative_target = (charge_target[:, 0] * charge_target[:, 1] > 0).to(
        output.relative_sign_logit.dtype
    )
    relative_sign_loss = F.binary_cross_entropy_with_logits(
        output.relative_sign_logit,
        relative_target,
    )
    structure_loss = (
        weights.position * position_loss
        + weights.magnitude * magnitude_loss
        + weights.relative_sign * relative_sign_loss
    )

    has_g05 = g05_mask.sum(dim=(1, 2)) > 0
    global_sign_loss: torch.Tensor | None = None
    total_loss = structure_loss
    if torch.any(has_g05):
        global_target = (charge_target[has_g05, 0] > 0).to(
            output.global_sign_logit.dtype
        )
        global_sign_loss = F.binary_cross_entropy_with_logits(
            output.global_sign_logit[has_g05],
            global_target,
        )
        total_loss = total_loss + weights.global_sign * global_sign_loss

    return BatchLoss(
        total=total_loss,
        structure=structure_loss,
        position=position_loss,
        magnitude=magnitude_loss,
        relative_sign=relative_sign_loss,
        global_sign=global_sign_loss,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    weights: LossWeights,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochLoss:
    is_training = optimizer is not None
    model.train(mode=is_training)
    sums = {
        "structure": 0.0,
        "position": 0.0,
        "magnitude": 0.0,
        "relative_sign": 0.0,
        "global_sign": 0.0,
    }
    sample_count = 0
    observed_sample_count = 0

    for g00, g05, g05_mask, position_target, charge_target in loader:
        g00 = g00.to(device, non_blocking=True)
        g05 = g05.to(device, non_blocking=True)
        g05_mask = g05_mask.to(device, non_blocking=True)
        position_target = position_target.to(device, non_blocking=True)
        charge_target = charge_target.to(device, non_blocking=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_training):
            output = model(g00, g05, g05_mask)
            losses = calculate_losses(
                output,
                position_target,
                charge_target,
                g05_mask,
                weights,
            )
            if optimizer is not None:
                losses.total.backward()
                optimizer.step()

        current_batch_size = g00.shape[0]
        sample_count += current_batch_size
        sums["structure"] += losses.structure.item() * current_batch_size
        sums["position"] += losses.position.item() * current_batch_size
        sums["magnitude"] += losses.magnitude.item() * current_batch_size
        sums["relative_sign"] += losses.relative_sign.item() * current_batch_size
        if losses.global_sign is not None:
            current_observed = int((g05_mask.sum(dim=(1, 2)) > 0).sum().item())
            observed_sample_count += current_observed
            sums["global_sign"] += losses.global_sign.item() * current_observed

    if sample_count == 0:
        raise ValueError("Cannot run an epoch on an empty DataLoader")
    global_sign = (
        sums["global_sign"] / observed_sample_count
        if observed_sample_count > 0
        else None
    )
    structure = sums["structure"] / sample_count
    total = structure + (0.0 if global_sign is None else weights.global_sign * global_sign)
    return EpochLoss(
        total=total,
        structure=structure,
        position=sums["position"] / sample_count,
        magnitude=sums["magnitude"] / sample_count,
        relative_sign=sums["relative_sign"] / sample_count,
        global_sign=global_sign,
    )


def align_global_charge_sign(
    prediction: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Oracle alignment only for samples where global sign is unidentifiable."""

    direct_error = np.mean((prediction - target) ** 2, axis=1)
    flipped_error = np.mean((-prediction - target) ** 2, axis=1)
    alignment = np.where(flipped_error < direct_error, -1.0, 1.0)[:, None]
    return prediction * alignment


def evaluate_model(
    model: nn.Module,
    dataset: TensorDataset,
    stats: physics.NormalizationStats,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    loader = make_loader(
        dataset,
        batch_size=batch_size,
        device=device,
        shuffle=False,
    )
    model.eval()
    predictions: dict[str, list[np.ndarray]] = {
        "position": [],
        "magnitude": [],
        "relative_logit": [],
        "global_logit": [],
        "position_target": [],
        "charge_target": [],
        "mask": [],
    }
    with torch.inference_mode():
        for g00, g05, g05_mask, position_target, charge_target in loader:
            output = model(
                g00.to(device, non_blocking=True),
                g05.to(device, non_blocking=True),
                g05_mask.to(device, non_blocking=True),
            )
            predictions["position"].append(output.position.cpu().numpy())
            predictions["magnitude"].append(output.magnitude.cpu().numpy())
            predictions["relative_logit"].append(
                output.relative_sign_logit.cpu().numpy()
            )
            predictions["global_logit"].append(
                output.global_sign_logit.cpu().numpy()
            )
            predictions["position_target"].append(position_target.numpy())
            predictions["charge_target"].append(charge_target.numpy())
            predictions["mask"].append(g05_mask.numpy())

    position_prediction = (
        np.concatenate(predictions["position"]) * stats.position_std
        + stats.position_mean
    )
    position_target = (
        np.concatenate(predictions["position_target"]) * stats.position_std
        + stats.position_mean
    )
    magnitude_prediction = (
        np.concatenate(predictions["magnitude"]) * stats.charge_scale
    )
    charge_target = np.concatenate(predictions["charge_target"]) * stats.charge_scale
    relative_logit = np.concatenate(predictions["relative_logit"])
    global_logit = np.concatenate(predictions["global_logit"])
    mask = np.concatenate(predictions["mask"])
    g05_counts = mask.sum(axis=(1, 2))
    observed = g05_counts > 0

    relative_sign = np.where(relative_logit >= 0, 1.0, -1.0)
    global_sign = np.where(global_logit >= 0, 1.0, -1.0)
    charge_prediction = np.column_stack(
        (
            magnitude_prediction[:, 0] * global_sign,
            magnitude_prediction[:, 1] * global_sign * relative_sign,
        )
    )
    evaluated_charge_prediction = charge_prediction.copy()
    if np.any(~observed):
        evaluated_charge_prediction[~observed] = align_global_charge_sign(
            charge_prediction[~observed],
            charge_target[~observed],
        )

    position_mae = np.mean(np.abs(position_prediction - position_target), axis=0)
    position_error_1 = float(
        np.linalg.norm(
            position_prediction[:, 0:3] - position_target[:, 0:3],
            axis=1,
        ).mean()
    )
    position_error_2 = float(
        np.linalg.norm(
            position_prediction[:, 3:6] - position_target[:, 3:6],
            axis=1,
        ).mean()
    )
    charge_mae = np.mean(
        np.abs(evaluated_charge_prediction - charge_target),
        axis=0,
    )
    target_relative_sign = np.sign(charge_target[:, 0] * charge_target[:, 1])
    metrics: dict[str, Any] = {
        "position_mae": position_mae,
        **{name: float(value) for name, value in zip(POSITION_MAE_NAMES, position_mae)},
        "mean_position_mae": float(position_mae.mean()),
        "position_error_1": position_error_1,
        "position_error_2": position_error_2,
        "mean_position_3d_error": 0.5 * (position_error_1 + position_error_2),
        "charge_mae": charge_mae,
        "charge_mae_q1": float(charge_mae[0]),
        "charge_mae_q2": float(charge_mae[1]),
        "charge_magnitude_mae": float(
            np.mean(np.abs(magnitude_prediction - np.abs(charge_target)))
        ),
        "relative_sign_accuracy": float(
            np.mean(relative_sign == target_relative_sign)
        ),
        "global_sign_bce": None,
        "global_sign_accuracy": None,
        "absolute_sign_accuracy": None,
        "signed_pair_accuracy": None,
        "observed_sample_fraction": float(observed.mean()),
        "observations_per_sample": float(g05_counts.mean()),
        "charge_error_sign_policy": (
            "oracle global-sign alignment only for samples with no observed G05"
        ),
    }

    if np.any(observed):
        target_global_positive = charge_target[observed, 0] > 0
        logits = global_logit[observed].astype(np.float64)
        targets = target_global_positive.astype(np.float64)
        metrics["global_sign_accuracy"] = float(
            np.mean((logits >= 0) == target_global_positive)
        )
        metrics["global_sign_bce"] = float(
            np.mean(
                np.maximum(logits, 0.0)
                - logits * targets
                + np.log1p(np.exp(-np.abs(logits)))
            )
        )
        prediction_sign = np.sign(charge_prediction[observed])
        target_sign = np.sign(charge_target[observed])
        metrics["absolute_sign_accuracy"] = float(
            np.mean(prediction_sign == target_sign)
        )
        metrics["signed_pair_accuracy"] = float(
            np.mean(np.all(prediction_sign == target_sign, axis=1))
        )
    return json_ready(metrics)


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def epoch_loss_dict(loss: EpochLoss) -> dict[str, float | None]:
    return asdict(loss)


def finite_epoch_loss(loss: EpochLoss) -> bool:
    values = [
        loss.total,
        loss.structure,
        loss.position,
        loss.magnitude,
        loss.relative_sign,
    ]
    if loss.global_sign is not None:
        values.append(loss.global_sign)
    return all(np.isfinite(value) for value in values)


def fraction_label(fraction: float) -> str:
    return f"{int(round(fraction * 100)):03d}pct"


def run_configuration(
    *,
    protocol_fingerprint: str,
    code_sha256: str,
    spec: ModelSpec,
    fraction: float,
    g05_count: int,
    candidate_count: int,
    seed: int,
    settings: TrainingSettings,
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_fingerprint": protocol_fingerprint,
        "code_sha256": code_sha256,
        "model": {
            "name": spec.name,
            "description": spec.description,
            "input_policy": spec.input_policy,
        },
        "observation": {
            "name": f"g05_{fraction_label(fraction)}",
            "g05_fraction": fraction,
            "g05_count_per_sample": g05_count,
            "candidate_count": candidate_count,
            "selection": "fixed spatially-balanced nested prefix",
            "full_fraction_note": (
                "all stored candidate sensors, not the complete potential grid"
                if fraction == 1.0
                else None
            ),
        },
        "training": {
            "seed": seed,
            "batch_size": settings.batch_size,
            "max_epochs": settings.max_epochs,
            "optimizer": "AdamW",
            "learning_rate": settings.learning_rate,
            "weight_decay": settings.weight_decay,
            "loss_weights": asdict(settings.loss_weights),
            "checkpoint_selection": {
                selection: selection_policy(
                    selection, g05_count=g05_count,
                    global_sign_weight=settings.loss_weights.global_sign,
                )
                for selection in CHECKPOINT_SELECTIONS
            },
        },
    }


def run_id_for(config: Mapping[str, Any]) -> str:
    fingerprint = object_fingerprint(config)
    model_name = str(config["model"]["name"])
    fraction = float(config["observation"]["g05_fraction"])
    seed = int(config["training"]["seed"])
    return (
        f"{model_name}__g05_{fraction_label(fraction)}__seed_{seed}"
        f"__{fingerprint[:12]}"
    )


def selection_policy(
    selection: str, *, g05_count: int | None = None, global_sign_weight: float = 1.0,
) -> dict[str, Any]:
    """A null total inclusion flag means the shared protocol has no run G05 count."""
    if selection not in CHECKPOINT_SELECTIONS:
        raise ValueError(f"Unknown checkpoint selection: {selection!r}")
    is_structure = selection == "structure"
    includes_global_sign: bool | None = False
    if not is_structure and global_sign_weight != 0.0:
        includes_global_sign = None if g05_count is None else g05_count > 0
    return {
        "checkpoint_selection": selection,
        "selection_objective": f"validation_loss.{selection}",
        "selection_note": (
            f"One complete model state minimizing validation {selection} loss; "
            "no cross-epoch component composition. Equal losses keep the first epoch."
        ),
        "primary_metrics": STRUCTURE_METRIC_NAMES if is_structure else METRIC_NAMES,
        "global_sign_in_selection_objective": includes_global_sign,
        "global_sign_metrics_note": (
            "Global-sign performance was not optimized by checkpoint selection. "
            "Global/absolute/signed-pair sign metrics and signed charge MAE are "
            "secondary diagnostics; training still uses the unchanged total loss."
            if is_structure
            else "Selection includes global-sign loss only when G05 is observed and "
            "its loss weight is nonzero, together with structure loss; "
            "it does not optimize global sign alone."
        ),
    }


def run_metadata(run_config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run_id_for(run_config),
        "run_fingerprint": object_fingerprint(run_config),
        "protocol_fingerprint": run_config["protocol_fingerprint"],
        "protocol_version": run_config["protocol_version"],
        "model_name": run_config["model"]["name"],
        "g05_fraction": run_config["observation"]["g05_fraction"],
        "g05_count_per_sample": run_config["observation"]["g05_count_per_sample"],
        "seed": run_config["training"]["seed"],
    }


def run_checkpoint_paths(checkpoint_run_dir: Path) -> dict[str, Path]:
    return {
        "latest": checkpoint_run_dir / "latest.pt",
        **{
            selection: checkpoint_run_dir / f"best_{selection}.pt"
            for selection in CHECKPOINT_SELECTIONS
        },
    }


def copy_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    # Best snapshots survive subsequent optimizer steps, including CPU training.
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def update_best_checkpoints(
    best_checkpoints: dict[str, dict[str, Any]],
    *,
    run_config: Mapping[str, Any],
    epoch: int,
    validation_loss: EpochLoss,
    model_state: dict[str, torch.Tensor],
) -> tuple[str, ...]:
    """Select each objective independently from the same complete epoch state."""

    updated: list[str] = []
    for selection in CHECKPOINT_SELECTIONS:
        value = float(getattr(validation_loss, selection))
        if not np.isfinite(value):
            raise FloatingPointError(f"Non-finite validation {selection} loss")
        previous = best_checkpoints.get(selection)
        if previous is not None and value >= previous["selected_validation_loss"]:
            continue
        best_checkpoints[selection] = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_kind": "selected",
            **run_metadata(run_config),
            **run_config["training"]["checkpoint_selection"][selection],
            "selected_epoch": epoch,
            "selected_validation_loss": value,
            "validation_losses": epoch_loss_dict(validation_loss),
            # Preserve familiar scalar fields, scoped by checkpoint_selection.
            "epoch": epoch,
            "validation_loss": value,
            "model_state_dict": model_state,
        }
        updated.append(selection)
    return tuple(updated)


def best_tracking_fields(
    best_checkpoints: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for selection in CHECKPOINT_SELECTIONS:
        checkpoint = best_checkpoints.get(selection, {})
        fields[f"best_{selection}_loss"] = checkpoint.get("selected_validation_loss")
        fields[f"best_{selection}_epoch"] = checkpoint.get("selected_epoch")
    # Legacy resume/status aliases always refer to total selection.
    fields["best_validation_loss"] = fields["best_total_loss"]
    fields["best_epoch"] = fields["best_total_epoch"]
    return fields


def make_resume_checkpoint(
    *,
    run_config: Mapping[str, Any],
    epoch: int,
    model_state: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    shuffle_generator: torch.Generator,
    best_checkpoints: dict[str, dict[str, Any]],
    history: list[dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    if set(best_checkpoints) != set(CHECKPOINT_SELECTIONS):
        raise RuntimeError("Resume state requires both total and structure checkpoints")
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_kind": "latest",
        **run_metadata(run_config),
        "checkpoint_selection": "latest",
        "selection_objective": "latest completed epoch for resuming training only",
        "selected_epoch": epoch,
        "selected_validation_loss": None,  # No minimization objective for latest.
        "validation_losses": history[-1]["validation"],
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "shuffle_generator_state": shuffle_generator.get_state(),
        "rng_state": capture_rng_state(),
        **best_tracking_fields(best_checkpoints),
        # A single atomic commit covers training state AND both best snapshots.
        # The standalone best files can be rebuilt after an interrupted publish.
        "best_checkpoints": best_checkpoints,
        "elapsed_seconds": elapsed_seconds,
        "history": history,
    }


def validate_selected_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    run_config: Mapping[str, Any],
    selection: str,
    expected_epoch: int,
    expected_loss: float,
) -> None:
    expected = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_kind": "selected",
        **run_metadata(run_config),
        **run_config["training"]["checkpoint_selection"][selection],
        "selected_epoch": expected_epoch,
        "selected_validation_loss": expected_loss,
        "epoch": expected_epoch,
        "validation_loss": expected_loss,
    }
    for key, value in expected.items():
        if canonical_json(checkpoint.get(key)) != canonical_json(value):
            raise RuntimeError(f"Invalid {selection} checkpoint metadata: {key}")
    if checkpoint.get("validation_losses", {}).get(selection) != expected_loss:
        raise RuntimeError(f"Invalid {selection} checkpoint validation loss")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError(f"Missing complete model state in {selection} checkpoint")


def validate_resume_checkpoint(
    checkpoint: Mapping[str, Any], run_config: Mapping[str, Any]
) -> None:
    if (
        checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION
        or checkpoint.get("checkpoint_kind") != "latest"
    ):
        raise RuntimeError(
            "This is not a dual-selection latest checkpoint. Legacy best.pt/latest.pt "
            "cannot recover an unsaved historical structure optimum. Keep the legacy "
            "run unchanged and use a new --experiment-name for dual selection."
        )
    for key, value in run_metadata(run_config).items():
        if checkpoint.get(key) != value:
            raise RuntimeError(f"Latest checkpoint metadata mismatch: {key}")
    required = {
        "epoch", "model_state_dict", "optimizer_state_dict", "rng_state",
        "shuffle_generator_state", "history", "best_checkpoints", "elapsed_seconds",
        "best_total_loss", "best_total_epoch", "best_structure_loss", "best_structure_epoch",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise RuntimeError(f"Incomplete resume state: {sorted(missing)}")
    epoch = int(checkpoint["epoch"])
    history = checkpoint["history"]
    if (
        not 1 <= epoch <= int(run_config["training"]["max_epochs"])
        or len(history) != epoch
        or [row["epoch"] for row in history] != list(range(1, epoch + 1))
    ):
        raise RuntimeError("Latest checkpoint epoch/history mismatch")
    best_checkpoints = checkpoint["best_checkpoints"]
    if set(best_checkpoints) != set(CHECKPOINT_SELECTIONS):
        raise RuntimeError("Latest checkpoint must contain both selection snapshots")
    for key, value in best_tracking_fields(best_checkpoints).items():
        if checkpoint.get(key) != value:
            raise RuntimeError(f"Latest checkpoint best tracker mismatch: {key}")
    for selection in CHECKPOINT_SELECTIONS:
        best_row = min(history, key=lambda row: row["validation"][selection])
        best = best_checkpoints[selection]
        validate_selected_checkpoint(
            best,
            run_config=run_config,
            selection=selection,
            expected_epoch=int(best_row["epoch"]),
            expected_loss=float(best_row["validation"][selection]),
        )
        if best["validation_losses"] != best_row["validation"]:
            raise RuntimeError(f"{selection} checkpoint does not match its history epoch")
        if best["model_state_dict"].keys() != checkpoint["model_state_dict"].keys():
            raise RuntimeError(f"{selection} checkpoint is not a complete model state")


def save_status(
    path: Path,
    *,
    status: str,
    run_id: str,
    **extra: Any,
) -> None:
    atomic_write_json(
        path,
        {
            "run_id": run_id,
            "status": status,
            "updated_at": utc_now(),
            **extra,
        },
    )


def train_and_evaluate_run(
    *,
    spec: ModelSpec,
    train_dataset: TensorDataset,
    validation_dataset: TensorDataset,
    test_dataset: TensorDataset,
    stats: physics.NormalizationStats,
    run_config: dict[str, Any],
    experiment_results_dir: Path,
    experiment_checkpoint_dir: Path,
    settings: TrainingSettings,
    device: torch.device,
) -> tuple[dict[str, Any], bool]:
    run_id = run_id_for(run_config)
    run_fingerprint = object_fingerprint(run_config)
    result_run_dir = experiment_results_dir / "runs" / run_id
    checkpoint_run_dir = experiment_checkpoint_dir / run_id
    config_path = result_run_dir / "config.json"
    status_path = result_run_dir / "status.json"
    history_path = result_run_dir / "history.json"
    result_path = result_run_dir / "result.json"
    checkpoint_paths = run_checkpoint_paths(checkpoint_run_dir)
    latest_path = checkpoint_paths["latest"]

    if result_path.exists():
        with result_path.open("r", encoding="utf-8") as handle:
            existing_result = json.load(handle)
        if existing_result.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError(f"Result fingerprint mismatch: {result_path}")
        completed_result_evaluations(existing_result)
        completed_evaluations = existing_result["evaluations"]
        recovery_checkpoint = None
        for selection in CHECKPOINT_SELECTIONS:
            best_path = checkpoint_paths[selection]
            needs_restore = not best_path.exists()
            if needs_restore:
                if recovery_checkpoint is None:
                    try:
                        recovery_checkpoint = load_torch_checkpoint(latest_path, device)
                    except FileNotFoundError as error:
                        raise RuntimeError(
                            f"Cannot restore {best_path.name}; latest checkpoint is missing: {latest_path}"
                        ) from error
                    validate_resume_checkpoint(recovery_checkpoint, run_config)
                    if recovery_checkpoint["epoch"] != run_config["training"]["max_epochs"]:
                        raise RuntimeError("Cannot restore a completed run from an unfinished latest checkpoint")
                best = recovery_checkpoint["best_checkpoints"][selection]
            else:
                best = load_torch_checkpoint(best_path, device)
            evaluation = completed_evaluations[selection]
            validate_selected_checkpoint(
                best, run_config=run_config, selection=selection,
                expected_epoch=evaluation["selected_epoch"],
                expected_loss=evaluation["selected_validation_loss"],
            )
            if best["validation_losses"] != evaluation["validation_losses"]:
                raise RuntimeError(f"{selection} checkpoint losses differ from the completed result")
            if needs_restore:
                atomic_torch_save(best, best_path)
        save_status(
            status_path, status="completed", run_id=run_id,
            **best_tracking_fields(completed_evaluations),
            result_path=str(result_path.resolve()),
        )
        print(f"SKIP completed: {run_id}")
        return existing_result, True

    result_run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_run_dir.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            existing_config = json.load(handle)
        if object_fingerprint(existing_config) != run_fingerprint:
            raise RuntimeError(f"Run configuration mismatch: {config_path}")
    else:
        atomic_write_json(config_path, run_config)

    seed = int(run_config["training"]["seed"])
    set_reproducibility(seed)
    model = spec.factory().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    shuffle_generator = torch.Generator().manual_seed(seed)
    train_loader = make_loader(
        train_dataset,
        batch_size=settings.batch_size,
        device=device,
        shuffle=True,
        generator=shuffle_generator,
    )
    validation_loader = make_loader(
        validation_dataset,
        batch_size=settings.batch_size,
        device=device,
        shuffle=False,
    )

    start_epoch = 1
    best_checkpoints: dict[str, dict[str, Any]] = {}
    history: list[dict[str, Any]] = []
    elapsed_before_resume = 0.0
    resumed = False
    if latest_path.exists():
        checkpoint = load_torch_checkpoint(latest_path, device)
        validate_resume_checkpoint(checkpoint, run_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        optimizer_to_device(optimizer, device)
        shuffle_generator.set_state(checkpoint["shuffle_generator_state"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_checkpoints = dict(checkpoint["best_checkpoints"])
        history = list(checkpoint["history"])
        elapsed_before_resume = float(checkpoint.get("elapsed_seconds", 0.0))
        resumed = True
        # latest.pt is authoritative if publishing either best file was interrupted.
        for selection in CHECKPOINT_SELECTIONS:
            atomic_torch_save(best_checkpoints[selection], checkpoint_paths[selection])
        atomic_write_json(history_path, history)

    save_status(
        status_path,
        status="running",
        run_id=run_id,
        resumed=resumed,
        next_epoch=start_epoch,
        **best_tracking_fields(best_checkpoints),
    )
    print(
        f"\nRUN {run_id} | device={device} | "
        f"resume_epoch={start_epoch if resumed else 'new'}"
    )
    started_at = time.perf_counter()
    for epoch in range(start_epoch, settings.max_epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            device=device,
            weights=settings.loss_weights,
            optimizer=optimizer,
        )
        validation_loss = run_epoch(
            model,
            validation_loader,
            device=device,
            weights=settings.loss_weights,
        )
        if not finite_epoch_loss(validation_loss):
            raise FloatingPointError(
                f"Non-finite validation loss in {run_id} at epoch {epoch}"
            )
        epoch_row = {
            "epoch": epoch,
            "train": epoch_loss_dict(train_loss),
            "validation": epoch_loss_dict(validation_loss),
        }
        history.append(epoch_row)
        epoch_state = copy_model_state(model)
        updated_selections = update_best_checkpoints(
            best_checkpoints,
            run_config=run_config,
            epoch=epoch,
            validation_loss=validation_loss,
            model_state=epoch_state,
        )

        elapsed_seconds = elapsed_before_resume + (time.perf_counter() - started_at)
        atomic_torch_save(
            make_resume_checkpoint(
                run_config=run_config,
                epoch=epoch,
                model_state=epoch_state,
                optimizer=optimizer,
                shuffle_generator=shuffle_generator,
                best_checkpoints=best_checkpoints,
                history=history,
                elapsed_seconds=elapsed_seconds,
            ),
            latest_path,
        )
        # Publish selected files only after the resume state commits atomically.
        for selection in updated_selections:
            atomic_torch_save(best_checkpoints[selection], checkpoint_paths[selection])
        atomic_write_json(history_path, history)
        save_status(
            status_path,
            status="running",
            run_id=run_id,
            resumed=resumed,
            next_epoch=epoch + 1,
            **best_tracking_fields(best_checkpoints),
        )
        global_text = (
            "N/A"
            if validation_loss.global_sign is None
            else f"{validation_loss.global_sign:.6f}"
        )
        print(
            f"  epoch={epoch:03d} train={train_loss.total:.6f} "
            f"val={validation_loss.total:.6f} "
            f"val_structure={validation_loss.structure:.6f} "
            f"val_global={global_text} "
            + " ".join(
                f"best_{selection}={best_checkpoints[selection]['selected_validation_loss']:.6f}"
                f"@{best_checkpoints[selection]['selected_epoch']}"
                for selection in CHECKPOINT_SELECTIONS
            )
        )

    # Test data has no role in training or either checkpoint selection.
    evaluations: dict[str, dict[str, Any]] = {}
    for selection in CHECKPOINT_SELECTIONS:
        best_path = checkpoint_paths[selection]
        if not best_path.exists():
            raise RuntimeError(f"No valid {selection} checkpoint was produced: {run_id}")
        best_checkpoint = load_torch_checkpoint(best_path, device)
        validate_selected_checkpoint(
            best_checkpoint,
            run_config=run_config,
            selection=selection,
            expected_epoch=best_checkpoints[selection]["selected_epoch"],
            expected_loss=best_checkpoints[selection]["selected_validation_loss"],
        )
        model.load_state_dict(best_checkpoint["model_state_dict"])
        metrics = evaluate_model(
            model,
            test_dataset,
            stats,
            batch_size=settings.batch_size,
            device=device,
        )
        evaluations[selection] = {
            key: value for key, value in best_checkpoint.items() if key != "model_state_dict"
        }
        evaluations[selection].update(
            checkpoint_path=str(best_path.resolve()),
            test_metrics=metrics,
        )
        print(
            f"  TEST selection={selection} "
            f"epoch={best_checkpoint['selected_epoch']} | "
            f"position_mae={metrics['mean_position_mae']:.6f} | "
            f"position_3d={metrics['mean_position_3d_error']:.6f} | "
            f"magnitude_mae={metrics['charge_magnitude_mae']:.6f}"
        )
    elapsed_seconds = elapsed_before_resume + (time.perf_counter() - started_at)
    result = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        **run_metadata(run_config),
        "status": "completed",
        "completed_at": utc_now(),
        "configuration": run_config,
        "parameter_count": parameter_counts(model),
        "training_result": {
            **best_tracking_fields(best_checkpoints),
            "legacy_best_fields_selection": "total",
            "epochs_completed": settings.max_epochs,
            "elapsed_seconds": elapsed_seconds,
            "resumed": resumed,
            "selection_note": (
                "See evaluations.total and evaluations.structure; "
                "both use complete epoch states."
            ),
        },
        "evaluations": evaluations,
        "artifacts": {
            "config": str(config_path.resolve()),
            "history": str(history_path.resolve()),
            "latest_checkpoint": str(latest_path.resolve()),
            **{
                f"best_{selection}_checkpoint": str(checkpoint_paths[selection].resolve())
                for selection in CHECKPOINT_SELECTIONS
            },
        },
    }
    atomic_write_json(result_path, result)
    save_status(
        status_path,
        status="completed",
        run_id=run_id,
        **best_tracking_fields(best_checkpoints),
        result_path=str(result_path.resolve()),
    )
    print(f"DONE {run_id} | total and structure evaluations saved")
    return result, False


def completed_result_evaluations(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten a complete dual-evaluation run without interpreting legacy results."""

    if (
        result.get("result_schema_version") != RESULT_SCHEMA_VERSION
        or result.get("status") != "completed"
        or set(result.get("evaluations", {})) != set(CHECKPOINT_SELECTIONS)
    ):
        raise RuntimeError(
            "Expected a completed dual-selection result with both evaluations; "
            "legacy total-only results must stay in their original experiment."
        )
    config = result["configuration"]
    metadata = run_metadata(config)
    for key, value in metadata.items():
        if result.get(key) != value:
            raise RuntimeError(f"Result metadata mismatch: {key}")
    records: list[dict[str, Any]] = []
    common = {key: value for key, value in result.items() if key != "evaluations"}
    for selection in CHECKPOINT_SELECTIONS:
        evaluation = result["evaluations"][selection]
        expected = {
            **metadata,
            **config["training"]["checkpoint_selection"][selection],
            "selected_epoch": result["training_result"][f"best_{selection}_epoch"],
            "selected_validation_loss": result["training_result"][f"best_{selection}_loss"],
        }
        for key, value in expected.items():
            if canonical_json(evaluation.get(key)) != canonical_json(value):
                raise RuntimeError(f"Invalid {selection} result metadata: {key}")
        if evaluation.get("validation_losses", {}).get(selection) != expected["selected_validation_loss"]:
            raise RuntimeError(f"Invalid {selection} result validation loss")
        if (
            not isinstance(evaluation.get("test_metrics"), Mapping)
            or not evaluation.get("checkpoint_path")
        ):
            raise RuntimeError(f"Missing {selection} test metrics/checkpoint path")
        records.append({**common, **evaluation})
    return records


def result_to_row(result: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one explicitly selected evaluation, not a whole training run."""
    config = result["configuration"]
    model = config["model"]
    observation = config["observation"]
    training = config["training"]
    training_result = result["training_result"]
    metrics = result["test_metrics"]
    return {
        "run_id": result["run_id"],
        "run_fingerprint": result["run_fingerprint"],
        "protocol_fingerprint": result["protocol_fingerprint"],
        "checkpoint_selection": result["checkpoint_selection"],
        "selection_objective": result["selection_objective"],
        "selected_epoch": result["selected_epoch"],
        "selected_validation_loss": result["selected_validation_loss"],
        "selected_validation_total_loss": result["validation_losses"]["total"],
        "selected_validation_structure_loss": result["validation_losses"]["structure"],
        "global_sign_in_selection_objective": result["global_sign_in_selection_objective"],
        "global_sign_metrics_note": result["global_sign_metrics_note"],
        "primary_metrics": canonical_json(result["primary_metrics"]),
        "model": model["name"],
        "model_description": model["description"],
        "input_policy": canonical_json(model["input_policy"]),
        "g05_fraction": observation["g05_fraction"],
        "g05_count_per_sample": observation["g05_count_per_sample"],
        "seed": training["seed"],
        "parameter_count": result["parameter_count"]["total"],
        **{name: metrics.get(name) for name in METRIC_NAMES},
        "observed_sample_fraction": metrics["observed_sample_fraction"],
        "observations_per_sample": metrics["observations_per_sample"],
        # Keep old CSV column names as aliases for THIS row's selection.
        "best_validation_loss": result["selected_validation_loss"],
        "best_epoch": result["selected_epoch"],
        "elapsed_seconds": training_result["elapsed_seconds"],
        "best_checkpoint": result["checkpoint_path"],
    }


def sample_std(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def finite_metric_values(
    results: Sequence[Mapping[str, Any]],
    metric_name: str,
) -> list[float]:
    values: list[float] = []
    for result in results:
        value = result["test_metrics"].get(metric_name)
        if value is not None and np.isfinite(float(value)):
            values.append(float(value))
    return values


def comparison_key(result: Mapping[str, Any]) -> tuple[str, str, str, float, int]:
    config = result["configuration"]
    selection = str(result["checkpoint_selection"])
    if selection not in CHECKPOINT_SELECTIONS:
        raise ValueError(f"Missing or invalid report selection: {selection!r}")
    return (
        str(config["protocol_fingerprint"]),
        selection,
        str(config["model"]["name"]),
        float(config["observation"]["g05_fraction"]),
        int(config["training"]["seed"]),
    )


def selected_metadata_by_seed(results: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {
        column: canonical_json({
            str(item["configuration"]["training"]["seed"]): item[field]
            for item in results
        })
        for column, field in (
            ("run_fingerprints_by_seed", "run_fingerprint"),
            ("selected_epochs_by_seed", "selected_epoch"),
            ("selected_validation_losses_by_seed", "selected_validation_loss"),
        )
    }


def build_summary_rows(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, float], list[Mapping[str, Any]]] = {}
    seen: set[tuple[str, str, str, float, int]] = set()
    for result in results:
        key = comparison_key(result)
        if key in seen:
            raise RuntimeError(f"Duplicate seed/selection in summary: {key}")
        seen.add(key)
        grouped.setdefault(key[:-1], []).append(result)

    rows: list[dict[str, Any]] = []
    for (protocol_fingerprint, selection, model_name, fraction), group in sorted(grouped.items()):
        group = sorted(group, key=lambda item: int(item["configuration"]["training"]["seed"]))
        first = group[0]
        row: dict[str, Any] = {
            "protocol_fingerprint": protocol_fingerprint,
            "checkpoint_selection": selection,
            "selection_objective": first["selection_objective"],
            "global_sign_in_selection_objective": first["global_sign_in_selection_objective"],
            "global_sign_metrics_note": first["global_sign_metrics_note"],
            "primary_metrics": canonical_json(first["primary_metrics"]),
            **selected_metadata_by_seed(group),
            "model": model_name,
            "model_description": first["configuration"]["model"]["description"],
            "input_policy": canonical_json(
                first["configuration"]["model"]["input_policy"]
            ),
            "g05_fraction": fraction,
            "g05_count_per_sample": first["configuration"]["observation"][
                "g05_count_per_sample"
            ],
            "run_count": len(group),
            "seeds": ",".join(
                str(item["configuration"]["training"]["seed"]) for item in group
            ),
            "parameter_count": first["parameter_count"]["total"],
            "std_definition": "sample std across independent seeds (ddof=1)",
        }
        for metric_name in METRIC_NAMES:
            values = finite_metric_values(group, metric_name)
            row[f"{metric_name}_mean"] = (
                float(np.mean(values)) if values else None
            )
            row[f"{metric_name}_std"] = sample_std(values)
        rows.append(row)
    return rows


def build_pairwise_rows(
    results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, float, int], Mapping[str, Any]] = {}
    discovered_model_names: set[str] = set()
    contexts: set[tuple[str, str, float]] = set()
    for result in results:
        key = comparison_key(result)
        if key in by_key:
            raise RuntimeError(f"Duplicate seed/selection in paired comparison: {key}")
        protocol_fingerprint, selection, model_name, fraction, _ = key
        by_key[key] = result
        discovered_model_names.add(model_name)
        contexts.add((protocol_fingerprint, selection, fraction))

    default_order = {name: index for index, name in enumerate(DEFAULT_MODELS)}
    model_names = sorted(
        discovered_model_names,
        key=lambda name: (default_order.get(name, len(default_order)), name),
    )

    rows: list[dict[str, Any]] = []
    for model_a, model_b in itertools.combinations(model_names, 2):
        for protocol_fingerprint, selection, fraction in sorted(contexts):
            seeds_a = {
                seed
                for protocol, policy, name, current_fraction, seed in by_key
                if (protocol, policy, name, current_fraction)
                == (protocol_fingerprint, selection, model_a, fraction)
            }
            seeds_b = {
                seed
                for protocol, policy, name, current_fraction, seed in by_key
                if (protocol, policy, name, current_fraction)
                == (protocol_fingerprint, selection, model_b, fraction)
            }
            common_seeds = sorted(seeds_a.intersection(seeds_b))
            for metric_name in METRIC_NAMES:
                deltas: list[float] = []
                used_seeds: list[int] = []
                used_a: list[Mapping[str, Any]] = []
                used_b: list[Mapping[str, Any]] = []
                for seed in common_seeds:
                    result_a = by_key[(protocol_fingerprint, selection, model_a, fraction, seed)]
                    result_b = by_key[(protocol_fingerprint, selection, model_b, fraction, seed)]
                    value_a = result_a["test_metrics"].get(metric_name)
                    value_b = result_b["test_metrics"].get(metric_name)
                    if value_a is None or value_b is None:
                        continue
                    if not (np.isfinite(float(value_a)) and np.isfinite(float(value_b))):
                        continue
                    deltas.append(float(value_b) - float(value_a))
                    used_seeds.append(seed)
                    used_a.append(result_a)
                    used_b.append(result_b)
                if not deltas:
                    continue
                delta_mean = float(np.mean(deltas))
                improvement = (
                    -delta_mean if metric_name in LOWER_IS_BETTER else delta_mean
                )
                rows.append(
                    {
                        "protocol_fingerprint": protocol_fingerprint,
                        "checkpoint_selection": selection,
                        "selection_objective": used_a[0]["selection_objective"],
                        "global_sign_in_selection_objective": used_a[0]["global_sign_in_selection_objective"],
                        "global_sign_metrics_note": used_a[0]["global_sign_metrics_note"],
                        "model_a": model_a,
                        "model_b": model_b,
                        "g05_fraction": fraction,
                        "g05_count_per_sample": used_a[0]["configuration"]["observation"][
                            "g05_count_per_sample"
                        ],
                        "metric": metric_name,
                        "metric_role": (
                            "primary" if metric_name in used_a[0]["primary_metrics"]
                            else "secondary"
                        ),
                        "primary_research_metric": (
                            selection == "structure"
                            and metric_name in {"mean_position_mae", "mean_position_3d_error"}
                        ),
                        "paired_seed_count": len(deltas),
                        "paired_seeds": ",".join(map(str, used_seeds)),
                        **{f"{key}_a": value for key, value in selected_metadata_by_seed(used_a).items()},
                        **{f"{key}_b": value for key, value in selected_metadata_by_seed(used_b).items()},
                        "delta_b_minus_a_mean": delta_mean,
                        "delta_b_minus_a_std": sample_std(deltas),
                        "improvement_b_over_a_mean": improvement,
                        "improvement_b_over_a_by_seed": canonical_json({
                            str(seed): -delta if metric_name in LOWER_IS_BETTER else delta
                            for seed, delta in zip(used_seeds, deltas)
                        }),
                        "improvement_definition": (
                            "positive means model_b is better; sign is reversed for "
                            "error/loss metrics"
                        ),
                    }
                )
    return rows


def load_completed_results(
    experiment_results_dir: Path,
    protocol_fingerprint: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    runs_dir = experiment_results_dir / "runs"
    if not runs_dir.exists():
        return results
    for result_path in runs_dir.glob("*/result.json"):
        try:
            with result_path.open("r", encoding="utf-8") as handle:
                result = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            print(f"WARNING: reports unchanged; unreadable result {result_path}: {error}")
            raise
        if (
            result.get("status") == "completed"
            and result.get("configuration", {}).get("protocol_fingerprint")
            == protocol_fingerprint
        ):
            results.extend(completed_result_evaluations(result))
    return results


def refresh_reports(
    experiment_results_dir: Path,
    protocol_fingerprint: str,
) -> None:
    try:
        results = load_completed_results(experiment_results_dir, protocol_fingerprint)
    except (OSError, json.JSONDecodeError):
        return
    run_rows = sorted(
        (result_to_row(result) for result in results),
        key=lambda row: (row["model"], row["g05_fraction"], row["seed"], row["checkpoint_selection"]),
    )
    # Build all tables before publishing any; empty tables clear existing rows.
    reports = (
        ("runs.csv", run_rows),
        ("summary.csv", build_summary_rows(results)),
        ("pairwise_comparisons.csv", build_pairwise_rows(results)),
    )
    for name, rows in reports:
        atomic_write_csv(experiment_results_dir / name, rows)


def state_dicts_are_identical(first: nn.Module, second: nn.Module) -> bool:
    first_state = first.state_dict()
    second_state = second.state_dict()
    return first_state.keys() == second_state.keys() and all(
        torch.equal(first_state[name], second_state[name]) for name in first_state
    )


def has_nonzero_gradient(model: nn.Module, prefixes: tuple[str, ...]) -> bool:
    return any(
        name.startswith(prefixes)
        and parameter.grad is not None
        and torch.any(parameter.grad != 0).item()
        for name, parameter in model.named_parameters()
    )


def validate_model_output(
    output: ModelOutput,
    sample_count: int,
    model_name: str,
) -> None:
    expected = {
        "position": (sample_count, 6),
        "magnitude": (sample_count, 2),
        "relative_sign_logit": (sample_count,),
        "global_sign_logit": (sample_count,),
    }
    for name, shape in expected.items():
        tensor = getattr(output, name, None)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            actual = None if tensor is None else tuple(tensor.shape)
            raise RuntimeError(
                f"Model {model_name!r} returned invalid {name}: {actual} != {shape}"
            )


def run_checkpoint_smoke_tests(spec: ModelSpec, candidate_count: int) -> None:
    """Exercise the real save/resume helpers without running training epochs."""

    settings = TrainingSettings(
        batch_size=4,
        max_epochs=3,
        learning_rate=physics.LEARNING_RATE,
        weight_decay=physics.WEIGHT_DECAY,
        loss_weights=LossWeights(),
    )
    config = run_configuration(
        protocol_fingerprint="smoke-protocol",
        code_sha256="smoke-code",
        spec=spec,
        fraction=0.75,
        g05_count=physics.g05_count_for_fraction(0.75, candidate_count),
        candidate_count=candidate_count,
        seed=93,
        settings=settings,
    )
    set_reproducibility(93)
    model = spec.factory().cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay,
    )
    shuffle_generator = torch.Generator().manual_seed(93)
    torch.randperm(8, generator=shuffle_generator)
    best_checkpoints: dict[str, dict[str, Any]] = {}
    history: list[dict[str, Any]] = []
    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    # Total wins at epoch 1; structure wins at epoch 2. Epoch 3 tests ties.
    for epoch, (structure, global_sign) in enumerate(((2.0, 0.1), (1.0, 2.0), (1.0, 2.0)), 1):
        with torch.no_grad():
            next(model.parameters()).add_(0.01)
        loss = EpochLoss(structure + global_sign, structure, structure, 0.0, 0.0, global_sign)
        snapshots[epoch] = copy_model_state(model)
        history.append({"epoch": epoch, "train": epoch_loss_dict(loss), "validation": epoch_loss_dict(loss)})
        update_best_checkpoints(
            best_checkpoints, run_config=config, epoch=epoch,
            validation_loss=loss, model_state=snapshots[epoch],
        )
    resume = make_resume_checkpoint(
        run_config=config, epoch=3, model_state=snapshots[3], optimizer=optimizer,
        shuffle_generator=shuffle_generator, best_checkpoints=best_checkpoints,
        history=history, elapsed_seconds=0.0,
    )
    validate_resume_checkpoint(resume, config)
    if (resume["best_total_epoch"], resume["best_structure_epoch"]) != (1, 2):
        raise RuntimeError("Independent checkpoint selection/tie handling failed")
    with tempfile.TemporaryDirectory(prefix="model-experiment-checkpoints-") as directory:
        paths = run_checkpoint_paths(Path(directory))
        if len(set(paths.values())) != 3:
            raise RuntimeError("Latest/total/structure checkpoint paths collide")
        atomic_torch_save(resume, paths["latest"])
        loaded = load_torch_checkpoint(paths["latest"], torch.device("cpu"))
        validate_resume_checkpoint(loaded, config)
        if not torch.equal(loaded["shuffle_generator_state"], shuffle_generator.get_state()):
            raise RuntimeError("Shuffle generator did not survive checkpoint roundtrip")
        # Simulate a stop after latest committed but before either best file exists.
        for selection in CHECKPOINT_SELECTIONS:
            best = loaded["best_checkpoints"][selection]
            atomic_torch_save(best, paths[selection])
            restored = load_torch_checkpoint(paths[selection], torch.device("cpu"))
            validate_selected_checkpoint(
                restored, run_config=config, selection=selection,
                expected_epoch=best["selected_epoch"], expected_loss=best["selected_validation_loss"],
            )
            expected_state = snapshots[best["selected_epoch"]]
            if not all(
                torch.equal(value, restored["model_state_dict"][name])
                for name, value in expected_state.items()
            ):
                raise RuntimeError(f"{selection} checkpoint does not contain its full epoch state")
    print("  distinct checkpoint paths, independent selection, and complete resume state: OK")


def run_smoke_tests(
    *,
    arrays: physics.DatasetArrays,
    split: physics.DataSplit,
    stats: physics.NormalizationStats,
    fractions: Sequence[float],
    model_names: Sequence[str],
    weights: LossWeights,
    device: torch.device,
) -> None:
    print("\nRunning model-registry, forward, loss, and routing smoke tests...")
    smoke_indices = split.train[: min(8, len(split.train))]
    fractions_to_test = sorted(set((0.0, *fractions)))
    positive_fraction = next((value for value in fractions_to_test if value > 0), None)
    if positive_fraction is None:
        positive_fraction = 1.0
        fractions_to_test.append(positive_fraction)

    for fraction in fractions_to_test:
        dataset = physics.prepare_dataset(arrays, smoke_indices, stats, fraction)
        sample_count = min(4, len(dataset))
        tensors = [tensor[:sample_count].to(device) for tensor in dataset.tensors]
        g00, g05, g05_mask, position_target, charge_target = tensors
        for model_name in model_names:
            set_reproducibility(41)
            model = MODEL_REGISTRY[model_name].factory().to(device)
            output = model(g00, g05, g05_mask)
            validate_model_output(output, sample_count, model_name)
            losses = calculate_losses(
                output,
                position_target,
                charge_target,
                g05_mask,
                weights,
            )
            loss_tensors = [
                losses.total,
                losses.structure,
                losses.position,
                losses.magnitude,
                losses.relative_sign,
            ]
            if losses.global_sign is not None:
                loss_tensors.append(losses.global_sign)
            if not all(torch.isfinite(loss).item() for loss in loss_tensors):
                raise RuntimeError(
                    f"Non-finite smoke-test loss: model={model_name}, fraction={fraction}"
                )
            if fraction == 0.0 and losses.global_sign is not None:
                raise RuntimeError("G05=0 produced an invalid global-sign loss")
            if fraction > 0.0 and losses.global_sign is None:
                raise RuntimeError("Observed G05 did not produce global-sign supervision")
        print(f"  forward/loss fraction={fraction:.2f}: OK")

    default_pair = set(DEFAULT_MODELS)
    if default_pair.issubset(model_names):
        set_reproducibility(91)
        restricted = MODEL_REGISTRY["g05_sign_only"].factory().to(device)
        set_reproducibility(91)
        full = MODEL_REGISTRY["g05_full_reconstruction"].factory().to(device)
        if parameter_counts(restricted) != parameter_counts(full):
            raise RuntimeError("Default routing pair is not capacity matched")
        if not state_dicts_are_identical(restricted, full):
            raise RuntimeError("Default routing pair does not share initial state")

        zero_dataset = physics.prepare_dataset(arrays, smoke_indices, stats, 0.0)
        zero_tensors = [tensor[:4].to(device) for tensor in zero_dataset.tensors]
        with torch.no_grad():
            restricted_zero = restricted(*zero_tensors[:3])
            full_zero = full(*zero_tensors[:3])
        for output_name in (
            "position",
            "magnitude",
            "relative_sign_logit",
            "global_sign_logit",
        ):
            if not torch.equal(
                getattr(restricted_zero, output_name),
                getattr(full_zero, output_name),
            ):
                raise RuntimeError(
                    f"Default pair differs without G05 in output {output_name}"
                )

        positive_dataset = physics.prepare_dataset(
            arrays,
            smoke_indices,
            stats,
            positive_fraction,
        )
        positive_tensors = [tensor[:4].to(device) for tensor in positive_dataset.tensors]
        for model_name, should_reach_g05 in (
            ("g05_sign_only", False),
            ("g05_full_reconstruction", True),
        ):
            set_reproducibility(92)
            model = MODEL_REGISTRY[model_name].factory().to(device)
            model.zero_grad(set_to_none=True)
            output = model(*positive_tensors[:3])
            structural_objective = (
                output.position.sum()
                + output.magnitude.sum()
                + output.relative_sign_logit.sum()
            )
            structural_objective.backward()
            reaches_g05 = has_nonzero_gradient(
                model,
                ("g05_encoder.", "structure_context."),
            )
            if reaches_g05 != should_reach_g05:
                raise RuntimeError(
                    f"Unexpected structural G05 gradient route in {model_name}: "
                    f"{reaches_g05}"
                )
            # The real final projection starts at zero: on the first backward,
            # only that projection can receive structural G05-route gradients.
            # Warm up this disposable model once, then check the encoder AND
            # observed G05 input values using the actual structure loss.
            model.zero_grad(set_to_none=True)
            warmup_output = model(*positive_tensors[:3])
            calculate_losses(
                warmup_output, positive_tensors[3], positive_tensors[4],
                positive_tensors[2], LossWeights(),
            ).structure.backward()
            torch.optim.SGD(model.parameters(), lr=0.01).step()
            model.zero_grad(set_to_none=True)
            input_g05 = positive_tensors[1].detach().clone().requires_grad_(True)
            output = model(positive_tensors[0], input_g05, positive_tensors[2])
            calculate_losses(
                output, positive_tensors[3], positive_tensors[4],
                positive_tensors[2], LossWeights(),
            ).structure.backward()
            reaches_input = (
                input_g05.grad is not None
                and bool(torch.any(input_g05.grad[:, :, 2] != 0).item())
            )
            reaches_encoder = has_nonzero_gradient(model, ("g05_encoder.",))
            if reaches_input != should_reach_g05 or reaches_encoder != should_reach_g05:
                raise RuntimeError(f"Unexpected structure-loss G05 input/encoder gradient in {model_name}")
            if has_nonzero_gradient(model, ("global_sign_head.",)):
                raise RuntimeError("Structure loss reached the global-sign head")
            if input_g05.grad is not None:
                masked_gradient = input_g05.grad * (1.0 - positive_tensors[2])
                if torch.any(masked_gradient != 0).item():
                    raise RuntimeError("Structure loss reached an unobserved G05 point")
        print("  matched initialization/capacity and G05 routing: OK")
    run_checkpoint_smoke_tests(MODEL_REGISTRY[model_names[0]], arrays.g05.shape[1])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def normalization_payload(stats: physics.NormalizationStats) -> dict[str, Any]:
    return {
        "g00_mean": stats.g00_mean,
        "g00_std": stats.g00_std,
        "g05_value_scale": stats.g05_value_scale,
        "position_mean": stats.position_mean,
        "position_std": stats.position_std,
        "charge_scale": stats.charge_scale,
        "fit_scope": "training split only; shared by all models and G05 fractions",
    }


def build_protocol(
    *,
    data_path: Path,
    data_sha256: str,
    arrays: physics.DatasetArrays,
    split: physics.DataSplit,
    split_seed: int,
    stats: physics.NormalizationStats,
    settings: TrainingSettings,
    device: torch.device,
    code_sha256: str,
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "code": {
            "model_experiment_sha256": code_sha256,
            "physics_protocol_path": str(Path(physics.__file__).resolve()),
            "physics_protocol_sha256": file_sha256(Path(physics.__file__).resolve()),
        },
        "dataset": {
            "path": str(data_path.resolve()),
            "sha256": data_sha256,
            "sample_count": arrays.g00.shape[0],
            "g00_shape": arrays.g00.shape,
            "g05_shape": arrays.g05.shape,
            "target_shape": arrays.target.shape,
            "target_fields": physics.TARGET_FIELDS,
            "g05_fields": physics.G05_FIELDS,
            "target_ordering": "lexicographic x, then y, then z",
            "global_sign_anchor": "q1 after deterministic target ordering",
            "physical_equations": "G00=V^2 and G05=V",
        },
        "data_split": {
            "seed": split_seed,
            "train_count": len(split.train),
            "validation_count": len(split.validation),
            "test_count": len(split.test),
            "train_indices_sha256": hashlib.sha256(split.train.tobytes()).hexdigest(),
            "validation_indices_sha256": hashlib.sha256(
                split.validation.tobytes()
            ).hexdigest(),
            "test_indices_sha256": hashlib.sha256(split.test.tobytes()).hexdigest(),
        },
        "normalization": normalization_payload(stats),
        "training": {
            "batch_size": settings.batch_size,
            "max_epochs": settings.max_epochs,
            "optimizer": "AdamW",
            "learning_rate": settings.learning_rate,
            "weight_decay": settings.weight_decay,
            "loss_weights": asdict(settings.loss_weights),
            "model_selection": {
                selection: selection_policy(
                    selection, global_sign_weight=settings.loss_weights.global_sign,
                )
                for selection in CHECKPOINT_SELECTIONS
            },
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "resume_authority": "atomic latest.pt includes both complete best snapshots",
            "shuffle": "seeded per run; generator state is checkpointed",
        },
        "evaluation": {
            "test_set_used_only_after_validation_selection": True,
            "test_evaluations_per_run": len(CHECKPOINT_SELECTIONS),
            "same_test_dataset_for_both_selections": True,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "report_grouping": "protocol, checkpoint_selection, model, G05 fraction; pair equal seeds",
            "structure_primary_metrics": STRUCTURE_METRIC_NAMES,
            "g05_zero_global_sign_metrics": "N/A",
            "unobserved_charge_mae": "oracle aligned over the global +/- symmetry",
            "multi_seed_std": "sample standard deviation (ddof=1)",
            "metrics": METRIC_NAMES,
        },
        "reproducibility": {
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else platform.processor()
            ),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "deterministic_algorithms": "enabled with warn_only=True",
            "cuda_limitation": (
                "warn_only permits unsupported CUDA kernels; exact bitwise "
                "reproducibility can still depend on GPU, driver, and PyTorch"
            ),
        },
        "fairness_constraints": {
            "same_data_split_normalization_training_loss_and_evaluation": True,
            "same_seed_means_same_initial_state_and_batch_order": True,
            "default_pair_same_parameter_count": True,
            "default_pair_only_route_difference": "G05 access to structural outputs",
            "cross_epoch_component_composition": False,
            "global_sign_excluded_from_structure_checkpoint_selection": True,
            "training_objective_remains_total_loss": True,
        },
    }


def initialize_experiment_artifacts(
    *,
    experiment_results_dir: Path,
    experiment_checkpoint_dir: Path,
    protocol: dict[str, Any],
    split: physics.DataSplit,
    stats: physics.NormalizationStats,
    selected_specs: Sequence[ModelSpec],
) -> str:
    protocol_fingerprint = object_fingerprint(protocol)
    protocol_document = {
        "protocol_fingerprint": protocol_fingerprint,
        "created_or_verified_at": utc_now(),
        "protocol": protocol,
        "registered_models": [
            {
                "name": spec.name,
                "description": spec.description,
                "input_policy": spec.input_policy,
            }
            for spec in selected_specs
        ],
    }
    protocol_path = experiment_results_dir / "protocol.json"
    if protocol_path.exists():
        with protocol_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("protocol_fingerprint") != protocol_fingerprint:
            raise RuntimeError(
                "This experiment name already contains a different common "
                f"protocol: {protocol_path}. Use a new --experiment-name so "
                "incompatible runs are never mixed."
            )
    else:
        atomic_write_json(protocol_path, protocol_document)

    experiment_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    split_path = experiment_results_dir / "split_indices.npz"
    if not split_path.exists():
        atomic_save_npz(
            split_path,
            train=split.train,
            validation=split.validation,
            test=split.test,
        )
    normalization_path = experiment_results_dir / "normalization.json"
    if not normalization_path.exists():
        atomic_write_json(normalization_path, normalization_payload(stats))
    return protocol_fingerprint


def mark_run_failure(
    *,
    experiment_results_dir: Path,
    run_config: Mapping[str, Any],
    status: str,
    error: BaseException,
) -> None:
    run_id = run_id_for(run_config)
    status_path = experiment_results_dir / "runs" / run_id / "status.json"
    save_status(
        status_path,
        status=status,
        run_id=run_id,
        error_type=type(error).__name__,
        error_message=str(error),
        traceback="".join(traceback.format_exception(error)),
        recovery=(
            "Re-run the same command. The latest complete epoch checkpoint "
            "will be resumed and completed results will be skipped."
        ),
    )


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(requested)


def parse_csv_values(value: str, converter: Callable[[str], Any]) -> tuple[Any, ...]:
    return tuple(converter(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run fair, resumable multi-model/multi-G05/multi-seed experiments."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
    )
    parser.add_argument(
        "--models",
        type=lambda value: parse_csv_values(value, str),
        default=DEFAULT_MODELS,
        help="Comma-separated registered model names",
    )
    parser.add_argument(
        "--fractions",
        type=lambda value: parse_csv_values(value, float),
        default=physics.G05_FRACTIONS,
        help="Comma-separated nested G05 fractions",
    )
    parser.add_argument(
        "--seeds",
        type=lambda value: parse_csv_values(value, int),
        default=physics.EXPERIMENT_SEEDS,
        help="Comma-separated model/shuffle seeds",
    )
    parser.add_argument("--split-seed", type=int, default=physics.DATA_SPLIT_SEED)
    parser.add_argument("--epochs", type=int, default=physics.MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=physics.BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=physics.LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=physics.WEIGHT_DECAY)
    parser.add_argument(
        "--position-loss-weight",
        type=float,
        default=physics.POSITION_LOSS_WEIGHT,
    )
    parser.add_argument(
        "--magnitude-loss-weight",
        type=float,
        default=physics.MAGNITUDE_LOSS_WEIGHT,
    )
    parser.add_argument(
        "--relative-sign-loss-weight",
        type=float,
        default=physics.RELATIVE_SIGN_LOSS_WEIGHT,
    )
    parser.add_argument(
        "--global-sign-loss-weight",
        type=float,
        default=physics.GLOBAL_SIGN_LOSS_WEIGHT,
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record a failed run and continue with the remaining matrix",
    )
    return parser.parse_args()


def validate_cli(args: argparse.Namespace) -> None:
    if not args.experiment_name or args.experiment_name in {".", ".."}:
        raise ValueError("--experiment-name must be a non-empty directory name")
    if Path(args.experiment_name).name != args.experiment_name:
        raise ValueError("--experiment-name must not contain path separators")
    if not args.models or not args.fractions or not args.seeds:
        raise ValueError("At least one model, fraction, and seed are required")
    unknown_models = set(args.models).difference(MODEL_REGISTRY)
    if unknown_models:
        raise ValueError(
            f"Unknown models {sorted(unknown_models)}; available={sorted(MODEL_REGISTRY)}"
        )
    if len(set(args.models)) != len(args.models):
        raise ValueError("--models contains duplicates")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds contains duplicates")
    if len(set(args.fractions)) != len(args.fractions):
        raise ValueError("--fractions contains duplicates")
    if tuple(sorted(args.fractions)) != tuple(args.fractions):
        raise ValueError("--fractions must be sorted in ascending order")
    if any(not 0.0 <= fraction <= 1.0 for fraction in args.fractions):
        raise ValueError("Every G05 fraction must be between 0 and 1")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive")
    numeric_nonnegative = (
        args.learning_rate,
        args.weight_decay,
        args.position_loss_weight,
        args.magnitude_loss_weight,
        args.relative_sign_loss_weight,
        args.global_sign_loss_weight,
    )
    if any(not np.isfinite(value) or value < 0 for value in numeric_nonnegative):
        raise ValueError("Learning rate, weight decay, and loss weights must be finite >= 0")
    if args.learning_rate == 0:
        raise ValueError("--learning-rate must be greater than zero")


def print_registered_models() -> None:
    print("Registered models:")
    for name, spec in MODEL_REGISTRY.items():
        print(f"  {name}: {spec.description}")
        for output_name, inputs in spec.input_policy.items():
            print(f"    {output_name}: {inputs}")


def main() -> None:
    args = parse_args()
    if args.list_models:
        print_registered_models()
        return
    validate_cli(args)
    device = resolve_device(args.device)
    settings = TrainingSettings(
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        loss_weights=LossWeights(
            position=args.position_loss_weight,
            magnitude=args.magnitude_loss_weight,
            relative_sign=args.relative_sign_loss_weight,
            global_sign=args.global_sign_loss_weight,
        ),
    )
    selected_specs = [MODEL_REGISTRY[name] for name in args.models]
    data_path = args.data.resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    print("Device:", device)
    print("Data:", data_path)
    print("Models:", ", ".join(args.models))
    print("G05 fractions:", tuple(args.fractions))
    print("Seeds:", tuple(args.seeds))
    arrays = physics.load_dataset(data_path)
    split = physics.create_data_split(arrays.g00.shape[0], args.split_seed)
    stats = physics.calculate_normalization_stats(arrays, split.train)
    run_smoke_tests(
        arrays=arrays,
        split=split,
        stats=stats,
        fractions=args.fractions,
        model_names=args.models,
        weights=settings.loss_weights,
        device=device,
    )
    if args.smoke_only:
        print("Smoke-only run complete")
        return

    code_sha256 = file_sha256(Path(__file__).resolve())
    data_sha256 = file_sha256(data_path)
    protocol = build_protocol(
        data_path=data_path,
        data_sha256=data_sha256,
        arrays=arrays,
        split=split,
        split_seed=args.split_seed,
        stats=stats,
        settings=settings,
        device=device,
        code_sha256=code_sha256,
    )
    experiment_results_dir = args.results_root.resolve() / args.experiment_name
    experiment_checkpoint_dir = (
        args.checkpoint_root.resolve() / args.experiment_name
    )
    protocol_fingerprint = initialize_experiment_artifacts(
        experiment_results_dir=experiment_results_dir,
        experiment_checkpoint_dir=experiment_checkpoint_dir,
        protocol=protocol,
        split=split,
        stats=stats,
        selected_specs=selected_specs,
    )
    refresh_reports(experiment_results_dir, protocol_fingerprint)

    candidate_count = arrays.g05.shape[1]
    planned_runs = len(args.fractions) * len(args.seeds) * len(args.models)
    print(f"\nExperiment matrix: {planned_runs} runs")
    print("Results:", experiment_results_dir)
    print("Checkpoints:", experiment_checkpoint_dir)

    completed_now = 0
    skipped = 0
    failed = 0
    for fraction in args.fractions:
        g05_count = physics.g05_count_for_fraction(fraction, candidate_count)
        print(
            f"\nPreparing common datasets: G05 fraction={fraction:.2f}, "
            f"points={g05_count}/{candidate_count}"
        )
        train_dataset = physics.prepare_dataset(arrays, split.train, stats, fraction)
        validation_dataset = physics.prepare_dataset(
            arrays,
            split.validation,
            stats,
            fraction,
        )
        test_dataset = physics.prepare_dataset(arrays, split.test, stats, fraction)
        for seed in args.seeds:
            for spec in selected_specs:
                run_config = run_configuration(
                    protocol_fingerprint=protocol_fingerprint,
                    code_sha256=code_sha256,
                    spec=spec,
                    fraction=fraction,
                    g05_count=g05_count,
                    candidate_count=candidate_count,
                    seed=seed,
                    settings=settings,
                )
                try:
                    _, was_skipped = train_and_evaluate_run(
                        spec=spec,
                        train_dataset=train_dataset,
                        validation_dataset=validation_dataset,
                        test_dataset=test_dataset,
                        stats=stats,
                        run_config=run_config,
                        experiment_results_dir=experiment_results_dir,
                        experiment_checkpoint_dir=experiment_checkpoint_dir,
                        settings=settings,
                        device=device,
                    )
                except KeyboardInterrupt as error:
                    mark_run_failure(
                        experiment_results_dir=experiment_results_dir,
                        run_config=run_config,
                        status="interrupted",
                        error=error,
                    )
                    refresh_reports(experiment_results_dir, protocol_fingerprint)
                    print("\nInterrupted. Completed results and latest epoch are preserved.")
                    raise
                except Exception as error:
                    failed += 1
                    mark_run_failure(
                        experiment_results_dir=experiment_results_dir,
                        run_config=run_config,
                        status="failed",
                        error=error,
                    )
                    refresh_reports(experiment_results_dir, protocol_fingerprint)
                    if not args.continue_on_error:
                        raise
                    print(
                        f"ERROR recorded for {run_id_for(run_config)}: "
                        f"{type(error).__name__}: {error}"
                    )
                    continue
                if was_skipped:
                    skipped += 1
                else:
                    completed_now += 1
                # Reports are durable after every individual run.
                refresh_reports(experiment_results_dir, protocol_fingerprint)
        del train_dataset, validation_dataset, test_dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nExperiment matrix complete")
    print(
        f"Completed now={completed_now}, already complete={skipped}, failed={failed}"
    )
    print("Per-run table:", experiment_results_dir / "runs.csv")
    print("Multi-seed summary:", experiment_results_dir / "summary.csv")
    pairwise_path = experiment_results_dir / "pairwise_comparisons.csv"
    if pairwise_path.exists():
        print("Paired model comparison:", pairwise_path)


if __name__ == "__main__":
    main()

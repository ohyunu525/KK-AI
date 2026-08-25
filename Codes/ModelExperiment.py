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
        "ModelExperiment.py must remain beside Codes/NewLearning8.py so the "
        "validated project physics protocol can be imported."
    ) from error


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = physics.DEFAULT_DATA_PATH
DEFAULT_RESULTS_ROOT = PROJECT_DIR / "Results" / "model_experiments"
DEFAULT_CHECKPOINT_ROOT = PROJECT_DIR / "Models" / "model_experiments"
DEFAULT_EXPERIMENT_NAME = "g05_routing_comparison_v1"
DEFAULT_MODELS = ("g05_sign_only", "g05_full_reconstruction")
PROTOCOL_VERSION = "model-experiment-v1"
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
    if not rows:
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
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
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
            "checkpoint_selection": (
                "single epoch minimizing validation total loss for every model"
            ),
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
    latest_path = checkpoint_run_dir / "latest.pt"
    best_path = checkpoint_run_dir / "best.pt"

    if result_path.exists():
        with result_path.open("r", encoding="utf-8") as handle:
            existing_result = json.load(handle)
        if existing_result.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError(f"Result fingerprint mismatch: {result_path}")
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
    best_validation_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []
    elapsed_before_resume = 0.0
    resumed = False
    if latest_path.exists():
        checkpoint = load_torch_checkpoint(latest_path, device)
        if checkpoint.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError(f"Latest checkpoint fingerprint mismatch: {latest_path}")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        optimizer_to_device(optimizer, device)
        shuffle_generator.set_state(checkpoint["shuffle_generator_state"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_loss = float(checkpoint["best_validation_loss"])
        best_epoch = int(checkpoint["best_epoch"])
        history = list(checkpoint.get("history", []))
        elapsed_before_resume = float(checkpoint.get("elapsed_seconds", 0.0))
        resumed = True

    save_status(
        status_path,
        status="running",
        run_id=run_id,
        resumed=resumed,
        next_epoch=start_epoch,
        best_epoch=best_epoch or None,
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
        if validation_loss.total < best_validation_loss:
            best_validation_loss = validation_loss.total
            best_epoch = epoch
            atomic_torch_save(
                {
                    "run_fingerprint": run_fingerprint,
                    "epoch": epoch,
                    "validation_loss": validation_loss.total,
                    "model_state_dict": {
                        name: value.detach().cpu()
                        for name, value in model.state_dict().items()
                    },
                },
                best_path,
            )

        elapsed_seconds = elapsed_before_resume + (time.perf_counter() - started_at)
        atomic_torch_save(
            {
                "run_fingerprint": run_fingerprint,
                "epoch": epoch,
                "model_state_dict": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "shuffle_generator_state": shuffle_generator.get_state(),
                "rng_state": capture_rng_state(),
                "best_validation_loss": best_validation_loss,
                "best_epoch": best_epoch,
                "elapsed_seconds": elapsed_seconds,
                "history": history,
            },
            latest_path,
        )
        atomic_write_json(history_path, history)
        save_status(
            status_path,
            status="running",
            run_id=run_id,
            resumed=resumed,
            next_epoch=epoch + 1,
            best_epoch=best_epoch,
            best_validation_loss=best_validation_loss,
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
            f"val_global={global_text} best={best_validation_loss:.6f}@{best_epoch}"
        )

    if not best_path.exists():
        raise RuntimeError(f"No valid best checkpoint was produced: {run_id}")
    best_checkpoint = load_torch_checkpoint(best_path, device)
    if best_checkpoint.get("run_fingerprint") != run_fingerprint:
        raise RuntimeError(f"Best checkpoint fingerprint mismatch: {best_path}")
    model.load_state_dict(best_checkpoint["model_state_dict"])
    metrics = evaluate_model(
        model,
        test_dataset,
        stats,
        batch_size=settings.batch_size,
        device=device,
    )
    elapsed_seconds = elapsed_before_resume + (time.perf_counter() - started_at)
    result = {
        "run_id": run_id,
        "run_fingerprint": run_fingerprint,
        "status": "completed",
        "completed_at": utc_now(),
        "configuration": run_config,
        "parameter_count": parameter_counts(model),
        "training_result": {
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "epochs_completed": settings.max_epochs,
            "elapsed_seconds": elapsed_seconds,
            "resumed": resumed,
            "selection_note": (
                "One complete model state from the epoch with minimum validation "
                "total loss; no cross-epoch component composition."
            ),
        },
        "test_metrics": metrics,
        "artifacts": {
            "config": str(config_path.resolve()),
            "history": str(history_path.resolve()),
            "latest_checkpoint": str(latest_path.resolve()),
            "best_checkpoint": str(best_path.resolve()),
        },
    }
    atomic_write_json(result_path, result)
    save_status(
        status_path,
        status="completed",
        run_id=run_id,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        result_path=str(result_path.resolve()),
    )
    print(
        f"DONE {run_id} | best={best_validation_loss:.6f}@{best_epoch} | "
        f"position_mae={metrics['mean_position_mae']:.6f} | "
        f"magnitude_mae={metrics['charge_magnitude_mae']:.6f}"
    )
    return result, False


def result_to_row(result: Mapping[str, Any]) -> dict[str, Any]:
    config = result["configuration"]
    model = config["model"]
    observation = config["observation"]
    training = config["training"]
    training_result = result["training_result"]
    metrics = result["test_metrics"]
    return {
        "run_id": result["run_id"],
        "model": model["name"],
        "model_description": model["description"],
        "input_policy": canonical_json(model["input_policy"]),
        "g05_fraction": observation["g05_fraction"],
        "g05_count_per_sample": observation["g05_count_per_sample"],
        "seed": training["seed"],
        "parameter_count": result["parameter_count"]["total"],
        **{name: metrics.get(name) for name in METRIC_NAMES},
        "position_mae_x1": metrics["position_mae"][0],
        "position_mae_y1": metrics["position_mae"][1],
        "position_mae_z1": metrics["position_mae"][2],
        "position_mae_x2": metrics["position_mae"][3],
        "position_mae_y2": metrics["position_mae"][4],
        "position_mae_z2": metrics["position_mae"][5],
        "observed_sample_fraction": metrics["observed_sample_fraction"],
        "observations_per_sample": metrics["observations_per_sample"],
        "best_validation_loss": training_result["best_validation_loss"],
        "best_epoch": training_result["best_epoch"],
        "elapsed_seconds": training_result["elapsed_seconds"],
        "best_checkpoint": result["artifacts"]["best_checkpoint"],
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


def build_summary_rows(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[Mapping[str, Any]]] = {}
    for result in results:
        config = result["configuration"]
        key = (
            str(config["model"]["name"]),
            float(config["observation"]["g05_fraction"]),
        )
        grouped.setdefault(key, []).append(result)

    rows: list[dict[str, Any]] = []
    for (model_name, fraction), group in sorted(grouped.items()):
        group = sorted(group, key=lambda item: int(item["configuration"]["training"]["seed"]))
        first = group[0]
        row: dict[str, Any] = {
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
    by_key: dict[tuple[str, float, int], Mapping[str, Any]] = {}
    discovered_model_names: set[str] = set()
    fractions: set[float] = set()
    for result in results:
        config = result["configuration"]
        model_name = str(config["model"]["name"])
        fraction = float(config["observation"]["g05_fraction"])
        seed = int(config["training"]["seed"])
        by_key[(model_name, fraction, seed)] = result
        discovered_model_names.add(model_name)
        fractions.add(fraction)

    default_order = {name: index for index, name in enumerate(DEFAULT_MODELS)}
    model_names = sorted(
        discovered_model_names,
        key=lambda name: (default_order.get(name, len(default_order)), name),
    )

    rows: list[dict[str, Any]] = []
    for model_a, model_b in itertools.combinations(model_names, 2):
        for fraction in sorted(fractions):
            seeds_a = {
                seed
                for name, current_fraction, seed in by_key
                if name == model_a and current_fraction == fraction
            }
            seeds_b = {
                seed
                for name, current_fraction, seed in by_key
                if name == model_b and current_fraction == fraction
            }
            common_seeds = sorted(seeds_a.intersection(seeds_b))
            for metric_name in METRIC_NAMES:
                deltas: list[float] = []
                used_seeds: list[int] = []
                for seed in common_seeds:
                    value_a = by_key[(model_a, fraction, seed)]["test_metrics"].get(
                        metric_name
                    )
                    value_b = by_key[(model_b, fraction, seed)]["test_metrics"].get(
                        metric_name
                    )
                    if value_a is None or value_b is None:
                        continue
                    if not (np.isfinite(float(value_a)) and np.isfinite(float(value_b))):
                        continue
                    deltas.append(float(value_b) - float(value_a))
                    used_seeds.append(seed)
                if not deltas:
                    continue
                delta_mean = float(np.mean(deltas))
                improvement = (
                    -delta_mean if metric_name in LOWER_IS_BETTER else delta_mean
                )
                rows.append(
                    {
                        "model_a": model_a,
                        "model_b": model_b,
                        "g05_fraction": fraction,
                        "metric": metric_name,
                        "paired_seed_count": len(deltas),
                        "paired_seeds": ",".join(map(str, used_seeds)),
                        "delta_b_minus_a_mean": delta_mean,
                        "delta_b_minus_a_std": sample_std(deltas),
                        "improvement_b_over_a_mean": improvement,
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
            print(f"WARNING: ignoring unreadable result {result_path}: {error}")
            continue
        if (
            result.get("status") == "completed"
            and result.get("configuration", {}).get("protocol_fingerprint")
            == protocol_fingerprint
        ):
            results.append(result)
    return results


def refresh_reports(
    experiment_results_dir: Path,
    protocol_fingerprint: str,
) -> None:
    results = load_completed_results(experiment_results_dir, protocol_fingerprint)
    if not results:
        return
    run_rows = sorted(
        (result_to_row(result) for result in results),
        key=lambda row: (row["model"], row["g05_fraction"], row["seed"]),
    )
    atomic_write_csv(experiment_results_dir / "runs.csv", run_rows)
    atomic_write_csv(
        experiment_results_dir / "summary.csv",
        build_summary_rows(results),
    )
    pairwise_rows = build_pairwise_rows(results)
    if pairwise_rows:
        atomic_write_csv(experiment_results_dir / "pairwise_comparisons.csv", pairwise_rows)


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
        print("  matched initialization/capacity and G05 routing: OK")
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
            "model_selection": "minimum validation total loss at one epoch",
            "shuffle": "seeded per run; generator state is checkpointed",
        },
        "evaluation": {
            "test_set_used_once_after_validation selection": True,
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

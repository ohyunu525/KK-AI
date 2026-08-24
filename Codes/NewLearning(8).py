from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "Models"
RESULTS_DIR = PROJECT_DIR / "Results"
DEFAULT_DATA_PATH = MODELS_DIR / "charge_dataset_multipoint_v2.npz"
CHECKPOINT_DIR = MODELS_DIR / "g05_separated_branch_checkpoints_v3"
MODEL_G00_PATH = MODELS_DIR / "model_G00_structure_branch_v3.pt"
MODEL_G00_G05_PATH = MODELS_DIR / "model_G00_G05_composed_v3.pt"
NORMALIZATION_PATH = MODELS_DIR / "normalization_stats_v3.npz"
RUN_RESULTS_PATH = RESULTS_DIR / "g05_separated_branch_results_v3.csv"
SUMMARY_RESULTS_PATH = RESULTS_DIR / "g05_separated_branch_summary_v3.csv"

PLOT_PATHS = {
    "mean_position_mae": RESULTS_DIR / "g05_structure_position_mae_v3.png",
    "mean_position_3d_error": RESULTS_DIR / "g05_structure_position_3d_v3.png",
    "charge_magnitude_mae": RESULTS_DIR / "g05_structure_magnitude_mae_v3.png",
    "relative_sign_accuracy": RESULTS_DIR / "g05_structure_relative_sign_v3.png",
    "global_sign_accuracy": RESULTS_DIR / "g05_global_sign_accuracy_v3.png",
    "absolute_sign_accuracy": RESULTS_DIR / "g05_absolute_sign_accuracy_v3.png",
    "global_sign_bce": RESULTS_DIR / "g05_global_sign_bce_v3.png",
}

DATA_SPLIT_SEED = 42
EXPERIMENT_SEEDS = (41, 42, 43)
G05_FRACTIONS = (0.00, 0.10, 0.25, 0.50, 0.75, 1.00)
BATCH_SIZE = 128
MAX_EPOCHS = 300
PATIENCE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
POSITION_LOSS_WEIGHT = 1.0
MAGNITUDE_LOSS_WEIGHT = 1.0
RELATIVE_SIGN_LOSS_WEIGHT = 1.0
GLOBAL_SIGN_LOSS_WEIGHT = 1.0
EPSILON = 1e-8
POSITION_INDICES = (0, 1, 2, 4, 5, 6)
CHARGE_INDICES = (3, 7)
TARGET_FIELDS = ("x1", "y1", "z1", "q1", "x2", "y2", "z2", "q2")
G05_FIELDS = ("grid_x_index", "grid_y_index", "signed_potential")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class DatasetArrays:
    g00: np.ndarray
    g05: np.ndarray
    target: np.ndarray
    grid_x: np.ndarray
    grid_y: np.ndarray
    epsilon_0: float
    target_fields: tuple[str, ...] | None
    g05_fields: tuple[str, ...] | None


@dataclass(frozen=True)
class DataSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class NormalizationStats:
    g00_mean: float
    g00_std: float
    g05_value_scale: float
    position_mean: np.ndarray
    position_std: np.ndarray
    charge_scale: float


@dataclass(frozen=True)
class ModelOutput:
    position: torch.Tensor
    magnitude: torch.Tensor
    relative_sign_logit: torch.Tensor
    global_sign_logit: torch.Tensor


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
class TrainingResult:
    model: ChargeNet
    seed: int
    g05_fraction: float
    g05_count: int
    best_structure_loss: float
    best_structure_epoch: int
    best_position_loss: float
    best_position_epoch: int
    best_magnitude_loss: float
    best_magnitude_epoch: int
    best_global_sign_loss: float | None
    best_global_sign_epoch: int | None
    structure_checkpoint_path: Path
    position_checkpoint_path: Path
    magnitude_checkpoint_path: Path
    global_sign_checkpoint_path: Path | None
    checkpoint_path: Path


@dataclass(frozen=True)
class EvaluationResult:
    position_mae: np.ndarray
    mean_position_mae: float
    position_error_1: float
    position_error_2: float
    mean_position_3d_error: float
    charge_mae: np.ndarray
    charge_magnitude_mae: float
    relative_sign_accuracy: float
    global_sign_accuracy: float | None
    global_sign_bce: float | None
    absolute_sign_accuracy: float | None
    signed_pair_accuracy: float | None
    observed_sample_fraction: float
    observations_per_sample: float


def set_reproducibility(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _optional_string_tuple(
    archive: np.lib.npyio.NpzFile,
    key: str,
) -> tuple[str, ...] | None:
    if key not in archive:
        return None
    return tuple(str(value) for value in archive[key].tolist())


def load_dataset(path: Path) -> DatasetArrays:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Run Codes/generate_charge_dataset.py first."
        )
    with np.load(path, allow_pickle=False) as archive:
        required_keys = {"G00", "G05", "target"}
        missing_keys = required_keys.difference(archive.files)
        if missing_keys:
            raise ValueError(f"Dataset keys are missing: {sorted(missing_keys)}")
        g00 = archive["G00"].astype(np.float32)
        g05 = archive["G05"].astype(np.float32)
        target = archive["target"].astype(np.float32)
        grid_x = (
            archive["grid_x"].astype(np.float64)
            if "grid_x" in archive
            else np.linspace(-2.0, 2.0, g00.shape[2])
        )
        grid_y = (
            archive["grid_y"].astype(np.float64)
            if "grid_y" in archive
            else np.linspace(-2.0, 2.0, g00.shape[1])
        )
        epsilon_0 = (
            float(archive["epsilon_0"].item()) if "epsilon_0" in archive else 1.0
        )
        arrays = DatasetArrays(
            g00=g00,
            g05=g05,
            target=target,
            grid_x=grid_x,
            grid_y=grid_y,
            epsilon_0=epsilon_0,
            target_fields=_optional_string_tuple(archive, "target_fields"),
            g05_fields=_optional_string_tuple(archive, "g05_fields"),
        )
    validate_dataset(arrays)
    verify_physical_consistency(arrays)
    return arrays


def validate_dataset(arrays: DatasetArrays) -> None:
    g00, g05, target = arrays.g00, arrays.g05, arrays.target
    if g00.ndim != 3:
        raise ValueError(f"Invalid G00 shape: {g00.shape}")
    if g05.ndim != 3 or g05.shape[-1] != len(G05_FIELDS):
        raise ValueError(f"Invalid G05 shape: {g05.shape}")
    if target.ndim != 2 or target.shape[-1] != len(TARGET_FIELDS):
        raise ValueError(f"Invalid target shape: {target.shape}")
    if not (g00.shape[0] == g05.shape[0] == target.shape[0]):
        raise ValueError("G00, G05, and target sample counts differ")
    if g00.shape[0] < 10:
        raise ValueError("At least 10 samples are required")
    if g05.shape[1] < 10:
        raise ValueError(
            "At least 10 candidate G05 points per sample are required; regenerate "
            "the legacy (N, 1, 3) dataset."
        )
    if arrays.target_fields is not None and arrays.target_fields != TARGET_FIELDS:
        raise ValueError(
            f"Target semantics mismatch: {arrays.target_fields} != {TARGET_FIELDS}"
        )
    if arrays.g05_fields is not None and arrays.g05_fields != G05_FIELDS:
        raise ValueError(f"G05 semantics mismatch: {arrays.g05_fields} != {G05_FIELDS}")
    if not all(np.isfinite(value).all() for value in (g00, g05, target)):
        raise ValueError("Dataset contains non-finite values")
    if arrays.grid_x.shape != (g00.shape[2],) or arrays.grid_y.shape != (
        g00.shape[1],
    ):
        raise ValueError("Grid metadata does not match the G00 shape")

    x_index, y_index = g05[:, :, 0], g05[:, :, 1]
    if not np.allclose(x_index, np.rint(x_index)) or not np.allclose(
        y_index, np.rint(y_index)
    ):
        raise ValueError("G05 coordinates must be integer grid indices")
    if np.any((x_index < 0) | (x_index >= g00.shape[2])) or np.any(
        (y_index < 0) | (y_index >= g00.shape[1])
    ):
        raise ValueError("G05 grid index is out of range")
    candidate_positions = g05[0, :, 0:2]
    if not np.all(g05[:, :, 0:2] == candidate_positions[np.newaxis, :, :]):
        raise ValueError("G05 candidate locations must be fixed across samples")
    if np.unique(candidate_positions, axis=0).shape[0] != g05.shape[1]:
        raise ValueError("Candidate G05 sensor locations contain duplicates")
    x1, y1, z1 = target[:, 0], target[:, 1], target[:, 2]
    x2, y2, z2 = target[:, 4], target[:, 5], target[:, 6]
    lexicographically_misordered = (x1 > x2) | (
        (x1 == x2) & ((y1 > y2) | ((y1 == y2) & (z1 > z2)))
    )
    if np.any(lexicographically_misordered):
        raise ValueError("Targets are not deterministically ordered by x, then y, then z")
    if np.any(np.abs(target[:, CHARGE_INDICES]) <= EPSILON):
        raise ValueError("Zero charge is outside this experiment's target semantics")


def verify_physical_consistency(
    arrays: DatasetArrays,
    verification_sample_count: int = 16,
) -> None:
    """Verify target ordering and the equations G00=V^2 and G05=V."""
    sample_indices = np.linspace(
        0,
        arrays.target.shape[0] - 1,
        min(verification_sample_count, arrays.target.shape[0]),
        dtype=np.int64,
    )
    grid_x, grid_y = np.meshgrid(arrays.grid_x, arrays.grid_y)
    factor = 1.0 / (4.0 * np.pi * arrays.epsilon_0)
    for sample_index in sample_indices:
        target = arrays.target[sample_index]
        potential = np.zeros_like(grid_x, dtype=np.float64)
        for base_index in (0, 4):
            distance = np.sqrt(
                (grid_x - target[base_index]) ** 2
                + (grid_y - target[base_index + 1]) ** 2
                + target[base_index + 2] ** 2
            )
            potential += factor * target[base_index + 3] / distance
        if not np.allclose(
            potential**2,
            arrays.g00[sample_index],
            rtol=2e-4,
            atol=2e-6,
        ):
            raise ValueError("G00 is inconsistent with target=[x1,y1,z1,q1,x2,y2,z2,q2]")
        x_index = np.rint(arrays.g05[sample_index, :, 0]).astype(np.int64)
        y_index = np.rint(arrays.g05[sample_index, :, 1]).astype(np.int64)
        expected_g05 = potential[y_index, x_index]
        if not np.allclose(
            expected_g05,
            arrays.g05[sample_index, :, 2],
            rtol=2e-4,
            atol=2e-6,
        ):
            raise ValueError("G05 signed potential is inconsistent with the target")


def create_data_split(sample_count: int, seed: int) -> DataSplit:
    indices = np.random.default_rng(seed).permutation(sample_count)
    train_end = int(sample_count * 0.8)
    validation_end = int(sample_count * 0.9)
    return DataSplit(
        train=indices[:train_end],
        validation=indices[train_end:validation_end],
        test=indices[validation_end:],
    )


def calculate_normalization_stats(
    arrays: DatasetArrays,
    train_indices: np.ndarray,
) -> NormalizationStats:
    train_g00 = arrays.g00[train_indices]
    train_g05_values = arrays.g05[train_indices, :, 2]
    train_positions = arrays.target[train_indices][:, POSITION_INDICES]
    train_charges = arrays.target[train_indices][:, CHARGE_INDICES]
    return NormalizationStats(
        g00_mean=float(train_g00.mean()),
        g00_std=float(train_g00.std()) + EPSILON,
        # All fractions share one train-only physical scale.
        g05_value_scale=float(np.sqrt(np.mean(train_g05_values**2))) + EPSILON,
        position_mean=train_positions.mean(axis=0),
        position_std=train_positions.std(axis=0) + EPSILON,
        charge_scale=float(np.sqrt(np.mean(train_charges**2))) + EPSILON,
    )


def g05_count_for_fraction(g05_fraction: float, candidate_count: int) -> int:
    if not 0.0 <= g05_fraction <= 1.0:
        raise ValueError(f"Invalid G05 fraction: {g05_fraction}")
    if g05_fraction == 0.0:
        return 0
    return max(1, int(round(g05_fraction * candidate_count)))


def create_g05_mask(
    sample_count: int,
    candidate_count: int,
    g05_fraction: float,
) -> np.ndarray:
    observed_count = g05_count_for_fraction(g05_fraction, candidate_count)
    mask = np.zeros((sample_count, candidate_count, 1), dtype=np.float32)
    mask[:, :observed_count, 0] = 1.0
    return mask


def prepare_dataset(
    arrays: DatasetArrays,
    indices: np.ndarray,
    stats: NormalizationStats,
    g05_fraction: float,
) -> TensorDataset:
    g00 = (arrays.g00[indices] - stats.g00_mean) / stats.g00_std
    g00 = g00[:, np.newaxis, :, :]
    g05 = arrays.g05[indices].copy()
    g05[:, :, 0] = 2.0 * g05[:, :, 0] / (arrays.g00.shape[2] - 1) - 1.0
    g05[:, :, 1] = 2.0 * g05[:, :, 1] / (arrays.g00.shape[1] - 1) - 1.0
    g05[:, :, 2] /= stats.g05_value_scale
    g05_mask = create_g05_mask(len(indices), g05.shape[1], g05_fraction)
    positions = arrays.target[indices][:, POSITION_INDICES]
    positions = (positions - stats.position_mean) / stats.position_std
    charges = arrays.target[indices][:, CHARGE_INDICES] / stats.charge_scale
    return TensorDataset(
        torch.from_numpy(np.ascontiguousarray(g00)),
        torch.from_numpy(np.ascontiguousarray(g05)),
        torch.from_numpy(g05_mask),
        torch.from_numpy(np.ascontiguousarray(positions)),
        torch.from_numpy(np.ascontiguousarray(charges)),
    )


def create_data_loader(
    dataset: TensorDataset,
    shuffle: bool,
    seed: int | None = None,
) -> DataLoader:
    generator = None
    if shuffle:
        if seed is None:
            raise ValueError("shuffle=True requires a seed")
        generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
        generator=generator,
    )


class ChargeNet(nn.Module):
    """Physics-separated structure and global-sign branches.

    Branch A receives only G00 and predicts ordered positions, non-negative
    magnitudes, and sign(q1*q2). Branch B receives only masked G05 points and
    predicts sign(q1), the global orientation anchor. The two branches share no
    parameters, so global-sign gradients cannot alter position features.
    """

    def __init__(self) -> None:
        super().__init__()
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

        # Coordinates are included with signed V before mask-aware pooling.
        self.g05_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        # masked mean + max + std is more robust than mean-only pooling when
        # the number of observed points grows from a few points to all 32
        # candidates. "fraction=1" means all candidates, not a full 32x32 map.
        self.global_sign_head = nn.Sequential(
            nn.Linear(32 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    @staticmethod
    def _masked_g05_summary(
        point_features: torch.Tensor,
        g05_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = g05_mask.to(dtype=point_features.dtype)
        observed_count = mask.sum(dim=1).clamp_min(1.0)
        mean = (point_features * mask).sum(dim=1) / observed_count
        centered = point_features - mean.unsqueeze(1)
        variance = ((centered**2) * mask).sum(dim=1) / observed_count
        standard_deviation = torch.sqrt(variance.clamp_min(0.0) + 1e-6)

        masked_for_max = point_features.masked_fill(~g05_mask.bool(), -torch.inf)
        maximum = masked_for_max.max(dim=1).values
        has_observation = g05_mask.sum(dim=(1, 2), keepdim=False) > 0
        maximum = torch.where(has_observation[:, None], maximum, torch.zeros_like(maximum))
        summary = torch.cat((mean, maximum, standard_deviation), dim=1)
        # This also removes sqrt(epsilon) from the no-observation std feature.
        return summary * has_observation[:, None].to(summary.dtype)

    def forward(
        self,
        g00: torch.Tensor,
        g05: torch.Tensor,
        g05_mask: torch.Tensor,
    ) -> ModelOutput:
        if g05.shape[:2] != g05_mask.shape[:2] or g05_mask.shape[-1] != 1:
            raise ValueError(f"G05/mask shape mismatch: {g05.shape}, {g05_mask.shape}")
        structure_features = self.g00_encoder(self.g00_cnn(g00))
        position = self.position_head(structure_features)
        # Softplus makes the semantics of this head explicitly |q| >= 0.
        magnitude = F.softplus(self.magnitude_head(structure_features))
        relative_sign_logit = self.relative_sign_head(structure_features).squeeze(1)

        point_features = self.g05_encoder(g05)
        g05_summary = self._masked_g05_summary(point_features, g05_mask)
        global_sign_logit = self.global_sign_head(g05_summary).squeeze(1)
        return ModelOutput(
            position=position,
            magnitude=magnitude,
            relative_sign_logit=relative_sign_logit,
            global_sign_logit=global_sign_logit,
        )


def binary_sign(logit: torch.Tensor) -> torch.Tensor:
    """Map a binary logit to {-1,+1}; ties use +1."""
    return torch.where(logit >= 0, torch.ones_like(logit), -torch.ones_like(logit))


def reconstruct_charges(
    magnitude: torch.Tensor,
    relative_sign_logit: torch.Tensor,
    global_sign_logit: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct ordered [q1,q2] with q1 as the global-sign anchor."""
    relative_sign = binary_sign(relative_sign_logit)
    global_sign = binary_sign(global_sign_logit)
    q1 = magnitude[:, 0] * global_sign
    q2 = magnitude[:, 1] * global_sign * relative_sign
    return torch.stack((q1, q2), dim=1)


def calculate_losses(
    output: ModelOutput,
    position_target: torch.Tensor,
    charge_target: torch.Tensor,
    g05_mask: torch.Tensor,
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
        POSITION_LOSS_WEIGHT * position_loss
        + MAGNITUDE_LOSS_WEIGHT * magnitude_loss
        + RELATIVE_SIGN_LOSS_WEIGHT * relative_sign_loss
    )

    has_g05 = g05_mask.sum(dim=(1, 2)) > 0
    global_sign_loss: torch.Tensor | None
    if torch.any(has_g05):
        global_target = (charge_target[has_g05, 0] > 0).to(
            output.global_sign_logit.dtype
        )
        global_sign_loss = F.binary_cross_entropy_with_logits(
            output.global_sign_logit[has_g05],
            global_target,
        )
        total_loss = structure_loss + GLOBAL_SIGN_LOSS_WEIGHT * global_sign_loss
    else:
        # No fabricated global-sign target is used when G05=0.
        global_sign_loss = None
        total_loss = structure_loss
    return BatchLoss(
        total=total_loss,
        structure=structure_loss,
        position=position_loss,
        magnitude=magnitude_loss,
        relative_sign=relative_sign_loss,
        global_sign=global_sign_loss,
    )


def run_epoch(
    model: ChargeNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochLoss:
    is_training = optimizer is not None
    model.train(mode=is_training)
    structure_sum = 0.0
    position_sum = 0.0
    magnitude_sum = 0.0
    relative_sign_sum = 0.0
    global_sign_sum = 0.0
    sample_count = 0
    global_sign_sample_count = 0

    for g00, g05, g05_mask, position_target, charge_target in loader:
        g00_device = g00.to(DEVICE, non_blocking=True)
        g05_device = g05.to(DEVICE, non_blocking=True)
        mask_device = g05_mask.to(DEVICE, non_blocking=True)
        position_target_device = position_target.to(DEVICE, non_blocking=True)
        charge_target_device = charge_target.to(DEVICE, non_blocking=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_training):
            output = model(g00_device, g05_device, mask_device)
            losses = calculate_losses(
                output,
                position_target_device,
                charge_target_device,
                mask_device,
            )
            if optimizer is not None:
                losses.total.backward()
                optimizer.step()

        current_batch_size = g00.shape[0]
        structure_sum += losses.structure.item() * current_batch_size
        position_sum += losses.position.item() * current_batch_size
        magnitude_sum += losses.magnitude.item() * current_batch_size
        relative_sign_sum += losses.relative_sign.item() * current_batch_size
        sample_count += current_batch_size
        if losses.global_sign is not None:
            observed_count = int((g05_mask.sum(dim=(1, 2)) > 0).sum().item())
            global_sign_sum += losses.global_sign.item() * observed_count
            global_sign_sample_count += observed_count

    if sample_count == 0:
        raise ValueError("Empty DataLoader")
    structure_average = structure_sum / sample_count
    global_sign_average = (
        global_sign_sum / global_sign_sample_count
        if global_sign_sample_count > 0
        else None
    )
    total_average = structure_average
    if global_sign_average is not None:
        total_average += GLOBAL_SIGN_LOSS_WEIGHT * global_sign_average
    return EpochLoss(
        total=total_average,
        structure=structure_average,
        position=position_sum / sample_count,
        magnitude=magnitude_sum / sample_count,
        relative_sign=relative_sign_sum / sample_count,
        global_sign=global_sign_average,
    )


def fraction_label(g05_fraction: float) -> str:
    return f"{int(round(g05_fraction * 100)):03d}pct"


def checkpoint_metadata(
    arrays: DatasetArrays,
    stats: NormalizationStats,
) -> dict[str, object]:
    return {
        "model_architecture": "physics-separated G00 structure + G05 global-sign v3",
        "grid_shape": tuple(arrays.g00.shape[1:]),
        "g05_candidate_count": arrays.g05.shape[1],
        "position_indices": POSITION_INDICES,
        "charge_indices": CHARGE_INDICES,
        "target_fields": TARGET_FIELDS,
        "g05_fields": G05_FIELDS,
        "target_ordering": "q1/q2 ordered lexicographically by x, then y, then z",
        "global_sign_anchor": "sign(q1)",
        "relative_sign_target": "1 iff q1*q2 > 0",
        "final_q_reconstruction": "q1=|q1|*s, q2=|q2|*s*r",
        "g00_mean": stats.g00_mean,
        "g00_std": stats.g00_std,
        "g05_value_scale": stats.g05_value_scale,
        "position_mean": stats.position_mean,
        "position_std": stats.position_std,
        "charge_scale": stats.charge_scale,
        "partial_g05_definition": "fixed spatially-balanced nested sensor prefixes",
        "full_fraction_definition": (
            "all stored G05 candidate sensors, not the complete potential grid"
        ),
        "g05_pooling": "mask-aware mean + max + population std",
        "gradient_policy": "global-sign branch shares no parameters with G00 branch",
        "checkpoint_policy": (
            "compose best structure epoch and best global-sign epoch; safe because "
            "branches are parameter- and input-independent"
        ),
    }


STRUCTURE_PREFIXES = (
    "g00_cnn.",
    "g00_encoder.",
    "position_head.",
    "magnitude_head.",
    "relative_sign_head.",
)
GLOBAL_SIGN_PREFIXES = ("g05_encoder.", "global_sign_head.")


def component_state_dict(
    model: ChargeNet,
    prefixes: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name.startswith(prefixes)
    }


def full_state_dict(model: ChargeNet) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }


def save_component_checkpoint(
    path: Path,
    model: ChargeNet,
    prefixes: tuple[str, ...],
    component: str,
    objective: str,
    epoch: int,
    validation_loss: float,
    run_metadata: dict[str, object],
) -> None:
    torch.save(
        {
            "component_state_dict": component_state_dict(model, prefixes),
            "component": component,
            "objective": objective,
            "epoch": epoch,
            "validation_loss": validation_loss,
            **run_metadata,
        },
        path,
    )


def load_checkpoint(path: Path) -> dict[str, object]:
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def load_component(model: ChargeNet, path: Path) -> None:
    checkpoint = load_checkpoint(path)
    component = checkpoint["component_state_dict"]
    current_state = model.state_dict()
    unknown_keys = set(component).difference(current_state)
    if unknown_keys:
        raise KeyError(f"Unknown component state keys: {sorted(unknown_keys)}")
    current_state.update(component)
    model.load_state_dict(current_state)


def train_model(
    train_dataset: TensorDataset,
    validation_dataset: TensorDataset,
    g05_fraction: float,
    g05_count: int,
    seed: int,
    max_epochs: int,
    patience: int,
    metadata: dict[str, object],
) -> TrainingResult:
    set_reproducibility(seed)
    train_loader = create_data_loader(train_dataset, shuffle=True, seed=seed)
    validation_loader = create_data_loader(validation_dataset, shuffle=False)
    model = ChargeNet().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = CHECKPOINT_DIR / (
        f"g05_{fraction_label(g05_fraction)}_seed_{seed}"
    )
    structure_path = Path(f"{prefix}_best_structure.pt")
    position_path = Path(f"{prefix}_best_position.pt")
    magnitude_path = Path(f"{prefix}_best_magnitude.pt")
    global_sign_path = (
        Path(f"{prefix}_best_global_sign.pt") if g05_count > 0 else None
    )
    composed_path = Path(f"{prefix}_composed.pt")
    run_metadata = {
        "g05_fraction": g05_fraction,
        "g05_count": g05_count,
        "seed": seed,
        **metadata,
    }

    best_structure_loss = float("inf")
    best_structure_epoch = 0
    best_position_loss = float("inf")
    best_position_epoch = 0
    best_magnitude_loss = float("inf")
    best_magnitude_epoch = 0
    best_global_sign_loss = float("inf") if g05_count > 0 else None
    best_global_sign_epoch: int | None = None
    epochs_since_structure_improvement = 0
    epochs_since_sign_improvement = 0

    print("\n" + "=" * 88)
    print(
        f"Training: G05 fraction={g05_fraction:.2f}, points={g05_count} | seed={seed}"
    )
    print("=" * 88)
    for epoch in range(1, max_epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer)
        validation_loss = run_epoch(model, validation_loader)
        finite_values = (
            validation_loss.structure,
            validation_loss.position,
            validation_loss.magnitude,
            validation_loss.relative_sign,
        )
        if validation_loss.global_sign is not None:
            finite_values += (validation_loss.global_sign,)
        if not all(np.isfinite(value) for value in finite_values):
            raise FloatingPointError(
                f"Non-finite validation loss: fraction={g05_fraction}, "
                f"seed={seed}, epoch={epoch}"
            )
        train_global = (
            "N/A" if train_loss.global_sign is None else f"{train_loss.global_sign:.6f}"
        )
        val_global = (
            "N/A"
            if validation_loss.global_sign is None
            else f"{validation_loss.global_sign:.6f}"
        )
        print(
            f"Epoch {epoch:03d} | Train structure={train_loss.structure:.6f} "
            f"pos={train_loss.position:.6f} mag={train_loss.magnitude:.6f} "
            f"rel={train_loss.relative_sign:.6f} global={train_global} | "
            f"Val structure={validation_loss.structure:.6f} "
            f"pos={validation_loss.position:.6f} mag={validation_loss.magnitude:.6f} "
            f"rel={validation_loss.relative_sign:.6f} global={val_global}"
        )

        structure_improved = validation_loss.structure < best_structure_loss
        if structure_improved:
            best_structure_loss = validation_loss.structure
            best_structure_epoch = epoch
            epochs_since_structure_improvement = 0
            save_component_checkpoint(
                structure_path,
                model,
                STRUCTURE_PREFIXES,
                "structure",
                "position+magnitude+relative_sign",
                epoch,
                validation_loss.structure,
                run_metadata,
            )
        else:
            epochs_since_structure_improvement += 1

        if validation_loss.position < best_position_loss:
            best_position_loss = validation_loss.position
            best_position_epoch = epoch
            save_component_checkpoint(
                position_path,
                model,
                STRUCTURE_PREFIXES,
                "structure",
                "position",
                epoch,
                validation_loss.position,
                run_metadata,
            )
        if validation_loss.magnitude < best_magnitude_loss:
            best_magnitude_loss = validation_loss.magnitude
            best_magnitude_epoch = epoch
            save_component_checkpoint(
                magnitude_path,
                model,
                STRUCTURE_PREFIXES,
                "structure",
                "magnitude",
                epoch,
                validation_loss.magnitude,
                run_metadata,
            )

        if validation_loss.global_sign is not None:
            sign_improved = validation_loss.global_sign < best_global_sign_loss
            if sign_improved:
                best_global_sign_loss = validation_loss.global_sign
                best_global_sign_epoch = epoch
                epochs_since_sign_improvement = 0
                assert global_sign_path is not None
                save_component_checkpoint(
                    global_sign_path,
                    model,
                    GLOBAL_SIGN_PREFIXES,
                    "global_sign",
                    "global_sign_bce",
                    epoch,
                    validation_loss.global_sign,
                    run_metadata,
                )
            else:
                epochs_since_sign_improvement += 1

        structure_stopped = epochs_since_structure_improvement >= patience
        sign_stopped = (
            validation_loss.global_sign is None
            or epochs_since_sign_improvement >= patience
        )
        if structure_stopped and sign_stopped:
            print(
                f"Early stopping at epoch {epoch}; best structure epoch="
                f"{best_structure_epoch}, best global-sign epoch="
                f"{best_global_sign_epoch if best_global_sign_epoch is not None else 'N/A'}"
            )
            break

    # Different-epoch composition is safe here: the sign component uses only
    # G05 and shares neither parameters nor input features with the G00 component.
    load_component(model, structure_path)
    if global_sign_path is not None:
        load_component(model, global_sign_path)
    torch.save(
        {
            "model_state_dict": full_state_dict(model),
            "component_sources": {
                "structure": str(structure_path),
                "global_sign": (
                    None if global_sign_path is None else str(global_sign_path)
                ),
            },
            "best_structure_loss": best_structure_loss,
            "best_structure_epoch": best_structure_epoch,
            "best_position_loss": best_position_loss,
            "best_position_epoch": best_position_epoch,
            "best_magnitude_loss": best_magnitude_loss,
            "best_magnitude_epoch": best_magnitude_epoch,
            "best_global_sign_loss": best_global_sign_loss,
            "best_global_sign_epoch": best_global_sign_epoch,
            **run_metadata,
        },
        composed_path,
    )
    return TrainingResult(
        model=model,
        seed=seed,
        g05_fraction=g05_fraction,
        g05_count=g05_count,
        best_structure_loss=best_structure_loss,
        best_structure_epoch=best_structure_epoch,
        best_position_loss=best_position_loss,
        best_position_epoch=best_position_epoch,
        best_magnitude_loss=best_magnitude_loss,
        best_magnitude_epoch=best_magnitude_epoch,
        best_global_sign_loss=best_global_sign_loss,
        best_global_sign_epoch=best_global_sign_epoch,
        structure_checkpoint_path=structure_path,
        position_checkpoint_path=position_path,
        magnitude_checkpoint_path=magnitude_path,
        global_sign_checkpoint_path=global_sign_path,
        checkpoint_path=composed_path,
    )


def align_global_charge_sign(
    prediction: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Oracle alignment used only for sign-unidentifiable G05=0 charge MAE."""
    direct_error = np.mean((prediction - target) ** 2, axis=1)
    flipped_error = np.mean((-prediction - target) ** 2, axis=1)
    alignment = np.where(flipped_error < direct_error, -1.0, 1.0)[:, None]
    return prediction * alignment


def evaluate_model(
    model: ChargeNet,
    test_dataset: TensorDataset,
    stats: NormalizationStats,
) -> EvaluationResult:
    test_loader = create_data_loader(test_dataset, shuffle=False)
    model.eval()
    position_predictions: list[np.ndarray] = []
    magnitude_predictions: list[np.ndarray] = []
    relative_logits: list[np.ndarray] = []
    global_logits: list[np.ndarray] = []
    position_targets: list[np.ndarray] = []
    charge_targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    with torch.inference_mode():
        for g00, g05, g05_mask, position_target, charge_target in test_loader:
            output = model(
                g00.to(DEVICE, non_blocking=True),
                g05.to(DEVICE, non_blocking=True),
                g05_mask.to(DEVICE, non_blocking=True),
            )
            position_predictions.append(output.position.cpu().numpy())
            magnitude_predictions.append(output.magnitude.cpu().numpy())
            relative_logits.append(output.relative_sign_logit.cpu().numpy())
            global_logits.append(output.global_sign_logit.cpu().numpy())
            position_targets.append(position_target.numpy())
            charge_targets.append(charge_target.numpy())
            masks.append(g05_mask.numpy())

    position_prediction = (
        np.concatenate(position_predictions) * stats.position_std
        + stats.position_mean
    )
    position_target = (
        np.concatenate(position_targets) * stats.position_std
        + stats.position_mean
    )
    magnitude_prediction = np.concatenate(magnitude_predictions) * stats.charge_scale
    charge_target = np.concatenate(charge_targets) * stats.charge_scale
    relative_logit = np.concatenate(relative_logits)
    global_logit = np.concatenate(global_logits)
    g05_mask = np.concatenate(masks)
    g05_counts = g05_mask.sum(axis=(1, 2))
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
    charge_magnitude_mae = float(
        np.mean(np.abs(magnitude_prediction - np.abs(charge_target)))
    )
    target_relative_sign = np.sign(charge_target[:, 0] * charge_target[:, 1])
    relative_sign_accuracy = float(np.mean(relative_sign == target_relative_sign))

    if np.any(observed):
        target_global_positive = charge_target[observed, 0] > 0
        global_sign_accuracy = float(
            np.mean((global_logit[observed] >= 0) == target_global_positive)
        )
        # Stable BCE-with-logits in NumPy: max(x,0)-x*y+log1p(exp(-abs(x))).
        logits = global_logit[observed].astype(np.float64)
        targets = target_global_positive.astype(np.float64)
        global_sign_bce = float(
            np.mean(
                np.maximum(logits, 0.0)
                - logits * targets
                + np.log1p(np.exp(-np.abs(logits)))
            )
        )
        prediction_sign = np.sign(charge_prediction[observed])
        target_sign = np.sign(charge_target[observed])
        absolute_sign_accuracy = float(np.mean(prediction_sign == target_sign))
        signed_pair_accuracy = float(
            np.mean(np.all(prediction_sign == target_sign, axis=1))
        )
    else:
        global_sign_accuracy = None
        global_sign_bce = None
        absolute_sign_accuracy = None
        signed_pair_accuracy = None

    return EvaluationResult(
        position_mae=position_mae,
        mean_position_mae=float(position_mae.mean()),
        position_error_1=position_error_1,
        position_error_2=position_error_2,
        mean_position_3d_error=0.5 * (position_error_1 + position_error_2),
        charge_mae=charge_mae,
        charge_magnitude_mae=charge_magnitude_mae,
        relative_sign_accuracy=relative_sign_accuracy,
        global_sign_accuracy=global_sign_accuracy,
        global_sign_bce=global_sign_bce,
        absolute_sign_accuracy=absolute_sign_accuracy,
        signed_pair_accuracy=signed_pair_accuracy,
        observed_sample_fraction=float(observed.mean()),
        observations_per_sample=float(g05_counts.mean()),
    )


def sample_std(values: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """Sample standard deviation across independent seeds (ddof=1)."""
    values = np.asarray(values)
    sample_count = values.size if axis is None else values.shape[axis]
    if sample_count < 2:
        reduced = np.mean(values, axis=axis)
        if np.ndim(reduced) == 0:
            return float("nan")
        return np.full(np.shape(reduced), np.nan, dtype=np.float64)
    return np.std(values, axis=axis, ddof=1)


def optional_value(value: float | int | Path | None) -> float | int | str:
    return "" if value is None else str(value) if isinstance(value, Path) else value


def print_evaluation(
    result: EvaluationResult,
    g05_fraction: float,
    g05_count: int,
    seed: int,
) -> None:
    print("\n" + "-" * 88)
    print(
        f"Test: G05 fraction={g05_fraction:.2f} | points={g05_count} | seed={seed}"
    )
    print("G00 structure position MAE [x1,y1,z1,x2,y2,z2]:", result.position_mae)
    print("Mean coordinate position MAE:", result.mean_position_mae)
    print(
        "Charge 1/2/mean 3D position error:",
        result.position_error_1,
        result.position_error_2,
        result.mean_position_3d_error,
    )
    print("G00 structure charge magnitude MAE:", result.charge_magnitude_mae)
    print("Relative sign accuracy:", result.relative_sign_accuracy)
    print("Global sign BCE/accuracy:", result.global_sign_bce, result.global_sign_accuracy)
    print("Absolute/pair sign accuracy:", result.absolute_sign_accuracy, result.signed_pair_accuracy)


def run_result_row(
    training: TrainingResult,
    evaluation: EvaluationResult,
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "g05_fraction": training.g05_fraction,
        "g05_count_per_sample": training.g05_count,
        "seed": training.seed,
        "mean_position_mae": evaluation.mean_position_mae,
        "position_mae_x1": evaluation.position_mae[0],
        "position_mae_y1": evaluation.position_mae[1],
        "position_mae_z1": evaluation.position_mae[2],
        "position_mae_x2": evaluation.position_mae[3],
        "position_mae_y2": evaluation.position_mae[4],
        "position_mae_z2": evaluation.position_mae[5],
        "charge1_position_error": evaluation.position_error_1,
        "charge2_position_error": evaluation.position_error_2,
        "mean_position_3d_error": evaluation.mean_position_3d_error,
        "charge_mae_q1": evaluation.charge_mae[0],
        "charge_mae_q2": evaluation.charge_mae[1],
        "charge_magnitude_mae": evaluation.charge_magnitude_mae,
        "relative_sign_accuracy": evaluation.relative_sign_accuracy,
        "global_sign_bce": optional_value(evaluation.global_sign_bce),
        "global_sign_accuracy": optional_value(evaluation.global_sign_accuracy),
        "absolute_sign_accuracy": optional_value(evaluation.absolute_sign_accuracy),
        "signed_pair_accuracy": optional_value(evaluation.signed_pair_accuracy),
        "observed_sample_fraction": evaluation.observed_sample_fraction,
        "observations_per_sample": evaluation.observations_per_sample,
        "best_structure_loss": training.best_structure_loss,
        "best_structure_epoch": training.best_structure_epoch,
        "best_position_loss": training.best_position_loss,
        "best_position_epoch": training.best_position_epoch,
        "best_magnitude_loss": training.best_magnitude_loss,
        "best_magnitude_epoch": training.best_magnitude_epoch,
        "best_global_sign_loss": optional_value(training.best_global_sign_loss),
        "best_global_sign_epoch": optional_value(training.best_global_sign_epoch),
        "structure_checkpoint_path": str(training.structure_checkpoint_path),
        "position_checkpoint_path": str(training.position_checkpoint_path),
        "magnitude_checkpoint_path": str(training.magnitude_checkpoint_path),
        "global_sign_checkpoint_path": optional_value(
            training.global_sign_checkpoint_path
        ),
        "composed_checkpoint_path": str(training.checkpoint_path),
    }
    return row


def save_csv(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    if not rows:
        raise ValueError("Cannot save an empty result table")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scalar_metric(result: EvaluationResult, name: str) -> float:
    value = getattr(result, name)
    return np.nan if value is None else float(value)


def build_summary_rows(
    results_by_fraction: dict[float, list[EvaluationResult]],
    candidate_count: int,
) -> list[dict[str, float | int | str]]:
    metric_names = (
        "mean_position_mae",
        "position_error_1",
        "position_error_2",
        "mean_position_3d_error",
        "charge_magnitude_mae",
        "relative_sign_accuracy",
        "global_sign_bce",
        "global_sign_accuracy",
        "absolute_sign_accuracy",
        "signed_pair_accuracy",
    )
    rows: list[dict[str, float | int | str]] = []
    for fraction, results in results_by_fraction.items():
        row: dict[str, float | int | str] = {
            "g05_fraction": fraction,
            "g05_count_per_sample": g05_count_for_fraction(
                fraction,
                candidate_count,
            ),
            "g05_full_fraction_note": (
                "all candidate sensors; not full potential grid"
                if fraction == 1.0
                else ""
            ),
            "run_count": len(results),
            "std_definition": "sample std (ddof=1)",
        }
        position_mae = np.stack([result.position_mae for result in results])
        for index, label in enumerate(("x1", "y1", "z1", "x2", "y2", "z2")):
            row[f"position_mae_{label}_mean"] = float(position_mae[:, index].mean())
            row[f"position_mae_{label}_std"] = float(
                sample_std(position_mae[:, index])
            )
        for metric_name in metric_names:
            values = np.asarray(
                [scalar_metric(result, metric_name) for result in results],
                dtype=np.float64,
            )
            valid_values = values[np.isfinite(values)]
            row[f"{metric_name}_mean"] = (
                "" if valid_values.size == 0 else float(valid_values.mean())
            )
            row[f"{metric_name}_std"] = (
                ""
                if valid_values.size < 2
                else float(sample_std(valid_values))
            )
        rows.append(row)
    return rows


def save_metric_plot(
    results_by_fraction: dict[float, list[EvaluationResult]],
    metric_name: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    # Plotting is imported lazily so --smoke-only and model inference do not
    # require matplotlib in a minimal PyTorch environment.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fractions = np.asarray(tuple(results_by_fraction), dtype=np.float64)
    means = np.full_like(fractions, np.nan)
    stds = np.full_like(fractions, np.nan)
    for index, fraction in enumerate(results_by_fraction):
        values = np.asarray(
            [scalar_metric(result, metric_name) for result in results_by_fraction[fraction]],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]
        if values.size:
            means[index] = values.mean()
        if values.size >= 2:
            stds[index] = sample_std(values)
    valid = np.isfinite(means)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    if np.any(valid):
        yerr = np.where(np.isfinite(stds[valid]), stds[valid], 0.0)
        axis.errorbar(
            fractions[valid],
            means[valid],
            yerr=yerr,
            fmt="o-",
            capsize=4,
            linewidth=1.5,
        )
    axis.set_xlabel("G05 fraction")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.set_xticks(fractions)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_summary_plots(
    results_by_fraction: dict[float, list[EvaluationResult]],
) -> None:
    plot_specs = {
        "mean_position_mae": (
            "Mean coordinate position MAE",
            "G00-only structure position MAE",
        ),
        "mean_position_3d_error": (
            "Mean 3D position error",
            "G00-only structure 3D position error",
        ),
        "charge_magnitude_mae": (
            "Charge magnitude MAE",
            "G00-only charge magnitude MAE",
        ),
        "relative_sign_accuracy": (
            "Relative sign accuracy",
            "G00-only relative sign accuracy",
        ),
        "global_sign_accuracy": (
            "Global sign accuracy",
            "G05-only global sign accuracy",
        ),
        "absolute_sign_accuracy": (
            "Absolute sign accuracy",
            "Reconstructed charge sign accuracy",
        ),
        "global_sign_bce": (
            "Global sign BCE",
            "G05-only global sign BCE",
        ),
    }
    for metric_name, (ylabel, title) in plot_specs.items():
        save_metric_plot(
            results_by_fraction,
            metric_name,
            ylabel,
            title,
            PLOT_PATHS[metric_name],
        )


def save_normalization_stats(stats: NormalizationStats) -> None:
    NORMALIZATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        NORMALIZATION_PATH,
        G00_mean=stats.g00_mean,
        G00_std=stats.g00_std,
        G05_value_scale=stats.g05_value_scale,
        position_mean=stats.position_mean,
        position_std=stats.position_std,
        charge_scale=stats.charge_scale,
        position_indices=np.asarray(POSITION_INDICES, dtype=np.int64),
        charge_indices=np.asarray(CHARGE_INDICES, dtype=np.int64),
        target_fields=np.asarray(TARGET_FIELDS),
        global_sign_anchor=np.asarray("q1 after lexicographic x/y/z target ordering"),
        standard_deviation_definition=np.asarray(
            "sample std (ddof=1) for multi-seed reports"
        ),
    )


def save_canonical_checkpoint(
    training: TrainingResult,
    output_path: Path,
    metadata: dict[str, object],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": full_state_dict(training.model),
            "g05_fraction": training.g05_fraction,
            "g05_count": training.g05_count,
            "seed": training.seed,
            "best_structure_loss": training.best_structure_loss,
            "best_structure_epoch": training.best_structure_epoch,
            "best_position_loss": training.best_position_loss,
            "best_position_epoch": training.best_position_epoch,
            "best_magnitude_loss": training.best_magnitude_loss,
            "best_magnitude_epoch": training.best_magnitude_epoch,
            "best_global_sign_loss": training.best_global_sign_loss,
            "best_global_sign_epoch": training.best_global_sign_epoch,
            "source_checkpoint": str(training.checkpoint_path),
            **metadata,
        },
        output_path,
    )


def assert_gradient_isolation(
    model: ChargeNet,
    tensors: list[torch.Tensor],
) -> None:
    g00, g05, g05_mask, position_target, charge_target = tensors
    if not torch.any(g05_mask > 0):
        return
    structure_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith(STRUCTURE_PREFIXES)
    ]
    sign_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith(GLOBAL_SIGN_PREFIXES)
    ]

    model.zero_grad(set_to_none=True)
    output = model(g00, g05, g05_mask)
    losses = calculate_losses(output, position_target, charge_target, g05_mask)
    assert losses.global_sign is not None
    losses.global_sign.backward()
    if any(
        parameter.grad is not None and torch.any(parameter.grad != 0)
        for parameter in structure_parameters
    ):
        raise RuntimeError("Global-sign gradient leaked into the G00 structure branch")
    if not any(
        parameter.grad is not None and torch.any(parameter.grad != 0)
        for parameter in sign_parameters
    ):
        raise RuntimeError("Global-sign loss produced no G05 branch gradient")

    model.zero_grad(set_to_none=True)
    output = model(g00, g05, g05_mask)
    F.mse_loss(output.position, position_target).backward()
    if any(
        parameter.grad is not None and torch.any(parameter.grad != 0)
        for parameter in sign_parameters
    ):
        raise RuntimeError("Position gradient leaked into the G05 global-sign branch")
    model.zero_grad(set_to_none=True)


def run_smoke_tests(
    datasets_by_fraction: dict[
        float,
        tuple[TensorDataset, TensorDataset, TensorDataset],
    ],
) -> None:
    print("\nRunning forward/loss/gradient-isolation smoke tests...")
    gradient_isolation_checked = False
    for fraction, dataset_group in datasets_by_fraction.items():
        dataset = dataset_group[0]
        sample_count = min(4, len(dataset))
        tensors = [tensor[:sample_count].to(DEVICE) for tensor in dataset.tensors]
        g00, g05, g05_mask, position_target, charge_target = tensors
        set_reproducibility(41)
        model = ChargeNet().to(DEVICE)
        with torch.no_grad():
            output = model(g00, g05, g05_mask)
            losses = calculate_losses(output, position_target, charge_target, g05_mask)
            charge_prediction = reconstruct_charges(
                output.magnitude,
                output.relative_sign_logit,
                output.global_sign_logit,
            )
        expected_shapes = {
            "position": (sample_count, 6),
            "magnitude": (sample_count, 2),
            "relative_sign_logit": (sample_count,),
            "global_sign_logit": (sample_count,),
            "reconstructed_charge": (sample_count, 2),
        }
        actual_shapes = {
            "position": tuple(output.position.shape),
            "magnitude": tuple(output.magnitude.shape),
            "relative_sign_logit": tuple(output.relative_sign_logit.shape),
            "global_sign_logit": tuple(output.global_sign_logit.shape),
            "reconstructed_charge": tuple(charge_prediction.shape),
        }
        for name, expected_shape in expected_shapes.items():
            if actual_shapes[name] != expected_shape:
                raise RuntimeError(
                    f"{name} smoke-test shape error at fraction={fraction}: "
                    f"{actual_shapes[name]} != {expected_shape}"
                )
        loss_values = (
            losses.total,
            losses.structure,
            losses.position,
            losses.magnitude,
            losses.relative_sign,
        )
        if losses.global_sign is not None:
            loss_values += (losses.global_sign,)
        if not all(torch.isfinite(loss).item() for loss in loss_values):
            raise RuntimeError(f"Non-finite smoke-test loss: {fraction}")
        if fraction == 0.0 and losses.global_sign is not None:
            raise RuntimeError("G05=0 must not produce a global-sign loss")
        if fraction > 0.0 and not gradient_isolation_checked:
            assert_gradient_isolation(model, tensors)
            gradient_isolation_checked = True
        print(f"  G05 fraction={fraction:.2f}: OK")
    if not gradient_isolation_checked:
        print("  Gradient isolation: skipped (no positive G05 fraction requested)")
    else:
        print("  Gradient isolation: OK")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_number_list(value: str, converter: type) -> tuple:
    return tuple(converter(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the physics-separated G00 structure / G05 global-sign experiment."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument(
        "--seeds",
        type=lambda value: parse_number_list(value, int),
        default=EXPERIMENT_SEEDS,
    )
    parser.add_argument(
        "--fractions",
        type=lambda value: parse_number_list(value, float),
        default=G05_FRACTIONS,
    )
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


def training_selection_score(training: TrainingResult) -> float:
    sign_loss = (
        0.0
        if training.best_global_sign_loss is None
        else training.best_global_sign_loss
    )
    return training.best_structure_loss + GLOBAL_SIGN_LOSS_WEIGHT * sign_loss


def main() -> None:
    args = parse_args()
    experiment_seeds = tuple(args.seeds)
    g05_fractions = tuple(args.fractions)
    if not experiment_seeds or not g05_fractions:
        raise ValueError("At least one seed and fraction are required")
    if tuple(sorted(g05_fractions)) != g05_fractions:
        raise ValueError("Fractions must be sorted")
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs and patience must be positive")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    set_reproducibility(DATA_SPLIT_SEED)
    print("Device:", DEVICE)
    print("Data:", args.data.resolve())
    arrays = load_dataset(args.data)
    data_split = create_data_split(arrays.g00.shape[0], DATA_SPLIT_SEED)
    stats = calculate_normalization_stats(arrays, data_split.train)
    metadata = checkpoint_metadata(arrays, stats)
    print("G00/G05/target:", arrays.g00.shape, arrays.g05.shape, arrays.target.shape)
    print("Target semantics:", TARGET_FIELDS)
    print("G05 semantics:", G05_FIELDS)
    print("Target ordering: lexicographic x/y/z; q1 is the global-sign anchor")
    print("G05 pooling: masked mean + max + std over [x_index,y_index,V]")
    print("G05 fraction=1.0 means all stored candidate sensors, not a full field")

    datasets_by_fraction: dict[
        float,
        tuple[TensorDataset, TensorDataset, TensorDataset],
    ] = {}
    for fraction in g05_fractions:
        datasets_by_fraction[fraction] = (
            prepare_dataset(arrays, data_split.train, stats, fraction),
            prepare_dataset(arrays, data_split.validation, stats, fraction),
            prepare_dataset(arrays, data_split.test, stats, fraction),
        )
        count = g05_count_for_fraction(fraction, arrays.g05.shape[1])
        note = (
            " (all candidate sensors; not the complete grid)"
            if fraction == 1.0
            else ""
        )
        print(f"G05 fraction={fraction:.2f}: {count} points per sample{note}")

    run_smoke_tests(datasets_by_fraction)
    if args.smoke_only:
        print("Smoke-only run complete")
        return

    results_by_fraction: dict[float, list[EvaluationResult]] = {
        fraction: [] for fraction in g05_fractions
    }
    run_rows: list[dict[str, float | int | str]] = []
    best_by_fraction: dict[float, TrainingResult] = {}
    for fraction in g05_fractions:
        train_dataset, validation_dataset, test_dataset = datasets_by_fraction[fraction]
        g05_count = g05_count_for_fraction(fraction, arrays.g05.shape[1])
        for seed in experiment_seeds:
            training = train_model(
                train_dataset,
                validation_dataset,
                fraction,
                g05_count,
                seed,
                args.epochs,
                args.patience,
                metadata,
            )
            evaluation = evaluate_model(training.model, test_dataset, stats)
            results_by_fraction[fraction].append(evaluation)
            run_rows.append(run_result_row(training, evaluation))
            print_evaluation(evaluation, fraction, g05_count, seed)
            current_best = best_by_fraction.get(fraction)
            if current_best is None or training_selection_score(
                training
            ) < training_selection_score(current_best):
                best_by_fraction[fraction] = training

    summary_rows = build_summary_rows(results_by_fraction, arrays.g05.shape[1])
    save_csv(run_rows, RUN_RESULTS_PATH)
    save_csv(summary_rows, SUMMARY_RESULTS_PATH)
    save_summary_plots(results_by_fraction)
    save_normalization_stats(stats)
    if 0.0 in best_by_fraction:
        save_canonical_checkpoint(best_by_fraction[0.0], MODEL_G00_PATH, metadata)
    if 1.0 in best_by_fraction:
        save_canonical_checkpoint(best_by_fraction[1.0], MODEL_G00_G05_PATH, metadata)

    print("\nExperiment complete")
    print("Per-run CSV:", RUN_RESULTS_PATH)
    print("Summary CSV:", SUMMARY_RESULTS_PATH)
    print("Purpose-specific and composed checkpoints:", CHECKPOINT_DIR)
    print("Plots:", RESULTS_DIR)


if __name__ == "__main__":
    main()

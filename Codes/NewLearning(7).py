from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "Models"
RESULTS_DIR = PROJECT_DIR / "Results"
DEFAULT_DATA_PATH = MODELS_DIR / "charge_dataset_multipoint_v2.npz"
CHECKPOINT_DIR = MODELS_DIR / "g05_fraction_checkpoints_v2"
MODEL_G00_PATH = MODELS_DIR / "model_G00_capacity_matched_v2.pt"
MODEL_G00_G05_PATH = MODELS_DIR / "model_G00_G05_full_v2.pt"
NORMALIZATION_PATH = MODELS_DIR / "normalization_stats_v2.npz"
RUN_RESULTS_PATH = RESULTS_DIR / "g05_fraction_results_v2.csv"
SUMMARY_RESULTS_PATH = RESULTS_DIR / "g05_fraction_summary_v2.csv"

PLOT_PATHS = {
    "absolute_sign_accuracy": RESULTS_DIR
    / "g05_fraction_absolute_sign_accuracy_v2.png",
    "global_sign_accuracy": RESULTS_DIR / "g05_fraction_global_sign_accuracy_v2.png",
    "relative_sign_accuracy": RESULTS_DIR
    / "g05_fraction_relative_sign_accuracy_v2.png",
    "charge_magnitude_mae": RESULTS_DIR / "g05_fraction_charge_magnitude_mae_v2.png",
    "mean_position_error": RESULTS_DIR / "g05_fraction_mean_position_error_v2.png",
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
CHARGE_LOSS_WEIGHT = 1.0
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
class EpochLoss:
    total: float
    position: float
    charge: float


@dataclass(frozen=True)
class TrainingResult:
    model: ChargeNet
    best_validation_loss: float
    best_epoch: int
    seed: int
    g05_fraction: float
    g05_count: int
    checkpoint_path: Path


@dataclass(frozen=True)
class EvaluationResult:
    position_mae: np.ndarray
    position_error_1: float
    position_error_2: float
    charge_mae: np.ndarray
    charge_magnitude_mae: float
    relative_sign_accuracy: float
    absolute_sign_accuracy: float | None
    global_sign_accuracy: float | None
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
            float(archive["epsilon_0"].item())
            if "epsilon_0" in archive
            else 1.0
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
    if g05.ndim != 3 or g05.shape[-1] != 3:
        raise ValueError(f"Invalid G05 shape: {g05.shape}")
    if target.ndim != 2 or target.shape[-1] != len(TARGET_FIELDS):
        raise ValueError(f"Invalid target shape: {target.shape}")
    if not (g00.shape[0] == g05.shape[0] == target.shape[0]):
        raise ValueError("G00, G05, and target sample counts differ")
    if g00.shape[0] < 10:
        raise ValueError("At least 10 samples are required")
    if g05.shape[1] < 10:
        raise ValueError(
            "At least 10 candidate G05 points per sample are required. "
            "The legacy (N, 1, 3) dataset measures sample availability, not an "
            "in-sample information fraction; regenerate the dataset."
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

    x_index = g05[:, :, 0]
    y_index = g05[:, :, 1]
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
        raise ValueError(
            "Candidate G05 sensor locations must be fixed across samples for a "
            "fair nested-fraction experiment"
        )
    if np.unique(candidate_positions, axis=0).shape[0] != g05.shape[1]:
        raise ValueError("Candidate G05 sensor locations contain duplicates")
    if np.any(target[:, 0] > target[:, 4]):
        raise ValueError("Targets are not deterministically ordered by charge x")
    if np.any(np.abs(target[:, CHARGE_INDICES]) <= EPSILON):
        raise ValueError("Zero charge is outside this experiment's target semantics")


def verify_physical_consistency(
    arrays: DatasetArrays,
    verification_sample_count: int = 16,
) -> None:
    """Numerically verify target layout and both observation equations."""
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
            raise ValueError(
                "G00 is inconsistent with target=[x1,y1,z1,q1,x2,y2,z2,q2]"
            )

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
        # A common train-only physical scale is intentionally shared by all
        # fractions so normalization itself is not a fraction-dependent change.
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
    """One capacity-matched architecture for every G05 fraction."""

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
        self.g05_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.shared_encoder = nn.Sequential(nn.Linear(160, 128), nn.ReLU())
        self.position_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
        )
        self.charge_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(
        self,
        g00: torch.Tensor,
        g05: torch.Tensor,
        g05_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if g05.shape[:2] != g05_mask.shape[:2] or g05_mask.shape[-1] != 1:
            raise ValueError(f"G05/mask shape mismatch: {g05.shape}, {g05_mask.shape}")
        g00_features = self.g00_encoder(self.g00_cnn(g00))
        point_features = self.g05_encoder(g05) * g05_mask
        observed_count = g05_mask.sum(dim=1).clamp_min(1.0)
        g05_features = point_features.sum(dim=1) / observed_count
        shared_features = self.shared_encoder(
            torch.cat((g00_features, g05_features), dim=1)
        )
        return self.position_head(shared_features), self.charge_head(shared_features)


def samplewise_charge_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    g05_mask: torch.Tensor,
) -> torch.Tensor:
    """Use signed loss only where that sample actually has signed G05 data."""
    direct_error = torch.mean((prediction - target) ** 2, dim=1)
    flipped_error = torch.mean((prediction + target) ** 2, dim=1)
    invariant_error = torch.minimum(direct_error, flipped_error)
    has_g05 = g05_mask.sum(dim=(1, 2)) > 0
    return torch.where(has_g05, direct_error, invariant_error).mean()


def calculate_losses(
    position_prediction: torch.Tensor,
    charge_prediction: torch.Tensor,
    position_target: torch.Tensor,
    charge_target: torch.Tensor,
    g05_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    position_loss = nn.functional.mse_loss(position_prediction, position_target)
    charge_loss = samplewise_charge_mse(
        charge_prediction,
        charge_target,
        g05_mask,
    )
    total_loss = (
        POSITION_LOSS_WEIGHT * position_loss
        + CHARGE_LOSS_WEIGHT * charge_loss
    )
    return total_loss, position_loss, charge_loss


def run_epoch(
    model: ChargeNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochLoss:
    is_training = optimizer is not None
    model.train(mode=is_training)
    total_loss_sum = 0.0
    position_loss_sum = 0.0
    charge_loss_sum = 0.0
    sample_count = 0

    for g00, g05, g05_mask, position_target, charge_target in loader:
        g00_device = g00.to(DEVICE, non_blocking=True)
        g05_device = g05.to(DEVICE, non_blocking=True)
        mask_device = g05_mask.to(DEVICE, non_blocking=True)
        position_target_device = position_target.to(DEVICE, non_blocking=True)
        charge_target_device = charge_target.to(DEVICE, non_blocking=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_training):
            position_prediction, charge_prediction = model(
                g00_device,
                g05_device,
                mask_device,
            )
            total_loss, position_loss, charge_loss = calculate_losses(
                position_prediction,
                charge_prediction,
                position_target_device,
                charge_target_device,
                mask_device,
            )
            if optimizer is not None:
                total_loss.backward()
                optimizer.step()
        current_batch_size = g00.shape[0]
        total_loss_sum += total_loss.item() * current_batch_size
        position_loss_sum += position_loss.item() * current_batch_size
        charge_loss_sum += charge_loss.item() * current_batch_size
        sample_count += current_batch_size

    if sample_count == 0:
        raise ValueError("Empty DataLoader")
    return EpochLoss(
        total=total_loss_sum / sample_count,
        position=position_loss_sum / sample_count,
        charge=charge_loss_sum / sample_count,
    )


def fraction_label(g05_fraction: float) -> str:
    return f"{int(round(g05_fraction * 100)):03d}pct"


def checkpoint_metadata(
    arrays: DatasetArrays,
    stats: NormalizationStats,
) -> dict[str, object]:
    return {
        "model_architecture": "capacity-matched masked G00+G05 v2",
        "grid_shape": tuple(arrays.g00.shape[1:]),
        "g05_candidate_count": arrays.g05.shape[1],
        "position_indices": POSITION_INDICES,
        "charge_indices": CHARGE_INDICES,
        "target_fields": TARGET_FIELDS,
        "g05_fields": G05_FIELDS,
        "g00_mean": stats.g00_mean,
        "g00_std": stats.g00_std,
        "g05_value_scale": stats.g05_value_scale,
        "position_mean": stats.position_mean,
        "position_std": stats.position_std,
        "charge_scale": stats.charge_scale,
        "partial_g05_definition": "fixed spatially-balanced nested sensor prefixes",
        "normalization_definition": "common full-candidate train-only RMS scale",
        "charge_loss_definition": "samplewise signed if observed, sign-invariant if missing",
    }


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
    checkpoint_path = CHECKPOINT_DIR / (
        f"g05_{fraction_label(g05_fraction)}_seed_{seed}_best.pt"
    )
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    print("\n" + "=" * 72)
    print(
        f"Training: G05 fraction={g05_fraction:.2f}, "
        f"points={g05_count} | seed={seed}"
    )
    print("=" * 72)
    for epoch in range(1, max_epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer)
        validation_loss = run_epoch(model, validation_loader)
        if not np.isfinite(validation_loss.total):
            raise FloatingPointError(
                f"Non-finite validation loss: fraction={g05_fraction}, "
                f"seed={seed}, epoch={epoch}"
            )
        print(
            f"Epoch {epoch:03d} | "
            f"Train total={train_loss.total:.6f} "
            f"pos={train_loss.position:.6f} q={train_loss.charge:.6f} | "
            f"Val total={validation_loss.total:.6f} "
            f"pos={validation_loss.position:.6f} q={validation_loss.charge:.6f}"
        )
        if validation_loss.total < best_validation_loss:
            best_validation_loss = validation_loss.total
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": {
                        name: value.detach().cpu()
                        for name, value in model.state_dict().items()
                    },
                    "g05_fraction": g05_fraction,
                    "g05_count": g05_count,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_validation_loss,
                    **metadata,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}")
            break

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=False,
        )
    except TypeError:
        # Compatibility with PyTorch releases predating weights_only.
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    return TrainingResult(
        model=model,
        best_validation_loss=best_validation_loss,
        best_epoch=best_epoch,
        seed=seed,
        g05_fraction=g05_fraction,
        g05_count=g05_count,
        checkpoint_path=checkpoint_path,
    )


def align_global_charge_sign(
    prediction: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
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
    charge_predictions: list[np.ndarray] = []
    position_targets: list[np.ndarray] = []
    charge_targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    with torch.inference_mode():
        for g00, g05, g05_mask, position_target, charge_target in test_loader:
            position_prediction, charge_prediction = model(
                g00.to(DEVICE, non_blocking=True),
                g05.to(DEVICE, non_blocking=True),
                g05_mask.to(DEVICE, non_blocking=True),
            )
            position_predictions.append(position_prediction.cpu().numpy())
            charge_predictions.append(charge_prediction.cpu().numpy())
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
    charge_prediction = np.concatenate(charge_predictions) * stats.charge_scale
    charge_target = np.concatenate(charge_targets) * stats.charge_scale
    g05_mask = np.concatenate(masks)
    g05_counts = g05_mask.sum(axis=(1, 2))
    observed = g05_counts > 0

    evaluated_charge_prediction = charge_prediction.copy()
    if np.any(~observed):
        evaluated_charge_prediction[~observed] = align_global_charge_sign(
            charge_prediction[~observed],
            charge_target[~observed],
        )

    position_mae = np.mean(
        np.abs(position_prediction - position_target),
        axis=0,
    )
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
        np.mean(np.abs(np.abs(charge_prediction) - np.abs(charge_target)))
    )
    prediction_sign = np.sign(charge_prediction)
    target_sign = np.sign(charge_target)
    relative_sign_accuracy = float(
        np.mean(
            np.sign(charge_prediction[:, 0] * charge_prediction[:, 1])
            == np.sign(charge_target[:, 0] * charge_target[:, 1])
        )
    )

    if np.any(observed):
        # Absolute accuracy is per charge. Global accuracy uses q1, the
        # deterministic left-most charge, as the orientation anchor.
        absolute_sign_accuracy = float(
            np.mean(prediction_sign[observed] == target_sign[observed])
        )
        global_sign_accuracy = float(
            np.mean(prediction_sign[observed, 0] == target_sign[observed, 0])
        )
        signed_pair_accuracy = float(
            np.mean(
                np.all(
                    prediction_sign[observed] == target_sign[observed],
                    axis=1,
                )
            )
        )
    else:
        absolute_sign_accuracy = None
        global_sign_accuracy = None
        signed_pair_accuracy = None

    return EvaluationResult(
        position_mae=position_mae,
        position_error_1=position_error_1,
        position_error_2=position_error_2,
        charge_mae=charge_mae,
        charge_magnitude_mae=charge_magnitude_mae,
        relative_sign_accuracy=relative_sign_accuracy,
        absolute_sign_accuracy=absolute_sign_accuracy,
        global_sign_accuracy=global_sign_accuracy,
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


def optional_value(value: float | None) -> float | str:
    return "" if value is None else value


def print_evaluation(
    result: EvaluationResult,
    g05_fraction: float,
    seed: int,
) -> None:
    print("\n" + "-" * 72)
    print(f"Test: G05 fraction={g05_fraction:.2f} | seed={seed}")
    print("Position MAE [x1,y1,z1,x2,y2,z2]:", result.position_mae)
    print("Charge 1/2 position error:", result.position_error_1, result.position_error_2)
    print("Charge magnitude MAE:", result.charge_magnitude_mae)
    print("Relative sign accuracy:", result.relative_sign_accuracy)
    print("Absolute/global/pair sign accuracy:", result.absolute_sign_accuracy,
          result.global_sign_accuracy, result.signed_pair_accuracy)


def run_result_row(
    fraction: float,
    training: TrainingResult,
    evaluation: EvaluationResult,
) -> dict[str, float | int | str]:
    return {
        "g05_fraction": fraction,
        "g05_count_per_sample": training.g05_count,
        "seed": training.seed,
        "position_mae_x1": evaluation.position_mae[0],
        "position_mae_y1": evaluation.position_mae[1],
        "position_mae_z1": evaluation.position_mae[2],
        "position_mae_x2": evaluation.position_mae[3],
        "position_mae_y2": evaluation.position_mae[4],
        "position_mae_z2": evaluation.position_mae[5],
        "charge1_position_error": evaluation.position_error_1,
        "charge2_position_error": evaluation.position_error_2,
        "mean_position_error": 0.5
        * (evaluation.position_error_1 + evaluation.position_error_2),
        "charge_mae_q1": evaluation.charge_mae[0],
        "charge_mae_q2": evaluation.charge_mae[1],
        "charge_magnitude_mae": evaluation.charge_magnitude_mae,
        "relative_sign_accuracy": evaluation.relative_sign_accuracy,
        "absolute_sign_accuracy": optional_value(evaluation.absolute_sign_accuracy),
        "global_sign_accuracy": optional_value(evaluation.global_sign_accuracy),
        "signed_pair_accuracy": optional_value(evaluation.signed_pair_accuracy),
        "observed_sample_fraction": evaluation.observed_sample_fraction,
        "best_epoch": training.best_epoch,
        "best_validation_loss": training.best_validation_loss,
        "checkpoint_path": str(training.checkpoint_path),
    }


def save_csv(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    if not rows:
        raise ValueError("Cannot save an empty result table")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scalar_metric(result: EvaluationResult, name: str) -> float:
    if name == "mean_position_error":
        return 0.5 * (result.position_error_1 + result.position_error_2)
    value = getattr(result, name)
    return np.nan if value is None else float(value)


def build_summary_rows(
    results_by_fraction: dict[float, list[EvaluationResult]],
    candidate_count: int,
) -> list[dict[str, float | int | str]]:
    metric_names = (
        "position_error_1",
        "position_error_2",
        "mean_position_error",
        "charge_magnitude_mae",
        "relative_sign_accuracy",
        "absolute_sign_accuracy",
        "global_sign_accuracy",
        "signed_pair_accuracy",
    )
    rows: list[dict[str, float | int | str]] = []
    for fraction in G05_FRACTIONS:
        results = results_by_fraction[fraction]
        row: dict[str, float | int | str] = {
            "g05_fraction": fraction,
            "g05_count_per_sample": g05_count_for_fraction(
                fraction,
                candidate_count,
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
    fractions = np.asarray(G05_FRACTIONS, dtype=np.float64)
    means = np.full_like(fractions, np.nan)
    stds = np.full_like(fractions, np.nan)
    for index, fraction in enumerate(G05_FRACTIONS):
        values = np.asarray(
            [scalar_metric(result, metric_name) for result in results_by_fraction[fraction]],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]
        if values.size:
            means[index] = values.mean()
        if values.size >= 2:
            stds[index] = sample_std(values)
    valid = np.isfinite(means) & np.isfinite(stds)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.errorbar(
        fractions[valid],
        means[valid],
        yerr=stds[valid],
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
        "absolute_sign_accuracy": ("Absolute sign accuracy", "G05 vs absolute sign"),
        "global_sign_accuracy": ("Global sign accuracy", "G05 vs global sign"),
        "relative_sign_accuracy": ("Relative sign accuracy", "G05 vs relative sign"),
        "charge_magnitude_mae": ("Charge magnitude MAE", "G05 vs charge magnitude"),
        "mean_position_error": (
            "Mean Euclidean position error",
            "G05 vs mean position error",
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
        std_definition=np.asarray("sample std (ddof=1) for multi-seed reports"),
    )


def save_canonical_checkpoint(
    training: TrainingResult,
    output_path: Path,
    metadata: dict[str, object],
) -> None:
    torch.save(
        {
            "model_state_dict": {
                name: value.detach().cpu()
                for name, value in training.model.state_dict().items()
            },
            "g05_fraction": training.g05_fraction,
            "g05_count": training.g05_count,
            "seed": training.seed,
            "best_epoch": training.best_epoch,
            "best_validation_loss": training.best_validation_loss,
            **metadata,
        },
        output_path,
    )


def run_smoke_tests(
    datasets_by_fraction: dict[
        float,
        tuple[TensorDataset, TensorDataset, TensorDataset],
    ],
) -> None:
    print("\nRunning forward/loss smoke tests...")
    for fraction in G05_FRACTIONS:
        dataset = datasets_by_fraction[fraction][0]
        sample_count = min(4, len(dataset))
        tensors = [tensor[:sample_count].to(DEVICE) for tensor in dataset.tensors]
        g00, g05, g05_mask, position_target, charge_target = tensors
        set_reproducibility(41)
        model = ChargeNet().to(DEVICE)
        with torch.no_grad():
            position_prediction, charge_prediction = model(g00, g05, g05_mask)
            losses = calculate_losses(
                position_prediction,
                charge_prediction,
                position_target,
                charge_target,
                g05_mask,
            )
        if position_prediction.shape != (sample_count, 6):
            raise RuntimeError(f"Position smoke-test shape error: {fraction}")
        if charge_prediction.shape != (sample_count, 2):
            raise RuntimeError(f"Charge smoke-test shape error: {fraction}")
        if not all(torch.isfinite(loss).item() for loss in losses):
            raise RuntimeError(f"Non-finite smoke-test loss: {fraction}")
        print(f"  G05 fraction={fraction:.2f}: OK")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_number_list(value: str, converter: type) -> tuple:
    return tuple(converter(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the capacity-matched nested partial-G05 experiment."
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


def main() -> None:
    global EXPERIMENT_SEEDS, G05_FRACTIONS
    args = parse_args()
    EXPERIMENT_SEEDS = tuple(args.seeds)
    G05_FRACTIONS = tuple(args.fractions)
    if not EXPERIMENT_SEEDS or not G05_FRACTIONS:
        raise ValueError("At least one seed and fraction are required")
    if tuple(sorted(G05_FRACTIONS)) != G05_FRACTIONS:
        raise ValueError("Fractions must be sorted")

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
    print("Partial G05: fixed spatially-balanced nested sensor prefixes")

    datasets_by_fraction: dict[
        float,
        tuple[TensorDataset, TensorDataset, TensorDataset],
    ] = {}
    for fraction in G05_FRACTIONS:
        datasets_by_fraction[fraction] = (
            prepare_dataset(arrays, data_split.train, stats, fraction),
            prepare_dataset(arrays, data_split.validation, stats, fraction),
            prepare_dataset(arrays, data_split.test, stats, fraction),
        )
        count = g05_count_for_fraction(fraction, arrays.g05.shape[1])
        print(f"G05 fraction={fraction:.2f}: {count} points for every sample")

    run_smoke_tests(datasets_by_fraction)
    if args.smoke_only:
        print("Smoke-only run complete")
        return

    results_by_fraction: dict[float, list[EvaluationResult]] = {
        fraction: [] for fraction in G05_FRACTIONS
    }
    run_rows: list[dict[str, float | int | str]] = []
    best_by_fraction: dict[float, TrainingResult] = {}

    for fraction in G05_FRACTIONS:
        train_dataset, validation_dataset, test_dataset = datasets_by_fraction[fraction]
        g05_count = g05_count_for_fraction(fraction, arrays.g05.shape[1])
        for seed in EXPERIMENT_SEEDS:
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
            run_rows.append(run_result_row(fraction, training, evaluation))
            print_evaluation(evaluation, fraction, seed)
            current_best = best_by_fraction.get(fraction)
            if (
                current_best is None
                or training.best_validation_loss < current_best.best_validation_loss
            ):
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
    print("Checkpoints:", CHECKPOINT_DIR)
    print("Plots:", RESULTS_DIR)


if __name__ == "__main__":
    main()

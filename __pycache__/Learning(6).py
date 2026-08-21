from __future__ import annotations

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


# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "charge_dataset.npz"
MODEL_G00_PATH = BASE_DIR / "model_G00_only.pt"
MODEL_G00_G05_PATH = BASE_DIR / "model_G00_G05.pt"
NORMALIZATION_PATH = BASE_DIR / "normalization_stats.npz"
CHECKPOINT_DIR = BASE_DIR / "g05_fraction_checkpoints"
CSV_PATH = BASE_DIR / "g05_fraction_results.csv"
PLOT_PATHS = {
    "absolute_sign_accuracy": BASE_DIR / "g05_fraction_absolute_sign_accuracy.png",
    "relative_sign_accuracy": BASE_DIR / "g05_fraction_relative_sign_accuracy.png",
    "charge_magnitude_mae": BASE_DIR / "g05_fraction_charge_magnitude_mae.png",
    "mean_position_error": BASE_DIR / "g05_fraction_mean_position_error.png",
}


# Experiment settings
DATA_SPLIT_SEED = 42
G05_COVERAGE_SEED = 42
EXPERIMENT_SEEDS = (41, 42, 43)
G05_FRACTIONS = (0.00, 0.10, 0.25, 0.50, 0.75, 1.00)
BATCH_SIZE = 128
EPOCHS = 300
PATIENCE = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
POSITION_LOSS_WEIGHT = 1.0
CHARGE_LOSS_WEIGHT = 1.0
EPSILON = 1e-8
POSITION_INDICES = (0, 1, 2, 4, 5, 6)
CHARGE_INDICES = (3, 7)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class DatasetArrays:
    g00: np.ndarray
    g05: np.ndarray
    target: np.ndarray


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


def set_reproducibility(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_dataset(path: Path) -> DatasetArrays:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with np.load(path) as archive:
        g00 = archive["G00"].astype(np.float32)
        g05 = archive["G05"].astype(np.float32)
        target = archive["target"].astype(np.float32)
    validate_dataset(g00, g05, target)
    return DatasetArrays(g00=g00, g05=g05, target=target)


def validate_dataset(
    g00: np.ndarray,
    g05: np.ndarray,
    target: np.ndarray,
) -> None:
    if g00.ndim != 3:
        raise ValueError(f"Invalid G00 shape: {g00.shape}")
    if g05.ndim != 3 or g05.shape[-1] != 3:
        raise ValueError(f"Invalid G05 shape: {g05.shape}")
    if target.ndim != 2 or target.shape[-1] != 8:
        raise ValueError(f"Invalid target shape: {target.shape}")
    sample_count = g00.shape[0]
    if sample_count < 10:
        raise ValueError("At least 10 samples are required")
    if g05.shape[0] != sample_count or target.shape[0] != sample_count:
        raise ValueError("G00, G05, and target sample counts differ")
    if g00.shape[1] < 4 or g00.shape[2] < 4:
        raise ValueError(f"G00 grid is too small: {g00.shape[1:]}")
    if g05.shape[1] != 1:
        raise ValueError(
            "This experiment expects exactly one G05 observation per sample; "
            f"received {g05.shape[1]}"
        )
    if not np.isfinite(g00).all():
        raise ValueError("G00 contains non-finite values")
    if not np.isfinite(g05).all():
        raise ValueError("G05 contains non-finite values")
    if not np.isfinite(target).all():
        raise ValueError("target contains non-finite values")

    grid_height, grid_width = g00.shape[1:]
    x_index = g05[:, :, 0]
    y_index = g05[:, :, 1]
    if not np.allclose(x_index, np.rint(x_index)):
        raise ValueError("G05 x coordinates are not integer grid indices")
    if not np.allclose(y_index, np.rint(y_index)):
        raise ValueError("G05 y coordinates are not integer grid indices")
    if np.any((x_index < 0) | (x_index >= grid_width)):
        raise ValueError("G05 x index is out of range")
    if np.any((y_index < 0) | (y_index >= grid_height)):
        raise ValueError("G05 y index is out of range")


def create_data_split(sample_count: int, seed: int) -> DataSplit:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(sample_count)
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
        g05_value_scale=float(np.sqrt(np.mean(train_g05_values**2))) + EPSILON,
        position_mean=train_positions.mean(axis=0),
        position_std=train_positions.std(axis=0) + EPSILON,
        charge_scale=float(np.sqrt(np.mean(train_charges**2))) + EPSILON,
    )


def create_g05_coverage_ranks(
    grid_height: int,
    grid_width: int,
    seed: int,
) -> np.ndarray:
    """Assign a deterministic, nested sensor-density rank to every grid cell."""
    cell_count = grid_height * grid_width
    permutation = np.random.default_rng(seed).permutation(cell_count)
    ranks = np.empty(cell_count, dtype=np.int64)
    ranks[permutation] = np.arange(cell_count, dtype=np.int64)
    return ranks.reshape(grid_height, grid_width)


def create_g05_observation_mask(
    arrays: DatasetArrays,
    indices: np.ndarray,
    g05_fraction: float,
    coverage_ranks: np.ndarray,
) -> np.ndarray:
    """Keep complete [ix, iy, V] observations only at enabled sensor cells.

    G05 has one observation per sample, not three interchangeable features.
    Fractions therefore control the density of observable grid locations. The
    fixed cell ranking makes lower-fraction sensor sets strict subsets of
    higher-fraction sets.
    """
    if not 0.0 <= g05_fraction <= 1.0:
        raise ValueError(f"Invalid G05 fraction: {g05_fraction}")
    enabled_cell_count = int(round(g05_fraction * coverage_ranks.size))
    x_index = np.rint(arrays.g05[indices, :, 0]).astype(np.int64)
    y_index = np.rint(arrays.g05[indices, :, 1]).astype(np.int64)
    observed = coverage_ranks[y_index, x_index] < enabled_cell_count
    return observed.astype(np.float32)[:, :, np.newaxis]


def prepare_dataset(
    arrays: DatasetArrays,
    indices: np.ndarray,
    stats: NormalizationStats,
    g05_fraction: float,
    coverage_ranks: np.ndarray,
) -> TensorDataset:
    g00 = (arrays.g00[indices] - stats.g00_mean) / stats.g00_std
    g00 = g00[:, np.newaxis, :, :]
    g05 = arrays.g05[indices].copy()
    grid_height, grid_width = arrays.g00.shape[1:]
    g05[:, :, 0] = 2.0 * g05[:, :, 0] / (grid_width - 1) - 1.0
    g05[:, :, 1] = 2.0 * g05[:, :, 1] / (grid_height - 1) - 1.0
    g05[:, :, 2] = g05[:, :, 2] / stats.g05_value_scale
    g05_mask = create_g05_observation_mask(
        arrays, indices, g05_fraction, coverage_ranks
    )
    positions = arrays.target[indices][:, POSITION_INDICES]
    positions = (positions - stats.position_mean) / stats.position_std
    charges = arrays.target[indices][:, CHARGE_INDICES] / stats.charge_scale
    return TensorDataset(
        torch.from_numpy(np.ascontiguousarray(g00)),
        torch.from_numpy(np.ascontiguousarray(g05)),
        torch.from_numpy(np.ascontiguousarray(g05_mask)),
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
    """Separate position and charge heads over G00 and optional G05 features."""

    def __init__(self, use_g05: bool = True) -> None:
        super().__init__()
        self.use_g05 = use_g05
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
            nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU()
        )
        self.g05_encoder: nn.Module | None = None
        if use_g05:
            self.g05_encoder = nn.Sequential(
                nn.Linear(3, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
            )
        fusion_input_size = 160 if use_g05 else 128
        self.shared_encoder = nn.Sequential(
            nn.Linear(fusion_input_size, 128), nn.ReLU()
        )
        self.position_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 6)
        )
        self.charge_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2)
        )

    def forward(
        self,
        g00: torch.Tensor,
        g05: torch.Tensor | None = None,
        g05_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        g00_features = self.g00_encoder(self.g00_cnn(g00))
        if self.use_g05:
            if g05 is None or g05_mask is None or self.g05_encoder is None:
                raise ValueError("G05 input and observation mask are required")
            if g05.shape[:2] != g05_mask.shape[:2] or g05_mask.shape[-1] != 1:
                raise ValueError(
                    f"G05/mask shape mismatch: {g05.shape}, {g05_mask.shape}"
                )
            point_features = self.g05_encoder(g05) * g05_mask
            observed_count = g05_mask.sum(dim=1).clamp_min(1.0)
            g05_features = point_features.sum(dim=1) / observed_count
            fused_features = torch.cat((g00_features, g05_features), dim=1)
        else:
            fused_features = g00_features
        shared_features = self.shared_encoder(fused_features)
        return (
            self.position_head(shared_features),
            self.charge_head(shared_features),
        )


def global_sign_invariant_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    direct_error = torch.mean((prediction - target) ** 2, dim=1)
    flipped_error = torch.mean((prediction + target) ** 2, dim=1)
    return torch.minimum(direct_error, flipped_error).mean()


def calculate_losses(
    position_prediction: torch.Tensor,
    charge_prediction: torch.Tensor,
    position_target: torch.Tensor,
    charge_target: torch.Tensor,
    use_g05: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    position_loss = nn.functional.mse_loss(position_prediction, position_target)
    if use_g05:
        charge_loss = nn.functional.mse_loss(charge_prediction, charge_target)
    else:
        charge_loss = global_sign_invariant_mse(charge_prediction, charge_target)
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
        g05_device = g05.to(DEVICE, non_blocking=True) if model.use_g05 else None
        mask_device = (
            g05_mask.to(DEVICE, non_blocking=True) if model.use_g05 else None
        )
        position_target_device = position_target.to(DEVICE, non_blocking=True)
        charge_target_device = charge_target.to(DEVICE, non_blocking=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_training):
            position_prediction, charge_prediction = model(
                g00_device, g05_device, mask_device
            )
            total_loss, position_loss, charge_loss = calculate_losses(
                position_prediction,
                charge_prediction,
                position_target_device,
                charge_target_device,
                model.use_g05,
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


def train_model(
    train_dataset: TensorDataset,
    validation_dataset: TensorDataset,
    g05_fraction: float,
    seed: int,
) -> TrainingResult:
    use_g05 = g05_fraction > 0.0
    set_reproducibility(seed)
    train_loader = create_data_loader(train_dataset, shuffle=True, seed=seed)
    validation_loader = create_data_loader(validation_dataset, shuffle=False)
    model = ChargeNet(use_g05=use_g05).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / (
        f"g05_{fraction_label(g05_fraction)}_seed_{seed}_best.pt"
    )
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    print("\n" + "=" * 72)
    print(f"Training: G05 fraction={g05_fraction:.2f} | seed={seed}")
    print("=" * 72)
    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer)
        validation_loss = run_epoch(model, validation_loader)
        if not np.isfinite(validation_loss.total):
            raise FloatingPointError(
                f"Non-finite validation loss: fraction={g05_fraction}, "
                f"seed={seed}, epoch={epoch}"
            )
        print(
            f"Epoch {epoch:02d} | "
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
                    "use_g05": use_g05,
                    "g05_fraction": g05_fraction,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_validation_loss,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}")
            break

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    return TrainingResult(
        model=model,
        best_validation_loss=best_validation_loss,
        best_epoch=best_epoch,
        seed=seed,
        g05_fraction=g05_fraction,
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
    with torch.inference_mode():
        for g00, g05, g05_mask, position_target, charge_target in test_loader:
            position_prediction, charge_prediction = model(
                g00.to(DEVICE, non_blocking=True),
                g05.to(DEVICE, non_blocking=True) if model.use_g05 else None,
                g05_mask.to(DEVICE, non_blocking=True) if model.use_g05 else None,
            )
            position_predictions.append(position_prediction.cpu().numpy())
            charge_predictions.append(charge_prediction.cpu().numpy())
            position_targets.append(position_target.numpy())
            charge_targets.append(charge_target.numpy())

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

    if model.use_g05:
        evaluated_charge_prediction = charge_prediction
        absolute_sign_accuracy = float(
            np.mean(np.sign(charge_prediction) == np.sign(charge_target))
        )
    else:
        evaluated_charge_prediction = align_global_charge_sign(
            charge_prediction, charge_target
        )
        absolute_sign_accuracy = None

    position_mae = np.mean(
        np.abs(position_prediction - position_target), axis=0
    )
    position_error_1 = float(
        np.linalg.norm(
            position_prediction[:, 0:3] - position_target[:, 0:3], axis=1
        ).mean()
    )
    position_error_2 = float(
        np.linalg.norm(
            position_prediction[:, 3:6] - position_target[:, 3:6], axis=1
        ).mean()
    )
    charge_mae = np.mean(
        np.abs(evaluated_charge_prediction - charge_target), axis=0
    )
    charge_magnitude_mae = float(
        np.mean(np.abs(np.abs(charge_prediction) - np.abs(charge_target)))
    )
    predicted_relative_sign = np.sign(
        charge_prediction[:, 0] * charge_prediction[:, 1]
    )
    target_relative_sign = np.sign(charge_target[:, 0] * charge_target[:, 1])
    relative_sign_accuracy = float(
        np.mean(predicted_relative_sign == target_relative_sign)
    )
    return EvaluationResult(
        position_mae=position_mae,
        position_error_1=position_error_1,
        position_error_2=position_error_2,
        charge_mae=charge_mae,
        charge_magnitude_mae=charge_magnitude_mae,
        relative_sign_accuracy=relative_sign_accuracy,
        absolute_sign_accuracy=absolute_sign_accuracy,
    )


def print_evaluation(
    result: EvaluationResult,
    g05_fraction: float,
    seed: int,
) -> None:
    print("\n" + "=" * 72)
    print(f"Test: G05 fraction={g05_fraction:.2f} | seed={seed}")
    print("=" * 72)
    print("Position MAE [x1, y1, z1, x2, y2, z2]:", result.position_mae)
    print("Charge 1 Euclidean position error:", result.position_error_1)
    print("Charge 2 Euclidean position error:", result.position_error_2)
    print("Charge magnitude MAE:", result.charge_magnitude_mae)
    print("Relative sign accuracy:", result.relative_sign_accuracy)
    if result.absolute_sign_accuracy is None:
        print("Absolute sign accuracy: N/A")
    else:
        print("Absolute sign accuracy:", result.absolute_sign_accuracy)


def aggregate_values(
    results: list[EvaluationResult],
) -> dict[str, tuple[np.ndarray | float, np.ndarray | float]]:
    position_mae = np.stack([result.position_mae for result in results])
    position_error_1 = np.array([result.position_error_1 for result in results])
    position_error_2 = np.array([result.position_error_2 for result in results])
    charge_magnitude = np.array(
        [result.charge_magnitude_mae for result in results]
    )
    relative_sign = np.array(
        [result.relative_sign_accuracy for result in results]
    )
    return {
        "position_mae": (position_mae.mean(axis=0), position_mae.std(axis=0)),
        "position_error_1": (position_error_1.mean(), position_error_1.std()),
        "position_error_2": (position_error_2.mean(), position_error_2.std()),
        "charge_magnitude_mae": (
            charge_magnitude.mean(),
            charge_magnitude.std(),
        ),
        "relative_sign_accuracy": (relative_sign.mean(), relative_sign.std()),
    }


def print_fraction_summaries(
    results_by_fraction: dict[float, list[EvaluationResult]],
) -> None:
    print("\n" + "#" * 72)
    print("FINAL TEST-METRIC SUMMARY (mean/std over seeds)")
    print("#" * 72)
    for fraction in G05_FRACTIONS:
        results = results_by_fraction[fraction]
        aggregate = aggregate_values(results)
        print(f"\nG05 fraction: {fraction:.2f}")
        print("Position MAE mean:", aggregate["position_mae"][0])
        print("Position MAE std :", aggregate["position_mae"][1])
        print(
            "Charge 1 position error mean/std:",
            *aggregate["position_error_1"],
        )
        print(
            "Charge 2 position error mean/std:",
            *aggregate["position_error_2"],
        )
        print(
            "Charge magnitude MAE mean/std:",
            *aggregate["charge_magnitude_mae"],
        )
        print(
            "Relative sign accuracy mean/std:",
            *aggregate["relative_sign_accuracy"],
        )
        if fraction == 0.0:
            print("Absolute sign accuracy: N/A")
        else:
            values = np.array(
                [result.absolute_sign_accuracy for result in results],
                dtype=np.float64,
            )
            print(
                "Absolute sign accuracy mean/std:", values.mean(), values.std()
            )


def save_results_csv(
    rows: list[dict[str, float | int | str]],
    output_path: Path,
) -> None:
    fieldnames = [
        "g05_fraction",
        "seed",
        "position_mae_x1",
        "position_mae_y1",
        "position_mae_z1",
        "position_mae_x2",
        "position_mae_y2",
        "position_mae_z2",
        "charge1_position_error",
        "charge2_position_error",
        "mean_position_error",
        "charge_magnitude_mae",
        "relative_sign_accuracy",
        "absolute_sign_accuracy",
        "best_epoch",
        "best_validation_loss",
        "observed_train_fraction",
        "observed_validation_fraction",
        "observed_test_fraction",
        "checkpoint_path",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_metric_plot(
    fractions: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
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
    fractions = np.array(G05_FRACTIONS, dtype=np.float64)
    absolute_mean = np.full_like(fractions, np.nan)
    absolute_std = np.full_like(fractions, np.nan)
    relative_mean = np.empty_like(fractions)
    relative_std = np.empty_like(fractions)
    magnitude_mean = np.empty_like(fractions)
    magnitude_std = np.empty_like(fractions)
    position_mean = np.empty_like(fractions)
    position_std = np.empty_like(fractions)

    for index, fraction in enumerate(G05_FRACTIONS):
        results = results_by_fraction[fraction]
        relative = np.array([r.relative_sign_accuracy for r in results])
        magnitude = np.array([r.charge_magnitude_mae for r in results])
        position = np.array(
            [0.5 * (r.position_error_1 + r.position_error_2) for r in results]
        )
        relative_mean[index], relative_std[index] = relative.mean(), relative.std()
        magnitude_mean[index], magnitude_std[index] = magnitude.mean(), magnitude.std()
        position_mean[index], position_std[index] = position.mean(), position.std()
        if fraction > 0.0:
            absolute = np.array(
                [r.absolute_sign_accuracy for r in results], dtype=np.float64
            )
            absolute_mean[index], absolute_std[index] = (
                absolute.mean(),
                absolute.std(),
            )

    save_metric_plot(
        fractions,
        absolute_mean,
        absolute_std,
        "Absolute sign accuracy",
        "G05 fraction vs absolute sign accuracy",
        PLOT_PATHS["absolute_sign_accuracy"],
    )
    save_metric_plot(
        fractions,
        relative_mean,
        relative_std,
        "Relative sign accuracy",
        "G05 fraction vs relative sign accuracy",
        PLOT_PATHS["relative_sign_accuracy"],
    )
    save_metric_plot(
        fractions,
        magnitude_mean,
        magnitude_std,
        "Charge magnitude MAE",
        "G05 fraction vs charge magnitude MAE",
        PLOT_PATHS["charge_magnitude_mae"],
    )
    save_metric_plot(
        fractions,
        position_mean,
        position_std,
        "Mean Euclidean position error",
        "G05 fraction vs mean position error",
        PLOT_PATHS["mean_position_error"],
    )


def save_model_checkpoint(
    training_result: TrainingResult,
    output_path: Path,
    arrays: DatasetArrays,
    stats: NormalizationStats,
) -> None:
    checkpoint = {
        "model_state_dict": {
            name: parameter.detach().cpu()
            for name, parameter in training_result.model.state_dict().items()
        },
        "use_g05": training_result.model.use_g05,
        "g05_fraction": training_result.g05_fraction,
        "seed": training_result.seed,
        "best_epoch": training_result.best_epoch,
        "best_validation_loss": training_result.best_validation_loss,
        "grid_shape": tuple(arrays.g00.shape[1:]),
        "position_indices": POSITION_INDICES,
        "charge_indices": CHARGE_INDICES,
        "g00_mean": stats.g00_mean,
        "g00_std": stats.g00_std,
        "g05_value_scale": stats.g05_value_scale,
        "position_mean": stats.position_mean,
        "position_std": stats.position_std,
        "charge_scale": stats.charge_scale,
        "g05_partial_definition": "nested deterministic sensor-cell coverage",
        "g05_coverage_seed": G05_COVERAGE_SEED,
    }
    torch.save(checkpoint, output_path)


def save_normalization_stats(stats: NormalizationStats) -> None:
    np.savez(
        NORMALIZATION_PATH,
        G00_mean=stats.g00_mean,
        G00_std=stats.g00_std,
        G05_value_scale=stats.g05_value_scale,
        position_mean=stats.position_mean,
        position_std=stats.position_std,
        charge_scale=stats.charge_scale,
        position_indices=np.array(POSITION_INDICES, dtype=np.int64),
        charge_indices=np.array(CHARGE_INDICES, dtype=np.int64),
    )


def observed_fraction(dataset: TensorDataset) -> float:
    g05_mask = dataset.tensors[2]
    return float(g05_mask.mean().item())


def run_smoke_tests(
    datasets_by_fraction: dict[float, tuple[TensorDataset, TensorDataset, TensorDataset]],
) -> None:
    print("\nRunning forward/loss smoke tests...")
    for fraction in G05_FRACTIONS:
        dataset = datasets_by_fraction[fraction][0]
        sample_count = min(4, len(dataset))
        g00, g05, g05_mask, position_target, charge_target = (
            tensor[:sample_count].to(DEVICE) for tensor in dataset.tensors
        )
        model = ChargeNet(use_g05=fraction > 0.0).to(DEVICE)
        with torch.no_grad():
            position_prediction, charge_prediction = model(
                g00,
                g05 if fraction > 0.0 else None,
                g05_mask if fraction > 0.0 else None,
            )
            losses = calculate_losses(
                position_prediction,
                charge_prediction,
                position_target,
                charge_target,
                fraction > 0.0,
            )
        if position_prediction.shape != (sample_count, 6):
            raise RuntimeError(f"Position smoke-test shape error: {fraction}")
        if charge_prediction.shape != (sample_count, 2):
            raise RuntimeError(f"Charge smoke-test shape error: {fraction}")
        if not all(torch.isfinite(loss).item() for loss in losses):
            raise RuntimeError(f"Non-finite smoke-test loss: {fraction}")
        print(f"  G05 fraction={fraction:.2f}: OK")
        del model, g00, g05, g05_mask, position_target, charge_target
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    set_reproducibility(DATA_SPLIT_SEED)
    print("Device:", DEVICE)
    print("Data:", DATA_PATH)
    arrays = load_dataset(DATA_PATH)
    data_split = create_data_split(arrays.g00.shape[0], DATA_SPLIT_SEED)
    stats = calculate_normalization_stats(arrays, data_split.train)
    grid_height, grid_width = arrays.g00.shape[1:]
    coverage_ranks = create_g05_coverage_ranks(
        grid_height, grid_width, G05_COVERAGE_SEED
    )

    print("G00:", arrays.g00.shape)
    print("G05:", arrays.g05.shape)
    print("G05 semantics: [grid_x_index, grid_y_index, signed potential V]")
    print("Partial G05: nested deterministic coverage of candidate sensor cells")
    print("Target:", arrays.target.shape)
    print(
        "Split sizes:",
        len(data_split.train),
        len(data_split.validation),
        len(data_split.test),
    )

    datasets_by_fraction: dict[
        float, tuple[TensorDataset, TensorDataset, TensorDataset]
    ] = {}
    for fraction in G05_FRACTIONS:
        datasets = (
            prepare_dataset(
                arrays, data_split.train, stats, fraction, coverage_ranks
            ),
            prepare_dataset(
                arrays, data_split.validation, stats, fraction, coverage_ranks
            ),
            prepare_dataset(
                arrays, data_split.test, stats, fraction, coverage_ranks
            ),
        )
        datasets_by_fraction[fraction] = datasets
        print(
            f"G05 fraction={fraction:.2f} realized observation rates "
            f"train/val/test="
            f"{observed_fraction(datasets[0]):.4f}/"
            f"{observed_fraction(datasets[1]):.4f}/"
            f"{observed_fraction(datasets[2]):.4f}"
        )

    run_smoke_tests(datasets_by_fraction)

    results_by_fraction: dict[float, list[EvaluationResult]] = {
        fraction: [] for fraction in G05_FRACTIONS
    }
    structured_results: dict[float, dict[int, EvaluationResult]] = {
        fraction: {} for fraction in G05_FRACTIONS
    }
    csv_rows: list[dict[str, float | int | str]] = []
    best_canonical: dict[float, TrainingResult] = {}

    # Fraction-major order keeps every fraction on the same seed-defined shuffle.
    for fraction in G05_FRACTIONS:
        train_dataset, validation_dataset, test_dataset = (
            datasets_by_fraction[fraction]
        )
        for seed in EXPERIMENT_SEEDS:
            training_result = train_model(
                train_dataset,
                validation_dataset,
                g05_fraction=fraction,
                seed=seed,
            )
            evaluation_result = evaluate_model(
                training_result.model, test_dataset, stats
            )
            results_by_fraction[fraction].append(evaluation_result)
            structured_results[fraction][seed] = evaluation_result
            print_evaluation(evaluation_result, fraction, seed)
            current_best = best_canonical.get(fraction)
            if (
                current_best is None
                or training_result.best_validation_loss
                < current_best.best_validation_loss
            ):
                best_canonical[fraction] = training_result

            csv_rows.append(
                {
                    "g05_fraction": fraction,
                    "seed": seed,
                    "position_mae_x1": evaluation_result.position_mae[0],
                    "position_mae_y1": evaluation_result.position_mae[1],
                    "position_mae_z1": evaluation_result.position_mae[2],
                    "position_mae_x2": evaluation_result.position_mae[3],
                    "position_mae_y2": evaluation_result.position_mae[4],
                    "position_mae_z2": evaluation_result.position_mae[5],
                    "charge1_position_error": evaluation_result.position_error_1,
                    "charge2_position_error": evaluation_result.position_error_2,
                    "mean_position_error": 0.5
                    * (
                        evaluation_result.position_error_1
                        + evaluation_result.position_error_2
                    ),
                    "charge_magnitude_mae": evaluation_result.charge_magnitude_mae,
                    "relative_sign_accuracy": evaluation_result.relative_sign_accuracy,
                    "absolute_sign_accuracy": ""
                    if evaluation_result.absolute_sign_accuracy is None
                    else evaluation_result.absolute_sign_accuracy,
                    "best_epoch": training_result.best_epoch,
                    "best_validation_loss": training_result.best_validation_loss,
                    "observed_train_fraction": observed_fraction(train_dataset),
                    "observed_validation_fraction": observed_fraction(
                        validation_dataset
                    ),
                    "observed_test_fraction": observed_fraction(test_dataset),
                    "checkpoint_path": str(training_result.checkpoint_path),
                }
            )

    print_fraction_summaries(results_by_fraction)
    save_results_csv(csv_rows, CSV_PATH)
    save_summary_plots(results_by_fraction)
    save_model_checkpoint(
        best_canonical[0.0], MODEL_G00_PATH, arrays, stats
    )
    save_model_checkpoint(
        best_canonical[1.0], MODEL_G00_G05_PATH, arrays, stats
    )
    save_normalization_stats(stats)

    print("\nExperiment complete")
    print("Structured fractions:", sorted(structured_results))
    print("CSV:", CSV_PATH)
    for plot_path in PLOT_PATHS.values():
        print("Plot:", plot_path)


if __name__ == "__main__":
    main()

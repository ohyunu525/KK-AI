"""수치를 변경하지 않고 추출한 최종 g05_full_reconstruction 모델.

출처: Codes/ModelExperiment10.py(전체 라우팅을 포함한 RoutedChargeNet)와
Codes/NewLearning9.py(데이터 준비, 5전하 손실, 복호화, 지표).
최종 설정은 Modelexperiment11/FIXED_SEED_RUNS.md에 기록된 다음 값이다.
AdamW(lr=0.001, weight_decay=0.0001), 배치 크기=128, 최대 150 epoch,
구조 dropout=0.2, 네 손실 가중치=1, 두 검증 목표의 patience=20.

이 저장소에는 대회 전용 제출 API가 없다. 따라서 이 파일은 기존
tensor/ModelOutput 및 load_trained_model API를 유지하며, 채점기의
predict() 서명이나 제출 CSV 형식을 가정하지 않는다. NumPy와 PyTorch만
필요하고 다른 프로젝트 Python 파일은 가져오지 않는다. 검증한 환경은
Python 3.12.13, NumPy 2.5.2, PyTorch 2.7.0+cu126이다.

추론(정답 또는 데이터셋 파일 불필요):
    model, stats, checkpoint = load_trained_model(checkpoint_path, device)
    fraction = checkpoint["configuration"]["observation"]["g05_fraction"]
    g00, g05, mask = prepare_inputs(raw_g00, raw_g05, stats, fraction)
    with torch.no_grad():
        output = model(g00.to(device), g05.to(device), mask.to(device))
        charges = denormalize_output(output, stats)  # 배치별 전하 5개의 (x, y, z, q) 값

raw_g00은 float32 [B,H,W]의 V**2이고, raw_g05는 원래 후보 순서를 유지한
float32 [B,K,3]의 (grid_x_index, grid_y_index, 부호 있는 V)이다. 모델은
정규화된 [B,1,H,W], [B,K,3], [B,K,1] 접두사 마스크를 입력받는다. 네 개의
원시 출력과 순서 없는 전하 슬롯 다섯 개는 변경하지 않는다.
fraction=0이면 전체 logit은 0이다. 이때 +1은 절대 부호를 식별한 결과가
아니라 기존의 결정론적 동률 대표값일 뿐이며, 절대 부호 지표는 N/A로 남는다.
정렬, 클리핑, 알 수 없는 정답과의 대응, 앙상블은 추가하지 않는다.

학습된 모델에는 외부의 신뢰할 수 있는 .pt 파일이 필요하며, 가중치를 이
파일에 포함하거나 자동 선택하지 않는다. 제출 추론에는 seed 42의 검증
best_structure만 허용한다. 테스트 점수가 아니라 필요한 관측 조건에 맞는
checkpoint를 선택한다. 예를 들어 기존의 전체 관측 checkpoint는 다음과 같다.
    Modelexperiment11/fraction_sweep_checkpoints/seed42_g05_sweep_dropout020/
    g05_full_reconstruction__g05_100pct__seed_42__2d332d1730d1/best_structure.pt
checkpoint에 저장된 정규화와 fraction을 사용한다. .pt 파일의 이동 또는 이름
변경은 가능하다. 상대 경로는 이 Python 파일이 있는 디렉터리를 기준으로 해석한다.

선택적인 단일 실행 학습도 기존 프로토콜을 유지하며 탐색이나 sweep은 하지 않는다.
    python FinalCode.py --train --fraction 1.0 --output-dir FinalCode_outputs/run42
    python FinalCode.py --evaluate-only --checkpoint path/to/best_structure.pt
학습은 항상 seed 42에서 새로 시작하며, 고정된 80/10/10 분할(seed 42)을 쓰고,
통계는 훈련 분할에서만 적합한다. 변경하지 않은 전체 손실을 최적화하면서 두
검증 목표를 모두 감시한다. 두 목표의 전체 epoch 최적 상태를 저장하되, 최종
테스트 평가는 선택이 끝난 뒤 best_structure만 사용한다.
기존 출력 디렉터리는 재사용하거나 덮어쓰지 않는다. 자동 재개와 실험 집계는
의도적으로 제외했다. JSON 파일에는 이 실행의 설정, 분할, 이력, 선택한
checkpoint, 최종 지표만 기록된다. import할 때는 학습, 데이터 로드, 테스트
평가, 출력 파일 쓰기를 수행하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import random
import time
import uuid
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "Models" / "charge_dataset_5charges_v9.npz"
MODEL_NAME = "g05_full_reconstruction"
CHARGE_COUNT = 5
TRAINING_SEED = 42
DATA_SPLIT_SEED = 42
EPSILON = 1e-8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROTOCOL_VERSION = "new-learning10-g05-routing-v1"
CHECKPOINT_SCHEMA_VERSION = 2
CHECKPOINT_SELECTIONS = ("total", "structure")
# 데이터 파일의 위치와 무관하게 내용으로 고정 데이터셋을 식별한다.
DATA_SHA256 = "f90880bbbcd31c528c0603a2aa8043c0ddea567c393d83fc59ae3afd3ac24863"
TARGET_FIELDS = tuple(
    f"{field}{index}" for index in range(1, CHARGE_COUNT + 1)
    for field in ("x", "y", "z", "q")
)
G05_FIELDS = ("grid_x_index", "grid_y_index", "signed_potential")
STRUCTURE_METRIC_NAMES = (
    "position_mae_x", "position_mae_y", "position_mae_z", "mean_position_mae",
    "mean_position_3d_error", "charge_magnitude_mae", "relative_sign_accuracy",
    "relative_configuration_accuracy", "pairwise_relative_sign_accuracy",
)
METRIC_NAMES = STRUCTURE_METRIC_NAMES + (
    "global_sign_accuracy", "global_sign_bce", "absolute_sign_accuracy",
    "absolute_sign_set_accuracy", "charge_mae", "global_invariant_charge_mae",
)
INPUT_POLICY = {
    "position": "G00 + masked even G05 summary",
    "magnitude": "G00 + masked even G05 summary",
    "relative_sign": "G00 + masked even G05 summary",
    "global_sign": "masked G05 only; odd in V",
}


@dataclass(frozen=True)
class DatasetArrays:
    g00: np.ndarray
    g05: np.ndarray
    target: np.ndarray  # [샘플, 순서 없는 전하, (x, y, z, q)]
    grid_x: np.ndarray
    grid_y: np.ndarray
    epsilon_0: float = 1.0
    target_fields: tuple[str, ...] | None = None
    g05_fields: tuple[str, ...] | None = None


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
    position_mean: np.ndarray  # [3], 다섯 전하 슬롯 전체가 공유하는 축별 평균
    position_std: np.ndarray   # [3], 다섯 전하 슬롯 전체가 공유하는 축별 표준편차
    charge_scale: float

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["position_mean"] = self.position_mean.tolist()
        values["position_std"] = self.position_std.tolist()
        return values


@dataclass(frozen=True)
class ModelOutput:
    position: torch.Tensor              # [B, 5, 3], 정규화된 위치
    magnitude: torch.Tensor             # [B, 5], 정규화된 |q|
    relative_sign_logit: torch.Tensor   # [B, 5], 복호화 부호의 곱이 +1
    global_sign_logit: torch.Tensor     # [B], 전하 곱의 부호에 대한 로짓


@dataclass(frozen=True)
class LossWeights:
    position: float = 1.0
    magnitude: float = 1.0
    relative_sign: float = 1.0
    global_sign: float = 1.0


@dataclass(frozen=True)
class BatchLoss:
    total: torch.Tensor
    structure: torch.Tensor
    position: torch.Tensor
    magnitude: torch.Tensor
    relative_sign: torch.Tensor
    global_sign: torch.Tensor | None
    assignment: torch.Tensor  # 예측 슬롯 -> 정답 슬롯의 일대일 대응


@dataclass(frozen=True)
class EpochLoss:
    total: float
    structure: float
    position: float
    magnitude: float
    relative_sign: float
    global_sign: float | None


@dataclass(frozen=True)
class TrainingSettings:
    max_epochs: int = 150
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


@dataclass(frozen=True)
class RegularizationSettings:
    """확정된 dropout 값과 두 목표 기준의 조기 종료 설정을 담는다."""

    structure_dropout: float = 0.2
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.0
    early_stopping_min_epochs: int = 0

    def __post_init__(self) -> None:
        for name in ("structure_dropout", "early_stopping_min_delta"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.structure_dropout < 1:
            raise ValueError("structure_dropout must be in [0, 1)")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be nonnegative")
        for name in ("early_stopping_patience", "early_stopping_min_epochs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")


class DualObjectiveEarlyStopping:
    """검증 구조 손실 또는 전체 손실 중 하나라도 개선되면 대기 횟수를 초기화한다."""

    def __init__(self, settings: RegularizationSettings) -> None:
        self.settings = settings
        self.epoch = 0
        self.best_losses: dict[str, float | None] = dict.fromkeys(CHECKPOINT_SELECTIONS)
        self.last_improvement_epoch = 0
        self.bad_epochs = 0
        self.stopped = False

    def update(self, epoch: int, validation: EpochLoss | Mapping[str, Any]) -> bool:
        if self.stopped or type(epoch) is not int or epoch != self.epoch + 1:
            raise RuntimeError("Early stopping requires consecutive epochs and cannot continue after stopping")
        values = asdict(validation) if isinstance(validation, EpochLoss) else validation
        # 먼저 두 값을 모두 검증한다. 두 번째 값이 NaN일 때 첫 번째 최솟값만
        # 갱신된 불완전 상태가 남지 않게 하기 위한 순서다.
        scores = {name: values[name] for name in CHECKPOINT_SELECTIONS}
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in scores.values()):
            raise FloatingPointError("Early stopping requires finite validation total and structure losses")
        improved = False
        for name, score in scores.items():
            previous = self.best_losses[name]
            if previous is None or score < previous - self.settings.early_stopping_min_delta:
                # min_delta>0이면 유의미했던 최솟값을 기준으로 누적 개선을 비교한다.
                # 작은 개선마다 기준을 움직여 patience가 영원히 리셋되지 않는 오류를 피한다.
                self.best_losses[name] = float(score)
                improved = True
        self.epoch = epoch
        if improved:
            self.last_improvement_epoch, self.bad_epochs = epoch, 0
        else:
            self.bad_epochs += 1
        self.stopped = bool(self.settings.early_stopping_patience > 0
                            and epoch >= self.settings.early_stopping_min_epochs
                            and self.bad_epochs >= self.settings.early_stopping_patience)
        return self.stopped

    def state_dict(self) -> dict[str, Any]:
        return {"epoch": self.epoch, "best_losses": dict(self.best_losses),
                "last_improvement_epoch": self.last_improvement_epoch,
                "bad_epochs": self.bad_epochs, "stopped": self.stopped}



def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_dataset(path: Path) -> DatasetArrays:
    """5전하 데이터와 메타데이터를 읽고 형상·물리 일관성을 함께 검증한다.

    이후 학습 코드가 G00=V², G05=V, 전하 수 5개라는 전제를 안전하게 사용할 수
    있도록, 단순한 파일 존재 여부가 아니라 배열 형식과 일부 샘플의 물리식까지
    확인한다. 잘못된 2전하 데이터나 다른 생성 규칙의 파일을 조기에 막는다.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Five-charge dataset not found: {path}. "
            "Supply the existing five-charge NPZ with --data; this file does not generate data."
        )
    with np.load(path, allow_pickle=False) as archive:
        missing = {"G00", "G05", "target"}.difference(archive.files)
        if missing:
            raise ValueError(f"Missing dataset keys: {sorted(missing)}")
        g00 = archive["G00"].astype(np.float32)
        if g00.ndim != 3:
            raise ValueError(f"G00 must have shape [N,H,W], received {g00.shape}")
        target = archive["target"].astype(np.float32)
        if target.ndim == 2 and target.shape[1] == CHARGE_COUNT * 4:
            target = target.reshape(-1, CHARGE_COUNT, 4)
        if "charge_count" in archive and int(archive["charge_count"].item()) != CHARGE_COUNT:
            raise ValueError("FinalCode requires exactly five charges per sample")
        arrays = DatasetArrays(
            g00=g00,
            g05=archive["G05"].astype(np.float32),
            target=target,
            grid_x=(archive["grid_x"].astype(np.float64) if "grid_x" in archive
                    else np.linspace(-2.0, 2.0, g00.shape[2])),
            grid_y=(archive["grid_y"].astype(np.float64) if "grid_y" in archive
                    else np.linspace(-2.0, 2.0, g00.shape[1])),
            epsilon_0=float(archive["epsilon_0"].item()) if "epsilon_0" in archive else 1.0,
            target_fields=tuple(archive["target_fields"].tolist()) if "target_fields" in archive else None,
            g05_fields=tuple(archive["g05_fields"].tolist()) if "g05_fields" in archive else None,
        )
    validate_dataset(arrays)
    verify_physical_consistency(arrays)
    return arrays


def validate_dataset(arrays: DatasetArrays) -> None:
    g00, g05, target = arrays.g00, arrays.g05, arrays.target
    if g00.ndim != 3 or min(g00.shape[1:]) < 4:
        raise ValueError(f"Invalid G00 shape: {g00.shape}; grid must be at least 4x4")
    if g05.ndim != 3 or g05.shape[1] < 1 or g05.shape[2] != 3:
        raise ValueError(f"G05 must have shape [N,K,3] with K >= 1, received {g05.shape}")
    if target.ndim != 3 or target.shape[1:] != (CHARGE_COUNT, 4):
        raise ValueError(
            f"FinalCode needs five charges: target [N,20] or [N,5,4], "
            f"received {target.shape}. Do not use the old two-charge dataset."
        )
    if not (g00.shape[0] == g05.shape[0] == target.shape[0]) or g00.shape[0] < 10:
        raise ValueError("G00/G05/target must have the same sample count, at least 10")
    if arrays.target_fields is not None and arrays.target_fields != TARGET_FIELDS:
        raise ValueError(f"Expected target fields {TARGET_FIELDS}, got {arrays.target_fields}")
    if arrays.g05_fields is not None and arrays.g05_fields != G05_FIELDS:
        raise ValueError(f"Expected G05 fields {G05_FIELDS}, got {arrays.g05_fields}")
    if not all(np.isfinite(a).all() for a in (g00, g05, target, arrays.grid_x, arrays.grid_y)):
        raise ValueError("Dataset contains non-finite values")
    if not math.isfinite(arrays.epsilon_0) or arrays.epsilon_0 <= 0:
        raise ValueError("epsilon_0 must be finite and positive")
    if arrays.grid_x.shape != (g00.shape[2],) or arrays.grid_y.shape != (g00.shape[1],):
        raise ValueError("Grid metadata does not match G00")
    if np.any(np.diff(arrays.grid_x) <= 0) or np.any(np.diff(arrays.grid_y) <= 0):
        raise ValueError("Grid axes must be strictly increasing")
    if np.any(g00 < 0):
        raise ValueError("G00 is V**2 and cannot be negative")
    if np.any(target[:, :, 2] <= 0):
        raise ValueError("Use the upper-half-space convention z > 0 to remove mirror ambiguity")
    if np.any(np.abs(target[:, :, 3]) <= EPSILON):
        raise ValueError("All five charges must be nonzero for the canonical sign targets")
    coordinates = g05[:, :, :2]
    if not np.array_equal(coordinates, np.rint(coordinates)):
        raise ValueError("G05 coordinates must be integer grid indices")
    if np.any((coordinates[:, :, 0] < 0) | (coordinates[:, :, 0] >= g00.shape[2])) or np.any(
        (coordinates[:, :, 1] < 0) | (coordinates[:, :, 1] >= g00.shape[1])
    ):
        raise ValueError("G05 coordinate is outside the grid")
    if not np.all(coordinates == coordinates[:1]):
        raise ValueError("Candidate G05 locations must be fixed across samples")
    if np.unique(coordinates[0], axis=0).shape[0] != g05.shape[1]:
        raise ValueError("Candidate G05 locations contain duplicates")


def verify_physical_consistency(arrays: DatasetArrays, sample_count: int = 16) -> None:
    grid_x, grid_y = np.meshgrid(arrays.grid_x, arrays.grid_y)
    indices = np.linspace(0, len(arrays.target) - 1, min(sample_count, len(arrays.target)), dtype=int)
    for index in indices:
        charges = arrays.target[index].astype(np.float64)
        distance = np.sqrt(
            (grid_x[..., None] - charges[:, 0]) ** 2
            + (grid_y[..., None] - charges[:, 1]) ** 2
            + charges[:, 2] ** 2
        )
        potential = np.sum(charges[:, 3] / (4 * np.pi * arrays.epsilon_0 * distance), axis=-1)
        if not np.allclose(potential**2, arrays.g00[index], rtol=2e-4, atol=2e-6):
            raise ValueError(f"G00 != V**2 for five-charge target at sample {index}")
        x = arrays.g05[index, :, 0].astype(np.int64)
        y = arrays.g05[index, :, 1].astype(np.int64)
        if not np.allclose(potential[y, x], arrays.g05[index, :, 2], rtol=2e-4, atol=2e-6):
            raise ValueError(f"G05 != signed V at sample {index}")


def create_data_split(sample_count: int, seed: int = DATA_SPLIT_SEED) -> DataSplit:
    if sample_count < 10:
        raise ValueError("At least 10 samples are required for the 80/10/10 split")
    # 하나의 고정 순열을 80/10/10으로 자르면, 재실행과 모든 G05 비율에서 같은
    # 훈련·검증·시험 샘플을 사용하게 된다.
    indices = np.random.default_rng(seed).permutation(sample_count)
    return DataSplit(
        train=indices[:int(sample_count * 0.8)],
        validation=indices[int(sample_count * 0.8):int(sample_count * 0.9)],
        test=indices[int(sample_count * 0.9):],
    )


def calculate_normalization_stats(arrays: DatasetArrays, train_indices: np.ndarray) -> NormalizationStats:
    if len(train_indices) == 0:
        raise ValueError("Cannot fit normalization on an empty training split")
    # 정규화 통계는 반드시 훈련 분할만으로 맞춘다. 특히 위치 통계는 전하 번호별
    # 다섯 세트를 합쳐 축마다 하나씩 계산하므로, 임의의 슬롯 순서에 의존하지 않는다.
    positions = arrays.target[train_indices, :, :3].astype(np.float64)
    charges = arrays.target[train_indices, :, 3].astype(np.float64)
    g00 = arrays.g00[train_indices].astype(np.float64)
    g05 = arrays.g05[train_indices, :, 2].astype(np.float64)
    return NormalizationStats(
        g00_mean=float(g00.mean()),
        g00_std=max(float(g00.std()), EPSILON),
        g05_value_scale=max(float(np.sqrt(np.mean(g05**2))), EPSILON),
        position_mean=positions.mean(axis=(0, 1)).astype(np.float32),
        position_std=np.maximum(positions.std(axis=(0, 1)), EPSILON).astype(np.float32),
        charge_scale=max(float(np.sqrt(np.mean(charges**2))), EPSILON),
    )


def g05_count_for_fraction(fraction: float, candidate_count: int) -> int:
    if not math.isfinite(fraction) or not 0 <= fraction <= 1 or candidate_count < 1:
        raise ValueError("G05 fraction must be in [0,1] and candidate_count must be positive")
    return 0 if fraction == 0 else max(1, int(round(fraction * candidate_count)))


def create_g05_mask(sample_count: int, candidate_count: int, fraction: float) -> np.ndarray:
    mask = np.zeros((sample_count, candidate_count, 1), dtype=np.float32)
    # 모든 샘플에서 같은 후보 목록의 앞부분을 사용한다. 비율이 커질수록 이전
    # 관측을 포함하는 중첩된 앞부분이므로 G05 조건 간 비교가 공정하다.
    mask[:, :g05_count_for_fraction(fraction, candidate_count), 0] = 1
    return mask


def prepare_dataset(
    arrays: DatasetArrays, indices: np.ndarray, stats: NormalizationStats, fraction: float,
) -> TensorDataset:
    """원시 배열을 모델 입력/정답 tensor로 바꾼다.

    G00은 채널 하나짜리 2D CNN 입력으로 정규화하고, G05의 격자 인덱스는 [-1, 1]
    좌표로 바꾼다. 위치와 전하도 훈련 분할 통계로만 정규화하며, 마스크는 G05
    값 자체를 지우지 않고 모델이 관측 여부를 판단하도록 별도로 전달한다.
    """
    g00 = ((arrays.g00[indices] - stats.g00_mean) / stats.g00_std)[:, None]
    g05 = arrays.g05[indices].copy()
    g05[:, :, 0] = 2 * g05[:, :, 0] / (arrays.g00.shape[2] - 1) - 1
    g05[:, :, 1] = 2 * g05[:, :, 1] / (arrays.g00.shape[1] - 1) - 1
    g05[:, :, 2] /= stats.g05_value_scale
    mask = create_g05_mask(len(indices), g05.shape[1], fraction)
    positions = (arrays.target[indices, :, :3] - stats.position_mean) / stats.position_std
    charges = arrays.target[indices, :, 3] / stats.charge_scale
    return TensorDataset(*(torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
                           for a in (g00, g05, mask, positions, charges)))


def create_data_loader(
    dataset: TensorDataset, batch_size: int, *, shuffle: bool = False,
    seed: int = DATA_SPLIT_SEED, device: torch.device = DEVICE,
) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


class SpatialAveragePool(nn.Module):
    """비결정적 CUDA adaptive-pool 역전파 없이 4×4 평균 풀링을 수행한다."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        height, width = features.shape[-2:]
        if height % 4 == 0 and width % 4 == 0:
            return F.avg_pool2d(features, kernel_size=(height // 4, width // 4))
        # 다른 격자 크기에서도 적응형 풀링의 바닥/천장 경계 구간 정의를 따른다.
        # 각 평균의 역전파는 결정론적이며, 겹치는 구간의 기울기는 그래프 순서로 더해진다.
        rows = []
        for row in range(4):
            top, bottom = row * height // 4, ((row + 1) * height + 3) // 4
            columns = []
            for column in range(4):
                left, right = column * width // 4, ((column + 1) * width + 3) // 4
                columns.append(features[:, :, top:bottom, left:right].mean(dim=(-2, -1)))
            rows.append(torch.stack(columns, dim=-1))
        return torch.stack(rows, dim=-2)


class RoutedChargeNet(nn.Module):
    """전체 복원 전용 모델로, G00과 짝대칭 G05 구조 특징 및 홀대칭 G05 전체 부호를 사용한다.

    공유 G05 인코더는 구조 헤드와 전체 부호 헤드 모두에서 기울기를 받는다.
    매개변수 이름·순서·형상·초기화 방식·forward 연산은 확정된 전체 모델
    (매개변수 406,969개)과 같고, dropout은 0.2로 고정한다.
    """

    def __init__(self) -> None:
        super().__init__()
        # G00은 전위 제곱의 2차원 격자이므로 3×3 합성곱과 풀링으로 공간 특징을
        # 추출한다. 마지막 4×4 요약은 입력 격자 크기가 달라도 MLP 입력 크기를
        # 고정한다.
        self.g00_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), SpatialAveragePool(),
        )
        # 하나의 G00 임베딩에서 전하 5개에 대한 구조 출력 세 종류를 분리한다.
        # 위치 15개(5×3), 크기 5개, 상대 부호 로짓 5개가 한 슬롯의 대응을 유지한다.
        self.g00_encoder = nn.Sequential(nn.Flatten(), nn.Linear(64 * 4 * 4, 256), nn.ReLU())
        self.position_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 15))
        self.magnitude_head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 5))
        self.relative_sign_head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 5))
        # G05는 (정규화 x, 정규화 y, 부호 있는 V) 후보점 목록이다. 각 점을
        # 다층 퍼셉트론(MLP)으로 인코딩한 후 관측된 점의 통계량만 요약해 전체 부호를 판별한다.
        self.g05_encoder = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.global_sign_head = nn.Sequential(
            nn.Linear(32 * 3, 64), nn.ReLU(), nn.Linear(64, 1, bias=False),
        )
        # 원래 모듈 등록 및 난수 초기화 순서를 그대로 보존한다.
        self.structure_context = nn.Sequential(
            nn.Linear(32 * 3, 128), nn.ReLU(), nn.Linear(128, 256),
        )
        nn.init.zeros_(self.structure_context[-1].weight)
        nn.init.zeros_(self.structure_context[-1].bias)
        self.structure_dropout = nn.Dropout(p=RegularizationSettings().structure_dropout)

    @staticmethod
    def _masked_summary(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """관측된 G05 특징의 평균·최댓값·표준편차를 고정 길이 벡터로 만든다.

        결측값은 모든 통계량에서 제외하고, 관측이 전혀 없으면 영벡터를 반환한다.
        이 규칙 덕분에 G05=0은 임의의 결측 대체값이 아니라 ``부호를 알 수 없음``
        이라는 명시적인 입력 상태가 된다.
        """
        mask = mask.to(features.dtype)
        count = mask.sum(dim=1).clamp_min(1)
        mean = (features * mask).sum(dim=1) / count
        variance = ((features - mean[:, None]) ** 2 * mask).sum(dim=1) / count
        deviation = torch.sqrt(variance.clamp_min(0) + 1e-6)
        has_observation = mask.sum(dim=(1, 2)) > 0
        maximum = features.masked_fill(~mask.bool(), -torch.inf).max(dim=1).values
        maximum = torch.where(has_observation[:, None], maximum, torch.zeros_like(maximum))
        return torch.cat((mean, maximum, deviation), dim=1) * has_observation[:, None]

    def forward_global_sign(self, g05: torch.Tensor, g05_mask: torch.Tensor) -> torch.Tensor:
        """원래의 G05 전용 전체 부호 함수로, V -> -V 변환에 대해 홀대칭이다."""
        if (g05.ndim != 3 or g05.shape[-1] != 3 or g05.shape[1] < 1
                or g05_mask.shape != (*g05.shape[:2], 1)):
            raise ValueError(f"G05/mask shape mismatch: {g05.shape}, {g05_mask.shape}")
        # 인코더 앞에서 마스크를 적용해 결측 자리에 어떤 큰 값이 있어도 특징으로
        # 새어 들어가지 못하게 한다.
        observed_g05 = g05.masked_fill(~g05_mask.bool(), 0)
        reversed_g05 = observed_g05 * observed_g05.new_tensor((1, 1, -1))
        both = torch.cat((observed_g05, reversed_g05), dim=0)
        both_mask = torch.cat((g05_mask, g05_mask), dim=0)
        summary = self._masked_summary(self.g05_encoder(both), both_mask)
        positive_score, negative_score = self.global_sign_head(summary).squeeze(-1).chunk(2)
        # 정확한 물리 대칭: 측정 V를 반전하면 전체 부호 로짓도 반전되어야 한다.
        return (positive_score - negative_score) * 0.5

    def forward(self, g00: torch.Tensor, g05: torch.Tensor, g05_mask: torch.Tensor) -> ModelOutput:
        if (g05.ndim != 3 or g05.shape[-1] != 3 or g05.shape[1] < 1
                or g05_mask.shape != (*g05.shape[:2], 1) or g05.shape[0] != g00.shape[0]):
            raise ValueError(f"G00/G05/mask shape mismatch: {g00.shape}, {g05.shape}, {g05_mask.shape}")
        structure = self.g00_encoder(self.g00_cnn(g00))
        # 원래의 G05 전용 홀함수는 의도적으로 변경하지 않는다.
        global_logit = self.forward_global_sign(g05, g05_mask)
        observed = g05_mask.sum(dim=(1, 2)) > 0
        if torch.any(observed):
            points = g05.masked_fill(~g05_mask.bool(), 0)
            reversed_points = points * points.new_tensor((1, 1, -1))
            summaries = self._masked_summary(
                self.g05_encoder(torch.cat((points, reversed_points), dim=0)),
                torch.cat((g05_mask, g05_mask), dim=0),
            )
            positive, negative = summaries.chunk(2)
            # 구조 출력은 관측 집합 전체의 부호 반전에 짝대칭이고, 전체 부호는 홀대칭이어야 한다.
            even_summary = (positive + negative) * 0.5
            context = self.structure_context(even_summary) * observed[:, None]
            structure = structure + context
        # 부호 반전 대칭을 만드는 G05의 +V/-V 쌍에는 난수를 넣지 않는다. 대칭적인
        # 구조 특징을 합친 뒤 한 번만 드롭아웃한다. global_logit은 이 연산과 무관하다.
        # eval()에서는 항등 함수이므로 기존 결정론적 추론·전하 순열·짝·홀 대칭을 유지한다.
        # train()에서 독립적으로 두 번 forward하면 마스크가 달라 구조 출력도 다를 수 있다. 학습 중
        # 대칭 검증이 필요하면 같은 RNG 상태(동일 마스크)로 비교해야 한다.
        structure = self.structure_dropout(structure)
        return ModelOutput(
            position=self.position_head(structure).reshape(-1, CHARGE_COUNT, 3),
            magnitude=F.softplus(self.magnitude_head(structure)),
            relative_sign_logit=self.relative_sign_head(structure),
            global_sign_logit=global_logit,
        )


@lru_cache(maxsize=8)
def charge_permutations(device: torch.device) -> torch.Tensor:
    return torch.tensor(list(itertools.permutations(range(CHARGE_COUNT))), dtype=torch.long, device=device)


@lru_cache(maxsize=8)
def relative_sign_patterns(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    patterns = [p for p in itertools.product((-1, 1), repeat=CHARGE_COUNT) if math.prod(p) == 1]
    return torch.tensor(patterns, dtype=dtype, device=device)


def canonical_sign_targets(charges: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """순서와 전체 부호 반전에 강한 상대 부호·전체 부호 정답을 만든다.

    g는 다섯 부호의 곱이므로 전하 순서에 무관하다. r_i = sign(q_i)·g는 모든
    전하 부호를 함께 뒤집어도 변하지 않고 그 곱은 +1이다. 따라서 슬롯 하나를
    임의의 기준 전하로 정하지 않아도 5전하 부호 구성을 표현할 수 있다.
    """
    signs = torch.where(charges > 0, torch.ones_like(charges), -torch.ones_like(charges))
    global_sign = signs.prod(dim=1)
    relative_sign = signs * global_sign[:, None]
    return (relative_sign > 0).to(charges.dtype), (global_sign > 0).to(charges.dtype)


def relative_pattern_scores(logits: torch.Tensor) -> torch.Tensor:
    patterns = (relative_sign_patterns(logits.device, logits.dtype) + 1) * 0.5
    return (logits[:, None, :] * patterns[None, :, :]).sum(dim=-1)


def relative_sign_nll(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """곱이 +1인 16개 상대 부호 패턴에 대한 샘플별 조건부 NLL/5를 계산한다."""
    patterns = (relative_sign_patterns(logits.device, logits.dtype) + 1) * 0.5
    target_index = (target[:, None, :] == patterns[None, :, :]).all(dim=-1).long().argmax(dim=1)
    return F.cross_entropy(relative_pattern_scores(logits), target_index, reduction="none") / CHARGE_COUNT


def decode_relative_signs(logits: torch.Tensor) -> torch.Tensor:
    """가능한 상대 부호 패턴 중 우도가 가장 큰 것을 복호화한다.

    logit을 각 전하마다 독립 임계값으로 자르면 부호 곱이 -1이 될 수 있다. 여기서는
    16개 유효 조합만 비교하므로 항상 물리적으로 일관된 다섯 부호를 반환한다.
    """
    index = relative_pattern_scores(logits).argmax(dim=1)
    return relative_sign_patterns(logits.device, logits.dtype)[index]


def reconstruct_charges(output: ModelOutput) -> torch.Tensor:
    """예측 슬롯 순서의 정규화된 전하를 재구성한다.

    G05가 없는 경우 전체 부호는 원리적으로 식별할 수 없으므로 logit 0의 tie는
    +1 대표값으로 정한다. 이는 절대 부호를 알아냈다는 뜻이 아니다.
    """
    global_sign = torch.where(output.global_sign_logit >= 0, 1.0, -1.0)
    return output.magnitude * decode_relative_signs(output.relative_sign_logit) * global_sign[:, None]


@torch.no_grad()
def minimum_cost_assignment(pair_cost: torch.Tensor) -> torch.Tensor:
    """샘플별로 정답을 한 번씩만 쓰는 정확한 일대일 대응을 찾는다.

    전하가 5개이므로 120개 순열을 모두 열거해도 작고, 계산을 장치 안에서 끝낼
    수 있다. 이는 CPU/SciPy 왕복 없이 Hungarian 할당과 같은 최소 비용 목표를
    푸는 방법이며, 여러 예측이 같은 정답을 고르는 최근접점 손실이 아니다.
    """
    if pair_cost.ndim != 3 or pair_cost.shape[1:] != (CHARGE_COUNT, CHARGE_COUNT):
        raise ValueError(f"Expected pair cost [B,5,5], received {pair_cost.shape}")
    permutations = charge_permutations(pair_cost.device)
    rows = torch.arange(CHARGE_COUNT, device=pair_cost.device)
    costs = pair_cost[:, rows, permutations].sum(dim=-1)
    return permutations[costs.argmin(dim=1)]


def matching_cost(
    output: ModelOutput, position_target: torch.Tensor, charge_target: torch.Tensor,
    weights: LossWeights = LossWeights(),
) -> torch.Tensor:
    relative_target, _ = canonical_sign_targets(charge_target)
    position = (output.position[:, :, None, :] - position_target[:, None, :, :]).square().mean(dim=-1)
    magnitude = (output.magnitude[:, :, None] - charge_target.abs()[:, None, :]).square()
    logits = output.relative_sign_logit[:, :, None]
    relative = F.softplus(logits) - logits * relative_target[:, None, :]
    # 조건부 상대 부호 음의 로그우도(NLL)는 이 이진 교차 엔트로피(BCE) 비용과
    # 샘플별 분할 상수만 다르므로, 이 대응이 실제 구조 손실도 최소화한다.
    # G05 및 전체 부호 비용은 의도적으로 제외한다.
    # 전체 복원에서도 이 구조 전체의 일대일 대응을 그대로 사용한다.
    return weights.position * position + weights.magnitude * magnitude + weights.relative_sign * relative


def matched_targets(
    position: torch.Tensor, charge: torch.Tensor, assignment: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        position.gather(1, assignment[:, :, None].expand(-1, -1, 3)),
        charge.gather(1, assignment),
    )


def calculate_losses(
    output: ModelOutput, position_target: torch.Tensor, charge_target: torch.Tensor,
    g05_mask: torch.Tensor, weights: LossWeights = LossWeights(),
) -> BatchLoss:
    """변경하지 않은 120개 순열 기반 구조 손실과 관측된 G05의 전체 부호 BCE를 계산한다."""
    with torch.no_grad():
        assignment = minimum_cost_assignment(matching_cost(output, position_target, charge_target, weights))
        aligned_position, aligned_charge = matched_targets(position_target, charge_target, assignment)
        relative_target, global_target = canonical_sign_targets(aligned_charge)
    position = F.mse_loss(output.position, aligned_position)
    magnitude = F.mse_loss(output.magnitude, aligned_charge.abs())
    relative = relative_sign_nll(output.relative_sign_logit, relative_target).mean()
    structure = weights.position * position + weights.magnitude * magnitude + weights.relative_sign * relative
    observed = g05_mask.sum(dim=(1, 2)) > 0
    global_loss = (
        F.binary_cross_entropy_with_logits(output.global_sign_logit[observed], global_target[observed])
        if torch.any(observed) else None
    )
    total = structure if global_loss is None else structure + weights.global_sign * global_loss
    return BatchLoss(total, structure, position, magnitude, relative, global_loss, assignment)


def run_epoch(
    model: RoutedChargeNet, loader: DataLoader, optimizer: torch.optim.Optimizer | None = None,
    weights: LossWeights = LossWeights(),
) -> EpochLoss:
    """변경하지 않은 결합 전체 손실로 한 학습/검증 epoch를 수행한다.

    전체 복원 모델은 구조 헤드와 전체 부호 헤드가 G05 인코더를 공유한다.
    기울기 절단, 스케줄러, 증강, 혼합 정밀도는 추가하지 않으며 전체 부호 BCE는
    G05가 관측된 샘플에만 적용한다.
    """
    model.train(optimizer is not None)
    device = next(model.parameters()).device
    sums = dict.fromkeys(("structure", "position", "magnitude", "relative_sign"), 0.0)
    sample_count = observed_count = 0
    global_sum = 0.0
    for batch in loader:
        g00, g05, mask, position, charge = (tensor.to(device, non_blocking=True) for tensor in batch)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            losses = calculate_losses(model(g00, g05, mask), position, charge, mask, weights)
            if not torch.isfinite(losses.total):
                raise FloatingPointError("Non-finite training/validation loss")
            if optimizer is not None:
                losses.total.backward()
                # 확정된 옵티마이저 단계를 유지하며 기울기 절단이나 학습률 스케줄러는 적용하지 않는다.
                optimizer.step()
        count = len(g00)
        sample_count += count
        for name in sums:
            sums[name] += float(getattr(losses, name).detach()) * count
        if losses.global_sign is not None:
            observed = int((mask.sum(dim=(1, 2)) > 0).sum())
            global_sum += float(losses.global_sign.detach()) * observed
            observed_count += observed
    if sample_count == 0:
        raise ValueError("Cannot run an epoch on an empty dataset")
    averages = {name: value / sample_count for name, value in sums.items()}
    global_average = global_sum / observed_count if observed_count else None
    total = averages["structure"] + weights.global_sign * (global_average or 0.0)
    return EpochLoss(total=total, global_sign=global_average, **averages)


@torch.no_grad()
def evaluate_model(
    model: RoutedChargeNet, dataset: TensorDataset, stats: NormalizationStats,
    batch_size: int = 128, weights: LossWeights = LossWeights(),
) -> dict[str, float | None]:
    """모든 전하 필드에 공유되는 하나의 일대일 대응으로 집합 지표를 계산한다.

    G05가 관측된 샘플의 charge_mae는 절대 부호 전하를 사용한다. 미관측 샘플은
    전하 다섯 개를 함께 뒤집은 두 전체 방향 중 더 좋은 경우만 허용하며, 개별
    전하를 따로 뒤집어 오차를 줄이지 않는다. 어느 지표도 고정 슬롯 번호를
    전하의 실제 ID로 해석하지 않는다.
    """
    model.eval()
    device = next(model.parameters()).device
    sums = dict.fromkeys((
        "position_mae_x", "position_mae_y", "position_mae_z", "mean_position_mae",
        "mean_position_3d_error", "charge_magnitude_mae", "relative_sign_accuracy",
        "relative_configuration_accuracy", "pairwise_relative_sign_accuracy",
        "charge_mae", "global_invariant_charge_mae",
    ), 0.0)
    observed_sums = dict.fromkeys((
        "global_sign_accuracy", "global_sign_bce", "absolute_sign_accuracy", "absolute_sign_set_accuracy",
    ), 0.0)
    sample_count = observed_count = 0
    observations = 0.0
    position_std = torch.as_tensor(stats.position_std, device=device)
    pairs = torch.triu_indices(CHARGE_COUNT, CHARGE_COUNT, offset=1, device=device)
    for batch in create_data_loader(dataset, batch_size, device=device):
        g00, g05, mask, position, charge = (t.to(device) for t in batch)
        output = model(g00, g05, mask)
        assignment = minimum_cost_assignment(matching_cost(output, position, charge, weights))
        position, charge = matched_targets(position, charge, assignment)
        relative_target, global_target = canonical_sign_targets(charge)
        relative = decode_relative_signs(output.relative_sign_logit)
        relative_correct = relative == (relative_target * 2 - 1)
        predicted_q = reconstruct_charges(output) * stats.charge_scale
        target_q = charge * stats.charge_scale
        position_error = (output.position - position) * position_std
        absolute_error = position_error.abs()
        direct_mae = (predicted_q - target_q).abs().mean(dim=1)
        flipped_mae = (-predicted_q - target_q).abs().mean(dim=1)
        invariant_mae = torch.minimum(direct_mae, flipped_mae)
        observed = mask.sum(dim=(1, 2)) > 0
        target_relative_sign = relative_target * 2 - 1
        pair_correct = (relative[:, pairs[0]] * relative[:, pairs[1]] ==
                        target_relative_sign[:, pairs[0]] * target_relative_sign[:, pairs[1]])
        values = {
            **{f"position_mae_{axis}": absolute_error[:, :, i].mean(dim=1) for i, axis in enumerate("xyz")},
            "mean_position_mae": absolute_error.mean(dim=(1, 2)),
            "mean_position_3d_error": position_error.norm(dim=-1).mean(dim=1),
            "charge_magnitude_mae": (output.magnitude * stats.charge_scale - target_q.abs()).abs().mean(dim=1),
            "relative_sign_accuracy": relative_correct.float().mean(dim=1),
            "relative_configuration_accuracy": relative_correct.all(dim=1).float(),
            "pairwise_relative_sign_accuracy": pair_correct.float().mean(dim=1),
            "charge_mae": torch.where(observed, direct_mae, invariant_mae),
            "global_invariant_charge_mae": invariant_mae,
        }
        for name, value in values.items():
            sums[name] += float(value.sum())
        if torch.any(observed):
            global_correct = (output.global_sign_logit[observed] >= 0) == global_target[observed].bool()
            absolute_correct = (predicted_q[observed] > 0) == (target_q[observed] > 0)
            observed_sums["global_sign_accuracy"] += float(global_correct.float().sum())
            observed_sums["global_sign_bce"] += float(F.binary_cross_entropy_with_logits(
                output.global_sign_logit[observed], global_target[observed], reduction="sum"))
            observed_sums["absolute_sign_accuracy"] += float(absolute_correct.float().mean(dim=1).sum())
            observed_sums["absolute_sign_set_accuracy"] += float(absolute_correct.all(dim=1).float().sum())
        sample_count += len(g00)
        observed_count += int(observed.sum())
        observations += float(mask.sum())
    if not sample_count:
        raise ValueError("Cannot evaluate an empty dataset")
    return {
        **{name: value / sample_count for name, value in sums.items()},
        **{name: value / observed_count if observed_count else None for name, value in observed_sums.items()},
        "observed_sample_fraction": observed_count / sample_count,
        "observations_per_sample": observations / sample_count,
    }


def prepare_inputs(
    g00: np.ndarray, g05: np.ndarray, stats: NormalizationStats, fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """prepare_dataset과 같은 float32 입력 변환을 정답 없이 수행한다.

    마스킹된 점을 포함해 저장된 모든 G05 후보를 원래 순서로 유지한다. 격자
    인덱스 대신 물리 x/y 좌표를 입력하지 않는다.
    """
    g00 = np.asarray(g00, dtype=np.float32)
    g05 = np.asarray(g05, dtype=np.float32)
    if (g00.ndim != 3 or min(g00.shape[1:]) < 4 or len(g00) < 1
            or g05.ndim != 3 or g05.shape[0] != len(g00)
            or g05.shape[1] < 1 or g05.shape[2] != 3):
        raise ValueError(f"Expected G00 [B,H,W], G05 [B,K,3]; got {g00.shape}, {g05.shape}")
    height, width = g00.shape[1:]
    g00 = ((g00 - stats.g00_mean) / stats.g00_std)[:, None]
    g05 = g05.copy()
    g05[:, :, 0] = 2 * g05[:, :, 0] / (width - 1) - 1
    g05[:, :, 1] = 2 * g05[:, :, 1] / (height - 1) - 1
    g05[:, :, 2] /= stats.g05_value_scale
    mask = create_g05_mask(len(g00), g05.shape[1], fraction)
    return tuple(torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
                 for a in (g00, g05, mask))


def denormalize_output(output: ModelOutput, stats: NormalizationStats) -> torch.Tensor:
    """클리핑 없이 원래의 순서 없는 [B,5,(x,y,z,q)] 표현으로 되돌린다.

    전하 크기는 output.magnitude * stats.charge_scale이다. 상대 부호는
    decode_relative_signs로 복호화하고, 절대 부호에는 reconstruct_charges가
    적용하는 전체 logit의 >=0 동률 규칙도 사용한다. G05가 없으면 절대 방향은
    식별할 수 없으므로, 정답이 있는 손실/지표에는 일대일 대응을 사용하되 예측에는 쓰지 않는다.
    """
    position_std = torch.as_tensor(stats.position_std, device=output.position.device)
    position_mean = torch.as_tensor(stats.position_mean, device=output.position.device)
    position = output.position * position_std + position_mean
    charge = reconstruct_charges(output) * stats.charge_scale
    return torch.cat((position, charge[:, :, None]), dim=-1)


def resolve_path(path: Path | str) -> Path:
    """상대 경로 인수는 현재 작업 디렉터리가 아니라 FinalCode.py 옆에서 해석한다."""
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def require_final_configuration(config: Mapping[str, Any]) -> None:
    """모델, 설정, seed 또는 데이터 프로토콜이 다른 checkpoint를 거부한다."""
    if config.get("model", {}).get("name") != MODEL_NAME or config.get("charge_count") != CHARGE_COUNT:
        raise RuntimeError("Only the finalized g05_full_reconstruction model is supported")
    training = config.get("training", {})
    expected = {
        **asdict(TrainingSettings()), "seed": TRAINING_SEED, "optimizer": "AdamW",
        "loss_weights": asdict(LossWeights()), "regularization": asdict(RegularizationSettings()),
    }
    for key, value in expected.items():
        if training.get(key) != value:
            raise RuntimeError(f"Checkpoint does not use the finalized {key}: expected {value}")
    if config.get("model", {}).get("input_policy") != INPUT_POLICY:
        raise RuntimeError("Checkpoint input routing does not match full reconstruction")
    observation = config.get("observation", {})
    fraction = observation.get("g05_fraction")
    if (fraction not in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
            or observation.get("candidate_count") != 32
            or observation.get("g05_count_per_sample") != g05_count_for_fraction(fraction, 32)
            or observation.get("selection") != "fixed nested sensor prefix"):
        raise RuntimeError("Checkpoint observation condition is not one of the finalized conditions")
    if config.get("split_counts") != {"train": 8000, "validation": 1000, "test": 1000}:
        raise RuntimeError("Checkpoint does not use the fixed 80/10/10 data split")
    stats = normalization_from_config(config)
    if (stats.position_mean.shape != (3,) or stats.position_std.shape != (3,)
            or not np.isfinite(stats.position_mean).all()
            or not np.isfinite(stats.position_std).all() or np.any(stats.position_std <= 0)
            or not math.isfinite(stats.g00_mean)
            or any(not math.isfinite(v) or v <= 0 for v in
                   (stats.g00_std, stats.g05_value_scale, stats.charge_scale))):
        raise RuntimeError("Invalid saved train-only normalization")


def model_from_config(config: Mapping[str, Any]) -> RoutedChargeNet:
    require_final_configuration(config)
    return RoutedChargeNet()


def load_trained_model(
    path: Path | str, device: torch.device = torch.device("cpu"),
) -> tuple[RoutedChargeNet, NormalizationStats, dict[str, Any]]:
    """기존 loader 반환 규약을 유지하며 데이터셋·결과·소스 파일은 필요 없다.

    신뢰할 수 있는 프로젝트 checkpoint만 불러온다. 원래 형식은
    weights_only=False인 torch.load를 사용한다. 가중치를 검색·순위화·병합하지 않는다.
    """
    checkpoint = load_torch_checkpoint(resolve_path(path), device)
    if (checkpoint.get("protocol_version") != PROTOCOL_VERSION
            or checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION
            or not isinstance(checkpoint.get("configuration"), Mapping)):
        raise RuntimeError("Expected a complete five-charge full-reconstruction checkpoint")
    config = checkpoint["configuration"]
    require_final_configuration(config)
    if checkpoint.get("checkpoint_selection") != "structure":
        raise RuntimeError("Submission inference requires seed 42 validation best_structure.pt")
    validate_selected_checkpoint(
        checkpoint, config, selection="structure",
        expected_epoch=checkpoint["selected_epoch"],
        expected_loss=checkpoint["selected_validation_loss"],
    )
    model = model_from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, normalization_from_config(config), checkpoint


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
    """짧게 발생하는 Windows 파일 잠금에도 파일을 원자적으로 교체한다."""

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


def atomic_torch_save(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("wb") as handle:
            torch.save(dict(value), handle)
            handle.flush()
            os.fsync(handle.fileno())
        replace_with_retry(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_torch_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    # 체크포인트에는 CPU 전용 난수 생성기 상태 텐서도 포함된다. 전체 매핑을 CUDA로
    # 불러오면 Generator.set_state()/torch.set_rng_state()가 실패한다.
    # 모델 매개변수는 load_state_dict가 복사한다. AdamW.load_state_dict는 step 카운터를
    # CPU에 유지하면서 모멘트 버퍼를 각 매개변수의 장치로 복원한다.
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint is not a mapping: {path}")
    return checkpoint


def regularization_from_config(config: Mapping[str, Any]) -> RegularizationSettings:
    """저장된 정규화 설정의 모든 필드를 요구하며 checkpoint 검증용으로 복원한다."""
    try:
        values = config["training"]["regularization"]
        if not isinstance(values, Mapping) or set(values) != set(RegularizationSettings.__dataclass_fields__):
            raise ValueError("missing or unknown regularization fields")
        return RegularizationSettings(**values)
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid v10 regularization configuration: {error}") from error


def normalization_from_config(config: Mapping[str, Any]) -> NormalizationStats:
    values = dict(config["normalization"])
    for name in ("position_mean", "position_std"):
        values[name] = np.asarray(values[name], dtype=np.float32)
    return NormalizationStats(**values)


def run_id_for(config: Mapping[str, Any]) -> str:
    return (f"{config['model']['name']}__g05_{round(config['observation']['g05_fraction'] * 100):03d}pct"
            f"__seed_{config['training']['seed']}__{object_fingerprint(config)[:12]}")


def run_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION, "protocol_fingerprint": config["protocol_fingerprint"],
        "run_fingerprint": object_fingerprint(config), "run_id": run_id_for(config),
        "model_name": config["model"]["name"], "charge_count": CHARGE_COUNT,
        "g05_fraction": config["observation"]["g05_fraction"],
        "g05_count_per_sample": config["observation"]["g05_count_per_sample"], "seed": config["training"]["seed"],
    }


def validate_model_state(state: Any, config: Mapping[str, Any]) -> None:
    expected = config["model"]["state_shapes"]
    if not isinstance(state, Mapping) or set(state) != set(expected):
        raise RuntimeError("Checkpoint must contain the complete model state, not a branch/component")
    for name, spec in expected.items():
        value = state[name]
        if (not isinstance(value, torch.Tensor) or list(value.shape) != spec["shape"]
                or str(value.dtype) != spec["dtype"] or not torch.isfinite(value).all()):
            raise RuntimeError(f"Invalid model state tensor: {name}")


def validate_identity(value: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    if config.get("protocol_version") != PROTOCOL_VERSION or config.get("charge_count") != CHARGE_COUNT:
        raise RuntimeError("Configuration is not for this five-charge routing experiment")
    regularization_from_config(config)
    for key, expected in run_metadata(config).items():
        if value.get(key) != expected:
            raise RuntimeError(f"Run metadata mismatch: {key}")
    if canonical_json(value.get("configuration")) != canonical_json(config):
        raise RuntimeError("Run configuration mismatch")


def validate_loss_values(losses: Any, config: Mapping[str, Any], label: str) -> None:
    if not isinstance(losses, Mapping) or set(losses) != set(EpochLoss.__dataclass_fields__):
        raise RuntimeError(f"Incomplete {label}")
    observed = config["observation"]["g05_count_per_sample"] > 0
    for name, value in losses.items():
        if name == "global_sign" and not observed:
            if value is not None:
                raise RuntimeError(f"Unobserved global sign must be N/A in {label}")
        elif value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise RuntimeError(f"Non-finite or missing {name} in {label}")


def update_best_checkpoints(
    best: dict[str, dict[str, Any]], *, config: Mapping[str, Any], epoch: int,
    validation: EpochLoss, model_state: dict[str, torch.Tensor],
) -> tuple[str, ...]:
    changed = []
    for selection in CHECKPOINT_SELECTIONS:
        score = float(getattr(validation, selection))
        if not math.isfinite(score):
            raise FloatingPointError(f"Non-finite validation {selection} loss")
        if selection in best and score >= best[selection]["selected_validation_loss"]:
            continue
        best[selection] = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION, "checkpoint_kind": "selected",
            **run_metadata(config), "configuration": config,
            **config["training"]["checkpoint_selection"][selection],
            "selected_epoch": epoch, "selected_validation_loss": score,
            "validation_losses": asdict(validation), "model_state_dict": model_state,
        }
        changed.append(selection)
    return tuple(changed)


def validate_selected_checkpoint(
    checkpoint: Mapping[str, Any], config: Mapping[str, Any], *, selection: str,
    expected_epoch: int, expected_loss: float,
) -> None:
    validate_identity(checkpoint, config)
    expected = {"checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION, "checkpoint_kind": "selected",
                **config["training"]["checkpoint_selection"][selection],
                "selected_epoch": expected_epoch, "selected_validation_loss": expected_loss}
    for key, value in expected.items():
        if canonical_json(checkpoint.get(key)) != canonical_json(value):
            raise RuntimeError(f"Invalid {selection} checkpoint metadata: {key}")
    if (not 1 <= expected_epoch <= config["training"]["max_epochs"] or not math.isfinite(expected_loss)
            or checkpoint.get("validation_losses", {}).get(selection) != expected_loss):
        raise RuntimeError(f"Invalid {selection} checkpoint epoch/loss")
    validate_loss_values(checkpoint["validation_losses"], config, f"{selection} checkpoint losses")
    validate_model_state(checkpoint.get("model_state_dict"), config)


def _load_fixed_dataset(path: Path | str) -> tuple[DatasetArrays, DataSplit]:
    path = resolve_path(path)
    if file_sha256(path) != DATA_SHA256:
        raise RuntimeError("Training/evaluation requires the unchanged finalized five-charge dataset")
    arrays = load_dataset(path)
    return arrays, create_data_split(len(arrays.target))


def _selection_policy(selection: str, g05_count: int) -> dict[str, Any]:
    return {
        "checkpoint_selection": selection,
        "selection_objective": f"validation_loss.{selection}",
        "selection_note": "One complete epoch state; first epoch wins ties; no component composition",
        "primary_metrics": list(STRUCTURE_METRIC_NAMES if selection == "structure" else METRIC_NAMES),
        "global_sign_in_selection_objective": selection == "total" and g05_count > 0,
        "global_sign_metrics_note": (
            "Global-sign and absolute-sign metrics are secondary diagnostics: structure selection "
            "does not optimize global-sign performance; training still uses the unchanged total loss."
            if selection == "structure" else
            "Total selection includes global-sign loss only with observed G05 and nonzero global-sign weight."
        ),
    }


def _run_configuration(
    model: RoutedChargeNet, arrays: DatasetArrays, split: DataSplit,
    stats: NormalizationStats, fraction: float, device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """실험 레지스트리 없이 단일 이식 가능한 실행의 출처 정보를 저장한다."""
    candidates = arrays.g05.shape[1]
    count = g05_count_for_fraction(fraction, candidates)
    protocol = json_ready({
        "protocol_version": PROTOCOL_VERSION, "data_sha256": DATA_SHA256,
        "source_sha256": file_sha256(Path(__file__)),
        "split_seed": DATA_SPLIT_SEED,
        "split_indices": {name: getattr(split, name) for name in ("train", "validation", "test")},
        "g00_shape": arrays.g00.shape, "g05_shape": arrays.g05.shape,
        "target_fields": TARGET_FIELDS, "g05_fields": G05_FIELDS,
        "normalization": stats.to_dict(),
        "matching": "exact 120-permutation minimum of position MSE + magnitude MSE + sign cost",
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms": "enabled, warn_only=True (unchanged baseline)",
        },
    })
    config = json_ready({
        "protocol_version": PROTOCOL_VERSION, "protocol_fingerprint": object_fingerprint(protocol),
        "charge_count": CHARGE_COUNT,
        "model": {
            "name": MODEL_NAME, "input_policy": INPUT_POLICY,
            "parameter_count": {
                "total": sum(p.numel() for p in model.parameters()),
                "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            },
            "state_shapes": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in model.state_dict().items()
            },
        },
        "observation": {
            "g05_fraction": fraction, "g05_count_per_sample": count, "candidate_count": candidates,
            "selection": "fixed nested sensor prefix",
            "full_fraction_definition": "all stored candidate sensors, not the full 32x32 field",
        },
        "training": {
            **asdict(TrainingSettings()), "seed": TRAINING_SEED, "optimizer": "AdamW",
            "loss_weights": asdict(LossWeights()), "regularization": asdict(RegularizationSettings()),
            "checkpoint_selection": {
                name: _selection_policy(name, count) for name in CHECKPOINT_SELECTIONS
            },
        },
        "normalization": stats.to_dict(),
        "split_counts": {name: len(getattr(split, name)) for name in ("train", "validation", "test")},
    })
    require_final_configuration(config)
    return protocol, config


def train_and_evaluate_run(
    *, fraction: float, output_dir: Path | str,
    data_path: Path | str = DEFAULT_DATA_PATH, device: torch.device = DEVICE,
) -> dict[str, Any]:
    """고정된 seed와 fraction으로 처음부터 학습하고 선택된 모델만 평가한다.

    중단된 실행이 남긴 디렉터리를 포함해 기존 디렉터리는 거부한다. 이 파일은
    이를 알리지 않고 재개·복구·덮어쓰기하지 않는다.
    """
    if fraction not in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        raise ValueError("Specify one finalized fraction: 0, 0.1, 0.25, 0.5, 0.75 or 1")
    output_dir = resolve_path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Use a NEW output directory; existing artifacts are protected: {output_dir}")
    arrays, split = _load_fixed_dataset(data_path)
    stats = calculate_normalization_stats(arrays, split.train)
    train = prepare_dataset(arrays, split.train, stats, fraction)
    validation = prepare_dataset(arrays, split.validation, stats, fraction)

    settings, weights = TrainingSettings(), LossWeights()
    set_reproducibility(TRAINING_SEED)
    model = RoutedChargeNet().to(device)
    protocol, config = _run_configuration(model, arrays, split, stats, fraction, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay,
    )
    train_loader = create_data_loader(
        train, settings.batch_size, shuffle=True, seed=TRAINING_SEED, device=device,
    )
    validation_loader = create_data_loader(validation, settings.batch_size, device=device)
    stopping = DualObjectiveEarlyStopping(RegularizationSettings())
    best: dict[str, dict[str, Any]] = {}
    history: list[dict[str, Any]] = []

    # 디렉터리를 원자적으로 만들면 동시에 실행된 호출이 같은 출력 경로를 공유하지 않는다.
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output_dir / "protocol.json", protocol)
    atomic_write_json(output_dir / "config.json", config)
    started = time.perf_counter()
    for epoch in range(1, settings.max_epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, weights)
        validation_loss = run_epoch(model, validation_loader, weights=weights)
        history.append({"epoch": epoch, "train": asdict(train_loss), "validation": asdict(validation_loss)})
        # 다음 epoch에서 갱신될 매개변수와 별칭을 공유하지 않도록 전체 상태를 CPU에 복제한다.
        state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        changed = update_best_checkpoints(
            best, config=config, epoch=epoch, validation=validation_loss, model_state=state,
        )
        stopping.update(epoch, validation_loss)
        for selection in changed:
            atomic_torch_save(best[selection], output_dir / f"best_{selection}.pt")
        atomic_write_json(output_dir / "history.json", history)
        print(
            f"epoch={epoch:03d}/{settings.max_epochs} train={train_loss.total:.6f} "
            f"val_structure={validation_loss.structure:.6f} val_total={validation_loss.total:.6f} "
            f"best_structure={best['structure']['selected_epoch']} no_improvement={stopping.bad_epochs}",
            flush=True,
        )
        if stopping.stopped:
            break

    # 모델 선택을 마친 뒤에만 테스트 TensorDataset을 준비해 테스트 데이터가 선택에 관여하지 않게 한다.
    checkpoint_path = output_dir / "best_structure.pt"
    selected = load_torch_checkpoint(checkpoint_path, device)
    validate_selected_checkpoint(
        selected, config, selection="structure",
        expected_epoch=best["structure"]["selected_epoch"],
        expected_loss=best["structure"]["selected_validation_loss"],
    )
    model.load_state_dict(selected["model_state_dict"], strict=True)
    test = prepare_dataset(arrays, split.test, stats, fraction)
    metrics = evaluate_model(model, test, stats, batch_size=settings.batch_size, weights=weights)
    result = {
        "model_name": MODEL_NAME, "seed": TRAINING_SEED, "configuration": config,
        "status": "completed", "epochs_completed": len(history),
        "stop_reason": "max_epochs" if len(history) == settings.max_epochs else "early_stopping",
        "early_stopping": stopping.state_dict(), "elapsed_seconds": time.perf_counter() - started,
        "checkpoint_selection": "structure", "checkpoint_path": "best_structure.pt",
        "selected_epoch": selected["selected_epoch"],
        "selected_validation_loss": selected["selected_validation_loss"],
        "validation_losses": selected["validation_losses"], "test_metrics": metrics,
    }
    atomic_write_json(output_dir / "result.json", result)
    return result


def evaluate_only_run(
    checkpoint_path: Path | str, data_path: Path | str = DEFAULT_DATA_PATH,
    device: torch.device = DEVICE,
) -> dict[str, Any]:
    """명시적으로 선택한 검증 checkpoint로 최종 테스트 평가만 수행하며 파일은 쓰지 않는다."""
    model, stats, checkpoint = load_trained_model(checkpoint_path, device)
    arrays, split = _load_fixed_dataset(data_path)
    fitted = calculate_normalization_stats(arrays, split.train)
    if canonical_json(stats.to_dict()) != canonical_json(fitted.to_dict()):
        raise RuntimeError("Saved normalization does not match the fixed TRAIN split")
    config = checkpoint["configuration"]
    test = prepare_dataset(arrays, split.test, stats, config["observation"]["g05_fraction"])
    return {
        "model_name": MODEL_NAME, "seed": TRAINING_SEED,
        "g05_fraction": config["observation"]["g05_fraction"], "checkpoint_selection": "structure",
        "selected_epoch": checkpoint["selected_epoch"],
        "selected_validation_loss": checkpoint["selected_validation_loss"],
        "test_metrics": evaluate_model(
            model, test, stats, batch_size=TrainingSettings().batch_size, weights=LossWeights(),
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--train", action="store_true", help="Fresh training of this one fixed model/seed")
    modes.add_argument("--evaluate-only", "--eval-only", action="store_true",
                       help="Read-only final test evaluation of a seed42 best_structure checkpoint")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--fraction", "--g05-fraction", type=float,
                       choices=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
                       help="One training condition; inference always uses the checkpoint condition")
    parser.add_argument("--checkpoint", type=Path, help="Explicit best_structure.pt; no auto-selection")
    parser.add_argument("--output-dir", type=Path, help="New directory for one training run only")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = DEVICE if args.device == "auto" else torch.device(args.device)
    if args.train:
        if args.fraction is None or args.output_dir is None or args.checkpoint is not None:
            parser.error("--train requires --fraction and --output-dir, and cannot load --checkpoint")
        result = train_and_evaluate_run(
            fraction=args.fraction, output_dir=args.output_dir, data_path=args.data, device=device,
        )
    else:
        if args.checkpoint is None or args.fraction is not None or args.output_dir is not None:
            parser.error("--evaluate-only requires --checkpoint; --fraction/--output-dir are training-only")
        result = evaluate_only_run(args.checkpoint, args.data, device)
    print(json.dumps(json_ready(result), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

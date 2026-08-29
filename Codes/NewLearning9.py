"""G05 sign-only 경로를 엄격히 분리한 5전하 집합 예측 모델.

실행: python Codes/NewLearning9.py --generate-data
전체 학습 없이 검사: --smoke-only를 추가한다.

각 출력 슬롯은 위치, |q|, 상대 부호 logit을 가진다. 정답 전하의 저장 순서는
어떠해도 되며, 손실은 현재 torch 장치에서 5! = 120개 순열을 모두 비교해 정확한
일대일 대응을 찾는다(SciPy 불필요). 위치 정규화는 전하 번호별이 아니라 모든
슬롯이 공유한다.

0이 아닌 다섯 전하에 대해 g = product(sign(q_i))를 순서 불변 전체 부호로 두고,
r_i = sign(q_i) * g로 정의한다. 그러면 product(r_i) = +1이며 모든 전하의 부호를
함께 반전해도 r_i는 변하지 않는다. G00은 위치·크기·r_i만 예측하고, G05는 g만
예측한다. 부호 디코더와 우도는 가능한 16개 상대 부호 조합만 허용하며, 특정 전하
슬롯을 기준점으로 삼지 않는다.

G00만으로는 g를 알 수 없다. 따라서 G05가 관측되지 않은 샘플의 절대 부호 지표는
N/A이며, sign-only는 |q|를 1로 고정한다는 뜻이 아니다.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

# 이 프로세스의 첫 CUDA 행렬 곱 전에 cuBLAS 작업 공간을 고정해, 가능한 범위에서
# GPU 연산의 재현성을 높인다.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


PROJECT_DIR = Path(__file__).resolve().parent.parent
CHARGE_COUNT = 5
DEFAULT_DATA_PATH = PROJECT_DIR / "Models" / "charge_dataset_5charges_v9.npz"
CHECKPOINT_DIR = PROJECT_DIR / "Models" / "new_learning9"
RESULTS_DIR = PROJECT_DIR / "Results" / "new_learning9"
PROTOCOL_VERSION = "new-learning9-five-charge-sign-only-v1"
TARGET_FIELDS = tuple(
    f"{field}{index}"
    for index in range(1, CHARGE_COUNT + 1)
    for field in ("x", "y", "z", "q")
)
G05_FIELDS = ("grid_x_index", "grid_y_index", "signed_potential")
DATA_SPLIT_SEED = 42
EXPERIMENT_SEEDS = (41, 42, 43)
G05_FRACTIONS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
EPSILON = 1e-8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STRUCTURE_PREFIXES = (
    "g00_cnn.", "g00_encoder.", "position_head.",
    "magnitude_head.", "relative_sign_head.",
)
GLOBAL_SIGN_PREFIXES = ("g05_encoder.", "global_sign_head.")


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
    global_sign_logit: torch.Tensor     # [B], sign(product(q))에 대한 logit


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
    max_epochs: int = 300
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


@dataclass(frozen=True)
class TrainingResult:
    model: ChargeNet
    checkpoint_path: Path
    best_structure_loss: float
    best_structure_epoch: int
    best_global_sign_loss: float | None
    best_global_sign_epoch: int | None


def set_reproducibility(seed: int) -> None:
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
            "Run Codes/NewLearning9.py --generate-data --smoke-only first."
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
            raise ValueError("NewLearning9 requires exactly five charges per sample")
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
            f"NewLearning9 needs five charges: target [N,20] or [N,5,4], "
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
        raise ValueError("All five charges must be nonzero for the sign-only targets")
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
    # 전하는 순서가 없는 집합이므로 의도적으로 사전식 정렬을 요구하지 않는다.


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
    # 관측을 포함하는 nested prefix이므로 G05 조건 간 비교가 공정하다.
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
        # 다른 격자 크기에서도 adaptive pooling의 floor/ceil bin 정의를 따른다.
        # 각 평균의 역전파는 결정론적이며, 겹치는 bin의 gradient는 그래프 순서로 더해진다.
        rows = []
        for row in range(4):
            top, bottom = row * height // 4, ((row + 1) * height + 3) // 4
            columns = []
            for column in range(4):
                left, right = column * width // 4, ((column + 1) * width + 3) // 4
                columns.append(features[:, :, top:bottom, left:right].mean(dim=(-2, -1)))
            rows.append(torch.stack(columns, dim=-1))
        return torch.stack(rows, dim=-2)


class ChargeNet(nn.Module):
    """G00 구조 분기와 G05 전체 부호 분기를 완전히 분리한 5전하 모델.

    G00 CNN은 다섯 전하의 위치·크기·상대 부호를 예측한다. G05는 전체 부호만
    예측하며 구조 특징으로 들어가는 순전파·gradient 경로가 전혀 없다. 따라서
    G05를 얼마나 관측했는지가 구조 복원 결과를 바꾸지 않는다.
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
        # 위치 15개(5×3), 크기 5개, 상대 부호 logit 5개가 한 슬롯의 대응을 유지한다.
        self.g00_encoder = nn.Sequential(nn.Flatten(), nn.Linear(64 * 4 * 4, 256), nn.ReLU())
        self.position_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 15))
        self.magnitude_head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 5))
        self.relative_sign_head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 5))
        # G05는 (정규화 x, 정규화 y, 부호 있는 V) 후보점 목록이다. 각 점을 MLP로
        # 인코딩한 후 관측된 점의 통계량만 요약해 전체 부호를 판별한다.
        self.g05_encoder = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.global_sign_head = nn.Sequential(
            nn.Linear(32 * 3, 64), nn.ReLU(), nn.Linear(64, 1, bias=False),
        )

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
        """G05만으로 전체 부호 logit을 계산한다.

        구조 분기를 재사용할 때에도 이 함수만 따로 학습할 수 있다. 또한 V를 모두
        반전하면 전하 전체 부호도 반전되어야 한다는 물리 대칭을 계산식에 강제한다.
        """
        if (g05.ndim != 3 or g05.shape[-1] != 3 or g05.shape[1] < 1
                or g05_mask.shape != (*g05.shape[:2], 1)):
            raise ValueError(f"G05/mask shape mismatch: {g05.shape}, {g05_mask.shape}")
        # encoder 앞에서 마스크를 적용해 결측 자리에 어떤 큰 값이 있어도 특징으로
        # 새어 들어가지 못하게 한다.
        observed_g05 = g05.masked_fill(~g05_mask.bool(), 0)
        reversed_g05 = observed_g05 * observed_g05.new_tensor((1, 1, -1))
        both = torch.cat((observed_g05, reversed_g05), dim=0)
        both_mask = torch.cat((g05_mask, g05_mask), dim=0)
        summary = self._masked_summary(self.g05_encoder(both), both_mask)
        positive_score, negative_score = self.global_sign_head(summary).squeeze(-1).chunk(2)
        # 정확한 물리 대칭: 측정 V를 반전하면 전체 부호 logit도 반전되어야 한다.
        return (positive_score - negative_score) * 0.5

    def forward(self, g00: torch.Tensor, g05: torch.Tensor, g05_mask: torch.Tensor) -> ModelOutput:
        if (g05.ndim != 3 or g05.shape[-1] != 3 or g05.shape[1] < 1
                or g05_mask.shape != (*g05.shape[:2], 1) or g05.shape[0] != g00.shape[0]):
            raise ValueError(f"G00/G05/mask shape mismatch: {g00.shape}, {g05.shape}, {g05_mask.shape}")
        # 구조 예측은 G00만 통과한다. 아래의 G05 함수는 별도의 전체 부호 출력만
        # 만들므로, 두 손실 사이에 공유 특징이나 gradient 경로가 생기지 않는다.
        structure = self.g00_encoder(self.g00_cnn(g00))
        global_logit = self.forward_global_sign(g05, g05_mask)
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
    # 조건부 상대 부호 NLL은 이 BCE 비용과 샘플별 분할 상수만 다르므로, 이 대응이
    # 실제 구조 손실도 최소화한다. G05/전체 부호 비용은 의도적으로 제외한다.
    # 전체 부호가 위치·크기·상대 부호의 구조 학습을 바꾸면 안 되기 때문이다.
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
    """순열 불변 구조 손실과 관측된 G05의 전체 부호 손실을 계산한다.

    먼저 gradient 없이 최소 비용 일대일 대응을 고정하고, 그 대응으로 위치 MSE·
    크기 MSE·상대 부호 NLL을 계산한다. 전체 부호 BCE는 G05 관측이 있는 샘플에만
    추가한다. 대응 선택에 전체 부호 비용을 넣지 않아 두 분기가 다시 연결되는 일을
    방지한다.
    """
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
    model: ChargeNet, loader: DataLoader, optimizer: torch.optim.Optimizer | None = None,
    weights: LossWeights = LossWeights(),
) -> EpochLoss:
    """구조와 전체 부호를 함께 학습하거나 검증하는 한 epoch을 실행한다.

    구조 손실은 모든 샘플에 대해 계산하지만, 전체 부호 BCE는 G05가 관측된 샘플에
    대해서만 계산한다. 두 분기는 파라미터·특징을 공유하지 않으므로 total loss로
    역전파해도 G05 부호 gradient가 G00 구조 branch로 들어가지 않는다.
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
                # 공동 gradient clipping이나 total-loss scheduler는 쓰지 않는다.
                # 그런 연산은 독립된 G05와 구조 최적화를 다시 연결할 수 있다.
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


def run_global_sign_epoch(
    model: ChargeNet, loader: DataLoader, optimizer: torch.optim.Optimizer | None = None,
    loss_weight: float = 1.0,
) -> float | None:
    """G05 전체 부호 분기만 따로 학습/검증한다.

    전체 부호는 전하 부호의 곱이므로 슬롯별 정답 대응을 구할 필요가 없다. 구조
    체크포인트를 재사용하는 G05 비율에서는 이 함수만 호출해 G00 CNN의 순전파와
    역전파를 모두 생략한다.
    """
    model.train(optimizer is not None)
    device = next(model.parameters()).device
    sample_count = observed_count = 0
    loss_sum = 0.0
    for _, g05, mask, _, charge in loader:
        g05, mask, charge = (t.to(device, non_blocking=True) for t in (g05, mask, charge))
        sample_count += len(g05)
        observed = mask.sum(dim=(1, 2)) > 0
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        if not torch.any(observed):
            continue
        # 전하 부호의 곱은 구조 슬롯 대응을 바꿔도 변하지 않는다.
        _, global_target = canonical_sign_targets(charge)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model.forward_global_sign(g05, mask)
            loss = F.binary_cross_entropy_with_logits(logits[observed], global_target[observed])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite global-sign training/validation loss")
            if optimizer is not None:
                (loss_weight * loss).backward()
                optimizer.step()
        count = int(observed.sum())
        observed_count += count
        loss_sum += float(loss.detach()) * count
    if sample_count == 0:
        raise ValueError("Cannot run an epoch on an empty dataset")
    return loss_sum / observed_count if observed_count else None


def checkpoint_metadata(
    arrays: DatasetArrays, stats: NormalizationStats, split: DataSplit, data_path: Path,
) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "charge_count": CHARGE_COUNT,
        "model_architecture": "five-charge G00 structure / G05-only global sign",
        "target_fields": list(TARGET_FIELDS),
        "target_ordering": "unordered set of five (x,y,z,q) tuples",
        "g05_fields": list(G05_FIELDS),
        "global_sign_anchor": "product(sign(q_i)) for five nonzero charges",
        "relative_sign_target": "sign(q_i) * global_sign; product(relative_signs)=+1",
        "relative_sign_loss": "conditional NLL over 16 valid patterns, divided by 5",
        "matching": "exact 120-permutation minimum of position MSE + magnitude MSE + sign cost",
        "metric_alignment": "same joint structure assignment as the loss; no G05/global-sign cost",
        "normalization": stats.to_dict(),
        "position_normalization": "train-only per-axis statistics pooled over all charge slots",
        "input_policy": {"position": "G00 only", "magnitude": "G00 only",
                         "relative_sign": "G00 only", "global_sign": "masked G05 only"},
        "global_sign_symmetry": "logit(-V) = -logit(V)",
        "gradient_policy": "no shared parameters, features, loss matching or optimizer scheduling",
        "checkpoint_policy": "compose independent best structure and best global-sign branches",
        "missing_g05": "global sign unidentifiable; absolute-sign metrics are N/A",
        "partial_g05_definition": "fixed nested sensor prefixes in every sample",
        "full_fraction_definition": "all stored candidate sensors, not necessarily the full grid",
        "data_path": str(data_path.resolve()),
        "grid_shape": list(arrays.g00.shape[1:]),
        "grid_x": arrays.grid_x.tolist(),
        "grid_y": arrays.grid_y.tolist(),
        "epsilon_0": arrays.epsilon_0,
        "g05_candidate_count": arrays.g05.shape[1],
        "split_seed": DATA_SPLIT_SEED,
        "split_indices": {name: getattr(split, name).tolist() for name in ("train", "validation", "test")},
    }


def copy_state(model: nn.Module, prefixes: tuple[str, ...] | None = None) -> dict[str, torch.Tensor]:
    # CPU clone은 필수다. 최적 스냅샷이 계속 학습되는 현재 파라미터를 참조하면
    # 이후 optimizer step으로 과거 최적 모델까지 바뀌기 때문이다.
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            if prefixes is None or name.startswith(prefixes)}


def save_checkpoint(checkpoint: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(checkpoint, temporary)
        for attempt in range(5):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * 2**attempt)
    finally:
        temporary.unlink(missing_ok=True)


def load_trained_model(
    path: Path, device: torch.device = DEVICE,
) -> tuple[ChargeNet, NormalizationStats, dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("protocol_version") != PROTOCOL_VERSION or checkpoint.get("charge_count") != CHARGE_COUNT:
        raise ValueError("This is not a NewLearning9 five-charge checkpoint")
    if "model_state_dict" not in checkpoint:
        raise ValueError("Use a composed checkpoint, not a single-branch checkpoint")
    values = checkpoint["normalization"]
    stats = NormalizationStats(
        g00_mean=values["g00_mean"], g00_std=values["g00_std"],
        g05_value_scale=values["g05_value_scale"], charge_scale=values["charge_scale"],
        position_mean=np.asarray(values["position_mean"], dtype=np.float32),
        position_std=np.asarray(values["position_std"], dtype=np.float32),
    )
    model = ChargeNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, stats, checkpoint


def fraction_label(fraction: float) -> str:
    return str(float(fraction)).replace(".", "p")


def create_run_directories(checkpoint_root: Path, results_root: Path) -> tuple[str, Path, Path]:
    """이전 실행을 건드리지 않고, 이번 실행의 산출물을 같은 run ID로 묶는다."""
    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:12]}"
    checkpoint_dir = checkpoint_root / run_id
    results_dir = results_root / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    if results_dir.resolve() != checkpoint_dir.resolve():
        results_dir.mkdir(parents=True, exist_ok=False)
    return run_id, checkpoint_dir, results_dir


def train_model(
    train_dataset: TensorDataset, validation_dataset: TensorDataset,
    fraction: float, seed: int, metadata: dict[str, object], *,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    settings: TrainingSettings = TrainingSettings(), weights: LossWeights = LossWeights(),
    device: torch.device = DEVICE, structure_source: Path | None = None,
) -> TrainingResult:
    """한 G05 비율·seed 조건을 새 폴더에 학습하고 최적 분기를 합성한다.

    첫 조건에서는 G00 구조와 G05 전체 부호를 함께 학습한다. 이후 같은 seed의
    다른 G05 비율은 검증된 G00 구조 스냅샷을 재사용하고 G05 branch만 학습한다.
    구조/전체 부호는 각각 최적 검증 epoch를 선택한 뒤 하나의 추론용 체크포인트로
    합성한다.
    """
    if settings.max_epochs < 1 or settings.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    set_reproducibility(seed)
    model = ChargeNet().to(device)
    run_dir = checkpoint_dir / f"seed_{seed}_g05_{fraction_label(fraction)}"
    run_metadata = {
        **metadata, "seed": seed, "g05_fraction": fraction,
        "settings": asdict(settings), "loss_weights": asdict(weights),
        "structure_source": str(structure_source.resolve()) if structure_source is not None else None,
    }
    best_structure_loss = math.inf
    best_structure_epoch = 0
    best_global_loss: float | None = None
    best_global_epoch: int | None = None
    best_structure: dict[str, torch.Tensor] | None = None
    best_global: dict[str, torch.Tensor] | None = None
    if structure_source is not None:
        # 구조 재사용은 seed·정규화·데이터 분할·학습 설정이 같은 경우에만 허용한다.
        # 하나라도 다르면 서로 다른 실험의 G00 구조를 섞게 되므로 즉시 중단한다.
        source = torch.load(structure_source, map_location="cpu", weights_only=True)
        mismatches = [key for key, value in run_metadata.items()
                      if key not in ("g05_fraction", "structure_source") and source.get(key) != value]
        if source.get("component") != "structure" or mismatches:
            raise ValueError(f"Incompatible structure checkpoint (metadata: {mismatches}): {structure_source}")
        best_structure = source["component_state_dict"]
        expected_keys = {key for key in model.state_dict() if key.startswith(STRUCTURE_PREFIXES)}
        if set(best_structure) != expected_keys:
            raise ValueError("Structure checkpoint must contain exactly the G00 branch parameters")
        best_structure_loss = float(source["validation_loss"])
        best_structure_epoch = int(source["epoch"])
        if not math.isfinite(best_structure_loss) or not 1 <= best_structure_epoch <= settings.max_epochs:
            raise ValueError("Invalid best structure loss or epoch in checkpoint")
        # 구조만 불러오기 전에 전체 모델을 같은 seed로 초기화한다. 그러면 독립된
        # G05 branch의 초기값과 셔플 조건이 원래 실험과 정확히 같게 유지된다.
        model.load_state_dict(best_structure, strict=False)
    optimizer = torch.optim.AdamW(
        [parameter for name, parameter in model.named_parameters()
         if structure_source is None or name.startswith(GLOBAL_SIGN_PREFIXES)],
        lr=settings.learning_rate, weight_decay=settings.weight_decay,
    )
    train_loader = create_data_loader(train_dataset, settings.batch_size, shuffle=True, seed=seed, device=device)
    validation_loader = create_data_loader(validation_dataset, settings.batch_size, device=device)
    # main을 거치지 않고 train_model을 직접 호출해도 기존 run을 덮어쓰지 않게 한다.
    run_dir.mkdir(parents=True, exist_ok=False)
    if structure_source is not None:
        save_checkpoint({**run_metadata, "component": "structure",
                         "component_state_dict": best_structure, "epoch": best_structure_epoch,
                         "validation_loss": best_structure_loss}, run_dir / "best_structure.pt")
        print(f"seed={seed} G05={fraction:.3f}: reusing G00 structure from {structure_source}", flush=True)
    history: list[dict[str, object]] = []
    for epoch in range(1, settings.max_epochs + 1):
        if structure_source is None:
            train_loss = run_epoch(model, train_loader, optimizer, weights)
            val_loss = run_epoch(model, validation_loader, weights=weights)
            history.append({"epoch": epoch, **{f"train_{k}": v for k, v in asdict(train_loss).items()},
                            **{f"validation_{k}": v for k, v in asdict(val_loss).items()}})
            if val_loss.structure < best_structure_loss:
                best_structure_loss, best_structure_epoch = val_loss.structure, epoch
                best_structure = copy_state(model, STRUCTURE_PREFIXES)
                save_checkpoint({**run_metadata, "component": "structure",
                                 "component_state_dict": best_structure, "epoch": epoch,
                                 "validation_loss": best_structure_loss}, run_dir / "best_structure.pt")
            val_global_loss = val_loss.global_sign
            progress = (f"train={train_loss.total:.5f} val_structure={val_loss.structure:.5f} "
                        f"pos={val_loss.position:.5f} mag={val_loss.magnitude:.5f} "
                        f"relative={val_loss.relative_sign:.5f}")
        else:
            train_global_loss = run_global_sign_epoch(model, train_loader, optimizer, weights.global_sign)
            val_global_loss = run_global_sign_epoch(model, validation_loader, loss_weight=weights.global_sign)
            # 빌려온 구조 곡선을 새 학습 결과처럼 기록하지 않는다. 원래 이력은
            # structure_source 옆에 보존되어 있으며 여기에는 G05 이력만 남긴다.
            history.append({"epoch": epoch, "train_global_sign": train_global_loss,
                            "validation_global_sign": val_global_loss})
            train_text = "N/A" if train_global_loss is None else f"{train_global_loss:.5f}"
            progress = f"train_global={train_text} reused_structure={best_structure_loss:.5f}"
        if val_global_loss is not None and (best_global_loss is None or val_global_loss < best_global_loss):
            best_global_loss, best_global_epoch = val_global_loss, epoch
            best_global = copy_state(model, GLOBAL_SIGN_PREFIXES)
            save_checkpoint({**run_metadata, "component": "global_sign",
                             "component_state_dict": best_global, "epoch": epoch,
                             "validation_loss": best_global_loss}, run_dir / "best_global_sign.pt")
        save_csv(history, run_dir / "history.csv")
        sign_text = "N/A" if val_global_loss is None else f"{val_global_loss:.5f}"
        print(f"seed={seed} G05={fraction:.3f} epoch={epoch:03d}/{settings.max_epochs} "
              f"{progress} global={sign_text}", flush=True)
    assert best_structure is not None
    # 두 branch의 최적 epoch가 다를 수 있으므로 마지막 epoch 전체 모델을 쓰지
    # 않는다. 구조·전체 부호의 각 최적 부분만 합쳐 추론용 composed.pt를 만든다.
    composed = copy_state(model)
    composed.update(best_structure)
    if best_global is not None:
        composed.update(best_global)
    model.load_state_dict(composed, strict=True)
    model.eval()
    path = run_dir / "composed.pt"
    save_checkpoint({
        **run_metadata, "model_state_dict": composed,
        "best_structure_loss": best_structure_loss, "best_structure_epoch": best_structure_epoch,
        "best_global_sign_loss": best_global_loss, "best_global_sign_epoch": best_global_epoch,
    }, path)
    return TrainingResult(model, path, best_structure_loss, best_structure_epoch, best_global_loss, best_global_epoch)


@torch.no_grad()
def evaluate_model(
    model: ChargeNet, dataset: TensorDataset, stats: NormalizationStats,
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


def save_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_summary_rows(
    run_rows: list[dict[str, object]], fractions: Sequence[float], metric_names: Sequence[str],
    candidate_count: int,
) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for fraction in fractions:
        group = [row for row in run_rows if row["g05_fraction"] == fraction]
        row: dict[str, object] = {"g05_fraction": fraction,
                                  "g05_count": g05_count_for_fraction(fraction, candidate_count),
                                  "run_count": len(group)}
        for name in metric_names:
            values = [item[name] for item in group if item[name] is not None]
            row[f"{name}_mean"] = float(np.mean(values)) if values else None
            row[f"{name}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if values else None)
        summary.append(row)
    return summary


def save_summary_plot(rows: list[dict[str, object]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = (
        ("mean_position_3d_error", "Position: mean 3D distance"),
        ("charge_magnitude_mae", "Charge magnitude MAE"),
        ("pairwise_relative_sign_accuracy", "Pairwise relative sign accuracy"),
        ("global_sign_accuracy", "Global sign accuracy (G05 only)"),
        ("absolute_sign_accuracy", "Absolute sign accuracy"),
        ("charge_mae", "Charge MAE (global-aligned when G05=0)"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, metrics):
        valid = [row for row in rows if row[f"{metric}_mean"] is not None]
        if valid:
            axis.errorbar([100 * row["g05_fraction"] for row in valid],
                          [row[f"{metric}_mean"] for row in valid],
                          yerr=[row[f"{metric}_std"] for row in valid], fmt="o-", capsize=4)
        else:
            axis.text(0.5, 0.5, "N/A: no observed G05", ha="center", transform=axis.transAxes)
        axis.set(title=title, xlabel="Observed G05 candidates (%)")
        axis.grid(alpha=0.25)
        if "accuracy" in metric:
            axis.set_ylim(-0.03, 1.03)
    figure.suptitle("NewLearning9 | 5 charges | exact set matching | G05 sign-only")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_smoke_tests(
    arrays: DatasetArrays, stats: NormalizationStats, train_indices: np.ndarray,
    fractions: Sequence[float] = G05_FRACTIONS, device: torch.device = DEVICE,
) -> None:
    indices = train_indices[:min(12, len(train_indices))]
    for fraction in fractions:
        set_reproducibility(123)
        dataset = prepare_dataset(arrays, indices, stats, fraction)
        g00, g05, mask, positions, charges = (t.to(device) for t in dataset.tensors)
        model = ChargeNet().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        output = model(g00, g05, mask)
        if output.position.shape != (len(indices), 5, 3) or output.magnitude.shape != (len(indices), 5):
            raise AssertionError("Five-charge output shape mismatch")
        losses = calculate_losses(output, positions, charges, mask)
        permutation = torch.tensor([4, 2, 0, 3, 1], device=device)
        permuted_loss = calculate_losses(output, positions[:, permutation], charges[:, permutation], mask)
        torch.testing.assert_close(losses.total, permuted_loss.total)
        # G05의 물리적 부호 반전 대칭을 포함해 두 branch의 순전파 분리를 검사한다.
        reversed_g05 = g05 * g05.new_tensor((1, 1, -1))
        reversed_output = model(g00, reversed_g05, mask)
        for name in ("position", "magnitude", "relative_sign_logit"):
            torch.testing.assert_close(getattr(output, name), getattr(reversed_output, name), rtol=0, atol=0)
        torch.testing.assert_close(output.global_sign_logit, -reversed_output.global_sign_logit, rtol=0, atol=0)
        decoded = decode_relative_signs(output.relative_sign_logit)
        torch.testing.assert_close(decoded.prod(dim=1), torch.ones(len(indices), device=device))
        if losses.global_sign is not None:
            losses.global_sign.backward(retain_graph=True)
            if any(p.grad is not None for name, p in model.named_parameters() if name.startswith(STRUCTURE_PREFIXES)):
                raise AssertionError("Global-sign gradients reached the G00 structure branch")
        elif fraction != 0:
            raise AssertionError("Observed G05 was not used for global-sign learning")
        optimizer.zero_grad(set_to_none=True)
        losses.structure.backward(retain_graph=True)
        if any(p.grad is not None for name, p in model.named_parameters() if name.startswith(GLOBAL_SIGN_PREFIXES)):
            raise AssertionError("Structure gradients reached the G05 branch")
        optimizer.zero_grad(set_to_none=True)
        losses.total.backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise AssertionError("Non-finite gradients")
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise AssertionError("Non-finite parameters after optimizer step")
        print(f"SMOKE PASS: G05={fraction:.3f}, points={g05_count_for_fraction(fraction, arrays.g05.shape[1])}, "
              "5-charge shape / permutation loss / sign-only gradients / optimizer", flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--generate-data", action="store_true",
                        help="Create a missing five-charge dataset at --data; never overwrite an existing dataset")
    parser.add_argument("--samples", type=int, default=10_000, help="Sample count for --generate-data")
    parser.add_argument("--g05-points", type=int, default=32, help="Candidate sensors for --generate-data")
    parser.add_argument("--data-seed", type=int, default=20_260_827)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seeds", type=lambda v: tuple(int(x.strip()) for x in v.split(",")), default=EXPERIMENT_SEEDS)
    parser.add_argument("--fractions", type=lambda v: tuple(float(x.strip()) for x in v.split(",")), default=G05_FRACTIONS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR,
                        help="Checkpoint root; each invocation creates a new run_* subdirectory")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="Results root; uses the same new run_* subdirectory name as checkpoints")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--smoke-only", action="store_true", help="Validate data, matching, gradients and one batch update; no full training")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--weight-decay must be finite and nonnegative")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(not 0 <= s < 2**32 for s in args.seeds):
        parser.error("--seeds must contain unique integers in [0, 2**32)")
    if (not args.fractions or len(set(args.fractions)) != len(args.fractions)
            or any(not math.isfinite(f) or not 0 <= f <= 1 for f in args.fractions)):
        parser.error("--fractions must contain unique finite values in [0,1]")
    args.fractions = tuple(sorted(args.fractions))
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    if args.generate_data and args.samples < 10:
        parser.error("--samples must be at least 10 for training/validation/test splits")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = DEVICE if args.device == "auto" else torch.device(args.device)
    if args.generate_data and not args.data.exists():
        from generate_charge_dataset import generate_dataset, save_dataset

        generated = generate_dataset(
            sample_count=args.samples, g05_point_count=args.g05_points,
            seed=args.data_seed, charge_count=CHARGE_COUNT,
        )
        save_dataset(generated, args.data)
        del generated
        print("Created five-charge dataset:", args.data.resolve(), flush=True)
    elif args.generate_data:
        print("Using existing dataset without overwriting:", args.data.resolve(), flush=True)
    arrays = load_dataset(args.data)
    split = create_data_split(len(arrays.target))
    stats = calculate_normalization_stats(arrays, split.train)
    metadata = checkpoint_metadata(arrays, stats, split, args.data)
    print("Device:", device)
    print("G00/G05/target:", arrays.g00.shape, arrays.g05.shape, arrays.target.shape)
    print("Unordered five-charge targets; exact 120-permutation joint matching")
    print("Sign-only: G00 -> position / magnitude / relative signs; G05 -> global sign only")
    print("Global sign = product(sign(q_i)); relative-sign product = +1")
    print("Fraction=1 uses all stored candidate sensors, not necessarily the entire grid", flush=True)
    run_smoke_tests(arrays, stats, split.train, args.fractions, device)
    if args.smoke_only:
        print("Smoke-only complete; no trained checkpoints or performance claims.")
        return

    settings = TrainingSettings(args.epochs, args.batch_size, args.learning_rate, args.weight_decay)
    run_id, checkpoint_dir, results_dir = create_run_directories(args.checkpoint_dir, args.results_dir)
    metadata = {**metadata, "run_id": run_id,
                "structure_training_policy": "train once per seed; reuse for subsequent G05 fractions"}
    print("Run:", run_id, flush=True)
    print("Checkpoints:", checkpoint_dir.resolve(), flush=True)
    print("Results:", results_dir.resolve(), flush=True)
    (results_dir / "protocol.json").write_text(json.dumps({
        **metadata, "settings": asdict(settings), "seeds": args.seeds, "fractions": args.fractions,
    }, indent=2, allow_nan=False), encoding="utf-8")
    np.savez_compressed(checkpoint_dir / "normalization.npz", **asdict(stats), charge_count=CHARGE_COUNT)
    run_rows: list[dict[str, object]] = []
    metric_names: tuple[str, ...] = ()
    structure_sources: dict[int, Path] = {}
    for fraction in args.fractions:
        # 모든 G05 비율의 전체 tensor를 메모리에 중복 생성하지 않고, 한 비율씩
        # 준비한다. 데이터가 커져도 GPU/메모리 사용량이 불필요하게 늘지 않는다.
        datasets = tuple(prepare_dataset(arrays, getattr(split, part), stats, fraction)
                         for part in ("train", "validation", "test"))
        best_selection_score = math.inf
        for seed in args.seeds:
            training = train_model(
                datasets[0], datasets[1], fraction, seed, metadata,
                checkpoint_dir=checkpoint_dir, settings=settings, device=device,
                structure_source=structure_sources.get(seed),
            )
            structure_sources.setdefault(seed, training.checkpoint_path.parent / "best_structure.pt")
            metrics = evaluate_model(training.model, datasets[2], stats, args.batch_size)
            metric_names = tuple(metrics)
            run_rows.append({
                "seed": seed, "g05_fraction": fraction,
                "g05_count": g05_count_for_fraction(fraction, arrays.g05.shape[1]),
                **metrics,
                "best_structure_loss": training.best_structure_loss,
                "best_structure_epoch": training.best_structure_epoch,
                "best_global_sign_loss": training.best_global_sign_loss,
                "best_global_sign_epoch": training.best_global_sign_epoch,
                "checkpoint_path": str(training.checkpoint_path.resolve()),
            })
            save_csv(run_rows, results_dir / "runs.csv")
            selection_score = training.best_structure_loss + (training.best_global_sign_loss or 0.0)
            if selection_score < best_selection_score:
                best_selection_score = selection_score
                best = torch.load(training.checkpoint_path, map_location="cpu", weights_only=True)
                best["selection_policy"] = "minimum validation structure + global-sign loss; test metrics excluded"
                save_checkpoint(best, checkpoint_dir / f"best_g05_{fraction_label(fraction)}.pt")
            print("Test set metrics:", json.dumps(metrics, allow_nan=False), flush=True)
    summary = build_summary_rows(run_rows, args.fractions, metric_names, arrays.g05.shape[1])
    save_csv(summary, results_dir / "summary.csv")
    if not args.no_plots:
        save_summary_plot(summary, results_dir / "summary.png")
    print("Completed. Checkpoints:", checkpoint_dir.resolve())
    print("Results:", results_dir.resolve())


if __name__ == "__main__":
    main()

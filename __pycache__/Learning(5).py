from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 경로 설정
# ============================================================

# 스크립트 기준 디렉터리.
BASE_DIR = Path(__file__).resolve().parent

# 학습 데이터 경로.
DATA_PATH = BASE_DIR / "charge_dataset.npz"

# G00-only 모델 경로.
MODEL_G00_PATH = BASE_DIR / "model_G00_only.pt"

# G00+G05 모델 경로.
MODEL_G00_G05_PATH = BASE_DIR / "model_G00_G05.pt"

# 정규화 통계 경로.
NORMALIZATION_PATH = BASE_DIR / "normalization_stats.npz"


# ============================================================
# 학습 설정
# ============================================================

# 데이터 분할 기준 시드.
DATA_SPLIT_SEED = 42

# 반복 실험 시드.
EXPERIMENT_SEEDS = (41, 42, 43)

# 미니배치 크기.
BATCH_SIZE = 128

# 최대 학습 epoch 수.
EPOCHS = 50

# 조기 종료 대기 epoch 수.
PATIENCE = 8

# AdamW 학습률.
LEARNING_RATE = 1e-3

# AdamW 가중치 감쇠율.
WEIGHT_DECAY = 1e-4

# 위치 손실 가중치.
POSITION_LOSS_WEIGHT = 1.0

# 전하 손실 가중치.
CHARGE_LOSS_WEIGHT = 1.0

# 수치 안정화 상수.
EPSILON = 1e-8

# 위치 target 열 인덱스.
POSITION_INDICES = (0, 1, 2, 4, 5, 6)

# 전하 target 열 인덱스.
CHARGE_INDICES = (3, 7)

# 학습 장치.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 데이터 구조
# ============================================================

@dataclass(frozen=True)
class DatasetArrays:
    """원본 데이터 배열."""

    # G00 격자 데이터.
    g00: np.ndarray

    # G05 관측점 데이터.
    g05: np.ndarray

    # 위치·전하 target 데이터.
    target: np.ndarray


@dataclass(frozen=True)
class DataSplit:
    """데이터 분할 인덱스."""

    # 학습 인덱스.
    train: np.ndarray

    # 검증 인덱스.
    validation: np.ndarray

    # 테스트 인덱스.
    test: np.ndarray


@dataclass(frozen=True)
class NormalizationStats:
    """학습 데이터 기반 정규화 통계."""

    # G00 전체 평균.
    g00_mean: float

    # G00 전체 표준편차.
    g00_std: float

    # G05 전위 RMS 크기.
    g05_value_scale: float

    # 위치 출력별 평균.
    position_mean: np.ndarray

    # 위치 출력별 표준편차.
    position_std: np.ndarray

    # 전하 공통 RMS 크기.
    charge_scale: float


@dataclass(frozen=True)
class EpochLoss:
    """epoch 평균 손실."""

    # 가중 합산 손실.
    total: float

    # 위치 손실.
    position: float

    # 전하 손실.
    charge: float


@dataclass(frozen=True)
class TrainingResult:
    """학습 완료 모델 정보."""

    # 검증 최적 모델.
    model: ChargeNet

    # 검증 최저 손실.
    best_validation_loss: float

    # 실험 시드.
    seed: int


@dataclass(frozen=True)
class EvaluationResult:
    """테스트 평가 지표."""

    # 위치 출력별 MAE.
    position_mae: np.ndarray

    # 첫 번째 전하 위치 거리 오차.
    position_error_1: float

    # 두 번째 전하 위치 거리 오차.
    position_error_2: float

    # 전하 출력별 MAE.
    charge_mae: np.ndarray

    # 전하 크기 MAE.
    charge_magnitude_mae: float

    # 상대 부호 정확도.
    relative_sign_accuracy: float

    # 절대 부호 정확도.
    absolute_sign_accuracy: float | None


# ============================================================
# 재현성 설정
# ============================================================

def set_reproducibility(seed: int) -> None:
    """난수와 연산 결정성 설정."""

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True, warn_only=True)

    # cuDNN 자동 최적화 비활성화.
    torch.backends.cudnn.benchmark = False

    # cuDNN 결정론 연산 활성화.
    torch.backends.cudnn.deterministic = True


# ============================================================
# 데이터 로드 및 검증
# ============================================================

def load_dataset(path: Path) -> DatasetArrays:
    """NPZ 데이터 로드."""

    if not path.exists():
        raise FileNotFoundError(f"데이터 파일 없음: {path}")

    with np.load(path) as archive:
        # G00 float32 배열.
        g00 = archive["G00"].astype(np.float32)

        # G05 float32 배열.
        g05 = archive["G05"].astype(np.float32)

        # target float32 배열.
        target = archive["target"].astype(np.float32)

    validate_dataset(g00, g05, target)

    return DatasetArrays(g00=g00, g05=g05, target=target)


def validate_dataset(
    g00: np.ndarray,
    g05: np.ndarray,
    target: np.ndarray,
) -> None:
    """데이터 shape·범위·유한값 검증."""

    if g00.ndim != 3:
        raise ValueError(f"G00 shape 오류: {g00.shape}")

    if g05.ndim != 3 or g05.shape[-1] != 3:
        raise ValueError(f"G05 shape 오류: {g05.shape}")

    if target.ndim != 2 or target.shape[-1] != 8:
        raise ValueError(f"target shape 오류: {target.shape}")

    # 전체 샘플 수.
    sample_count = g00.shape[0]

    if sample_count < 10:
        raise ValueError("최소 샘플 수 미달: 10")

    if g05.shape[0] != sample_count or target.shape[0] != sample_count:
        raise ValueError("G00, G05, target 샘플 수 불일치")

    if g00.shape[1] < 4 or g00.shape[2] < 4:
        raise ValueError(f"G00 격자 크기 부족: {g00.shape[1:]}")

    if g05.shape[1] < 1:
        raise ValueError("G05 관측점 수 부족: 1")

    if not np.isfinite(g00).all():
        raise ValueError("G00 비유한값 포함")

    if not np.isfinite(g05).all():
        raise ValueError("G05 비유한값 포함")

    if not np.isfinite(target).all():
        raise ValueError("target 비유한값 포함")

    # G00 격자 높이.
    grid_height = g00.shape[1]

    # G00 격자 너비.
    grid_width = g00.shape[2]

    if np.any((g05[:, :, 0] < 0) | (g05[:, :, 0] >= grid_width)):
        raise ValueError("G05 x 인덱스 범위 오류")

    if np.any((g05[:, :, 1] < 0) | (g05[:, :, 1] >= grid_height)):
        raise ValueError("G05 y 인덱스 범위 오류")


# ============================================================
# 데이터 분할 및 정규화
# ============================================================

def create_data_split(sample_count: int, seed: int) -> DataSplit:
    """고정 비율 데이터 분할."""

    # 분할 전용 난수 생성기.
    random_generator = np.random.default_rng(seed)

    # 무작위 전체 인덱스.
    shuffled_indices = random_generator.permutation(sample_count)

    # 학습 구간 끝 인덱스.
    train_end = int(sample_count * 0.8)

    # 검증 구간 끝 인덱스.
    validation_end = int(sample_count * 0.9)

    return DataSplit(
        train=shuffled_indices[:train_end],
        validation=shuffled_indices[train_end:validation_end],
        test=shuffled_indices[validation_end:],
    )


def calculate_normalization_stats(
    arrays: DatasetArrays,
    train_indices: np.ndarray,
) -> NormalizationStats:
    """학습 분할 전용 정규화 통계 계산."""

    # 학습 G00 데이터.
    train_g00 = arrays.g00[train_indices]

    # 학습 G05 전위 데이터.
    train_g05_values = arrays.g05[train_indices, :, 2]

    # 학습 위치 target 데이터.
    train_positions = arrays.target[train_indices][:, POSITION_INDICES]

    # 학습 전하 target 데이터.
    train_charges = arrays.target[train_indices][:, CHARGE_INDICES]

    # G00 전체 평균.
    g00_mean = float(train_g00.mean())

    # G00 전체 표준편차.
    g00_std = float(train_g00.std()) + EPSILON

    # 부호 대칭 보존용 G05 RMS 크기.
    g05_value_scale = float(np.sqrt(np.mean(train_g05_values**2))) + EPSILON

    # 위치 출력별 평균.
    position_mean = train_positions.mean(axis=0)

    # 위치 출력별 표준편차.
    position_std = train_positions.std(axis=0) + EPSILON

    # 부호 대칭 보존용 전하 RMS 크기.
    charge_scale = float(np.sqrt(np.mean(train_charges**2))) + EPSILON

    return NormalizationStats(
        g00_mean=g00_mean,
        g00_std=g00_std,
        g05_value_scale=g05_value_scale,
        position_mean=position_mean,
        position_std=position_std,
        charge_scale=charge_scale,
    )


def prepare_dataset(
    arrays: DatasetArrays,
    indices: np.ndarray,
    stats: NormalizationStats,
) -> TensorDataset:
    """모델 입력·target 텐서 생성."""

    # 선택 분할 G00 데이터.
    g00 = (arrays.g00[indices] - stats.g00_mean) / stats.g00_std

    # G00 채널 차원 추가.
    g00 = g00[:, np.newaxis, :, :]

    # 선택 분할 G05 데이터.
    g05 = arrays.g05[indices].copy()

    # G00 격자 높이.
    grid_height = arrays.g00.shape[1]

    # G00 격자 너비.
    grid_width = arrays.g00.shape[2]

    # G05 x 인덱스 정규화.
    g05[:, :, 0] = 2.0 * g05[:, :, 0] / (grid_width - 1) - 1.0

    # G05 y 인덱스 정규화.
    g05[:, :, 1] = 2.0 * g05[:, :, 1] / (grid_height - 1) - 1.0

    # G05 전위 크기 정규화.
    g05[:, :, 2] = g05[:, :, 2] / stats.g05_value_scale

    # 선택 분할 위치 target.
    positions = arrays.target[indices][:, POSITION_INDICES]

    # 위치 출력별 표준화.
    positions = (positions - stats.position_mean) / stats.position_std

    # 선택 분할 전하 target.
    charges = arrays.target[indices][:, CHARGE_INDICES]

    # 전하 공통 크기 정규화.
    charges = charges / stats.charge_scale

    return TensorDataset(
        torch.from_numpy(np.ascontiguousarray(g00)),
        torch.from_numpy(np.ascontiguousarray(g05)),
        torch.from_numpy(np.ascontiguousarray(positions)),
        torch.from_numpy(np.ascontiguousarray(charges)),
    )


def create_data_loader(
    dataset: TensorDataset,
    shuffle: bool,
    seed: int | None = None,
) -> DataLoader:
    """재현 가능한 DataLoader 생성."""

    # 셔플 전용 PyTorch 난수 생성기.
    loader_generator = None

    if shuffle:
        if seed is None:
            raise ValueError("shuffle=True인 경우 seed 필요")

        # 시드 고정 셔플 생성기.
        loader_generator = torch.Generator()
        loader_generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
        generator=loader_generator,
    )


# ============================================================
# 모델
# ============================================================

class ChargeNet(nn.Module):
    """G00·G05 기반 위치·전하 분리 회귀 모델."""

    def __init__(self, use_g05: bool = True) -> None:
        super().__init__()

        # G05 사용 여부.
        self.use_g05 = use_g05

        # G00 공간 특징 추출기.
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

        # G00 고정 길이 특징 변환기.
        self.g00_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
        )

        # G05 관측점 특징 추출기.
        self.g05_encoder = None

        if use_g05:
            # G05 사용 모델의 관측점 변환기.
            self.g05_encoder = nn.Sequential(
                nn.Linear(3, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
            )

        # 결합 특징 크기.
        fusion_input_size = 160 if use_g05 else 128

        # 공통 물리 특징 변환기.
        self.shared_encoder = nn.Sequential(
            nn.Linear(fusion_input_size, 128),
            nn.ReLU(),
        )

        # 위치 전용 출력 head.
        self.position_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
        )

        # 전하 전용 출력 head.
        self.charge_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(
        self,
        g00: torch.Tensor,
        g05: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """정규화 위치·전하 예측."""

        # G00 고정 길이 특징.
        g00_features = self.g00_encoder(self.g00_cnn(g00))

        if self.use_g05:
            if g05 is None or self.g05_encoder is None:
                raise ValueError("G05 입력 필요")

            # G05 관측점별 특징.
            point_features = self.g05_encoder(g05)

            # 순서 불변 G05 평균 특징.
            g05_features = point_features.mean(dim=1)

            # G00·G05 결합 특징.
            fused_features = torch.cat((g00_features, g05_features), dim=1)
        else:
            # G00-only 결합 특징.
            fused_features = g00_features

        # 공통 물리 특징.
        shared_features = self.shared_encoder(fused_features)

        # 정규화 위치 예측.
        position_prediction = self.position_head(shared_features)

        # 정규화 전하 예측.
        charge_prediction = self.charge_head(shared_features)

        return position_prediction, charge_prediction


# ============================================================
# 손실 함수
# ============================================================

def global_sign_invariant_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """전역 부호 반전에 불변인 전하 MSE."""

    # 원래 부호 방향의 샘플별 MSE.
    direct_error = torch.mean((prediction - target) ** 2, dim=1)

    # 전체 부호 반전 방향의 샘플별 MSE.
    flipped_error = torch.mean((prediction + target) ** 2, dim=1)

    return torch.minimum(direct_error, flipped_error).mean()


def calculate_losses(
    position_prediction: torch.Tensor,
    charge_prediction: torch.Tensor,
    position_target: torch.Tensor,
    charge_target: torch.Tensor,
    use_g05: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """위치·전하 분리 손실 계산."""

    # 위치 출력 MSE.
    position_loss = nn.functional.mse_loss(
        position_prediction,
        position_target,
    )

    if use_g05:
        # G05 기반 signed-q MSE.
        charge_loss = nn.functional.mse_loss(
            charge_prediction,
            charge_target,
        )
    else:
        # G00-only 전역 부호 불변 q MSE.
        charge_loss = global_sign_invariant_mse(
            charge_prediction,
            charge_target,
        )

    # 기능별 손실 가중 합.
    total_loss = (
        POSITION_LOSS_WEIGHT * position_loss
        + CHARGE_LOSS_WEIGHT * charge_loss
    )

    return total_loss, position_loss, charge_loss


# ============================================================
# epoch 실행
# ============================================================

def run_epoch(
    model: ChargeNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochLoss:
    """학습 또는 평가 epoch 실행."""

    # 학습 모드 여부.
    is_training = optimizer is not None

    model.train(mode=is_training)

    # 가중 합산 손실 누계.
    total_loss_sum = 0.0

    # 위치 손실 누계.
    position_loss_sum = 0.0

    # 전하 손실 누계.
    charge_loss_sum = 0.0

    # 처리 샘플 수.
    sample_count = 0

    for g00, g05, position_target, charge_target in loader:
        # 장치 배치 G00.
        g00_device = g00.to(DEVICE, non_blocking=True)

        # 장치 배치 G05.
        g05_device = (
            g05.to(DEVICE, non_blocking=True)
            if model.use_g05
            else None
        )

        # 장치 배치 위치 target.
        position_target_device = position_target.to(DEVICE, non_blocking=True)

        # 장치 배치 전하 target.
        charge_target_device = charge_target.to(DEVICE, non_blocking=True)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            # 정규화 위치·전하 예측.
            position_prediction, charge_prediction = model(
                g00_device,
                g05_device,
            )

            # 기능별 배치 손실.
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

        # 현재 배치 크기.
        current_batch_size = g00.shape[0]

        total_loss_sum += total_loss.item() * current_batch_size
        position_loss_sum += position_loss.item() * current_batch_size
        charge_loss_sum += charge_loss.item() * current_batch_size
        sample_count += current_batch_size

    if sample_count == 0:
        raise ValueError("빈 DataLoader")

    return EpochLoss(
        total=total_loss_sum / sample_count,
        position=position_loss_sum / sample_count,
        charge=charge_loss_sum / sample_count,
    )


# ============================================================
# 모델 학습
# ============================================================

def train_model(
    train_dataset: TensorDataset,
    validation_dataset: TensorDataset,
    use_g05: bool,
    seed: int,
) -> TrainingResult:
    """단일 시드 모델 학습."""

    set_reproducibility(seed)

    # 모델 표시 이름.
    model_name = "G00 + G05" if use_g05 else "G00 only"

    # 시드 고정 학습 DataLoader.
    train_loader = create_data_loader(
        train_dataset,
        shuffle=True,
        seed=seed,
    )

    # 고정 순서 검증 DataLoader.
    validation_loader = create_data_loader(
        validation_dataset,
        shuffle=False,
    )

    # 학습 대상 모델.
    model = ChargeNet(use_g05=use_g05).to(DEVICE)

    # AdamW 최적화기.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # 검증 최저 손실.
    best_validation_loss = float("inf")

    # 검증 최적 CPU 가중치.
    best_state: dict[str, torch.Tensor] | None = None

    # 연속 미개선 epoch 수.
    epochs_without_improvement = 0

    print()
    print("=" * 72)
    print(f"Training: {model_name} | seed={seed}")
    print("=" * 72)

    for epoch in range(1, EPOCHS + 1):
        # 학습 epoch 손실.
        train_loss = run_epoch(model, train_loader, optimizer)

        # 검증 epoch 손실.
        validation_loss = run_epoch(model, validation_loader)

        if not np.isfinite(validation_loss.total):
            raise FloatingPointError(
                f"검증 손실 비유한값: {model_name}, seed={seed}, epoch={epoch}"
            )

        print(
            f"Epoch {epoch:02d} | "
            f"Train total={train_loss.total:.6f} "
            f"pos={train_loss.position:.6f} q={train_loss.charge:.6f} | "
            f"Val total={validation_loss.total:.6f} "
            f"pos={validation_loss.position:.6f} q={validation_loss.charge:.6f}"
        )

        if validation_loss.total < best_validation_loss:
            # 갱신 검증 최저 손실.
            best_validation_loss = validation_loss.total

            # 갱신 검증 최적 CPU 가중치.
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }

            # 미개선 epoch 계수 초기화.
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping")
            break

    if best_state is None:
        raise RuntimeError("검증 최적 가중치 없음")

    model.load_state_dict(best_state)

    return TrainingResult(
        model=model,
        best_validation_loss=best_validation_loss,
        seed=seed,
    )


# ============================================================
# 모델 평가
# ============================================================

def align_global_charge_sign(
    prediction: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """평가용 전역 전하 부호 정렬."""

    # 원래 부호 방향의 샘플별 오차.
    direct_error = np.mean((prediction - target) ** 2, axis=1)

    # 전체 부호 반전 방향의 샘플별 오차.
    flipped_error = np.mean((-prediction - target) ** 2, axis=1)

    # 최소 오차 방향 계수.
    alignment = np.where(flipped_error < direct_error, -1.0, 1.0)[:, None]

    return prediction * alignment


def evaluate_model(
    model: ChargeNet,
    test_dataset: TensorDataset,
    stats: NormalizationStats,
) -> EvaluationResult:
    """실제 단위 테스트 평가."""

    # 고정 순서 테스트 DataLoader.
    test_loader = create_data_loader(test_dataset, shuffle=False)

    model.eval()

    # 정규화 위치 예측 배치 목록.
    position_prediction_batches: list[np.ndarray] = []

    # 정규화 전하 예측 배치 목록.
    charge_prediction_batches: list[np.ndarray] = []

    # 정규화 위치 target 배치 목록.
    position_target_batches: list[np.ndarray] = []

    # 정규화 전하 target 배치 목록.
    charge_target_batches: list[np.ndarray] = []

    with torch.inference_mode():
        for g00, g05, position_target, charge_target in test_loader:
            # 장치 배치 G00.
            g00_device = g00.to(DEVICE, non_blocking=True)

            # 장치 배치 G05.
            g05_device = (
                g05.to(DEVICE, non_blocking=True)
                if model.use_g05
                else None
            )

            # 정규화 배치 예측.
            position_prediction, charge_prediction = model(
                g00_device,
                g05_device,
            )

            position_prediction_batches.append(position_prediction.cpu().numpy())
            charge_prediction_batches.append(charge_prediction.cpu().numpy())
            position_target_batches.append(position_target.numpy())
            charge_target_batches.append(charge_target.numpy())

    # 전체 정규화 위치 예측.
    position_prediction_norm = np.concatenate(
        position_prediction_batches,
        axis=0,
    )

    # 전체 정규화 전하 예측.
    charge_prediction_norm = np.concatenate(
        charge_prediction_batches,
        axis=0,
    )

    # 전체 정규화 위치 target.
    position_target_norm = np.concatenate(position_target_batches, axis=0)

    # 전체 정규화 전하 target.
    charge_target_norm = np.concatenate(charge_target_batches, axis=0)

    # 실제 단위 위치 예측.
    position_prediction = (
        position_prediction_norm * stats.position_std + stats.position_mean
    )

    # 실제 단위 위치 target.
    position_target = (
        position_target_norm * stats.position_std + stats.position_mean
    )

    # 실제 단위 전하 예측.
    charge_prediction = charge_prediction_norm * stats.charge_scale

    # 실제 단위 전하 target.
    charge_target = charge_target_norm * stats.charge_scale

    if model.use_g05:
        # signed-q 평가용 전하 예측.
        evaluated_charge_prediction = charge_prediction

        # 절대 부호 정확도.
        absolute_sign_accuracy = float(
            np.mean(np.sign(charge_prediction) == np.sign(charge_target))
        )
    else:
        # 전역 부호 불변 평가용 전하 예측.
        evaluated_charge_prediction = align_global_charge_sign(
            charge_prediction,
            charge_target,
        )

        # G00-only 절대 부호 평가 제외.
        absolute_sign_accuracy = None

    # 위치 출력별 MAE.
    position_mae = np.mean(
        np.abs(position_prediction - position_target),
        axis=0,
    )

    # 첫 번째 전하 위치 거리 오차.
    position_error_1 = float(
        np.linalg.norm(
            position_prediction[:, 0:3] - position_target[:, 0:3],
            axis=1,
        ).mean()
    )

    # 두 번째 전하 위치 거리 오차.
    position_error_2 = float(
        np.linalg.norm(
            position_prediction[:, 3:6] - position_target[:, 3:6],
            axis=1,
        ).mean()
    )

    # 전하 출력별 MAE.
    charge_mae = np.mean(
        np.abs(evaluated_charge_prediction - charge_target),
        axis=0,
    )

    # 전하 크기 MAE.
    charge_magnitude_mae = float(
        np.mean(
            np.abs(
                np.abs(charge_prediction) - np.abs(charge_target)
            )
        )
    )

    # 예측 상대 부호.
    predicted_relative_sign = np.sign(
        charge_prediction[:, 0] * charge_prediction[:, 1]
    )

    # 정답 상대 부호.
    target_relative_sign = np.sign(
        charge_target[:, 0] * charge_target[:, 1]
    )

    # 상대 부호 정확도.
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
    use_g05: bool,
    seed: int,
) -> None:
    """단일 시드 테스트 지표 출력."""

    # 모델 표시 이름.
    model_name = "G00 + G05" if use_g05 else "G00 only"

    # 전하 MAE 표시 이름.
    charge_mae_name = "signed-q MAE" if use_g05 else "sign-invariant q MAE"

    print()
    print("=" * 72)
    print(f"Test: {model_name} | seed={seed}")
    print("=" * 72)
    print("Position MAE [x1, y1, z1, x2, y2, z2]")
    print(result.position_mae)
    print("Charge 1 position error:", result.position_error_1)
    print("Charge 2 position error:", result.position_error_2)
    print(f"{charge_mae_name} [q1, q2]:", result.charge_mae)
    print("Charge magnitude MAE:", result.charge_magnitude_mae)
    print("Relative sign accuracy:", result.relative_sign_accuracy)

    if result.absolute_sign_accuracy is not None:
        print("Absolute sign accuracy:", result.absolute_sign_accuracy)


def print_aggregate_evaluation(
    results: list[EvaluationResult],
    use_g05: bool,
) -> None:
    """다중 시드 평균·표준편차 출력."""

    # 모델 표시 이름.
    model_name = "G00 + G05" if use_g05 else "G00 only"

    # 위치 MAE 배열.
    position_mae_values = np.stack([result.position_mae for result in results])

    # 첫 번째 위치 거리 오차 배열.
    position_error_1_values = np.array(
        [result.position_error_1 for result in results]
    )

    # 두 번째 위치 거리 오차 배열.
    position_error_2_values = np.array(
        [result.position_error_2 for result in results]
    )

    # 전하 MAE 배열.
    charge_mae_values = np.stack([result.charge_mae for result in results])

    # 전하 크기 MAE 배열.
    charge_magnitude_values = np.array(
        [result.charge_magnitude_mae for result in results]
    )

    # 상대 부호 정확도 배열.
    relative_sign_values = np.array(
        [result.relative_sign_accuracy for result in results]
    )

    print()
    print("=" * 72)
    print(f"Multi-seed summary: {model_name}")
    print("=" * 72)
    print("Position MAE mean:", position_mae_values.mean(axis=0))
    print("Position MAE std :", position_mae_values.std(axis=0))
    print(
        "Charge 1 position error mean/std:",
        position_error_1_values.mean(),
        position_error_1_values.std(),
    )
    print(
        "Charge 2 position error mean/std:",
        position_error_2_values.mean(),
        position_error_2_values.std(),
    )
    print("Charge MAE mean:", charge_mae_values.mean(axis=0))
    print("Charge MAE std :", charge_mae_values.std(axis=0))
    print(
        "Charge magnitude MAE mean/std:",
        charge_magnitude_values.mean(),
        charge_magnitude_values.std(),
    )
    print(
        "Relative sign accuracy mean/std:",
        relative_sign_values.mean(),
        relative_sign_values.std(),
    )

    if use_g05:
        # 절대 부호 정확도 배열.
        absolute_sign_values = np.array(
            [result.absolute_sign_accuracy for result in results],
            dtype=np.float64,
        )

        print(
            "Absolute sign accuracy mean/std:",
            absolute_sign_values.mean(),
            absolute_sign_values.std(),
        )


# ============================================================
# 모델 및 통계 저장
# ============================================================

def save_model_checkpoint(
    training_result: TrainingResult,
    output_path: Path,
    arrays: DatasetArrays,
    stats: NormalizationStats,
) -> None:
    """추론 메타데이터 포함 모델 저장."""

    # CPU 모델 가중치.
    state_dict_cpu = {
        name: parameter.detach().cpu()
        for name, parameter in training_result.model.state_dict().items()
    }

    # 모델·전처리 메타데이터.
    checkpoint = {
        "model_state_dict": state_dict_cpu,
        "use_g05": training_result.model.use_g05,
        "seed": training_result.seed,
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
    }

    torch.save(checkpoint, output_path)


def save_normalization_stats(stats: NormalizationStats) -> None:
    """독립 정규화 통계 저장."""

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


# ============================================================
# 실행 진입점
# ============================================================

def main() -> None:
    """데이터 준비·다중 시드 학습·평가·저장."""

    set_reproducibility(DATA_SPLIT_SEED)

    print("Device:", DEVICE)
    print("Data:", DATA_PATH)

    # 검증 완료 원본 데이터.
    arrays = load_dataset(DATA_PATH)

    # 고정 데이터 분할.
    data_split = create_data_split(arrays.g00.shape[0], DATA_SPLIT_SEED)

    # 학습 분할 기반 정규화 통계.
    stats = calculate_normalization_stats(arrays, data_split.train)

    # 학습 TensorDataset.
    train_dataset = prepare_dataset(arrays, data_split.train, stats)

    # 검증 TensorDataset.
    validation_dataset = prepare_dataset(arrays, data_split.validation, stats)

    # 테스트 TensorDataset.
    test_dataset = prepare_dataset(arrays, data_split.test, stats)

    print("G00:", arrays.g00.shape)
    print("G05:", arrays.g05.shape)
    print("Target:", arrays.target.shape)
    print("Train:", len(train_dataset))
    print("Validation:", len(validation_dataset))
    print("Test:", len(test_dataset))

    # 모델 모드별 평가 결과.
    evaluation_results: dict[bool, list[EvaluationResult]] = {
        False: [],
        True: [],
    }

    # 모델 모드별 검증 최적 결과.
    best_training_results: dict[bool, TrainingResult] = {}

    for seed in EXPERIMENT_SEEDS:
        for use_g05 in (False, True):
            # 현재 시드 학습 결과.
            training_result = train_model(
                train_dataset,
                validation_dataset,
                use_g05=use_g05,
                seed=seed,
            )

            # 현재 시드 테스트 결과.
            evaluation_result = evaluate_model(
                training_result.model,
                test_dataset,
                stats,
            )

            evaluation_results[use_g05].append(evaluation_result)
            print_evaluation(evaluation_result, use_g05, seed)

            # 기존 검증 최적 결과.
            current_best = best_training_results.get(use_g05)

            if (
                current_best is None
                or training_result.best_validation_loss
                < current_best.best_validation_loss
            ):
                # 모델 모드별 검증 최적 결과 갱신.
                best_training_results[use_g05] = training_result

    print_aggregate_evaluation(evaluation_results[False], use_g05=False)
    print_aggregate_evaluation(evaluation_results[True], use_g05=True)

    save_model_checkpoint(
        best_training_results[False],
        MODEL_G00_PATH,
        arrays,
        stats,
    )
    save_model_checkpoint(
        best_training_results[True],
        MODEL_G00_G05_PATH,
        arrays,
        stats,
    )
    save_normalization_stats(stats)

    print()
    print("학습 완료")
    print("G00-only model:", MODEL_G00_PATH)
    print("G00+G05 model:", MODEL_G00_G05_PATH)
    print("Normalization:", NORMALIZATION_PATH)


if __name__ == "__main__":
    main()

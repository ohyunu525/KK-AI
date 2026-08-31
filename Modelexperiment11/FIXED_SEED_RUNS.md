# 확정 설정의 단일 seed 실행

`run_fixed_seed.ps1`은 기존 `Codes/ModelExperiment10.py`를 그대로 실행하는 PowerShell 진입점이다. 학습 코드, 모델, 데이터, loss, matching 또는 평가 방식을 바꾸지 않는다. 다른 seed의 결과나 가중치를 읽지 않는다.

프로젝트 루트에서 이 컴퓨터의 seed 42 실행:

```powershell
.\Modelexperiment11\run_fixed_seed.ps1 -Seed 42
```

다른 컴퓨터에서는 동일한 코드·데이터·실행 스크립트를 복사하고, 해당 컴퓨터에 배정한 **seed 숫자 하나만** 바꾼다. 실험 seed와 관계없이 데이터 분할 seed는 기존의 `42`로 고정된다. 같은 seed를 여러 컴퓨터에 중복 배정하지 않는다.

| 항목 | 고정값 |
|---|---|
| 모델 | `g05_sign_only`, `g05_full_reconstruction` |
| G05 fraction | `0.0, 0.1, 0.25, 0.5, 0.75, 1.0` |
| 관측 센서 수 | `0, 3, 8, 16, 24, 32`; 고정된 nested prefix |
| Optimizer | AdamW, lr `0.001`, weight decay `0.0001` |
| Structure dropout | `0.2`; global-sign 분기에는 dropout 없음 |
| Batch / 최대 epoch | `128` / `150` |
| 조기 종료 | validation total 또는 structure 개선 시 patience 초기화; patience `20`, min_delta `0`, min_epochs `0` |
| 데이터 분할 | seed `42`, train `8,000` / validation `1,000` / test `1,000` |
| 정규화 | train만으로 계산 |
| Loss | position, magnitude, relative-sign, global-sign 가중치 각각 `1.0`; 원래 loss 수식 유지 |
| Matching | 기존의 공동 비용을 최소화하는 5! = 120개 permutation 완전 탐색 |
| Checkpoint | validation structure와 total 각각의 첫 최솟값에 해당하는 전체 모델 상태 |
| 기본 보고 기준 | `best_structure.pt`; `best_total.pt`의 결과도 별도로 보존 |

모든 조건은 같은 seed에서 매번 새 모델과 optimizer로 시작한다. 다른 fraction이나 seed의 구조 가중치를 재사용하지 않는다. Test 추론과 지표 계산은 **각 실행의 학습과 validation 기반 checkpoint 선택이 끝난 뒤** 수행한다. Test 값으로 checkpoint·fraction·설정을 다시 고르지 않는다. 이 작업에는 추가 튜닝이나 seed 간 결과 집계가 없다.

## 출력 및 충돌 방지

출력 이름은 seed에서 자동으로 정해진다. 예를 들어 seed 42는 다음 경로만 사용한다.

- `Modelexperiment11/fraction_sweep_results/seed42_g05_sweep_dropout020/`: protocol, 분할 인덱스, 정규화, 조건별 설정·학습 이력·validation 선택·test 지표, CSV 집계.
- `Modelexperiment11/fraction_sweep_checkpoints/seed42_g05_sweep_dropout020/`: 각 모델·fraction·seed·설정 fingerprint가 포함된 폴더에 `latest.pt`, `best_structure.pt`, `best_total.pt`.
- `Modelexperiment11/fraction_sweep_launches/seed42_g05_sweep_dropout020/`: 실행 인수, 입력 SHA-256, 소스 스냅샷, 전체 콘솔 로그와 종료 코드.

기본 명령은 결과 또는 checkpoint 폴더가 이미 있으면 거부하므로 기존 실행을 덮어쓰거나 자동으로 재사용하지 않는다. 기존 실행의 중단 지점부터 이어갈 때에만 명시적으로 `-Resume`을 붙인다. 동일 seed의 동시 쓰기는 기존 OS 파일 잠금으로 차단한다.

```powershell
# 읽기 전용으로 실행 설정과 경로만 확인
.\Modelexperiment11\run_fixed_seed.ps1 -Seed 42 -DryRun

# 이 seed의 저장된 학습만 이어가기; 다른 seed를 탐색하거나 가져오지 않음
.\Modelexperiment11\run_fixed_seed.ps1 -Seed 42 -Resume

# 완료 산출물 검산; 학습·test 재추론 없음
.\.venv\Scripts\python.exe Modelexperiment11\verify_fixed_seed.py --seed 42 --require-fresh
```

`--require-fresh`는 모든 실행이 중단·재개 없이 epoch 1에서 시작했음까지 확인한다. 실제로 같은 seed를 중단 후 재개했다면 이 옵션을 빼고 검산한다. 결과 검산기는 기존 CSV를 재학습 없이 검증하고 `seed_audit.json`, `checkpoint_hashes.json`, `report.md`를 생성한다.

## 다른 컴퓨터에서의 재현

실행 스크립트는 확정된 두 학습 소스와 `Models/charge_dataset_5charges_v9.npz`의 SHA-256을 검사한다. 입력이 다르면 학습 전에 중단한다. 다른 컴퓨터의 seed 결과나 checkpoint를 복사해 초기화할 필요는 없다. 각 컴퓨터에 프로젝트 `.venv`를 준비하고 원래 환경과 같은 Python, NumPy, PyTorch/CUDA 버전을 사용한다. 실제 버전, 장치, 결정론 설정은 각 `protocol.json`에 기록된다.

`protocol_fingerprint`에는 원래 설계대로 절대 데이터 경로와 실행 환경도 포함되므로 컴퓨터가 다르면 달라질 수 있다. 검산 결과의 `portable_scientific_protocol_sha256`은 과학적 설정·분할·소스·데이터·조건이 같은지 확인하는 보조 키다. 원래 protocol이나 fingerprint는 수정하지 않는다. 이 보조 키가 같더라도 GPU/라이브러리 버전 차이를 숨기거나 컴퓨터 간 bitwise 동일성을 보장하지 않는다.

Test는 기존 프로젝트에서 사용했던 동일한 1,000개 표본이다. 새로운 독립 holdout으로 표현하지 않으며, 단일 seed만으로 분산·통계적 유의성이나 seed 전반의 성능 향상을 결론 내리지 않는다.

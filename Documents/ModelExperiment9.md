# ModelExperiment9: 5전하 G05 routing 비교

## 1. 기존 NewLearning9 구조

[NewLearning9.py](C:/Users/COM/Documents/KK-AI/Codes/NewLearning9.py)는 수정하지 않았다.
전하 수는 5개이고 target은 순서 없는 `(x, y, z, q)` 집합이다. `G00=V²`, `G05=V`이며,
예측 슬롯과 정답 사이의 일대일 대응은 기존 위치·크기·상대부호 비용으로 120개 순열을 모두 비교한다.
Global-sign 예측이나 손실은 assignment 비용에 들어가지 않는다.

부호 정의도 그대로다.

```text
s_i = sign(q_i)
g = product(s_i)
r_i = s_i * g
product(r_i) = +1
q_i = magnitude_i * r_i * g
```

상대부호 손실과 decoder는 16개 valid pattern을 사용한다. Global logit은 기존의
`(f(V) - f(-V)) / 2` 형태로 계산한다. 전하를 모두 반전하면 `g`만 바뀌고 `r_i`는 유지된다.

다음 함수는 복제하거나 재설계하지 않고 baseline에서 직접 호출한다.

| 영역 | 재사용하는 함수 |
| --- | --- |
| 데이터·물리 검사 | `load_dataset`, `validate_dataset`, `verify_physical_consistency` |
| 분할·정규화·센서 | `create_data_split`, `calculate_normalization_stats`, `prepare_dataset`, `create_data_loader` |
| Matching·부호 | `matching_cost`, `minimum_cost_assignment`, `canonical_sign_targets`, `decode_relative_signs` |
| 학습·평가 | `calculate_losses`, `run_epoch`, `evaluate_model` |

기본값은 split seed 42의 80/10/10 분할, 모든 전하 슬롯에 공통인 train-only 위치 정규화,
AdamW, learning rate `0.001`, weight decay `0.0001`, batch size `128`, loss weights 모두 `1.0`이다.
후보 센서는 동일한 fixed nested prefix를 사용한다. 현재 데이터는 32개 후보 센서이므로
75%는 24개, 100%는 32개이며 **32×32 field 전체 관측을 뜻하지 않는다**.

## 2. Full reconstruction을 위한 최소 변경

별도 [ModelExperiment9.py](C:/Users/COM/Documents/KK-AI/Codes/ModelExperiment9.py)에
`NewLearning9.ChargeNet`의 subclass를 추가했다. `super().__init__()`으로 기존 모듈을 먼저 만들고,
그 뒤에 두 모델 모두 동일한 `structure_context`를 만든다.

| 항목 | g05_sign_only | g05_full_reconstruction |
| --- | --- | --- |
| Position / magnitude / relative sign | G00 only | G00 + masked G05 context |
| Global sign | 기존 G05-only branch | 같은 G05-only branch |
| G05 encoder / mean·max·std pooling | 기존 구현 | 기존 구현 재사용 |
| 등록 파라미터 수 | 406,969 | 406,969 |
| Module 생성 순서·초기 state | 동일 | 동일 |
| `structure_context` | 생성하되 사용하지 않음 | Structure에 연결 |

추가 경로는 다음과 같다.

```text
summary_plus  = masked_mean_max_std(g05_encoder([x, y,  V]))
summary_minus = masked_mean_max_std(g05_encoder([x, y, -V]))
even_summary  = (summary_plus + summary_minus) / 2
context       = Linear(96,128) -> ReLU -> Linear(128,256)
structure     = g00_features + has_G05 * context(even_summary)
```

원래 global-sign branch는 그대로 호출한다. Structure는 전체 부호 반전에 불변이어야 하므로
G05 summary의 even 부분을 사용한다. 마지막 `Linear(128,256)`의 weight와 bias는 0으로 초기화한다.
따라서 두 모델의 초기 출력이 같고, G05가 없는 경우에는 학습된 context bias도 structure에 들어가지 않는다.
Mask는 encoder 전에 적용하므로 숨긴 센서의 큰 값이나 NaN이 forward에 새어 들어가지 않는다.

새 학습 loop는 저장·재개·비교를 담당하며 실제 epoch 학습과 test metric 계산은 기존 함수를 호출한다.
각 `(model, fraction, seed)`를 동일한 초기화·batch 순서로 학습한다. Baseline의 fraction 간
structure 재사용과 component composition은 이 비교 코드에서는 사용하지 않는다.

## 3. 해석 시 주의할 실험 설계 문제

1. **Full의 G05 encoder는 공유된다.** Structure와 global-sign 손실이 모두 이 encoder를 갱신한다.
   Global-sign loss의 backward가 G00 CNN/encoder, position/magnitude/relative-sign head,
   `structure_context`에 직접 도달하지는 않는다. 하지만 공유 encoder의 갱신은 다음 forward의
   structure context에 간접적으로 영향을 줄 수 있다. 따라서 branch별 최적 epoch를 합성하지 않는다.
2. **등록 파라미터 수와 활성 경로의 크기는 구분해야 한다.** Sign-only의 context는 사용하지 않는다.
   두 모델은 동일한 모듈·초기 state를 가지지만 full은 추가 경로를 실제로 사용한다.
3. **첫 seed42 비교는 탐색적 결과다.** 한 seed의 개선만으로 일반적 우위를 단정하지 않는다.
   `paired_seed_count=1`이면 sample standard deviation은 빈칸/N/A로 저장한다.
   이후 seed 41/42/43의 paired 개선량을 함께 확인한다.
4. **주 연구 비교는 structure 선택이다.** Total 선택의 위치 성능에는 global-sign validation loss에
   따른 선택 epoch 변화도 섞일 수 있다. Structure 선택의 global/absolute-sign 지표는 참고값으로 표시한다.
5. **대칭식과 부동소수 bitwise 동일성은 다르다.** 기존 baseline의 CPU batch 4에서도 global logit
   반전 시 최대 `1.257285475730896e-08`의 차이를 재현했다. Baseline 구현은 변경하지 않았고,
   대칭 검사에는 `atol=1e-7`, `rtol=1e-6`을 명시했다. 초기 가중치·동일 입력 출력·resume 상태 비교에는
   이러한 허용오차를 적용하지 않는다. 현재 환경의 CPU/CUDA resume는 오차 없이 일치했다.
6. **음의 결과는 이 fusion 설계에 대한 결과다.** Additive residual과 even summary가 유용하지 않았다는
   결과만으로 G05에 구조 정보가 전혀 없다고 결론 내릴 수는 없다. 주어진 baseline 수치도 강제 통과 기준이 아니다.

## 4. 수정된 전체 Python 코드

- [전체 실행 코드: ModelExperiment9.py](C:/Users/COM/Documents/KK-AI/Codes/ModelExperiment9.py)
- [회귀 테스트: test_model_experiment9.py](C:/Users/COM/Documents/KK-AI/Codes/test_model_experiment9.py)

실행 코드는 생략 없는 완전한 파일이며, 같은 디렉터리의 기존 `NewLearning9.py`를 사용한다.
새 외부 라이브러리는 추가하지 않았다. 기존 2전하 실험 코드도 수정하지 않았다.

추론 시에는 새 파일의 `load_trained_model()`을 사용한다. 두 선택 checkpoint는 normalization과
configuration을 포함하므로 별도의 branch 파일을 합성할 필요가 없다. 구형 baseline checkpoint는
명시적인 호환성 오류로 거부한다. 로더에는 이 프로젝트가 생성한 신뢰할 수 있는 checkpoint만 전달한다.

## 5. 저장 디렉터리와 재개

기준 경로는 `C:\Users\COM\Documents\KK-AI`다.

```text
Models/new_learning9_experiments/<experiment_name>/
└── <model>__g05_075pct__seed_42__<fingerprint-12>/
    ├── latest.pt
    ├── best_structure.pt
    └── best_total.pt

Results/new_learning9_experiments/<experiment_name>/
├── protocol.json
├── normalization.json
├── split_indices.npz
├── runs.csv
├── summary.csv
├── pairwise_comparisons.csv
├── pairwise_summary.csv
└── runs/<same-run-id>/
    ├── config.json
    ├── status.json
    ├── history.json
    └── result.json
```

| 산출물 | 의미 |
| --- | --- |
| `best_structure.pt` | Validation structure loss 최소 epoch의 complete model state |
| `best_total.pt` | Validation total loss 최소 epoch의 complete model state |
| `latest.pt` | 마지막 epoch의 model·optimizer·RNG·shuffle·history·두 best 전체 snapshot |
| `runs.csv` | Model run × selection; 첫 비교 완료 시 4행 |
| `summary.csv` | Protocol × model × fraction × selection의 seed 집계 |
| `pairwise_comparisons.csv` | 동일 protocol/fraction/seed/selection의 지표별 A/B 값과 개선량 |
| `pairwise_summary.csv` | 지표별 paired seed 수·목록, delta mean/std, improvement mean/std |

각 epoch에서 latest를 먼저 원자적으로 저장하고 두 best 파일을 게시한다. 파일 데이터는 flush/fsync 후
교체한다. 재개 시 latest에 포함된 두 best와 history를 복원하므로 중간 게시에서 중단되어도 다시 맞출 수 있다.
동점은 첫 epoch를 유지한다. Best는 서로 다른 epoch의 component를 섞지 않는다.

Test는 학습이 끝난 뒤 같은 test set으로 두 선택을 평가한다. 두 평가가 끝나야 `result.json`을 저장한다.
평가 도중 중단되었다면 학습을 반복하지 않고 평가를 다시 수행한다. 완료 run은 재실행 시 skip하되,
누락된 best는 유효한 latest에서 복원하고 status를 completed로 맞춘다. 정상 checkpoint와 result는 다시 쓰지 않는다.

코드 두 파일의 SHA256, 데이터 SHA256, 물리 protocol, normalization, optimizer 설정과 실행 환경을
fingerprint에 포함한다. 같은 이름에 다른 protocol을 섞지 않는다. **재개에는 같은 코드·데이터·환경·설정과
같은 experiment name이 필요하다.** Epoch 수 변경도 다른 protocol이므로 처음부터 `--epochs 300`을 지정한다.
Fractions와 seeds는 공통 protocol에서 제외하고 run configuration에 기록하므로 나중에 같은 이름으로 sweep을
확장할 수 있다. 이미 끝난 조건은 skip한다.

동일 결과 또는 checkpoint 디렉터리에 대한 동시 CLI 실행은 OS 파일 잠금으로 거부한다.
잠금 파일은 experiment 폴더 옆의 `.<experiment_name>.lock`이며, 프로세스가 종료되면 잠금이 풀린다.
파일이 남아 있어도 재개할 수 있으므로 실행 중에 잠금 파일을 수동으로 삭제하지 않는다.

결과 JSON을 읽지 못하면 CSV 갱신을 보류한다. 비교 대상이 없어지면 기존 CSV는 header만 남긴다.
손상된 history, 누락/비정상 metric, 잘못된 관측 수, 불완전한 모델 state는 조용히 집계하지 않고 오류로 처리한다.

`model_a=g05_sign_only`, `model_b=g05_full_reconstruction`이며,
`delta_b_minus_a = value_b - value_a`다. Error metric은 부호를 뒤집어
**`improvement_b_over_a > 0`이 항상 full reconstruction의 개선**을 뜻한다.
G05=0의 global/absolute-sign metric은 N/A이며 paired summary에서도 유효 seed 수를 0으로 기록한다.

## 6. Smoke-test 명령어

원본 5전하 데이터 `C:\Users\COM\Documents\KK-AI\Models\charge_dataset_5charges_v9.npz`를 재사용한다.
10,000개 표본·32개 후보 센서를 확인했다. 데이터를 다시 생성하지 않는다.

```powershell
& "C:\Users\COM\Documents\KK-AI\.venv\Scripts\python.exe" `
  "C:\Users\COM\Documents\KK-AI\Codes\ModelExperiment9.py" `
  --models "g05_sign_only,g05_full_reconstruction" `
  --fractions 0.75 --seeds 42 --epochs 300 --device cuda --smoke-only
```

Smoke는 train subset만 사용한다. G05=0과 양의 관측 조건을 모두 검사하며, zero projection의 upstream
gradient는 폐기할 모델에서 structure update 한 번 후 확인한다. 실제 학습 모델의 초기화를 바꾸지 않는다.
Shape·capacity·initial state·batch order·120개 assignment·16개 sign pattern·gradient 경로·부호 대칭·mask·
유한값·optimizer step·checkpoint roundtrip을 검사하고, test-set 평가나 본 실험 디렉터리는 만들지 않는다.

전체 회귀 테스트:

```powershell
& "C:\Users\COM\Documents\KK-AI\.venv\Scripts\python.exe" -m unittest discover `
  -s "C:\Users\COM\Documents\KK-AI\Codes" -p "test_*.py" -v
```

## 7. 75% / seed42 / 300 epoch 비교 명령어

```powershell
& "C:\Users\COM\Documents\KK-AI\.venv\Scripts\python.exe" `
  "C:\Users\COM\Documents\KK-AI\Codes\ModelExperiment9.py" `
  --experiment-name g05_routing_v1 `
  --models "g05_sign_only,g05_full_reconstruction" `
  --fractions 0.75 --seeds 42 --epochs 300 --device cuda
```

기본 CLI도 이 두 run만 실행하도록 설정했다. 중단 후 동일 명령을 다시 실행하면 재개한다.
완료 후에는 structure 선택의 세 핵심 오차부터 확인한다.

```powershell
Import-Csv "C:\Users\COM\Documents\KK-AI\Results\new_learning9_experiments\g05_routing_v1\pairwise_comparisons.csv" |
  Where-Object {
    $_.checkpoint_selection -eq "structure" -and
    $_.metric -in @("mean_position_mae", "mean_position_3d_error", "charge_magnitude_mae")
  } |
  Select-Object metric, seed, value_a, value_b, improvement_b_over_a, selected_epoch_a, selected_epoch_b
```

## 8. 이후 전체 sweep 명령어

첫 비교를 검토한 다음 실행한다. 6 fractions × 3 seeds × 2 models로 전체 36개 조건이며,
같은 이름으로 실행하면 이미 완료된 75% / seed42 조건은 skip한다.

```powershell
& "C:\Users\COM\Documents\KK-AI\.venv\Scripts\python.exe" `
  "C:\Users\COM\Documents\KK-AI\Codes\ModelExperiment9.py" `
  --experiment-name g05_routing_v1 `
  --models "g05_sign_only,g05_full_reconstruction" `
  --fractions "0,0.10,0.25,0.50,0.75,1.0" `
  --seeds "41,42,43" --epochs 300 --device cuda
```

## 검증과 구현 후 재리뷰 — 2026-08-28

- 새 회귀 테스트 26개 통과. 전체 **71개 중 68개 통과, 기존 3개 skip**.
  Skip 사유는 기존 legacy `train_g05_fraction_experiment` 모듈 부재다.
- 기존 baseline과 sign-only의 공통 초기 weights 및 소형 CPU 3-epoch 학습 궤적이 오차 없이 일치했다.
- G05=0에서 두 모델의 학습 상태·선택 결과가 일치했고, sign-only의 structure 궤적은 0/75/100%에서 같았다.
- 네 저장 경계에서의 중단/재개, CPU 및 RTX 3070 CUDA의 model·optimizer·RNG·shuffle·history·best 상태를
  오차 없이 비교했다. 이는 현재 검증한 런타임의 결과이며 모든 CUDA 환경의 bitwise 재현 보장은 아니다.
- 실제 원본 데이터의 train subset으로 CUDA smoke test를 통과했다.
- 별도 임시 20개 물리 표본으로 CUDA CLI를 실행했다: 두 모델 × 2 epochs, fraction 0.75, seed42.
  CSV 행 수 4/4/30/30과 재실행 시 두 run skip, checkpoint 여섯 파일의 내용 보존을 확인했다.
- 재리뷰에서 구형 checkpoint의 불명확한 오류, 비정상 history/누락 metric/잘못된 관측 수 검증 부족을
  테스트로 재현하고 수정했다. 동시 writer 잠금과 configuration에 맞는 고정 센서 mask 검사도 추가했다.
- Compile·전체 테스트·저장/재개 검사를 수행했고 검증용 임시 데이터와 결과는 정리했다.

**300-epoch 본 비교와 전체 sweep은 실행하지 않았다.** 따라서 5전하에서 실제 위치·크기·상대부호 개선이
발생했는지는 아직 판단하지 않았다. 위 명령으로 본 비교를 수행한 뒤 structure-selected paired 결과로 판단한다.

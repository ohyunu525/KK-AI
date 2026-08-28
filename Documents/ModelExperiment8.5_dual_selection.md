# ModelExperiment8.5: total / structure checkpoint 분리

## 문제와 수정 범위

기존 코드는 `validation_loss.total`이 가장 작은 epoch 하나만 `best.pt`로 저장했다.
따라서 G05가 structure branch에 들어가지 않는 `g05_sign_only`에서도 G05 fraction에 따른
global-sign validation loss 차이가 선택 epoch를 바꿀 수 있었다. 이때 위치 성능 차이에
checkpoint 선택 효과가 섞인다.

수정본은 **한 번의 동일한 학습 궤적에서 두 checkpoint를 독립적으로 선택**한다.

| 파일 | 선택 objective | 평가 목적 |
| --- | --- | --- |
| `best_total.pt` | `validation_loss.total` 최소 | 기존 전체 reconstruction 평가와의 연속성 |
| `best_structure.pt` | `validation_loss.structure` 최소 | position / magnitude / relative-sign reconstruction 비교 |
| `latest.pt` | 마지막으로 저장 완료한 epoch | 이어학습 전용; 최적 성능 평가용이 아님 |

각 best 파일에는 **해당 단일 epoch의 전체 `model_state_dict`**가 들어간다.
다른 epoch의 head나 component를 조합하지 않는다. 동점이면 기존 `<` 규칙처럼 먼저 나온 epoch를 유지한다.

변경하지 않은 항목: model architecture와 G05 pooling, parameter 수와 초기화,
G00=V² / G05=V, target 정의, split, train split으로만 계산하는 normalization,
fixed spatially-balanced nested sensor prefix, loss 식과 weight, AdamW 설정,
seed와 batch shuffle 처리, 기존 metric 계산식.
좌표별 MAE는 기존 `position_mae`에서 같은 값을 scalar key로도 제공하여 summary와 pairwise에도 포함했다.

**학습 objective는 여전히 total loss다.** Structure checkpoint에서 제거한 것은
global-sign loss의 *checkpoint 선택 영향*이다. 기존 full model의 공유 G05 encoder를 통한
학습 gradient까지 제거한 것은 아니다.

## 저장과 resume

매 epoch마다 다음 순서로 저장한다.

1. Total / structure 최적값을 각각 갱신한다. 전체 상태는 CPU로 복사하고 clone하여 이후 optimizer update와 분리한다.
2. 현재 epoch, model, optimizer, Python / NumPy / PyTorch / CUDA RNG,
   shuffle generator, history, elapsed time, 두 best loss / epoch와 두 best 전체 snapshot을
   `latest.pt`에 원자적으로 저장한다.
3. 갱신된 `best_total.pt` / `best_structure.pt`를 각각 원자적으로 저장한다.

재개할 때는 `latest.pt`를 기준으로 두 best 파일과 history를 복원한 뒤 다음 epoch를 진행한다.
따라서 latest 저장 후 한쪽 best 파일만 저장된 상태에서 중단되어도 일관되게 복구한다.
마지막 epoch까지 저장했고 test 평가 중에 중단됐다면 학습을 반복하지 않고 두 checkpoint를 다시 평가한다.
`result.json`은 두 평가가 모두 끝난 뒤 한 번에 저장되며, 완료된 run은 재실행 시 skip한다.
Skip 전에도 두 best 파일의 metadata를 확인한다. 누락된 best 파일은 검증된 마지막 epoch의
`latest.pt`에서 복원하고, `status.json`을 completed로 동기화한다. 이때 학습·test 평가를 반복하거나
정상 checkpoint와 `result.json`을 다시 쓰지 않는다. 복원할 latest가 없거나 유효하지 않으면 오류로 중단한다.

`latest.pt`에는 `best_total_loss`, `best_total_epoch`, `best_structure_loss`,
`best_structure_epoch`, `best_checkpoints.total`, `best_checkpoints.structure`가 있다.
기존 `best_validation_loss`, `best_epoch`는 total 기준 alias로 남겼다.
Best snapshot을 추가로 보관하므로 latest 파일 크기와 저장량은 늘어날 수 있다.

## 결과와 metadata

`result.json`의 `evaluations.total`, `evaluations.structure`에 각각 다음 정보가 저장된다.

- `checkpoint_selection`, `selection_objective`, `selected_epoch`, `selected_validation_loss`
- 선택 epoch의 모든 `validation_losses`와 전체 `test_metrics`
- model name, G05 fraction / point count, seed, run / protocol fingerprint
- checkpoint 경로, `primary_metrics`, `global_sign_in_selection_objective`, `global_sign_metrics_note`

`global_sign_in_selection_objective`는 해당 run에서 total 선택이고 G05 관측 수가 양수이며
global-sign loss weight가 0이 아닐 때만 true다. G05=0 또는 weight=0이면 false다.
관측 수가 아직 정해지지 않은 공통 protocol의 total 항목은 null이며, weight=0이면 false다.

Structure 평가에서도 기존 전체 지표를 계산하지만 global / absolute / signed-pair sign 지표와
signed charge MAE는 global-sign 성능을 최적화해 선택한 결과가 아니라는 점을 명시했다.
주된 structure 지표는 위치 MAE 6개, mean position MAE, 두 charge의 3D 위치 오차,
mean 3D 위치 오차, charge magnitude MAE, relative sign accuracy다.

| CSV | 행과 집계 단위 |
| --- | --- |
| `runs.csv` | run × `checkpoint_selection`: 학습 6회라면 평가 12행 |
| `summary.csv` | protocol × selection × model × fraction; seed 간 mean / sample std |
| `pairwise_comparisons.csv` | 같은 protocol / selection / fraction / seed끼리 비교 |

기존 CSV 지표 이름은 유지했다. `runs.csv`의 `best_epoch`, `best_validation_loss`,
`best_checkpoint`는 **해당 행의 selection**에 대응한다.
`summary.csv`와 pairwise에는 seed별 선택 epoch / validation loss / run fingerprint도 JSON 형식 컬럼으로 저장한다.
**외부 분석 코드도 반드시 `checkpoint_selection`으로 필터링하거나 그룹화해야 한다.**
결과 JSON 하나라도 읽을 수 없으면 세 CSV의 갱신을 모두 보류한다. 결과가 제거되어 집계나 비교 행이
없어지면 기존 CSV는 header만 남겨 이전 행이 최신 결과처럼 보이지 않도록 한다.

기본 모델 순서는 `model_a=g05_sign_only`, `model_b=g05_full_reconstruction`이다.
`improvement_b_over_a_mean > 0`이면 model_b가 더 좋다는 기존 의미를 유지했다.
`improvement_b_over_a_by_seed`에는 seed별 개선량을 저장하므로, 평균 개선뿐 아니라
41 / 42 / 43 모두 개선되었는지도 확인할 수 있다.
Structure pairwise의 `mean_position_mae`, `mean_position_3d_error`에는
`primary_research_metric=True`가 표시된다.

## 디렉터리 구조

아래 트리의 기준 경로는 `C:\Users\COM\Documents\KK-AI`다.

```text
Models/model_experiments/<experiment-name>/
└── <model>__g05_075pct__seed_<seed>__<run-fingerprint-12>/
    ├── latest.pt
    ├── best_total.pt
    └── best_structure.pt

Results/model_experiments/<experiment-name>/
├── protocol.json
├── split_indices.npz
├── normalization.json
├── runs.csv
├── summary.csv
├── pairwise_comparisons.csv
└── runs/<run-id>/
    ├── config.json
    ├── status.json
    ├── history.json
    └── result.json             # evaluations.total / evaluations.structure
```

## 기존 명령어와 구형 산출물 호환성

기존 CLI 옵션은 모두 유지된다. 선택 기준을 켜는 새 옵션은 필요하지 않다.
기본 experiment name만 `g05_routing_dual_selection_v2`로 바꿨고,
protocol version은 `model-experiment-v2-dual-selection`이다.
물리 protocol을 바꾼 것이 아니라 checkpoint / 결과 schema 변경을 구분하기 위한 버전이다.

명시적으로 구형 결과가 있는 `--experiment-name`을 재사용하면 기존 fingerprint 검사가 이를 거부한다.
새 이름으로 실행해야 하며, 구형 결과와 checkpoint는 수정하거나 삭제하지 않는다.
새 run에서는 **동일 코드·환경·설정·experiment name으로 같은 명령어를 다시 실행**하면 resume / skip한다.
Epoch 수 변경도 기존처럼 protocol 변경으로 취급하므로, 처음부터 최종 목표인 `--epochs 300`을 지정한다.

구형 `best.pt` / `latest.pt`에는 과거 structure 최적 epoch의 전체 상태가 없을 수 있다.
따라서 이를 새 dual-selection run으로 자동 변환하거나, total checkpoint를 structure 결과로
재표기하지 않는다. 구형 run의 이어학습은 그 run의 원본 코드로 진행해야 한다.
새 run에는 `best.pt` alias를 만들지 않으므로 직접 파일을 읽던 코드는 `best_total.pt`로 바꿔야 한다.
구형 JSON의 top-level `test_metrics`를 읽던 코드는 `evaluations.total.test_metrics` 또는
`evaluations.structure.test_metrics`를 명시적으로 선택해야 한다.

## 추천 실행 명령어 — PowerShell

검증 당시 기본 데이터셋 `C:\Users\COM\Documents\KK-AI\Models\charge_dataset_multipoint_v2.npz`가 없었다.
**기존 75% 실험에 사용한 원본 2전하 데이터를 복구하거나 `--data`에 그 파일의 실제 경로를 지정한 뒤 실행한다.**
현재 존재하는 5전하 데이터셋은 이 코드의 데이터 대체재가 아니다.
원본 데이터나 생성 seed를 임의로 바꾸면 과거 결과와의 비교 조건이 달라진다.

데이터를 복구하기 전에도 다음 회귀 테스트는 자체 임시 소형 데이터를 사용하므로 실행할 수 있다.

```powershell
& "C:\Users\COM\Documents\KK-AI\.venv\Scripts\python.exe" -m unittest discover `
  -s "C:\Users\COM\Documents\KK-AI\Codes" -p "test_*.py" -v
```

원본 데이터로 smoke test:

```powershell
& "C:\Users\COM\Documents\KK-AI\.venv\Scripts\python.exe" `
  "C:\Users\COM\Documents\KK-AI\Codes\ModelExperiment8.5.py" `
  --data "C:\Users\COM\Documents\KK-AI\Models\charge_dataset_multipoint_v2.npz" `
  --fractions 0.75 --seeds "41,42,43" --epochs 300 --device cuda --smoke-only
```

본 실험 — 75% × 3 seeds × 2 models, 모델당 300 epochs:

```powershell
& "C:\Users\COM\Documents\KK-AI\.venv\Scripts\python.exe" `
  "C:\Users\COM\Documents\KK-AI\Codes\ModelExperiment8.5.py" `
  --data "C:\Users\COM\Documents\KK-AI\Models\charge_dataset_multipoint_v2.npz" `
  --experiment-name g05_075_dual_v2 `
  --models "g05_sign_only,g05_full_reconstruction" `
  --fractions 0.75 --seeds "41,42,43" --epochs 300 --device cuda
```

완료 후 핵심 structure 결과만 확인:

```powershell
Import-Csv "C:\Users\COM\Documents\KK-AI\Results\model_experiments\g05_075_dual_v2\pairwise_comparisons.csv" |
  Where-Object {
    $_.checkpoint_selection -eq "structure" -and
    $_.metric -in @("mean_position_mae", "mean_position_3d_error")
  } |
  Select-Object metric, paired_seed_count, paired_seeds,
    improvement_b_over_a_mean, improvement_b_over_a_by_seed
```

## 실제 수행한 검증 (2026-08-28)

- `py_compile` 통과, `git diff --check` 통과.
- 전체 unittest 45개: **42개 통과, 3개 skip**. Skip은 원래 없던 legacy
  `train_g05_fraction_experiment` 모듈에 의존하는 테스트다.
- 새 회귀 테스트 20개: 서로 다른 best epoch와 동점 처리, global-sign validation 변화와
  structure 선택 독립성, 단일 epoch 전체 상태 저장, checkpoint 경로 분리,
  저장 전후 네 지점에서의 중단·재개, 평가 중단·재개, 완료 run skip,
  모델·optimizer·RNG·shuffle·history의 정확한 재현, metadata / 집계 분리,
  G05=0 출력, 동일 초기화·batch order, train-only normalization과 nested prefix를 확인했다.
- 리뷰 후 추가한 6개 테스트는 읽기 실패 시 CSV 보존, 결과 제거 시 기존 행 비우기,
  완료 run의 누락 checkpoint 복원과 유효하지 않은 latest 거부, result 저장 직후 중단된 status 복구,
  global-sign weight=0 metadata를 확인한다. 기존 G05=0 테스트에도 metadata 검증을 추가했다.
- 수정 전 소스와 18개 architecture / loss / training / RNG / normalization 정의의 AST가 일치했다.
  같은 현재 런타임의 소형 CPU 실험(20개 물리 표본, 3 epochs, 두 모델)에서 원본과
  새 `best_total`의 전체 상태, optimizer, RNG, history, 기존 test metric 값이 정확히 일치했다.
- 실제 CUDA CLI: 20개 임시 물리 표본, fraction 0.75, seeds 41 / 42 / 43,
  **2 epochs × 6 runs**. `runs.csv` 12행, summary 4행, pairwise 36행과 selection별
  paired seed 수 3을 확인했다. 재실행 시 6개 run 모두 skip하고 checkpoint 파일은 그대로였다.
- RTX 3070에서 full model을 3 epochs 학습하는 소형 중단·재개 실험도 상태와 지표가 정확히 일치했다.
  이는 검증한 현재 환경에서의 결과다. 기존 `warn_only=True` 설정의 CuBLAS 및
  `adaptive_avg_pool2d_backward_cuda` 비결정성 경고는 유지되어, 모든 CUDA 환경에서의
  bitwise 재현을 보장한다는 뜻은 아니다. 이 작업에서 kernel / seed 설정을 변경하지 않았다.
- 기존 checkpoint 테스트의 `import ModelExperiment`가 실제 파일명과 맞지 않아,
  `ModelExperiment8.5.py`를 경로로 import하도록 테스트만 수정했다.
- 기존 full model의 마지막 G05 structure projection은 0으로 초기화된다. 따라서 초기에는
  그 projection까지만 structure gradient가 도달한다. 강화한 smoke test는 폐기할 테스트 모델에서
  한 번의 structure update 후 G05 입력값 / encoder까지 gradient가 도달하는지 직접 확인한다.
  실제 모델 초기화와 학습 optimizer는 변경하지 않았다.

검증용 임시 데이터와 결과는 정리했고 기존 연구 결과는 건드리지 않았다.
**300-epoch 본 실험은 실행하지 않았으므로, 과거 3개 seed 모두의 위치 개선이 유지되는지는 아직 판단하지 않았다.**

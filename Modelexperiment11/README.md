# ModelExperiment11 — 검증 데이터로 선택하는 하이퍼파라미터 튜닝

원본 `ModelExperiment10.py` / `ModelExperiment9.py` / `NewLearning9.py`를 수정하지 않고, 새 실행기 `Codes/ModelExperiment11.py`와 이 폴더를 추가했다. 새 결과·설정·가중치는 모두 `Modelexperiment11/studies/<실험명>/`에 저장한다.

이번 실제 실행의 결과는 [한국어 결과 요약](RESULTS.md)과 [자동 생성 실험 보고서](studies/main/report.md)에 기록했다. 후보 전체는 [후보 목록](studies/main/candidates.csv), 학습 결과는 [trials.csv](studies/main/trials.csv), 설정 선택의 근거는 [selection.json](studies/main/selection.json)에 있다. 실행 과정은 [로그](audit/tuning.log)에 보존했다.

## 무엇을 분석했고 왜 튜닝하는가

[분석과 설계 근거](analysis.md), [현재 파일을 직접 조사한 결과](audit/existing_audit.json), [조사 스크립트](audit_existing.py)를 함께 보관했다.

- 기존 정식 학습 이력 36개와 최신 검증 전용 pilot 14개를 확인했다. 정식 학습은 모두 300 epoch였지만 구조 최적점은 25~42 epoch였다. 이후 학습 손실 감소와 검증 손실 증가가 함께 나타나므로 학습률·정규화 설정이 원인의 일부일 가능성이 높다.
- 기존 dropout 10%는 full 모델에는 개선 신호가 있었지만 sign-only에는 일관된 이득이 없었다. 한 모델에만 유리한 설정을 적용한 후 routing 효과로 해석하지 않도록 **두 모델에 공통인 설정 하나**를 선택한다.
- 이전 pilot은 가중치를 저장하지 않았다. 현재 데이터의 NPZ 바이트 해시는 pilot 기록과 다르고, 현재 PyTorch/CUDA도 다르다. 따라서 과거 수치에서 개선율을 계산하지 않고 **현재 환경에서 기준선을 새로 학습**한다.
- 튜닝이 전기적 역문제의 정보 부족이나 모호함을 해결한다고 가정하지 않는다. 손실의 개선과 개별 물리 지표의 개선을 구분한다.

## 고정한 실험 프로토콜

| 항목 | 설정 |
| --- | --- |
| 데이터 | `Models/charge_dataset_5charges_v9.npz`, 전하 5개, 10,000개 표본 |
| 분할 | 기존 split seed 42, train 8,000 / validation 1,000 / historical test 1,000 |
| 정규화 | train 인덱스만 사용해 한 번 계산, 모든 후보·모델·seed에 공유 |
| 구조 | 기존 full / sign-only routing, 모델 파라미터와 초기화 순서 유지 |
| 관측 | 기본 튜닝 범위 G05 100% = 고정 후보 센서 32개, 기존 nested prefix |
| 손실 | 위치 MSE + 크기 MSE + 상대부호 NLL/5, 전체부호 BCE, 모든 가중치 1 |
| 전하 대응 | 기존 5! = 120개 순열의 공동 최소 비용 대응 |
| 최적화 | AdamW, batch 128, 최대 150 epoch, 두 검증 손실 기반 patience 20 |
| seed | 모든 후보 screening seed 41; 상위 2개와 baseline은 seed 42·43 추가 |
| 일차 선택 지표 | `best_structure.pt`의 검증 구조 손실을 모델·관측 비율·seed에 같은 가중치로 평균 |
| 보조 지표 | 동일 epoch의 손실 성분·실제 단위 위치 오차·부호 정확도, 별도 `best_total.pt` |
| 실행 환경 | Python·NumPy·PyTorch·CUDA·cuDNN·GPU·스레드 수·TF32 등을 저장·대조 |
| 결정성 | Python/NumPy/Torch/CUDA/shuffle RNG 고정, deterministic algorithms는 경고가 아닌 오류 모드 |

Batch size, 손실 가중치, 모델 용량, clipping, augmentation, scheduler는 탐색하지 않는다. 특히 전체 검증 손실로 공통 학습률을 조정하면 sign-only 구조 분기가 G05 전체부호 손실의 영향을 받게 될 수 있으므로 새 scheduler를 넣지 않았다. 드롭아웃은 v10의 기존 구조 특징 위치만 사용한다.

평가 데이터에 맞춘 가중치 재학습도 하지 않는다. 선택된 epoch의 **전체 모델 상태**를 평가하며 구조와 부호 분기의 서로 다른 epoch를 조합하지 않는다.

## 탐색 범위와 선택 규칙

[search_config.json](search_config.json)에 실행 전에 모든 조건을 선언했다.

| 탐색 변수 | 범위 | 선택 이유 |
| --- | --- | --- |
| learning rate | `3e-4` ~ `3e-3` | 기존 `1e-3` 근처에서 수렴 속도와 최적점의 민감도를 확인 |
| weight decay | `1e-5` ~ `3e-3` | 기존 `1e-4`, pilot의 `1e-3`와 그 주변의 정규화 강도를 확인 |
| structure dropout | `0`, `0.05`, `0.1`, `0.2` | 기존 0과 pilot에서 검토된 낮은 dropout 범위 유지 |

기준선, dropout 0.1/0.2, decay 0.001, learning rate 0.0005/0.002의 **6개 사전 지정 후보**에 로그 균등 분포로 뽑은 **2개 조합**을 더한다. 난수 후보의 탐색 seed는 `20260831`이며 NumPy의 별도 Generator를 쓰므로 모델 초기화 RNG에 영향을 주지 않는다. 실제로 추출한 값은 study 초기화 시 `candidates.csv`와 `study.json`에 고정한다.

매 epoch 수를 줄여 탈락시키는 방식은 쓰지 않는다. 이전 dropout 20%가 늦게 수렴했던 점을 반영해 모든 후보에 같은 최대 150 epoch와 같은 조기 종료 규칙을 부여한다. 계산량은 seed 수를 단계적으로 늘려 제한한다.

1. 8개 후보 × 두 모델 × seed 41 = 16회 학습.
2. 검증 구조 손실 평균이 낮은 **비기준선 상위 2개**를 승격한다. baseline은 순위와 무관하게 반드시 유지한다.
3. 3개 후보 × 두 모델 × seed 42·43 = 12회 추가 학습.
4. 승격 후보의 세 seed 평균을 비교한다. 각 모델·관측 비율의 평균 구조 손실이 baseline보다 1% 넘게 악화되는 후보는 제외한다.
5. 적격 후보 중 평균 구조 손실이 최소이고 baseline보다 0.5% 이상 좋아진 후보를 선택한다. 그런 후보가 없으면 baseline을 유지한다.
6. 정확히 동률이면 baseline, 그다음 후보 ID 순으로 결정한다. epoch의 동률은 먼저 도달한 epoch를 유지한다.

1%/0.5%는 이번 실험에 사전 지정한 실용적 판단 기준이지 통계적 유의성 검정 기준이 아니다. 전체 탐색 공간의 전역 최적점을 보장하지 않는다. Screening seed도 최종 세 seed 평균에 포함되며, 같은 validation을 epoch와 하이퍼파라미터 선택에 재사용한다. 따라서 검증 개선율은 선택 편향을 포함하고, 독립 최종 평가로 별도 확인한다.

## 테스트 데이터의 분리

`train_validation_run()`은 `DevelopmentData(train, validation)`만 받는다. 테스트 TensorDataset/loader를 만들거나, 기존 자동 테스트 평가 함수를 호출하지 않는다. 파일 전체의 SHA-256과 데이터 형상·물리 일관성 검사는 데이터 식별·무결성 검사이며 테스트 성능 평가가 아니다.

`selection.json`에 후보 순위·선택 이유·평가할 seed·결과/가중치 해시를 고정한 후에만 `finalize`가 동작한다. 잠금 뒤에는 같은 study에서 추가 학습·설정 변경·재선택을 허용하지 않는다.

최종 평가에는 두 집합을 쓴다.

- `historical_test`: 기존의 1,000개. 과거 프로젝트 분석에 사용된 적이 있으므로 완전히 독립적인 새 증거로 표현하지 않는다.
- `fresh_test`: 선택 잠금 뒤 기존 생성기로 새 전하 1,000개를 생성한다. seed `2026083101`, 원래 학습의 센서 좌표, 같은 전하 범위와 물리식을 사용한다. 정규화는 계속 기존 train 통계다. 기존 표본과의 정확한 G00 중복 및 새 집합 내부 중복을 검사한다.

기준선과 선택 후보의 모든 seed, 두 모델, `best_structure`와 `best_total`을 동일하게 평가한다. Test 성적을 본 뒤 더 나은 seed나 checkpoint 기준을 선택하지 않는다. 다른 sensor fraction·배치·노이즈·실제 측정 데이터의 개선은 이 실험으로 입증하지 않는다.

**다른 설계를 새로 튜닝한다면 이미 본 fresh_test seed를 새 독립 테스트라고 재사용하지 말아야 한다.** 동일 seed로 새 study를 실행하는 것은 동일 실험의 재현이다. 테스트를 본 뒤 설계를 변경하는 후속 연구는 별도의 미사용 최종 데이터가 필요하다.

## 실행과 재개

프로젝트 루트 `C:\Users\COM\Documents\KK-AI`에서 실행한다. 기존 `.venv`의 NumPy/PyTorch만으로 학습·집계가 동작하며 새 라이브러리를 설치하지 않는다.

```powershell
# 초기화 → 탐색 → 설정 잠금 → 최종 평가 → 보고서
.\.venv\Scripts\python.exe Codes\ModelExperiment11.py run --device cuda

# 테스트를 열지 않고 선택까지만 실행
.\.venv\Scripts\python.exe Codes\ModelExperiment11.py tune --device cuda

# 선택이 잠긴 뒤 최종 평가만 실행
.\.venv\Scripts\python.exe Codes\ModelExperiment11.py finalize --device cuda

# 저장된 결과로 표와 보고서를 다시 생성 (학습/추론 없음)
.\.venv\Scripts\python.exe Codes\ModelExperiment11.py report
```

아무 인수 없이 실행하면 `run`과 같다. CUDA가 없으면 기본 `auto`는 CPU를 선택한다. 이미 CUDA에서 시작한 실험을 CPU로 이어 학습하는 것은 같은 실행으로 간주하지 않으며 거부한다.

중단되면 **같은 명령을 다시 실행**한다. 완료한 후보는 건너뛰고 진행 중 후보는 `latest.pt`의 모델, optimizer, shuffle RNG, Python/NumPy/Torch/CUDA RNG, 학습 이력, 두 best 상태, 조기 종료 상태를 함께 복원한다. 체크포인트 저장은 epoch마다 원자적으로 수행한다. 실행 중 한 명령만 같은 study에 쓸 수 있다.

선택이 이미 잠긴 상태의 `tune`은 재학습하지 않는다. 완료된 `finalize`도 테스트를 다시 추론하지 않고 저장된 결과를 검증해 반환한다. 실패한 후보는 성공값으로 집계하지 않으며 실패 내용은 해당 `status.json`에 남고 실행이 중단된다. 부분 결과로 설정을 확정하지 않는다.

새 실험은 설정 파일을 복사해 수정한 뒤 빈 폴더를 지정한다. 기존 study의 `resolved_config.json`, `study.json`, 저장 split, 소스 snapshot 등을 직접 수정해서 재개하면 안 된다.

```powershell
.\.venv\Scripts\python.exe Codes\ModelExperiment11.py run --config Modelexperiment11\search_config.json --study-dir Modelexperiment11\studies\reproduction --data Models\charge_dataset_5charges_v9.npz --device cuda
```

실행 환경이 다르면 새 study를 사용한다. 같은 seed라도 다른 PyTorch/CUDA 버전·CPU/GPU에서 동일한 수치를 보장하지 않는다. 버전과 장치를 포함한 재현성의 범위는 [PyTorch 공식 문서](https://docs.pytorch.org/docs/2.7/notes/randomness.html)와 같다.

## 파일 구조와 결과 해석

| 파일 | 용도 |
| --- | --- |
| `study.json`, `resolved_config.json`, `candidates.csv` | 불변 실험 정의, 데이터/소스/배열 해시, 실제 탐색 조합 |
| `sources/` | 실제 사용한 네 Python 소스의 원본 바이트 스냅샷 |
| `split_indices.npz`, `normalization.json` | 공통 분할과 train 통계 |
| `runs/<후보__모델__관측비율__seed>/config.json` | 각 학습의 완전한 설정과 fingerprint |
| 같은 폴더의 `history.json`, `history.csv` | 매 epoch train/validation 손실 성분 |
| 같은 폴더의 `latest.pt` | 중단·재개 상태, optimizer와 두 best 포함 |
| 같은 폴더의 `best_structure.pt`, `best_total.pt` | 검증 기준별 전체 모델 스냅샷 |
| 같은 폴더의 `result.json`, `status.json` | 검증 성능·선택 epoch·종료 이유 또는 실패 상태 |
| `trials.csv`, `screening.csv`, `confirmation.csv` | 전체 후보와 단계별 비교 |
| `promotion.json`, `selection.json` | 후보 승격, 최종 설정과 평가 대상의 불변 결정 |
| `final_evaluation_started.json` | 선택 잠금 후 테스트 평가 시작 기록 |
| `final/fresh_holdout.npz`, `fresh_holdout.json` | 새 최종 데이터, seed·해시·중복 검사 |
| `final/evaluations/`, `final/result.json` | 고정된 모델들의 최종 점수, 재개 가능한 개별 결과 |
| `paired_comparisons.csv` | 같은 seed/model/fraction/checkpoint에서 baseline 대 선택 후보 |
| `comparison_summary.csv` | seed별 비교의 평균, 표본 표준편차, 개선 seed 수 |
| `routing_comparisons.csv` | 같은 설정에서 full 대 sign-only의 기존 핵심 비교 |
| `report.md` | 선택 이유, 주·보조 성능 변화, 해석상의 한계 |

손실/오차의 개선은 `baseline − selected`, 정확도의 개선은 `selected − baseline`이다. 정확도 차이의 `accuracy_delta_pp`는 **퍼센트포인트**다. 개선율은 기준선으로 나눈 상대 변화율이다. 서로 다른 최적 epoch의 손실 성분을 합쳐 가상의 모델 점수를 만들지 않는다.

표준편차는 학습 seed에 대한 `ddof=1` 값이며, seed 하나인 경우 N/A다. 세 seed는 독립 데이터셋 세 개가 아니므로 작은 표준편차만으로 통계적 유의성이나 다른 분포에 대한 일반화를 주장하지 않는다.

체크포인트는 v10 로더와 호환된다. **이 프로젝트에서 생성한 신뢰할 수 있는 체크포인트만** 불러온다. `.pt`는 pickle을 포함하는 PyTorch 포맷이므로 외부의 알 수 없는 파일에 이 로더를 쓰지 않는다.

```python
import sys
from pathlib import Path
import torch
sys.path.insert(0, "Codes")
from ModelExperiment11 import load_trained_model

# 정확한 경로는 selection.json의 evaluation_runs와 각 result.json에서 확인
path = Path("Modelexperiment11/studies/main/runs/<trial>/best_structure.pt")
model, normalization, checkpoint = load_trained_model(path, torch.device("cpu"))
# model은 eval() 상태이며 저장된 dropout 설정과 정규화를 함께 복원한다.
```

기존 실행 명령과 기본값, 기존 results/checkpoints는 그대로 유지된다. 새 `.pt`와 최종 데이터 NPZ는 디스크에 보관하되 이 폴더의 `.gitignore`에서 Git 추적을 제외했다. 가중치를 공유·백업할 때는 해당 파일을 별도로 포함해야 한다.

## 검증

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s Codes -p test_model_experiment11.py -v
.\.venv\Scripts\python.exe Modelexperiment11\audit_existing.py --verify-preservation
```

신규 통합 테스트는 실제 합성 표본과 optimizer 업데이트로 데이터 누수 방지, 기준선의 v10 일치, dropout이 켜진 CPU/CUDA 중단·재개, 불변 선택·최종 평가의 재실행, 불완전 결과 거부, 공통 설정의 모델별 악화 제한 등을 검증한다. 작은 테스트 표본의 성적은 성능 개선의 근거로 사용하지 않는다.

현재 신규 테스트 13개가 통과했다. 최종 결과는 튜닝 실행기의 집계 함수를 호출하지 않는 별도 검산 스크립트로 다시 확인할 수 있다. 이 검산은 모든 예정 trial의 완료, 손실 성분 합, 첫 최적 epoch, 조기 종료 시점, 승격·공통 설정 선택, 가중치 해시, 최종 평가의 시간 순서와 비교 산식을 확인한다.

```powershell
.\.venv\Scripts\python.exe Modelexperiment11\verify_results.py
.\.venv\Scripts\python.exe Modelexperiment11\plot_results.py
```

그래프는 완료된 최종 결과만 읽으며 설정을 다시 선택하지 않는다. Matplotlib으로 PNG/SVG를 `studies/main/figures/`에 저장한다. 같은 환경의 정확한 중단·재개는 통합 테스트로 확인하고, 검산 스크립트는 저장 기록의 정합성을 별도로 검증한다.

변경 전 전체 기존 테스트 132개는 128개 통과·1개 오류·3개 skip이었다. 오류는 기존 `test_windows_long_unicode_atomic_save_load_replace_and_lock`의 Windows 긴 경로 생성 실패이며 신규 코드 추가 전부터 재현됐다. 원본 코드를 바꿔 숨기지 않았다. 근거는 [기존 테스트 로그](audit/legacy_tests.log), 신규 검증 로그는 [pipeline_tests.log](audit/pipeline_tests.log)다.

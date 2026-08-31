**ModelExperiment10: 기존 실험 기능을 유지한 과적합 제어**

[실행 코드](C:/Users/COM/Documents/GitHub/KK-AI/Codes/ModelExperiment10.py)는 최신 ModelExperiment9의 학습·평가 전용 실행·저장·재개·집계 기능을 별도 버전으로 유지한다. `ModelExperiment9.py`, `NewLearning9.py`, 기존 학습 결과와 체크포인트는 수정하지 않았다. 새 라이브러리도 추가하지 않았다.

2026-08-31 저장 구조 점검에서는 이 PC의 seed별 체크포인트 경로와 Python 3.14 소스 검사 문제를 10에서 보완했다. 학습 저장 형식과 아래 과적합 제어는 유지한다. 실제 파일 검사 결과와 이동한 실험의 평가 방법은 [저장 호환성 분석](C:/Users/COM/Documents/GitHub/KK-AI/Documents/ModelExperiment10_storage_compatibility.md)에 정리했다.

기본 변경은 **두 검증 목적을 함께 감시하는 조기 종료**다. 구조 분기 드롭아웃도 구현했지만 기본값은 **0.0(비활성)** 이다. 실제 비교에서 full 모델에는 도움이 되고 sign-only에는 일관되게 도움이 되지 않아, 공통 기본 설정으로 강제하지 않았다. 드롭아웃 확률을 지정하면 두 모델에 같은 값이 적용된다.

**왜 바꾸었는가**

[이전 학습 집중 분석](C:/Users/COM/Documents/GitHub/KK-AI/Results/training_analysis_20260831/report.md)에서 마지막 완료 실행은 full / G05 100% / seed 43의 300 epoch 학습이었다. 최적 구조·전체 체크포인트는 모두 26 epoch에 있었다. 이후 학습 구조 손실은 `0.51623 → 0.09293`으로 내려갔지만, 검증 구조 손실은 `0.59752 → 1.00291`로 67.85% 증가했다. 위치·크기·상대부호 분기에서 과적합이 나타났고 전체 부호 검증 BCE는 같은 양상으로 악화되지 않았다.

기존 코드도 최적 체크포인트는 올바르게 보존했다. 따라서 조기 종료만으로 기존 `best_structure.pt`의 정확도가 새로 좋아진다고 주장하지 않는다. 이미 악화 중인 마지막 가중치를 더 학습하는 비용을 줄이고, 완료 상태와 재개 동작을 명확하게 만드는 것이 첫 번째 개선이다.

**검토한 해결책과 적용 판단**

| 방법 | 판단과 적용 |
| --- | --- |
| 검증 기반 조기 종료 | 기본 활성화. `val_structure` 또는 `val_total` 중 하나라도 개선하면 대기를 초기화하고, 둘 다 20 epoch 동안 개선되지 않으면 종료한다. |
| 구조 특징 드롭아웃 | 구현하되 기본 비활성. G00/G05의 대칭적인 구조 특징을 합친 뒤, 위치·크기·상대부호 head 전에 한 번만 적용한다. |
| AdamW 가중치 감쇠 강화 | 기존 `--weight-decay` 옵션 유지. `1e-3`도 탐색했지만 한 seed의 소폭 개선만으로 기본 `1e-4`를 바꾸지 않았다. |
| 공통 `ReduceLROnPlateau(val_total)` | 채택하지 않음. sign-only의 G05 부호 손실이 구조 분기의 학습률에 영향을 주어 기존 분리 원칙을 깨뜨릴 수 있다. |
| 손실 교체·부호 label smoothing·후처리 제약 | 이번 버전에서는 채택하지 않음. 120개 순열의 assignment와 16개 상대부호 분포, 기존 지표의 의미를 유지한다. |
| 무작위 영상 변형·독립적인 G00/G05 잡음 | 채택하지 않음. 물리적으로 연결된 `G00=V²`, `G05=V`, 좌표·정답을 함께 변환하는 설계와 별도 검증이 필요하다. |

드롭아웃은 학습 중 일부 특징을 무작위로 제거해 특정 특징 간 의존을 줄이는 방법이다. 이 일반적인 근거만으로 이 데이터의 성능 개선을 보장할 수는 없으므로 아래 실제 검증 비교를 수행했다. [원 논문, Srivastava et al., JMLR 2014](https://jmlr.org/papers/v15/srivastava14a.html)

PyTorch `Dropout`은 학습 때 마스크와 `1/(1-p)` 배율을 적용하고 평가 때 항등 함수가 된다. 여기서는 G05의 `+V/-V` 쌍을 만드는 함수 안에 넣지 않아 전체 부호의 홀대칭을 유지한다. [PyTorch 2.11 Dropout 문서](https://docs.pytorch.org/docs/2.11/generated/torch.nn.Dropout.html)

AdamW는 적응적 최적화의 gradient와 가중치 감쇠를 분리하는 방식이며 이 프로젝트는 이미 사용 중이었다. 강도를 더 높이면 항상 낫다는 뜻은 아니다. [Loshchilov & Hutter, Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)

일반적인 조기 종료의 `patience`, `min_delta`, 시작 유예 개념을 사용하되, 이 실험의 두 독립 체크포인트 목적에 맞춰 제어기를 직접 구현했다. TensorFlow를 의존성으로 추가하지 않았다. [EarlyStopping 공식 설명](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)

검증 지표 정체에 따라 optimizer 학습률을 줄이는 스케줄러는 유용할 수 있지만, 이 코드에서 전체 손실을 공통 제어 신호로 쓰면 분리된 분기 사이에 새 의존 관계를 만든다. 이는 문서의 동작을 이 모델 구조에 적용한 판단이다. [PyTorch ReduceLROnPlateau](https://docs.pytorch.org/docs/2.11/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)

**종료·최적 선택·재개의 구체적인 의미**

`RegularizationSettings`의 값을 프로토콜과 각 실행의 configuration에 모두 기록한다. 설정을 바꾸면 실행 fingerprint도 달라져, 서로 다른 정규화 설정의 체크포인트를 같은 실행으로 재개하거나 같은 비교군으로 집계하지 않는다.

| 설정 | 기본값 | 의미 |
| --- | --- | --- |
| `--epochs` | `300` | 최대 epoch 수. 조기 종료가 먼저 발생할 수 있다. |
| `--early-stopping-patience` | `20` | 어느 검증 목적도 개선하지 못한 연속 epoch 수. `0`이면 비활성화. |
| `--early-stopping-min-delta` | `0.0` | patience를 초기화하기 위한 절대 손실 개선량. 상대 비율이 아니다. |
| `--early-stopping-min-epochs` | `0` | 이 epoch 전에는 종료하지 않는다. 그동안도 검증과 최적 저장은 수행한다. 최대 epoch 제한이 우선한다. |
| `--structure-dropout` | `0.0` | 구조 특징 드롭아웃 확률. 범위 `[0,1)`. `0.1`로 탐색 가능. |
| `--learning-rate` | `0.001` | 기존 AdamW 학습률 유지. |
| `--weight-decay` | `0.0001` | 기존 AdamW 감쇠 유지. |
| `--batch-size` | `128` | 학습 기본 크기. 평가 전용에서는 저장된 크기가 기본이다. |

`min_delta=0`이면 동일 손실은 개선으로 세지 않는다. 하나라도 새 최솟값을 기록해야 한다. `min_delta>0`이면 마지막으로 유의미하게 개선된 값을 기준으로 누적 개선을 비교한다. 작은 개선이 여러 번 쌓이면 임계값을 넘을 수 있다.

`best_total.pt`와 `best_structure.pt` 선택은 여전히 **실제 검증 손실의 엄격한 최솟값**이다. `min_delta`는 이 선택에 적용하지 않는다. 작은 개선도 최적 가중치 파일에 남으며, 동률이면 먼저 나온 epoch를 유지한다. 두 파일은 각각 해당 epoch의 전체 모델이고 분기별 가중치를 서로 합성하지 않는다.

각 epoch에서는 학습 → 검증 → 최적 선택 → 종료 상태 갱신 → `latest.pt` 원자적 저장 → 두 best와 history 게시 순서로 진행한다. `latest.pt`에는 모델, AdamW 상태, shuffle generator, Python/NumPy/PyTorch/CUDA 난수, 두 최적 스냅샷, 전체 이력, 조기 종료 카운터를 함께 저장한다.

재개 시 검증 이력을 재생하여 저장된 종료 상태가 맞는지 확인한다. 조기 종료 epoch의 `latest.pt`를 저장한 직후 중단됐어도 다음 epoch를 더 학습하지 않는다. 남은 두 최적 체크포인트의 테스트 평가와 결과 저장만 완료한다. 이미 완료된 실행은 학습·평가를 반복하지 않고, 필요한 경우 유실된 best 파일을 `latest.pt`에서 복구한다.

`result.json`과 `runs.csv`에는 실제 `epochs_completed`, `stop_reason` 및 정규화 설정을 남긴다. 이유는 `early_stopping` 또는 `max_epochs`이다. 최대 epoch와 조기 종료 조건이 같은 epoch에 도달하면 예산을 모두 사용한 `max_epochs`로 기록한다. 미완료 실행은 임의로 완료로 취급하지 않는다.

**유지한 기능**

| 영역 | 유지 내용 |
| --- | --- |
| 데이터 | 기존 NPZ 로드·물리 일관성 검사, 80/10/10 split seed 42, 학습 데이터로만 정규화 |
| 센서 | G05 고정 nested prefix, 0%의 관측 부재 처리, 현재 75%=24개·100%=32개 |
| 모델 비교 | sign-only / full 두 모델, 같은 초기 파라미터와 등록 파라미터 수 406,969개 |
| 물리·손실 | 5전하, 120개 순열 대응, 16개 상대부호 패턴, `global=product(sign(q))`, 기존 손실 가중치 |
| 분기 | sign-only 구조는 G00만, global은 G05만 사용. full의 G05 encoder 공유는 기존대로 유지 |
| 마스크·대칭 | 숨긴 G05의 NaN 차단, 관측이 없으면 global logit=0, 추론 시 구조 짝대칭과 global 홀대칭 |
| 추론 | `load_trained_model()`이 저장된 정규화·구조 dropout 확률을 복원하고 `eval()` 설정 |
| 저장·재개 | 전체 모델의 두 최적 선택, 원자적 latest 저장, optimizer·난수·shuffle·이력 재개, 손상·혼합 설정 거부 |
| 결과 | seed/model/fraction 실행, 원래 지표·개별/집계 CSV, paired 비교, sample std, 0%의 N/A |
| 운영 | CPU/CUDA/auto, smoke-only, evaluate-only/eval-only, 디렉터리 별칭, continue-on-error, 동시 쓰기 잠금 |
| 평가 전용 | 저장된 split/정규화/설정 사용, 별도 evaluations 출력, 원래 결과·가중치 변경 없음, 미완료 학습 자동 재개 없음 |

드롭아웃이 켜진 학습 중 독립적인 두 forward는 서로 다른 마스크 때문에 구조 출력이 다를 수 있다. 물리 대칭을 확인할 때는 `eval()`을 사용하거나 동일 RNG 상태로 같은 마스크를 적용한다. global branch 자체에는 dropout을 넣지 않는다. 학습 손실은 dropout이 켜진 값, 검증 손실은 꺼진 값이므로 단순한 둘의 차이만으로 정규화 강도를 판단하지 않는다.

`structure_dropout=0`과 `patience=0`에서는 두 경로 모두 ModelExperiment9와 epoch별 손실·가중치·AdamW 상태가 정확히 일치하는 회귀 테스트를 수행했다. 드롭아웃을 켠 상태에서도 sign-only의 구조 학습이 G05 비율과 무관하게 같은 epoch까지 동일함을 확인했다. 조기 종료 시점은 관측 비율별 검증 손실 때문에 달라질 수 있다.

**실제 데이터 탐색 결과**

모든 후보는 같은 8,000개 학습 데이터와 1,000개 검증 데이터, G05 100%=32개, AdamW `lr=0.001`, batch 128, 최대 100 epoch, patience 20으로 비교했다. 테스트셋은 이 탐색에서 dataset/loader를 생성하지 않았고 설정 선택에도 쓰지 않았다. 아래 값은 테스트 정확도가 아닌 **최저 검증 손실**이다.

| Full / seed 43 후보 | 최저 구조 손실 | 구조 선택 epoch | 최저 total | 실제 종료 epoch |
| --- | ---: | ---: | ---: | ---: |
| 기존 설정 + 조기 종료 | 0.597517 | 26 | 0.948644 | 46 |
| Dropout 10% | 0.573972 | 43 | 0.924231 | 66 |
| Dropout 20% | 0.573129 | 72 | 0.912720 | 92 |
| Weight decay 0.001 | 0.594047 | 26 | 0.945779 | 46 |

기존 설정의 최저 구조 손실과 epoch는 이전 완료 학습의 seed 43 결과와 정확히 일치했다. 10% dropout은 이 조건에서 구조 손실을 약 3.94% 낮췄다. 20%의 구조 손실 이득은 10% 대비 작았고 더 오래 학습했다. 가중치 감쇠 강화는 이 seed에서 약 0.58% 개선이었다.

두 모델과 세 seed의 추가 비교를 포함하여 총 14개 검증 전용 pilot을 수행했다. 기존 설정과 dropout 10%의 비교는 6개 `(모델, seed)` 쌍이며 나머지 2개는 첫 seed의 대안 탐색이다. 개선율은 `(기존 - dropout) / 기존 × 100`으로, 음수는 악화다.

| 모델 | Seed | 기존 최저 구조 손실 | Dropout 10% | 개선율 |
| --- | ---: | ---: | ---: | ---: |
| sign-only | 41 | 0.599689 | 0.615808 | −2.69% |
| sign-only | 42 | 0.594987 | 0.597558 | −0.43% |
| sign-only | 43 | 0.599906 | 0.588406 | +1.92% |
| full | 41 | 0.610524 | 0.600189 | +1.69% |
| full | 42 | 0.606753 | 0.587395 | +3.19% |
| full | 43 | 0.597517 | 0.573972 | +3.94% |

Full은 3/3 seed에서 개선했고, seed별 개선율 평균은 **+2.94%**였다. 평균 최저 구조 손실은 `0.604931 → 0.587185`였다. Sign-only는 1/3 seed만 개선했고 개선율 평균은 **−0.40%**, 평균 손실은 `0.598194 → 0.600591`이었다. 이 차이가 dropout 기본값을 0으로 둔 이유다. Full만을 위한 후속 실험이라면 0.1은 추가 검증할 근거가 있는 선택지다.

원자료와 재현 코드: [검증 요약](C:/Users/COM/Documents/GitHub/KK-AI/Results/model_experiment10_validation/validation_summary.json), [seed별 비교 CSV](C:/Users/COM/Documents/GitHub/KK-AI/Results/model_experiment10_validation/paired_validation.csv), [pilot 실행 코드](C:/Users/COM/Documents/GitHub/KK-AI/Results/model_experiment10_validation/validation_pilot.py), [집계·그래프 코드](C:/Users/COM/Documents/GitHub/KK-AI/Results/model_experiment10_validation/summarize_validation.py).

기존 36개 완료 학습의 이력을 새 조기 종료 코드로 재생한 결과, patience 20에서 기존 두 최적 epoch를 **36/36개 모두 보존**했다. 총 epoch는 `10,800 → 1,969`로 81.77% 감소했고 종료 지점은 46~68 epoch였다. 이는 **기존 이력에 대한 재생 결과**로, 드롭아웃 사용 시 같은 종료 지점이나 모든 미래 학습의 최적점 보존을 보장하지 않는다.

![정규화 검증 비교](C:/Users/COM/Documents/GitHub/KK-AI/Results/model_experiment10_validation/regularization_validation.png)

**실행 예시와 출력**

프로젝트 루트 `C:\Users\COM\Documents\GitHub\KK-AI`에서 실행한다. 먼저 실제 데이터의 작은 학습 부분집합만으로 모든 센서 비율을 확인하려면 다음 명령을 사용한다. `--smoke-only`는 테스트 평가나 실험 폴더를 만들지 않는다.

```powershell
.\.venv\Scripts\python.exe Codes\ModelExperiment10.py --fractions 0,0.1,0.25,0.5,0.75,1 --smoke-only
```

기존 두 모델 비교에 조기 종료를 적용하는 기본 실행이다. 기본 dropout은 꺼져 있다. 아래 명령은 예시이며 이 문서 작성 과정에서 36개 정식 학습을 새로 수행한 것은 아니다.

```powershell
.\.venv\Scripts\python.exe Codes\ModelExperiment10.py --fractions 0,0.1,0.25,0.5,0.75,1 --seeds 41,42,43 --epochs 300 --experiment-name g05_earlystop_v10
```

드롭아웃 10%를 두 모델에 똑같이 적용하여 추가 비교하려면 별도의 실험명을 사용한다.

```powershell
.\.venv\Scripts\python.exe Codes\ModelExperiment10.py --fractions 1 --seeds 41,42,43 --structure-dropout 0.1 --experiment-name g05_dropout010_v10
```

새 제어를 모두 끄고 기존 학습 동작을 재현하려면 다음과 같이 지정한다. 저장 프로토콜/경로는 계속 v10으로 분리된다.

```powershell
.\.venv\Scripts\python.exe Codes\ModelExperiment10.py --early-stopping-patience 0 --structure-dropout 0 --experiment-name g05_controls_off_v10
```

중단된 학습은 **동일한 코드·데이터·설정·실험명으로 동일 명령**을 다시 실행하면 이어진다. 이미 완료된 실험의 두 best만 다시 평가하려면 아래처럼 지정한다. 평가 전용에서 학습용 epochs/dropout/patience/lr 등의 CLI 값은 저장 설정을 덮어쓰지 않는다.

```powershell
.\.venv\Scripts\python.exe Codes\ModelExperiment10.py --evaluate-only --fractions 0,0.1,0.25,0.5,0.75,1 --seeds 41,42,43 --experiment-name g05_earlystop_v10
```

기본 결과는 `Results/new_learning10_experiments/<experiment-name>/`, 체크포인트는 `Models/new_learning10_experiments/<experiment-name>/` 아래에 저장된다. 각 실행에는 `latest.pt`, `best_structure.pt`, `best_total.pt`를 유지한다. 평가 전용은 결과 폴더 아래 `evaluations/<timestamp_uuid>/`에 별도로 저장한다. `--results-root/--results-dir`, `--checkpoint-root/--checkpoint-dir`로 위치를 바꿀 수 있다.

평가 전용에서는 `--evaluation-results-dir`로 `protocol.json`이 있는 **정확한 폴더**, `--evaluation-checkpoint-dir`로 run별 가중치 폴더들이 있는 **정확한 폴더**를 지정할 수 있다. 이 두 옵션은 실험명을 뒤에 붙이지 않는다. 암묵적인 결과 경로에 protocol이 없으면 단일 seed의 `_seedN` 폴더를 확인하며, 가중치는 공통/seed별 구조에서 저장된 protocol의 run ID가 일치하는 폴더를 찾는다. 여러 대체 후보가 일치하면 명시적인 경로가 필요하고, 여러 폴더의 가중치를 섞지 않는다.

PC 이동으로 데이터의 저장 경로만 사라졌을 때는 현재 프로젝트의 제한된 후보 중 **학습 당시 SHA256과 같은 파일만** 사용한다. 명시한 데이터 경로가 없거나 파일 내용이 다르면 다른 파일로 자동 대체하지 않는다. 물리 소스는 원본 파일 해시 일치를 우선하고, 주석·docstring·LF/CRLF 차이가 있을 때만 검증된 AST 동일성을 허용한다. 저장 protocol과 학습 실행 ID는 수정하지 않는다.

v9 가중치를 v10 실행에 섞어 재개하지 않는다. v9는 원래 v9 전용 로더/평가기를 사용해야 하며, 현재 v9 평가 명령에는 위 호환성 보고서에 기록한 경로·소스 검사 제한이 남아 있다. v10은 체크포인트 schema 2와 별도 프로토콜을 사용한다. 체크포인트 로더에는 이 프로젝트에서 생성한 신뢰할 수 있는 파일만 전달한다.

**검증 범위와 제한**

[전용 테스트](C:/Users/COM/Documents/GitHub/KK-AI/Codes/test_model_experiment10.py)는 기존 32개 회귀 시나리오에 조기 종료·dropout·손상 상태 거부 등을 추가한다. CPU와 RTX 3070 CUDA에서 dropout이 켜진 상태로 일반 epoch 및 종료 epoch의 저장 전/후 강제 중단을 재현하고, 재개한 모델·optimizer·RNG·shuffle·이력·두 최적 파일이 연속 실행과 정확히 같은지 검사했다. 중단된 테스트 평가 완료, 유실된 best 복구, 원본을 수정하지 않는 평가 전용 경로도 포함한다.

최초 과적합 제어 구현 시점에는 전체 121개 중 118개 통과, 3개 skip이었고 v10 전용 44개는 모두 통과했다. 당시 결과는 [초기 전체 테스트 로그](C:/Users/COM/Documents/GitHub/KK-AI/Results/model_experiment10_validation/final_all_tests.log)에 보존했다.

저장 호환성 보완 후 현재 검증은 **v10 전용 52개 모두 통과**, 전체 132개 중 128개 통과·1개 오류·3개 skip이다. 오류는 수정하지 않은 v9의 `test_evaluate_only_accepts_moved_seed_results_and_documented_legacy_source`로, 이번 수정 전에도 Python 3.14에서 같은 소스 해시 오류가 발생했다. Skip은 종전과 같은 구형 `train_g05_fraction_experiment` 모듈 부재다. [현재 v10 로그](C:/Users/COM/Documents/GitHub/KK-AI/Results/model_experiment10_storage_validation/final_model10_tests.log), [현재 전체 로그](C:/Users/COM/Documents/GitHub/KK-AI/Results/model_experiment10_storage_validation/final_all_tests.log), [새 저장 회귀 테스트](C:/Users/COM/Documents/GitHub/KK-AI/Codes/test_model_experiment10_storage.py).

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s Codes -p 'test_*.py' -v
```

실제 데이터에 대해서도 6개 센서 비율 × 두 모델의 smoke를 dropout 0과 0.1 각각 수행했다. 이는 작은 학습 부분집합으로 실행 경로를 검사한 것이며 36개 전체 학습을 뜻하지 않는다. [기본 설정 smoke](C:/Users/COM/Documents/GitHub/KK-AI/Results/model_experiment10_validation/full_data_smoke_default.log), [dropout smoke](C:/Users/COM/Documents/GitHub/KK-AI/Results/model_experiment10_validation/full_data_smoke_dropout.log).

[소스 보존 감사](C:/Users/COM/Documents/GitHub/KK-AI/Results/model_experiment10_validation/preservation_audit.json)에서도 v9 함수/메서드가 제거되지 않았고, `run_epoch`과 `evaluate_model`이 기존 물리 구현의 직접 별칭임을 확인했다. 소스 구조 비교는 행동 테스트를 보완하는 확인이며 단독으로 기능 동일성을 증명하지는 않는다. 현재 환경에서 검증한 재개 재현성을 다른 PyTorch/CUDA 버전까지 보장하지 않는다.

검증 결과는 동일 데이터 분할의 학습 seed 3개에서 얻었다. 독립적인 데이터셋 3개를 평가한 것이 아니며 테스트셋 성능 향상, 모든 G05 비율, 다른 데이터 분포까지 입증한 것이 아니다. 기존 평가의 target-assisted assignment와 상대부호 지표 의미도 바꾸지 않았다. 정식 비교 시 `best_structure`와 `best_total`을 구분하고 paired seed 결과를 함께 확인해야 한다.

추가 검증 데이터나 새 물리 조건 없이 hyperparameter 후보만 반복해서 늘리면 검증셋에 선택 편향이 생길 수 있다. 기본값은 보수적인 조기 종료로 두고, dropout은 비교 목적과 추가 검증 근거에 맞춰 선택하는 것이 이번 버전의 결론이다.

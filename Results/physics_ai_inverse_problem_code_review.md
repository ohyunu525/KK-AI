# 물리 기반 AI 역문제 코드 리뷰 및 수정 설계

검토일: 2026-08-21
검토 범위: 데이터 생성, NPZ 저장 의미, partial-G05 mask, 모델, loss, evaluation, 다중 seed 집계, 저장 경로

## 결론

기존 target index는 올바르다. 실제 target은

```text
[x1, y1, z1, q1, x2, y2, z2, q2]
```

이며 `POSITION_INDICES=(0,1,2,4,5,6)`, `CHARGE_INDICES=(3,7)`와 일치한다. 데이터 생성 코드 추적뿐 아니라 기존 `Models/charge_dataset.npz` 25개 샘플을 Coulomb 식으로 재계산해 확인했다. 최대 절대 재구성 오차는 G00에서 `7.47e-8`, G05에서 `7.86e-9`였다. 두 전하는 전체 10,000개 샘플에서 모두 x 좌표 오름차순으로 정렬되어 있었다.

반면 기존 fraction 실험의 10%는 각 샘플의 G05 10%가 아니다. 샘플당 후보가 한 점뿐이므로 test set에서 실제로는 약 9%의 샘플만 한 점을 받고 나머지 약 91%는 아무 G05도 받지 않았다. 그럼에도 fraction이 0보다 크면 전체 샘플에 signed-charge MSE를 적용했다. 이론적으로 절대 부호를 알 수 없는 약 91%의 샘플에도 절대 부호 정답을 강제한 것이다. 0%와 양의 fraction에서 모델 구조까지 달랐다. 이 둘이 기존 10%의 charge magnitude MAE 약 0.55, relative sign accuracy 약 0.76으로의 급락을 충분히 설명한다.

수정본은 샘플마다 32개 G05 후보점을 만들고, 고정된 공간 균형 nested 순서에서 fraction별로 `0/3/8/16/24/32`점을 모든 샘플에 제공한다. 모든 fraction이 같은 모델 구조를 사용하며, loss는 실제 sample mask를 보고 signed 또는 global-sign-invariant loss를 선택한다.

## A. 문제점 분류

### [P0] 결과의 타당성을 깨뜨리는 버그

1. **관측되지 않은 sample에 signed-q loss 적용**

   기존 `Learning(6).py`의 `train_model`은 `use_g05 = g05_fraction > 0`으로 정하고, `calculate_losses`는 `use_g05=True`이면 배치 전체에 signed MSE를 적용한다. 10% 조건의 test 관측률은 CSV상 약 0.09였으므로 대부분의 sample이 이론적으로 식별 불가능한 절대 부호를 supervision 받았다.

   수정본은 sample별로 다음 loss를 사용한다.

   ```text
   G05 mask에 관측점이 하나 이상 있음 -> signed MSE
   관측점이 없음                       -> min(MSE(q_hat,q), MSE(q_hat,-q))
   ```

2. **기존 결과를 “sample 내부 G05 fraction”으로 해석할 수 없음**

   `(N,1,3)`에서 한 점 전체를 위치 coverage로 켜고 끄므로 기존 CSV는 sensor-cell coverage/샘플 availability 실험 결과다. 요청된 “각 역문제 sample에서 G05 정보를 얼마나 줄였는가”에 대한 결과가 아니다. 따라서 기존 그래프는 새 v2 결과와 같은 곡선으로 직접 비교하면 안 된다.

### [P1] 실험 설계를 왜곡할 수 있는 문제

1. **0%만 다른 모델 용량**

   기존 0%는 G05 encoder가 없고 fusion 입력도 128이며, 양의 fraction은 G05 encoder가 있고 fusion 입력이 160이다. 0% 대 10% 차이에 정보량, parameter 수, 초기화 분포 차이가 섞였다. 수정본은 모든 fraction에서 동일한 160-input 구조와 all-zero mask를 사용한다.

2. **절대 부호 정확도의 평가 모집단**

   기존 양의 fraction은 G05를 못 받은 sample까지 포함해 전체 test set의 per-charge absolute sign accuracy를 계산했다. 이는 “관측된 sample의 부호 복원”과 “G05 없이 학습 분포에서 추측한 부호”를 섞는다. 새 설계는 양의 fraction에서 모든 sample이 최소 3점을 받으므로 이 혼합이 사라진다. 0% absolute/global sign accuracy는 원리적으로 N/A로 둔다.

3. **데이터 생성 재현성 부재**

   기존 generator는 전역 `np.random`을 seed 없이 사용했다. 수정본은 독립 generator와 저장 metadata에 generation seed를 기록한다.

4. **데이터 의미 metadata 부재**

   기존 NPZ에는 `G00`, `G05`, `target`만 있어 target 순서를 파일 자체로 확인할 수 없었다. 수정 generator는 `target_fields`, `g05_fields`, grid, epsilon, sensor 순서, generation seed를 함께 저장한다. 학습 코드는 metadata와 물리식을 모두 검증한다.

5. **실행 환경과 출력 경로가 소스 위치에 결합**

   기존 최신 fraction 코드가 `__pycache__` 아래에 있고 `BASE_DIR`에 모든 파일을 쓰도록 되어 있었다. 수정본은 코드 위치와 무관하게 project root를 구한 뒤 코드=`Codes`, dataset/checkpoint=`Models`, CSV/plot=`Results`로 고정한다.

6. **현재 가상환경은 실행 불가**

   `KK/pyvenv.cfg`가 존재하지 않는 `C:\Users\COM\...` Python을 가리킨다. 따라서 현 환경에서 generator/학습을 실행하려면 Python 3.12 환경을 새로 만들고 PyTorch, NumPy, Matplotlib을 설치해야 한다.

### [P2] 개선하면 좋은 사항

1. 기존 다중 seed 표준편차는 NumPy 기본값 `ddof=0`인 population std다. 독립 실행 3회의 산포 보고에는 sample std가 자연스러우므로 수정본은 `ddof=1`을 명시한다.
2. 기존 absolute sign accuracy는 두 전하를 모두 평균한 값이라 global sign ambiguity를 직접 보여주지 않는다. 수정본은 deterministic ordering의 첫 번째(왼쪽) 전하 q1을 orientation anchor로 한 `global_sign_accuracy`와 두 부호가 모두 맞는 `signed_pair_accuracy`를 추가한다.
3. K=32는 합리적인 기본 sensor budget이지 물리적으로 유일한 선택은 아니다. 최종 결론 전 K=16/32/64 민감도 실험과 관측 noise 실험을 권장한다.
4. `warn_only=True`인 deterministic CUDA 설정은 지원되지 않는 backward kernel을 실행하도록 허용한다. seed 재현성은 강화되지만 bitwise determinism을 보장하지는 않는다. 같은 GPU/driver에서 반복 오차를 확인하고 limitation으로 보고해야 한다.

### [OK] 현재 구현이 적절한 부분

1. target index와 generator의 저장 순서는 일치한다.
2. 전하를 `(x,y,z)` 순으로 정렬해 두 전하 permutation ambiguity를 제거한 방식은 현재 연속분포에서 적절하다.
3. G00 normalization, G05 signed-value RMS normalization, position/charge 분리 head, AdaptiveAvgPool2d는 유지할 수 있다.
4. train/validation/test 고정 split, seeds 41/42/43, seed별 DataLoader shuffle, early stopping, best-validation checkpoint, CUDA 자동 선택, main guard는 적절하다.
5. 0%에서 global-sign-invariant charge loss를 사용하고 평가 시 q와 -q 중 가까운 방향으로 정렬해 sign-invariant q error를 계산한 방식은 물리 대칭과 맞다.
6. G05 위치 index를 입력 feature에 포함하고 pointwise MLP 뒤 순서 불변 pooling을 사용하는 구조는 multi-point G05로 자연스럽게 확장 가능하다.
7. G05 normalization을 전체 train candidate에서 한 번 계산해 모든 fraction이 공유하는 방식은 test leakage가 아니며, fraction마다 scale을 바꾸는 confound를 피한다. 단, 보고서에 공통 train-only scale임을 명시해야 한다.
8. AdaptiveAvgPool2d의 deterministic warning만으로 pooling을 제거할 근거는 부족하다.

## B. 물리적·수학적 타당성

두 전하의 관측 평면 전위를 `V=V1+V2`라 두면 global charge flip에서

```text
(q1,q2) -> (-q1,-q2)
V       -> -V
G00=V^2 -> V^2
G05~V   -> -G05
```

가 된다. 따라서 G00만으로 global sign은 정보이론적으로 식별 불가능하다. 반면

```text
V^2 = V1^2 + V2^2 + 2 V1 V2
```

의 교차항은 `q1*q2`의 부호에 따라 달라지므로, 일반적인 비퇴화 배치에서는 G00이 charge magnitude와 relative sign을 담을 수 있다. 기존 0% 결과의 magnitude MAE 약 0.11, relative sign accuracy 약 1.00은 이 구조와 일관된다. 다만 연속 inverse problem의 유일성 전체를 이 식만으로 증명하는 것은 아니므로 “항상 완전 식별”이라고 주장해서는 안 된다.

G05의 nonzero signed potential 한 점은 noiseless 조건에서 두 global-sign 후보를 구분할 수 있다. 그러나 전위 node에 가까운 센서, noise, 모델 오차에서는 한 점이 약한 증거가 된다. 여러 공간점은 global sign 결정의 강건성을 높이고 본 연구 질문에도 맞다.

수정본은 charge representation을 magnitude/relative/global head로 완전히 분해하지 않았다. 현재 연구에서 먼저 제거해야 할 것은 잘못된 supervision과 불공정한 capacity이므로, end-to-end charge head를 유지하고 symmetry-aware samplewise loss만 적용하는 편이 변경 범위와 비교 가능성 면에서 낫다. 분해형 head는 이후 architecture ablation으로 비교할 수 있다.

## C. 최종 partial-G05 정의

추천 정의는 **고정된 공간 균형 sensor array의 nested prefix**다.

```text
G05 shape: (N, 32, 3)
point:      [grid_x_index, grid_y_index, signed potential V]

fraction 0.00 ->  0 points/sample
fraction 0.10 ->  3 points/sample
fraction 0.25 ->  8 points/sample
fraction 0.50 -> 16 points/sample
fraction 0.75 -> 24 points/sample
fraction 1.00 -> 32 points/sample
```

32개 위치는 32x32 grid에서 seeded maximin 순서로 선택한다. 각 prefix가 가능한 한 공간적으로 퍼지고 낮은 fraction은 높은 fraction의 엄격한 부분집합이다. 모든 sample에서 같은 위치를 사용하므로 센서 기하가 sample마다 달라지는 nuisance를 피하고, fraction 차이는 오직 signed measurement 수로 해석할 수 있다.

sample별 random point sampling은 다른 유용한 robustness 실험이지만, 주 실험으로 사용하면 같은 fraction에서도 sample별 난이도와 sensor geometry가 달라진다. 전체 1024-cell sensor coverage는 측정 예산의 의미가 지나치게 달라지고 계산·저장량도 커진다. 따라서 K=32 fixed nested array를 주 실험으로, random geometry와 K 민감도는 후속 ablation으로 두는 것이 타당하다.

## D. 현재 동작 → 문제 → 수정안 → 수정 후 의미

| 영역 | 현재 동작 | 문제 | 수정안 | 수정 후 의미 |
|---|---|---|---|---|
| 데이터 | `(N,1,3)` random 한 점 | fraction이 sample availability가 됨 | `(N,32,3)` fixed nested 후보 | fraction이 sample 내부 측정 budget |
| mask | cell coverage에 포함된 sample만 한 점 사용 | 많은 sample이 G05 없음 | 모든 sample에 동일 prefix mask | 양의 fraction에서 모든 sample 관측 |
| loss | fraction>0이면 전체 signed MSE | 미관측 sample에 불가능한 정답 강제 | sample mask 기반 hybrid loss | symmetry와 supervision 일치 |
| 모델 | 0%만 G05 branch 제거 | parameter/init confound | 전 fraction 같은 masked 모델 | 정보량 차이만 비교 |
| 평가 | 양의 fraction 전체에 absolute sign | 관측/미관측 효과 혼합 | 양의 fraction은 전 sample 관측, 0% N/A | metric 모집단 명확 |
| global sign | per-charge absolute만 출력 | global ambiguity를 직접 못 봄 | q1-anchor global 및 pair metric 추가 | symmetry breaking 직접 측정 |
| 통계 | `std(ddof=0)` | 3-run 산포를 population std로 보고 | `std(ddof=1)` | 독립 seed sample std |
| 경로 | source `BASE_DIR`에 혼합 저장 | Codes/Models/Results 분리 불안정 | project-root 기반 세 폴더 | 산출물 종류별 일관 저장 |
| 검증 | shape/range 중심 | target 의미 오류를 못 잡음 | metadata + Coulomb 재구성 검사 | index/physics mismatch 즉시 실패 |

## E. 구현된 코드

1. `Codes/generate_charge_dataset.py`
   - seeded data generation
   - 32개 spatially balanced fixed sensor
   - target/G05/grid/물리 metadata 저장
   - 기존 single-point dataset을 보존하는 기본 출력 `Models/charge_dataset_multipoint_v2.npz`

2. `Codes/train_g05_fraction_experiment.py`
   - 동일 architecture와 동일 seed initialization을 전 fraction에 적용
   - nested prefix mask
   - samplewise signed/sign-invariant charge loss
   - 기존 필수 위치·전하 metric 보존
   - global sign, signed pair metric 추가
   - 3-seed sample std (`ddof=1`)
   - checkpoint와 normalization은 `Models`, CSV와 plot은 `Results`
   - metadata/물리식/shape 검증과 forward/loss smoke test

3. `Codes/test_physics_pipeline.py`
   - target 순서와 두 관측식의 수치 회귀 테스트
   - fixed/unique sensor와 nested fraction count 테스트
   - sample별 hybrid charge loss 및 동일 초기화 테스트

기존 script와 기존 dataset·모델·그래프는 과거 결과 보존을 위해 삭제하거나 덮어쓰지 않았다. 새 산출물에는 `_v2` 이름을 사용한다. generator와 trainer의 기본 dataset 경로도 `Models/charge_dataset_multipoint_v2.npz`이므로 인자를 생략해도 기존 `Models/charge_dataset.npz`를 보존한다.

## F. 최종 실험 설계

```text
G05 fractions: 0, 0.10, 0.25, 0.50, 0.75, 1.00
G05 counts:    0, 3,    8,    16,   24,   32 points/sample
seeds:         41, 42, 43
split seed:    42
split:         80% train / 10% validation / 10% test
model:         모든 fraction에서 동일한 capacity-matched masked model
shuffle:       같은 seed에서 같은 DataLoader order
initialization:같은 seed에서 동일 parameter initialization
normalization: train-only, 모든 fraction 공통
training:      AdamW, max 300 epochs, patience 8
selection:     fraction/seed별 best-validation checkpoint
report std:    sample std, ddof=1
```

각 fraction에서 저장되는 핵심 metric은 다음과 같다.

```text
Position MAE [x1,y1,z1,x2,y2,z2]
Charge 1 Euclidean position error
Charge 2 Euclidean position error
Charge magnitude MAE
Relative sign accuracy
Absolute sign accuracy (0%는 N/A)
Global sign accuracy, q1 anchor (0%는 N/A)
Signed pair accuracy (0%는 N/A)
```

## 실행 순서

현재 `KK` 환경은 경로가 깨져 있으므로 먼저 Python 환경을 복구해야 한다. 그 후 project root에서 다음처럼 실행한다.

```powershell
python Codes/generate_charge_dataset.py --output Models/charge_dataset_multipoint_v2.npz
python Codes/train_g05_fraction_experiment.py --data Models/charge_dataset_multipoint_v2.npz
```

짧은 구조 검증만 하려면 다음을 사용한다.

```powershell
python Codes/train_g05_fraction_experiment.py `
  --data Models/charge_dataset_multipoint_v2.npz `
  --smoke-only
```

본 리뷰 중 새 generator는 12개 sample을 메모리에서 생성해 shape, deterministic target ordering, 32개 fixed/unique sensor를 확인했고 두 새 Python 파일의 문법 compile을 통과했다. 실제 v2 재학습은 PyTorch가 있는 정상 환경이 필요하므로 아직 수행하지 않았다. 기존 CSV/모델 수치는 설계 오류를 진단하는 증거로만 사용했으며 새 설계의 성능으로 재해석하지 않았다.

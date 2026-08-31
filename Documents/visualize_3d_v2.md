# Experiment11용 3D 시각화 로더 2

실행 파일은 `Codes/visualize_3d_v2.py`다. 기존 2전하용 `Codes/visualize_3d.py`와 학습 코드는 수정하지 않는다. **학습·튜닝·checkpoint 재선택 없이 이미 검증 기준으로 선택된 전체 모델을 읽어 시각화한다.**

## 현재 PC에서 바로 실행

현재 PC에서는 기존 `.venv`의 Python 실행 파일 연결이 끊겨 있어, 확인된 Python 런타임과 기존 `.venv` 패키지를 함께 사용한다. 새 패키지를 설치하거나 학습 환경을 변경하지 않는다.

```powershell
$projectRoot = 'C:\Users\COM\Documents\GitHub\KK-AI'
$visualPython = 'C:\Users\COM\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$env:PYTHONPATH = Join-Path $projectRoot '.venv\Lib\site-packages'
$checkpoint = Join-Path $projectRoot 'Modelexperiment11\studies\final_fraction_sweep_seed43_20260831\runs\dropout020__g05_full_reconstruction__g1__s43\best_structure.pt'

& $visualPython "$projectRoot\Codes\visualize_3d_v2.py" --checkpoint $checkpoint --sample-mode all --show-g00 --show-g05 --device cuda --no-show
```

이 명령은 full 모델, seed 43, fraction 1.0의 **기존 best_structure checkpoint**로 원래 test 1,000개를 한 번 추론하고 best/median/worst 표본의 그림 3개를 저장한다. 여기서 best/worst는 **시각화 표본의 오차 순위**이며 모델·checkpoint를 선택하는 기준이 아니다.

기본 `--sample-mode`는 `median`이다. 정렬 후 아래쪽 중앙 표본을 사용하고 동률은 원래 test 순서로 결정한다. GUI backend가 있는 환경에서는 `--no-show`를 생략하면 회전 가능한 Matplotlib 창도 열린다. 창을 사용할 수 없는 환경에서는 `--no-show`로 PNG를 저장한다.

모델, fraction, seed, dropout, loss 가중치를 덮어쓰는 인수는 없다. 다른 조건을 보려면 **그 조건의 checkpoint 경로만** 지정한다.

```powershell
# fraction 0.0의 sign-only 모델: 절대 부호는 N/A, 예측은 전체 ± 동치류
$checkpoint = Join-Path $projectRoot 'Modelexperiment11\studies\final_fraction_sweep_seed43_20260831\runs\dropout020__g05_sign_only__g0__s43\best_structure.pt'
& $visualPython "$projectRoot\Codes\visualize_3d_v2.py" --checkpoint $checkpoint --sample-mode all --device cuda --no-show

# 저장된 test split 내부의 0번 표본. 원본 데이터셋의 행 번호와 다름
& $visualPython "$projectRoot\Codes\visualize_3d_v2.py" --checkpoint $checkpoint --sample-index 0 --no-show

# 추론·그림 저장 없이 checkpoint / 설정 / 데이터 / 분할 무결성만 검사
& $visualPython "$projectRoot\Codes\visualize_3d_v2.py" --checkpoint $checkpoint --check-only --device cpu
```

정상적인 Python 환경을 사용하는 다른 컴퓨터에서는 `python Codes/visualize_3d_v2.py --checkpoint <경로> ...`로 실행하면 된다. 필요한 패키지는 기존 NumPy, PyTorch, Matplotlib이다.

## 개선한 호환성

- `ModelExperiment11.load_trained_model()`을 그대로 사용한다. 저장된 routing, dropout, 정규화와 전체 가중치를 복원하며 `eval()` + `torch.inference_mode()`로 추론한다.
- `(batch, 5, 3)` 위치와 전하 5개를 처리한다. 그림의 `P1..P5`는 예측 슬롯, `T1..T5`는 원본 정답 슬롯이다. 슬롯 번호는 물리적 전하 ID가 아니다.
- `NewLearning9.matching_cost()` → `minimum_cost_assignment()` → `matched_targets()`를 그대로 호출한다. **저장된 loss 가중치와 정규화 공간**에서 위치·크기·상대부호의 공동 비용을 사용하며, 위치만 가까운 순서로 다시 대응시키지 않는다. 모든 필드가 동일한 일대일 대응을 공유한다.
- 전하 재구성도 `NewLearning9.reconstruct_charges()`를 사용한다. 정답을 사용해 예측의 부호를 바꾸지 않는다. G05가 없으면 회색 예측과 전체 벡터의 `+/-` 동치류를 표시하고, 절대/전체 부호 정확도는 N/A로 기록한다. 전하 MAE에는 기존 평가처럼 **다섯 부호 전체를 함께** 뒤집는 비교만 허용한다.
- `best_structure.pt`와 `best_total.pt`의 선택 epoch·검증 손실·전체 checkpoint 해시를 확인한다. `latest.pt`, 불완전한 학습, 검증 선택 잠금이 없는 실행, 잠금에 포함되지 않은 후보는 거부한다. 서로 다른 epoch의 분기를 합성하지 않는다.
- 원래 test split과 train-only 정규화를 사용한다. 정규화를 재계산하거나 fit하지 않으며 데이터 재생성도 하지 않는다. standalone seed 실행에서는 저장 split archive를 고정 split seed로 **대조만** 한 후 archive의 인덱스를 사용한다.

표시하는 집계 지표는 기존 evaluator의 `mean_position_3d_error`, `mean_position_mae`, `charge_mae`, `global_invariant_charge_mae`, `charge_magnitude_mae`다. 공식 최종 평가표의 모든 지표를 새로 작성하는 기능은 아니며 기존 최종 결과를 대체하지 않는다.

## 지원하는 저장 형식과 이동

다음 두 형식을 지원한다.

1. `ModelExperiment11.py`가 만든 `study.json` + `selection.json` + `runs/<trial>/...`.
2. `run_finalized_fraction_sweep.py`가 만든 seed별 `protocol.json` + `validation_selection.json` + `split_indices.npz` + `runs/<trial>/...`. 현재 seed 43의 6-fraction 실행이 이 형식이다.

구형 2전하 모델, NewLearning9 단독 baseline, 임의의 다른 sweep manifest는 자동으로 추측해 로딩하지 않는다. tuning study에서는 **원래 historical test**만 사용한다. `final/fresh_holdout.npz`를 원래 test라고 바꾸어 넣을 수 없다.

checkpoint의 상위 폴더에서 study를 찾는다. checkpoint를 따로 옮겼다면 `--study-dir`로 해당 실행의 원래 메타데이터 폴더를 지정한다. 데이터 파일을 옮겼다면 `--data`를 지정할 수 있지만 원본과 SHA-256이 같아야 한다.

```powershell
& $visualPython "$projectRoot\Codes\visualize_3d_v2.py" --checkpoint 'D:\models\best_structure.pt' --study-dir 'D:\experiments\seed43' --data 'D:\data\charge_dataset_5charges_v9.npz' --no-show
```

`.pt` 한 개만으로는 저장 split과 검증 선택을 입증할 수 없다. 함께 보관해야 할 파일은 해당 study의 권한 문서(`study.json` 또는 `protocol.json`), 선택 잠금, `normalization.json`, 저장 split, 해당 trial의 `config.json`·`history.json`·`result.json`이다. `sources/`도 원본 그대로 보관하는 것이 좋다. 다른 seed의 checkpoint나 `result.json` 파일은 읽지 않는다.

원래 source hash와 현재 model/physics/loader 코드를 대조한다. Git의 LF/CRLF 차이만 허용하고 의미가 달라진 코드는 거부한다. 소스 snapshot을 자동으로 실행하지 않는다. CPU/GPU 또는 PyTorch/CUDA 버전이 다르면 수치의 비트 단위 일치까지 보장하지 않으며 실제 추론 환경을 저장한다.

**이 프로젝트에서 생성한 신뢰할 수 있는 checkpoint만 사용한다.** PyTorch `.pt`는 pickle을 포함할 수 있다.

## 출력과 재현 정보

기본 출력은 다음과 같이 분리된다. 같은 명령을 반복해도 고유 실행 폴더가 추가되므로 기존 그림·결과·checkpoint를 덮어쓰지 않는다.

```text
Results/visualizations_v2/
  seed43/<model>/g<fraction>/<selection>_<checkpoint-hash>/<UTC시간>_<고유값>/
    best_seed43_testXXXX_dataXXXXX.png
    best_seed43_testXXXX_dataXXXXX.json
    median_seed43_testXXXX_dataXXXXX.png
    median_seed43_testXXXX_dataXXXXX.json
    worst_seed43_testXXXX_dataXXXXX.png
    worst_seed43_testXXXX_dataXXXXX.json
    predictions_seed43.npz
    manifest.json
```

- PNG 옆 JSON: 전체 설정, checkpoint/검증 선택·데이터·split·소스 해시, 실제 표본 인덱스, 5전하 대응, 정답과 원시 예측, 전하별 오차, ±부호 처리.
- NPZ: 전체 test 인덱스, 예측 위치·전하, 공동 매칭한 정답, 대응 순열, 관측 mask, 위치 오차, global-sign logit, 표본별 표시 지표. `allow_pickle=False`로 읽을 수 있다.
- `manifest.json`: 생성 완료 표시, 설정·검증 정보, 실제 추론 batch size와 환경, 시각화 소스 hash, Matplotlib 버전, 그림·NPZ 파일 해시와 생성 옵션.

`--output-dir`는 이 seed/조건별 폴더를 담는 상위 경로만 바꾼다. `--batch-size`는 추론 메모리 조절용이며 기본값은 checkpoint에 저장된 batch size다. 학습 batch size나 설정 파일은 바뀌지 않는다.

## 검증

```powershell
& $visualPython -m unittest discover -s "$projectRoot\Codes" -p test_visualize_3d_v2.py -v
```

신규 테스트 14개는 두 저장 형식, 두 모델, 6개 fraction, 두 checkpoint 기준, native 평가와의 일치, 공동 매칭, fraction 0의 부호 처리, 원본 파일 보존, 저장 경로 충돌 방지, 변경된 데이터/정규화/split/가중치 거부를 검사한다. 테스트는 임시 합성 데이터와 가중치 fixture를 사용하며 실제 학습이나 optimizer 업데이트를 하지 않는다.

2026-08-31 실제 seed 43 검증에서는 24개 checkpoint 각각의 test 1,000개를 읽기 전용으로 대조했다. 표시 지표와 native evaluator 및 기존 저장 test 결과의 최대 절대 차이는 **2.2485852180231802e-8**로 허용 오차 `1e-6`보다 작았다. 기존 코드·실험 파일 140개는 검사 전후 해시가 같았고 다른 seed의 checkpoint는 로딩하지 않았다. 상세 기록은 `Results/visualizations_v2/validation_seed43.json`에 있다. 이 검증은 시각화 호환성을 확인한 것이며 추가 학습이나 모델 선택이 아니다.

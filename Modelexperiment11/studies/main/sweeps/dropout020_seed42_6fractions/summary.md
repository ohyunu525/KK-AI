# Experiment11 — seed 42, six-fraction training sweep

요청한 6개 fraction에서 5전하(5point) 모델 2종의 학습을 모두 완료했다. 기존 체크포인트를 재사용하지 않고 12개 실행을 새로 학습했으며, train/validation만 사용했다. Test 평가는 수행하지 않았다.

## 고정 설정

- 모델: `g05_sign_only`, `g05_full_reconstruction`
- Training seed: `42`
- Fractions: `0.0`, `0.1`, `0.25`, `0.5`, `0.75`, `1.0`
- Experiment11 선택 후보: `dropout020`
- AdamW learning rate: `0.001`; weight decay: `0.0001`; structure dropout: `0.2`
- 최대 150 epoch; batch size 128; early-stopping patience 20
- Python 3.12.13, NumPy 2.5.2, Torch 2.7.0+cu126, CUDA 12.6, RTX 3070
- 학습 구간 합산 시간: 약 569.8초

## 결과

아래 loss는 각 실행의 `best_structure.pt`를 선택한 validation structure loss이며, 작을수록 좋다. 모든 실행은 early stopping으로 종료됐다.

| Fraction | 관측 G05 수 / 32 | Sign-only epochs | Full-reconstruction epochs | Sign-only loss | Full-reconstruction loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0 | 75 | 75 | 0.5796431913 | 0.5796431913 |
| 0.1 | 3 | 83 | 64 | 0.5796431913 | 0.5814221292 |
| 0.25 | 8 | 83 | 64 | 0.5796431913 | 0.5746815014 |
| 0.5 | 16 | 80 | 77 | 0.5796431913 | 0.5794908967 |
| 0.75 | 24 | 80 | 76 | 0.5796431913 | 0.5745758429 |
| 1.0 | 32 | 80 | 69 | 0.5796431913 | 0.5800517778 |

`5point`는 예측하는 전하 수가 5개라는 뜻이다. Fraction은 저장된 G05 후보 32개 중 관측하는 수를 결정하며, `1.0`은 32×32 전체 장을 뜻하지 않는다. Sign-only의 구조 경로에는 G05가 들어가지 않으므로 동일 seed에서 구조 loss가 fraction별로 동일한 것은 설계대로다.

## 저장 파일

- `sweep.json`: 환경, 하이퍼파라미터, 데이터/소스/부모 연구의 감사 기록
- `sweep_result.json`: 완료된 12개 실행, validation 진단, 98개 산출물 SHA256
- `trials.csv`: 체크포인트 선택 기준별 결과 24행
- `verification.json`: 최종 무결성 및 5전하 출력 검증
- `training.log`: 전체 학습 로그
- `run_sweep.py`: Experiment11의 `train_validation_run`을 호출하는 재현/재개 드라이버
- `runs/<candidate>__<model>__g<fraction>__s42/`: 각 실행의 `config.json`, `history.json`, `history.csv`, `status.json`, `result.json`, `best_total.pt`, `best_structure.pt`, `latest.pt`

체크포인트는 선택용 24개와 재개용 12개, 총 36개다. 원래 `main/runs`는 변경하지 않았다.

## 검증 및 재현성

98개 산출물 해시를 검증했고, 24개 선택 체크포인트를 실제로 다시 불러와 validation 샘플 2개에 추론했다. 모든 출력이 위치 `[2, 5, 3]`, 크기/상대부호 `[2, 5]`, 전체부호 `[2]`의 유한값이었다. 두 모델 모두 fraction `1.0`에서 부모 연구의 전체 epoch 학습 이력이 정확히 재현됐다.

이 체크아웃의 부모 연구에는 원본 `.pt`와 `split_indices.npz`가 없다. 부모의 선택은 봉인된 validation JSON 이력으로 재검증했고, split은 저장된 seed로 재생성하여 `study.json`의 모든 인덱스와 대조했다. 데이터셋은 복원 후 파일 SHA256 및 모든 배열 SHA256이 부모 연구와 정확히 일치함을 확인했다.

부모 JSON과 일부 Python 파일은 Git 체크아웃의 LF/CRLF 변환 때문에 byte 해시가 달라져, 내용 변경 없이 줄바꿈 변환만 허용해 감사했다. 세 학습 핵심 소스는 이 검사에 통과했다. `generate_charge_dataset.py`의 과거 byte 해시는 현재 체크아웃에서 재현되지 않았지만 live와 snapshot은 동일하며, 실제 학습에 사용한 데이터 파일/배열 해시는 원본과 일치한다. 이 제한은 `sweep.json`의 source audit에 명시했다.

## 실행

동일 버전의 CUDA 환경에서 다음 드라이버를 실행하면, 동일 manifest를 확인한 후 완료된 실행은 검증하여 건너뛰고 미완료 실행만 `latest.pt`에서 재개한다.

```powershell
python -u -B Modelexperiment11/studies/main/sweeps/dropout020_seed42_6fractions/run_sweep.py
```

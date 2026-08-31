**ModelExperiment9 저장 구조의 현재 PC 적합성 및 ModelExperiment10 보완 — 2026-08-31**

저장 용량과 파일 형식은 이 컴퓨터에서 사용할 수 있지만, 변경된 ModelExperiment9의 **평가 경로 탐색과 소스 동일성 검사는 그대로 사용하기 어렵다**. 실제 seed별 저장 폴더와 Python 3.14에서 두 문제를 재현했다. 요청대로 해결 코드는 [ModelExperiment10.py](C:/Users/COM/Documents/GitHub/KK-AI/Codes/ModelExperiment10.py)에만 반영하고 원인·안전 조건을 한국어 주석으로 설명했다. ModelExperiment9/NewLearning9 및 기존 학습 파일은 수정하지 않았다.

| 확인 항목 | 이 PC에서 확인한 값/결과 |
| --- | --- |
| 프로젝트 | `C:\Users\COM\Documents\GitHub\KK-AI` |
| 실행 환경 | Windows 10 build 19045, `.venv` Python 3.14.7, PyTorch 2.11.0+cu128, CUDA 12.8 |
| 메모리 | RAM 약 31.85 GiB, RTX 3070 VRAM 약 8 GiB |
| 저장 장치 | C: NTFS, 감사 시 여유 약 824.5 GiB |
| 기존 가중치 | seed 43의 12개 실행 × 3파일 = 36개, 총 약 117.13 MiB, 모두 로드·내용 검증 성공 |
| 실제 쓰기 검사 | Results와 Models 각각 한글·공백 포함 320자/319자 경로에서 생성·원자적 교체·복원·배타 잠금 성공 |

감사 원자료는 [storage_audit.json](C:/Users/COM/Documents/GitHub/KK-AI/Results/model_experiment10_storage_validation/storage_audit.json), 재현 코드는 [audit_storage.py](C:/Users/COM/Documents/GitHub/KK-AI/Results/model_experiment10_storage_validation/audit_storage.py)다. 가중치·원본 소스·데이터·기존 결과 파일의 SHA256과 수정 시각을 검사 전후 대조하여 동일함을 확인했다. 쓰기 검사는 허용된 루트 안에 만든 임시 폴더에서만 했고 종료 후 정리했다.

**실제 저장 폴더와 경로 문제**

결과는 `Results/new_learning9_experiments/5point_routing_v1_seed41`, `_seed42`, `_seed43`에 각각 12개 실행씩 있다. 공통 `5point_routing_v1` 결과 폴더에는 protocol이 없다. 가중치는 `Models/new_learning9_experiments/5point_routing_v1_seed43`에만 있다.

현재 9는 단일 seed 결과 폴더를 찾은 뒤에도 가중치 기본 경로를 공통 `Models/.../5point_routing_v1`로 유지한다. 이 PC에는 그 폴더가 없어 평가 진입이 실패한다. 9에 추가된 정확한 경로 옵션으로 이 문제를 피할 수 있지만, 다음의 소스 검사 문제는 별도로 남는다. 저장된 `.pt` 파일이 손상되어 발생한 문제는 아니다.

10에는 같은 `--evaluation-results-dir`, `--evaluation-checkpoint-dir` 옵션을 추가했다. 평가 전용일 때만 사용할 수 있으며 지정 폴더 뒤에 실험명을 붙이지 않는다. 경로를 명시하지 않은 경우에는 다음 순서로 찾는다.

1. 원래 공통 결과 폴더에 protocol이 있으면 사용한다. 없고 seed가 하나일 때만 `_seedN` 결과 폴더를 확인한다.
2. 공통 체크포인트 폴더에 요청한 run ID의 파일이 있으면 기존 동작을 유지한다.
3. 없으면 결과 폴더와 같은 이름 또는 단일 seed의 `_seedN` 체크포인트 폴더를 확인한다. 이름만 비슷한 폴더가 아니라 **저장 protocol로 계산한 요청 run ID의 파일**이 있어야 한다.
4. 대체 후보 두 곳이 모두 일치하면 명시적인 체크포인트 경로를 요구한다. 개별 실행을 여러 폴더에서 모아 섞지 않으며 없는 seed의 가중치를 다른 seed로 대체하지 않는다.

여러 seed에 대해 `_seedN` 결과 폴더들을 자동 병합하지 않는다. 공통 protocol을 사용하거나, 각 seed를 명시해서 평가해야 한다. 사용자가 지정한 정확한 경로가 없으면 다른 폴더로 자동 전환하지 않는다. 파일의 존재는 폴더 선택 조건일 뿐이며, 선택 후 기존 checkpoint fingerprint/schema/구성 검증을 다시 수행한다.

**Python 3.14와 Windows 줄바꿈 문제**

9의 구형 소스 허용 목록에는 LF 파일 해시 `4768dd…`만 있었지만 seed 42·43에 기록된 해시는 같은 소스의 CRLF 버전 `8cd11a…`였다. 또한 이 PC에서 `ast.dump()`의 기본 출력 해시는 `d62c52…`인데 기존 허용 AST 해시는 `38a0d1…`였다. 계산 코드는 같은데 빈 필드의 출력 방식이 달라진 것이다.

Python 3.13부터 추가된 `show_empty`의 기본값은 False다. 10에서는 `show_empty=True`를 명시하고 형식 이름 `python-ast-show-empty-v1`을 protocol에 저장한다. 이 수정은 현재 원본의 AST 해시를 재현하기 위한 것으로, 모든 미래 Python 버전의 AST가 영구적으로 같다고 가정하지 않는다. [Python 공식 ast.dump 문서](https://docs.python.org/3.14/library/ast.html#ast.dump)

Git의 `ac59a6c`와 `f524bcf`에서 NewLearning9 원본을 직접 읽어, 두 버전의 LF/CRLF 파일 네 가지가 현재 코드와 같은 실행 AST를 가진다는 것을 확인했다. 초기 protocol에 AST가 없을 때는 이 검증된 네 해시만 대응표로 허용한다. 새 protocol은 AST와 형식을 함께 기록하며 형식 태그가 없는 과거 AST 기록은 확인 가능한 두 출력 관례만 비교한다.

전체 파일 SHA256 일치가 우선이다. 파일이 다르면 주석·docstring·줄바꿈을 제외한 AST가 일치해야 하고, 상수·계산식이 바뀌거나 원본을 확인할 근거가 없으면 계속 거부한다. AST 검사가 필요한데 형식 태그를 알 수 없는 경우도 거부한다. 사용한 검증 방법과 학습 당시/현재 소스 해시는 별도 `evaluation.json`에 남긴다. 원래 protocol의 fingerprint를 다시 계산해 덮어쓰거나 검사를 끄는 방식은 사용하지 않았다.

**이동한 데이터 경로와 복구할 수 없는 파일**

| Seed | 기록된 데이터 상태 | 현재 평가에 필요한 파일 |
| --- | --- | --- |
| 41 | 과거 `C:\Users\COM\Documents\KK-AI\Models\...` 경로가 이 PC에 없음. 기록된 SHA256은 `f90880…`. | 학습 당시 데이터의 동일한 복사본과 seed 41 체크포인트가 필요함. |
| 42 | 현재 프로젝트 경로는 존재하지만 파일 SHA256이 학습 당시 `f90880…`와 다름. | 학습 당시 데이터의 동일한 복사본과 seed 42 체크포인트가 필요함. |
| 43 | 현재 데이터 SHA256 `12453c…`가 protocol과 일치함. | 36개 체크포인트가 있고 정상적으로 읽힘. |

10은 저장된 데이터 경로가 사라졌을 때만 현재 기본 NPZ와 프로젝트 Models의 같은 파일명 두 후보를 확인한다. SHA256이 정확히 같은 복사본만 사용하고 원래 protocol의 경로·해시는 보존한다. `--data`로 지정한 경로가 없거나, 존재하는 파일의 해시가 다르면 다른 파일로 대체하지 않는다. 데이터 재생성, 재분할, 정규화 재계산도 하지 않는다.

seed 41·42와 현재 데이터의 **파일 바이트가 다르다**는 사실을 확인한 것이며, 사라진 NPZ 내부 배열이 어느 부분에서 다른지는 현재 파일만으로 판단할 수 없다. 경로 수정으로 사라진 가중치나 원본 데이터를 복원할 수는 없다. 또한 10의 보완은 9 가중치를 10 가중치로 변환하는 기능이 아니다. v9/v10의 protocol 및 checkpoint namespace 구분은 유지한다.

**Windows 저장 방식과 유지한 기능**

`latest.pt`, `best_total.pt`, `best_structure.pt`, JSON/CSV/NPZ 저장 형식을 유지했다. 같은 폴더의 임시 파일로 먼저 저장하고 `os.replace`로 교체하는 방식, 일시적인 Windows 파일 잠금 재시도, 실패 시 이전 파일 보존, 배타적 실험 잠금을 모두 유지·검증했다. CUDA를 요청해도 체크포인트는 먼저 CPU에 로드하고 필요한 모델/optimizer만 장치로 옮기는 기존 방식도 유지한다.

이 PC의 `LongPathsEnabled=1`을 읽어서 확인했고 레지스트리를 변경하지 않았다. Windows는 레지스트리 값과 앱의 긴 경로 지원이 함께 필요하므로 설정값만 보고 판단하지 않고 실제 Python/PyTorch 쓰기·읽기를 수행했다. 성공 범위는 이 실행 환경이며 모든 탐색기·외부 프로그램의 긴 경로 지원을 보장하는 것은 아니다. [Microsoft 공식 긴 경로 설명](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation)

기존 학습 루프, 두 목적의 조기 종료, 선택적 구조 dropout, 모델 분기·물리 손실·대칭, 두 최적 체크포인트, optimizer/RNG/shuffle 상태 재개, 집계·smoke·평가 전용 기능은 그대로다. 학습 재개는 여전히 **같은 코드·데이터·설정·환경의 기존 protocol**을 요구한다. 이번 경로 보완을 다른 환경에서 임의로 학습을 이어가기 위한 허용으로 확장하지 않았다.

**평가 명령 예시**

다음은 **10으로 학습한 해당 실험이 존재할 때** 사용할 예시다. 이 점검에서 정식 10 실험을 새로 학습하거나, 기존 9 폴더를 10으로 이름 변경하지 않았다. 프로젝트 루트에서 실행하며 폴더명은 실제 10 실험명에 맞춘다.

```powershell
.\.venv\Scripts\python.exe Codes\ModelExperiment10.py --evaluate-only --experiment-name g05_earlystop_v10 --seeds 43 --fractions 0,0.1,0.25,0.5,0.75,1
```

자동 후보에 해당하지 않는 이동 경로는 두 폴더를 정확하게 지정한다. 아래 경로도 예시이며 실제로 존재하는 10 실험 폴더를 넣어야 한다.

```powershell
$resultPath = 'C:\Users\COM\Documents\GitHub\KK-AI\Results\new_learning10_experiments\g05_earlystop_v10_seed43'
$checkpointPath = 'C:\Users\COM\Documents\GitHub\KK-AI\Models\new_learning10_experiments\g05_earlystop_v10_seed43'
.\.venv\Scripts\python.exe Codes\ModelExperiment10.py --evaluate-only --seeds 43 --fractions 0,0.1,0.25,0.5,0.75,1 --evaluation-results-dir $resultPath --evaluation-checkpoint-dir $checkpointPath
```

`--results-root`/`--checkpoint-root`는 실험명을 덧붙이는 **부모 폴더** 옵션이다. 위의 `--evaluation-…-dir`은 실험 폴더 자체를 지정하는 옵션이므로 서로 혼동하지 않아야 한다. 평가 결과는 선택한 결과 폴더의 `evaluations/<timestamp_uuid>/`에만 새로 작성하며 원본 학습 결과·체크포인트는 덮어쓰지 않는다.

**실제 검증 결과와 한계**

| 검증 | 결과 |
| --- | --- |
| 수정 전 전체 회귀 | 124개 중 120개 통과, v9 오류 1개, 종전 skip 3개 |
| 수정 후 v10 전용 | **52개 모두 통과**: 기존 44개 + 저장 호환성 8개 |
| 수정 후 전체 회귀 | 132개 중 128개 통과, 동일한 v9 오류 1개, 동일한 skip 3개 |
| 기존 파일 검사 | 체크포인트 36개 모두 내용 검증 성공, 감사한 원본 파일의 바이트/수정 시각 보존 |
| 기존 모델 추론 | full/G05 100%/seed 43의 best_structure(epoch 26)를 v9 로더로 읽고 학습 샘플 4개의 CPU/CUDA 출력이 유한함을 확인 |
| 실제 저장 검사 | 두 저장 루트 모두 긴 한글 경로의 저장·원자 교체·재로딩 및 배타 잠금 통과, 임시 폴더 정리 완료 |

남아 있는 v9 오류는 `test_evaluate_only_accepts_moved_seed_results_and_documented_legacy_source`의 소스 호환성 검사다. 변경 전에도 같은 오류를 재현했고, 요청 범위에 따라 9 자체를 수정하지 않아 그대로 남아 있다. Skip 3개는 구형 `train_g05_fraction_experiment` 모듈이 없는 기존 항목이다. 전체 테스트가 모두 통과했다고 해석하면 안 된다.

새 테스트는 공통/seed별 경로 각각에서 두 모델을 작은 합성 데이터로 실제 학습한 뒤 자동 경로와 명시 경로로 평가하고, 저장 당시와 같은 지표·설정을 사용하는지 확인한다. 평가 중 optimizer 생성·학습·smoke·가중치 저장·재분할·정규화 재계산을 호출하면 실패하도록 검사했다. 잘못된 seed 후보, 모호한 폴더, 해시 불일치, 알려지지 않은 소스, 상수 변경, 일시적/영구적 파일 잠금도 포함한다. 기존 CPU/CUDA 재개 재현성 및 조기 종료/dropout 회귀 검사 44개도 다시 실행했다.

이번 추가 작업에서 실제 데이터의 정식 학습이나 테스트셋 성능 비교는 수행하지 않았다. 이전 과적합 분석과 pilot 결과는 [ModelExperiment10 분석](C:/Users/COM/Documents/GitHub/KK-AI/Documents/ModelExperiment10.md)에 그대로 보존했다.

재현 로그: [수정 전 전체 테스트](C:/Users/COM/Documents/GitHub/KK-AI/Results/model_experiment10_storage_validation/baseline_tests.log), [수정 후 v10 테스트](C:/Users/COM/Documents/GitHub/KK-AI/Results/model_experiment10_storage_validation/final_model10_tests.log), [수정 후 전체 테스트](C:/Users/COM/Documents/GitHub/KK-AI/Results/model_experiment10_storage_validation/final_all_tests.log), [저장 감사 로그](C:/Users/COM/Documents/GitHub/KK-AI/Results/model_experiment10_storage_validation/storage_audit.log).

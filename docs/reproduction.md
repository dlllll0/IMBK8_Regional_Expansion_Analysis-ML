# 실행 안내

## 1. 데이터 없이 확인

저장소 폴더를 VS Code의 **파일 → 폴더 열기**로 엽니다. `README.md`를 열고 `Ctrl+Shift+V`로 미리보기를 확인합니다. 지수·검증 상세 영역은 접기 제목을 눌러 펼칠 수 있습니다.

`notebooks/01-analysis.ipynb`는 저장된 공개 집계 JSON과 이미지 파일만 읽습니다. VS Code의 Python·Jupyter 확장과 Python 커널이 필요합니다. 기본 셀은 실데이터·네트워크에 접근하거나 파일을 쓰지 않습니다.

저장소 최상위 터미널에서:

```powershell
python scripts/validate_repository.py
```

이 명령은 표준 라이브러리만으로 코드 문법·상대 링크·노트북 출력·공개 제외 파일·집계 산술을 검사합니다. 원자료 분석이나 공개 권한 검토를 대신하지는 않습니다.

## 2. 승인된 데이터로 핵심 경로 실행

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt`는 이번 로컬 재검증에서 실제 사용한 버전입니다. 원 프로젝트 당시 환경의 복원 목록은 아닙니다.

원자료는 저장소 밖에 보관하고 승인된 로컬 경로를 변수로 지정합니다. 아래 경로는 직접 입력하는 자리이며 실제 파일을 제공하는 예시가 아닙니다.

```powershell
$customerCsv = Read-Host '승인된 법인 거래 CSV의 전체 경로'
$rateCsv = Read-Host '승인된 월별 금리 CSV의 전체 경로'

.venv\Scripts\python.exe scripts/run_analysis.py `
  --data $customerCsv `
  --rates $rateCsv `
  --output outputs/local-validation.json `
  --with-regression `
  --jobs 2
```

입력은 [변수 설명](data-dictionary.md)과 [기계 판독 형식](input-schema.json)을 참고하세요. 법인 데이터는 CP949, 금리표는 UTF-8 인코딩을 사용합니다.

실행 순서는 **입력 검사 → 원코드 파생변수 → 기존 2단계 군집·시도 유사도 → 선택한 경우 기본 RF → 집계 JSON 저장**입니다. `--with-regression`을 빼면 회귀 학습은 생략합니다. 모델 학습 시간·메모리는 실행 환경에 따라 달라집니다.

출력은 집계 결과만 포함하며 30행 미만 지역은 해당 지역의 유사도와 관측 수 모두 제외합니다. 30은 포트폴리오 정리의 보수적 출력 기준이며 익명성을 수학적으로 보장하는 기준은 아닙니다. `outputs/`는 Git 제외 대상이며 새 실행 결과를 검토 없이 업로드하지 마세요.

기존 출력 파일이 있으면 덮어쓰지 않고 중단합니다. 새 파일명을 지정하세요. 원자료·모델·행 단위 예측은 저장하지 않으며 외부 전송 기능이 없습니다.

## 3. 원 비교·튜닝 실험

`src/model_selection.py`는 B 노트북의 모델 비교·튜닝 논리를 함수로 정리한 선택 실행 코드입니다. 이번 패키징에서 전체 CV·Optuna 실험을 다시 실행하지 않았습니다. 해당 의존성의 원 버전도 확인되지 않아 `requirements-optional.txt`에는 버전을 임의로 고정하지 않았습니다.

선택 실행 시 먼저 별도 환경에서 의존성을 설치하고, 승인된 데이터로 아래 함수를 호출할 수 있습니다. 데이터 프레임은 메모리에서만 유지하고 결과를 업로드하기 전에 검토해야 합니다.

```python
from pathlib import Path
from scripts.run_analysis import load_inputs
from src.pipeline import prepare
from src.model_selection import compare_and_tune

raw, rates = load_inputs(Path(customer_csv_path), Path(rate_csv_path))
summary = compare_and_tune(prepare(raw, rates), n_trials=30)
```

`customer_csv_path`와 `rate_csv_path`는 승인된 로컬 파일 경로 문자열로 직접 설정합니다. sampler seed가 없으므로 원 Optuna 시행 순서·결과와 동일성을 보장하지 않습니다. 새 평가를 설계할 때는 전처리를 CV 내부로 옮기고 기간·고객 단위 분할을 개선해야 합니다. 위 함수는 그 개선을 수행한 새 실험이 아니라 원 설계 보존용입니다.

## 4. 공개 그림 재생성

기존 노트북에서 가져온 PNG 3개는 보존된 이미지이며 재학습 없이 열람할 수 있습니다. 지역 순위 그림만 공개 집계 JSON에서 다시 그릴 수 있습니다.

```powershell
.venv\Scripts\python.exe scripts/render_figures.py --output outputs/regional-similarity-review.png
```

한글 폰트가 필요하며 Windows의 맑은 고딕 등을 확인합니다. 기존 그림은 덮어쓰지 않습니다.

## 실제 확인한 범위

- 전체 로컬 CSV의 행·열·기간과 입력 범주 검사.
- 원 산식 전처리, 기본 Random Forest, K=4 및 K=3 군집화, 시도 코사인 유사도 실행.
- 저장 군집 규모·중앙값·상위 유사도와 대조. 회귀 수치 차이는 별도 기록.
- 공개 노트북 기본 셀 실행, 그림 열람, 파일·경로·비밀정보 패턴 검사.
- 원 프로젝트 파일 및 사용한 입력 CSV의 해시 보존 확인.

확인하지 않은 범위: 원 노트북 전체 무수정 Run All, 전체 CV·Optuna 재탐색, SHAP 재산출, 실루엣 전체 재산출, 시군구 수정 순위, 기간 외·고객 외 검증, 실제 영업 실험·거점 수익성.


## 지역별 캠페인 사후 분석

저장소 밖의 승인된 CSV 경로를 위와 같이 지정한 뒤 실행합니다. 출력 상위 폴더를 먼저 만들고, 기존 파일이 없는 이름을 사용하세요.

```powershell
New-Item -ItemType Directory -Force outputs
.venv\Scripts\python.exe scripts/regional_campaign_analysis.py --data $customerCsv --rates $rateCsv --output outputs/regional-campaign-local.json
python scripts/render_regional_campaign.py
```

첫 명령의 분석은 기존 2단계 군집을 다시 적합한 뒤 집계만 저장합니다. 그림 스크립트는 공개 `assets/regional-campaign-results.json`만 읽고 `assets/regional-campaign-gaps.png`를 갱신합니다. 입력 원자료 수정·네트워크 전송·행별 결과 저장은 수행하지 않습니다.

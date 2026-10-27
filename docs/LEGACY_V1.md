> 이전 v1 설명의 보관본입니다. v2 코드로 v1 체크포인트 추론을 재개하지 마세요. 기존 소스는 GitHub commit `eb0737bedc62ee8327edf0a5c77db6cb60dcb939`에서 확인할 수 있습니다. 기존 결과 분석에는 현재 `audit` 명령을 사용합니다.

# ContractNLI 문맥 압축 메인 실험

현재 기본 실행 환경은 **Windows · PyCharm Jupyter Notebook · Python 3.11.9 · NVIDIA GPU 한 대**입니다. 실제 GPU는 실행 시 자동 인식합니다. [실행 노트북](notebooks/run_base_pycharm.ipynb)을 PyCharm에서 열고 순서대로 실행하세요.

**RTX 5070에서 새로 진행한다면 [RTX 5070 전용 노트북](notebooks/run_base_rtx5070_pycharm.ipynb)을 사용하세요.** PyTorch 2.8.0/CUDA 12.8과 bitsandbytes 0.48.1을 설치하며, 기존 CUDA 12.4 노트북과 설치 셀이 다릅니다. 실험 모델·4조건·3개 라벨·30% 목표는 같습니다.

Proposal의 메인 4조건만 비교합니다. Token-budget sweep, ablation, 의미적 evidence 보존 분석은 포함하지 않습니다.

| 조건 | 문맥 구성 | 사용 모델 | 호출 단위 |
|---|---|---|---|
| Original | 계약서 원문 전체 | 압축 없음 | 문서 |
| Naive | 질문을 보지 않는 일반 요약 | `Qwen/Qwen2.5-7B-Instruct` | 문서당 1회 |
| Contextualized | 조건·예외·부정·의무 주체 등을 보존하도록 요청하는 범용 요약 | 동일 Qwen2.5 | 문서당 1회 |
| LongLLMLingua | 가설에 관련된 문맥을 우선 선택하고 토큰 압축 | 공식 기본 scorer `NousResearch/Llama-2-7b-hf` | 문서–가설마다 1회 |

판단 모델은 네 조건 모두 **`Qwen/Qwen3-8B`**, thinking off, greedy decoding입니다. 판단 프롬프트도 문맥을 제외하면 같습니다. 모든 모델은 기본 **NF4 4bit**, 한 번에 하나만 로딩합니다. Naive·Contextualized 요약에는 가설·정답·gold evidence를 넣지 않습니다. 이 방식은 본 프로젝트의 프롬프트 기반 baseline이며 외부 논문의 특정 contextualized 요약 알고리즘을 재현했다는 뜻은 아닙니다.

## 평가 대상과 30%의 의미

공식 test split **123문서 × 17가설 = 2,091문제**를 모두 사용합니다.

- Entailment 968개, Contradiction 220개, NotMentioned 903개: **3개 라벨 모두 정답·출력에 유지**합니다.
- Gold evidence가 있는 1,188문제와 없는 NotMentioned 903문제를 구분합니다. 없는 근거를 보존율 0점으로 취급하지 않습니다.
- `retention_ratio = 0.30`은 **원문 토큰의 30%를 남기는 목표**입니다. 삭제 비율은 약 70%입니다. 판단 정답 보존율이나 evidence 보존율을 30%로 맞춘다는 뜻이 아닙니다.
- Qwen3 판단 토크나이저로 원문과 압축문 토큰 수를 공통 측정합니다. 가설·시스템 프롬프트는 문맥 길이에서 제외합니다.
- 요약의 요청 길이와 `max_new_tokens`는 `floor(원문 Qwen3 토큰 수 × 0.30)`입니다. Qwen2.5 생성 토크나이저와 공통 측정 토크나이저의 차이, 조기 EOS 때문에 실제 길이는 다를 수 있습니다.
- LongLLMLingua 내부 예산은 scorer 토큰 수의 30%입니다. 출력은 같은 Qwen3 토크나이저로 다시 측정합니다. 모델별 토크나이저가 다르므로 **엄밀한 동일 토큰 수 실험이 아니라 동일 목표 보존 비율 실험**입니다. 실제 평균·전체 토큰 비율과 예산 초과 건수를 보고합니다.
- 길이를 맞추기 위한 압축문 사후 절단이나 원문 자동 truncation은 하지 않습니다. 요약이 출력 한도에 도달했는지도 기록합니다.

## 설치

### RTX 5070 — Windows Blackwell

PyCharm의 프로젝트와 Jupyter 커널 모두 **Python 3.11.9 기반 새 가상환경**을 선택하고 [RTX 5070 전용 노트북](notebooks/run_base_rtx5070_pycharm.ipynb)을 여세요. 각 설치 셀은 다음과 같습니다.

```python
%pip install --upgrade pip
%pip install --upgrade torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
%pip install -e ".[inference,quantization,longllmlingua,rtx5070]"
```

설치 후 **커널을 재시작**하고 첫 환경 확인 셀부터 다시 진행합니다. GPU 확인 셀은 CUDA 연산과 bitsandbytes NF4 4bit 연산을 실제로 수행합니다. `CUDA available: True`만으로 호환성을 판단하지 않습니다. **`CUDA tensor test: PASS`와 `bitsandbytes NF4 test: PASS`가 모두 나와야 다음 단계로 진행하세요.**

이 노트북은 `configs/rtx5070.toml`을 읽고, smoke 결과는 `outputs/rtx5070_smoke/`, 전체 실험은 `outputs/rtx5070_base/`에 저장합니다. 이전 PC의 테스트 출력은 이 폴더로 복사하지 않습니다. 새 PC에서도 7번 smoke 셀의 `Smoke run completed: True` 및 실행 종료를 확인한 뒤 8번의 `RUN_FULL_EXPERIMENT`를 `True`로 바꿉니다.

스크린샷의 전체 그래픽 메모리는 공유 시스템 RAM을 포함합니다. 모델 적재에 사용하는 전용 VRAM은 약 12GB이므로 4bit·한 모델씩 로딩·배치 1은 유지합니다.

설치 조합의 근거: [공식 CUDA 12.8 wheel 목록](https://download.pytorch.org/whl/cu128/torch/), [PyTorch 2.8.0 빌드의 compute capability 12.0 지원](https://github.com/pytorch/pytorch/blob/v2.8.0/.ci/manywheel/build_cuda.sh), [bitsandbytes 0.48.1의 Windows sm120 지원](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.48.1/docs/source/installation.mdx).

### 이전 RTX 30/40 시리즈 환경

아래는 기존 `run_base_pycharm.ipynb`의 설치 안내입니다. RTX 5070은 위의 전용 설치 조합을 사용하세요.

PyCharm에서 프로젝트 인터프리터를 **Python 3.11.9 기반 가상환경**으로 설정하고, 노트북의 커널도 같은 환경으로 선택하세요. 예를 들어 프로젝트 환경이 `.venv`라면 커널의 Python 실행 파일은 `.venv\Scripts\python.exe`입니다. Jupyter 서버를 실행할 패키지가 없다면, 해당 환경을 선택한 **PyCharm Terminal**에서 다음을 실행하세요.

```powershell
python -m pip install jupyter ipykernel
```

[notebooks/run_base_pycharm.ipynb](notebooks/run_base_pycharm.ipynb)의 첫 셀에서 Python 버전·실행 파일 경로·프로젝트 루트를 확인합니다. 아래 설치 명령은 **Jupyter 코드 셀**에 각각 입력합니다. `python -m venv`나 `source`를 Python 셀에 입력하지 않습니다.

```python
%pip install --upgrade pip
%pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
%pip install -e ".[inference,quantization,longllmlingua]"
```

설치 후 **Restart Kernel**을 실행하세요. 그 다음 노트북의 첫 환경 확인 셀을 다시 실행하고 설치 셀을 건너뛰어 GPU 확인으로 이동합니다. 설치 직후 커널을 재시작하기 전에는 GPU 추론을 시작하지 않습니다.

노트북은 torch import 전에 `CUDA_VISIBLE_DEVICES=0`으로 GPU 한 개만 노출합니다. 실제 GPU 모델·VRAM·현재 여유 VRAM을 직접 출력합니다. 4bit·float16·SDPA attention·배치 1로 메모리를 줄이고, 요약/압축/판단 모델을 순서대로 하나씩 올립니다. **3070 한 대에서 모든 긴 입력이 적재된다는 보장은 없습니다.** 먼저 한 문서로 검증하세요. OOM이 발생하면 다른 GPU 프로그램을 종료하고 커널을 재시작한 뒤 저장된 결과부터 재개합니다. 길이 제한을 낮추거나 원문을 자르면 Original 조건의 의미가 바뀌므로 자동으로 수행하지 않습니다.

모델은 Hugging Face에서 최초 다운로드됩니다. 공개 모델 기준으로 API 키는 필수 설정이 아닙니다. 자체 서버/API를 호출하는 코드는 없습니다. 모델 가중치와 캐시를 위한 디스크 공간을 확보하세요.

## 데이터 준비와 첫 실행

RTX 5070 노트북에서는 아래와 같은 단계로 진행하되 `configs/rtx5070.toml`과 전용 출력 폴더를 사용합니다. 터미널에서 실행할 때는 다음과 같습니다.

```bash
python -m contractnli download
python -m contractnli run --config configs/rtx5070.toml --preflight
python tools/llmlingua_prefix_compat.py --config configs/rtx5070.toml --max-documents 1 --output-dir outputs/rtx5070_smoke
python tools/llmlingua_prefix_compat.py --config configs/rtx5070.toml
```

노트북에서는 5번부터 데이터 준비 → 입력 길이 검증 → 한 문서 smoke run → 전체 실행 순으로 진행합니다. **7번 셀을 중지하지 말고 `Smoke run completed: True` 출력 및 셀 실행 종료를 기다리세요.** 그 다음 8번 셀의 `RUN_FULL_EXPERIMENT`만 `True`로 바꾸고 8번 셀을 실행합니다. 최초 모델 다운로드가 끝난 뒤에도 압축·판단이 계속되므로 다운로드 100%를 실험 완료로 보지 않습니다. 완료되지 않은 smoke run은 전체 실행을 시작할 수 없도록 확인합니다. 노트북에서 데이터 준비와 실험 실행은 Python 함수를 직접 호출하므로 `python -m ...` 명령을 그대로 Python 셀에 입력할 필요가 없습니다.

중간에 중지했다면 같은 smoke 셀을 다시 실행하면 됩니다. 기존 결과와 모델 다운로드 캐시는 유지하세요. 노트북의 환경 확인 셀은 Hugging Face import 전에 다운로드 timeout 120초, 메타데이터 timeout 30초를 설정합니다. 이 값은 모델이나 실험 설정을 바꾸지 않습니다.

### LongLLMLingua `get_prefix_length`의 AssertionError

LLMLingua 0.2.2는 질문과 문맥의 토큰 경계를 찾은 뒤, 토큰을 decode한 문자열이 원문 앞 100문자와 정확히 같다고 검사합니다. 토크나이저의 공백·문장부호 정리 등으로 문자열 왕복 변환이 달라지면 이 검사가 실패할 수 있습니다.

두 실행 노트북에는 설정 셀 앞에 **3-1번 호환 보정 셀**이 있습니다. `tools/llmlingua_prefix_compat.py`가 기존 경계 계산을 먼저 사용하고, AssertionError가 난 경우에만 fast tokenizer의 원문 문자 위치로 경계를 검증합니다. 토큰이 질문과 문맥의 경계를 가로지르는 등 안전한 분리가 불가능하면 중단합니다. 원문이나 토큰 ID를 고치거나 assertion을 무조건 통과시키지 않습니다.

기존 결과를 재사용할 수 있도록 `src/contractnli` 본체는 바꾸지 않습니다. 기존 소스·데이터·설정 signature 및 모델/런타임 검사는 유지하고, 보정 파일의 버전과 별도 SHA256을 `run.json`의 `compatibility_patches`에 기록·검사합니다. 이전에 정상 생성된 요약·압축·판단 파일은 수정하지 않습니다. 모델 다운로드 캐시도 유지합니다.

진행 중 이 오류가 발생했다면 현재 노트북에 다음 셀을 추가한 뒤, 커널을 재시작하고 환경·GPU·보정·설정·데이터 셀을 실행하세요. 이 파일만 최신 코드에서 프로젝트의 `tools` 폴더로 복사하면 되며 패키지를 재설치할 필요가 없습니다.

```python
import runpy
patch = runpy.run_path(
    str(PROJECT_ROOT / "tools/llmlingua_prefix_compat.py"),
    run_name="contractnli_prefix_compat",
)
patch["apply_prefix_fix"]()
```

설정·데이터 셀 실행 후에는 캐시의 실제 토크나이저만으로 특정 실패 항목을 확인할 수도 있습니다. 예를 들어 마지막 완료 로그가 `442/2091: 49/nda-4`였다면 다음 항목은 아래와 같습니다. 이 검사는 가중치를 로딩하지 않습니다.

```python
print(patch["validate_prefix_case"](BASE_CONFIG, "50", "nda-11"))
```

7번 셀을 재실행해 완료된 smoke 결과를 불러오고, 8번 셀을 재실행하면 저장된 전체 실험의 마지막 완료 항목 다음부터 이어집니다. 기존 `outputs`와 Hugging Face 캐시는 유지하세요. 이 오류를 해결하려고 `condition_compare`나 LongLLMLingua 설정을 바꿀 필요가 없습니다.

터미널에서는 보정 적용 진입점으로 실행합니다.

```bash
python tools/llmlingua_prefix_compat.py --config configs/rtx5070.toml
```

아래는 노트북 대신 **프로젝트 가상환경을 선택한 터미널**에서 실행할 때의 명령입니다.

```bash
python -m contractnli download
python -m contractnli run --config configs/base.toml --dry-run
# 가중치를 로딩하기 전 토크나이저만으로 전체 입력 길이 확인
python -m contractnli run --config configs/base.toml --preflight
# 먼저 한 문서 smoke run: 본 실험과 출력 디렉터리를 구분
python tools/llmlingua_prefix_compat.py --config configs/base.toml --max-documents 1 --output-dir outputs/rtx3070_smoke
# 전체 123문서, 2,091문제 × 4조건
python tools/llmlingua_prefix_compat.py --config configs/base.toml
```

다운로더는 공식 저장소의 고정 commit ZIP을 받고 SHA256을 검증합니다. JSON과 LICENSE/TERMS만 추출하고 PDF는 추출하지 않습니다. 오프라인에서는 `download --archive /path/to/contract-nli.zip`을 사용할 수 있으며 동일한 체크섬을 검증합니다. 공식 ContractNLI 데이터는 CC BY 4.0이며 원 저자 표시와 데이터 이용 조건을 확인하세요.

전체 실험은 문서 요약 246회, 질문별 압축 2,091회, 판단 8,364회입니다. `--dry-run`은 실제 모델 추론이나 정확도 산출을 하지 않습니다. `--preflight`는 토큰 길이만 확인하며 GPU VRAM 검증은 smoke run이 담당합니다.

## GPU 한 대에서 순차 실행

`run --stage all` 또는 노트북의 `Experiment(BASE_CONFIG).run("all")`은 다음 순서로 실행합니다. **각 모델은 다음 모델을 로딩하기 전에 해제**되며, 모델 병렬화·여러 GPU·두 PC 간 파일 인계가 필요하지 않습니다.

1. Qwen2.5-7B-Instruct 로딩 → Naive·Contextualized 요약 생성 → 모델 해제.
2. 기본 Llama-2-7B scorer 로딩 → 가설별 LongLLMLingua 압축 → 모델 해제.
3. Qwen3-8B 로딩 → 4조건 판단 → 모델 해제.
4. 전체 예측이 완료되면 결과 JSON·CSV 생성.

필요하면 **같은 PC에서** 단계별로 나누어 실행할 수도 있습니다. 다음은 터미널 명령입니다.

```bash
# 문서별 요약
python tools/llmlingua_prefix_compat.py --stage summarize
# 질문별 LongLLMLingua 압축
python tools/llmlingua_prefix_compat.py --stage compress
# 동일한 네 문맥을 동일 판단 모델로 평가
python tools/llmlingua_prefix_compat.py --stage judge
python tools/llmlingua_prefix_compat.py --stage report
```

결과는 `outputs/rtx3070_base/`에 유지합니다. 여러 노트북이나 프로세스에서 같은 실험을 동시에 실행하지 마세요. 중단 후 같은 실행 셀이나 명령을 다시 실행하면 완료된 항목을 재사용합니다. `contexts.jsonl`, `predictions.jsonl`, `run.json`을 보존해야 합니다.

출력 폴더 이름의 `rtx3070`은 기존 결과와의 호환을 위해 유지한 이름이며 GPU 모델을 제한하지 않습니다. 실제 GPU는 `run.json`에 기록합니다. 다른 GPU에서도 실행할 수 있지만 새 장치에서 메모리 적재를 다시 확인해야 합니다.

데이터 내용·설정·소스가 달라지면 이전 결과 재사용을 거부합니다. 각 역할별 실제 모델 revision, 패키지 버전, 장치·양자화·attention 설정을 저장하고 변경을 감지합니다. 재현 실험에는 `revision = "main"` 대신 Hugging Face의 고정 commit SHA를 설정하는 편이 좋습니다. 최초 실행의 실제 revision은 `run.json`에서 확인할 수 있습니다. 이번 버전은 기본 출력 폴더를 새로 지정했으므로 이전 `outputs/base/`를 덮어쓰지 않습니다.

강제 종료로 `.run.lock`만 남았다면 해당 프로세스가 종료됐음을 확인한 후 잠금 파일만 제거하세요. JSONL 마지막 줄이 쓰다 끊겼다면 원본을 백업하고 불완전한 마지막 줄을 복구해야 합니다. 잘못된 줄을 조용히 무시하지 않습니다.

## LongLLMLingua 고정 설정

[공식 LongLLMLingua 사용 예제](https://github.com/microsoft/LLMLingua#-quick-start)의 설정을 사용합니다.

```python
condition_in_question="after_condition"
reorder_context="sort"
dynamic_context_compression_ratio=0.3
condition_compare=True
context_budget="+100"
rank_method="longllmlingua"
```

기본 `PromptCompressor` scorer인 Llama-2-7B를 사용하고, 위 값에 대한 ablation이나 튜닝은 하지 않습니다. `dynamic_context_compression_ratio=0.3`은 원문 보존 목표 30%와 별개의 공식 알고리즘 파라미터입니다. 메서드의 모든 인자를 생략하면 기본 `rank_method="llmlingua"`가 되므로 LongLLMLingua 실험이 되지 않습니다.

계약서는 가설·정답을 보지 않고 최대 512 scorer 토큰의 고정 청크로 나눕니다. 긴 줄은 문자 구간으로 나누며, 분할 자체는 모든 원문 문자를 보존합니다. 공식 구현이 청크를 연결하면서 문단 구분자를 추가할 수 있습니다. 질문에는 해당 NLI 가설만 사용하고 정답·gold evidence는 사용하지 않습니다. `concate_question=False`로 압축문에 가설을 중복 삽입하지 않습니다. 나머지 필터는 공식 기본값을 유지합니다.

Qwen3 지원을 위해 Transformers 4.51.3을 고정했습니다. LLMLingua 0.2.2의 legacy KV cache와 이 버전의 DynamicCache를 연결하는 어댑터가 있습니다. 캐시 표현만 변환하며 점수 계산·랭킹·압축 기준을 바꾸지 않습니다. LLMLingua-2로 대체하지 않습니다.

## 출력과 지표 정의

`outputs/rtx3070_base/` (전체 실험), `outputs/rtx3070_smoke/` (한 문서 검증):

- `run.json`: 설정, 데이터·코드 fingerprint, 모델 revision, 역할별 GPU·패키지 버전.
- `contexts.jsonl`: 재사용할 요약과 질문별 압축문, 원문/압축 토큰 수, 실제 보존 비율, 처리시간, 종료 사유.
- `predictions.jsonl`: 문서·가설 ID, 정답, 예측, 모델 원출력, 해당 문맥 key, gold evidence 유무 및 원문 근거.
- `metrics.json`, `metrics.csv`: 모든 예정 예측이 완료된 경우에만 생성되는 집계.

지표는 다음과 같습니다. 수치는 0–1 범위이며 CSV의 0.8은 80%입니다.

- **Accuracy**: 세 클래스 전체 정답률. 파싱 불가능한 출력도 오답으로 포함합니다.
- **Macro-F1**: 세 클래스 F1의 단순 평균. invalid 출력은 해당 정답 클래스의 FN입니다. 클래스별 precision/recall/F1과 confusion matrix도 저장합니다.
- **Decision flip rate**: Original과 압축 조건의 출력 라벨이 다른 비율. 양쪽 모두 유효 라벨인 동일 문항만 분모에 넣고 `valid_pair_n`을 함께 보고합니다.
- **Decision agreement rate**: 같은 분모에서 Original과 라벨이 같은 비율. Original의 오답을 그대로 유지해도 일치로 세므로 정확도와 구분합니다.
- **Correct decision retention**: Original이 맞힌 전체 문항 중 압축 조건도 맞힌 비율. 압축 결과의 invalid도 실패로 셉니다. 분모가 0이면 null입니다.
- **Actual token retention**: 압축문/원문 토큰 수. 문항별 비율 평균과 전체 토큰 합의 비율을 모두 제공합니다.
- **Total inference seconds**: 해당 조건의 실제 압축 호출 시간 합 + 판단 시간 합. Naive·Contextualized 문서 요약은 한 번만 계산하므로 17번 중복 합산하지 않습니다. LongLLMLingua는 가설마다 계산합니다. 문항당 상각 시간도 제공합니다.

시간에는 호출 내부 토큰화와 추론을 포함하며 모델 다운로드·로딩, 체크포인트 I/O는 제외합니다. 선택한 GPU 한 대에서 모든 조건을 실행하고 역할별 장치를 기록합니다. 같은 명령 재실행은 저장된 시간을 재사용하며 새 속도 측정이 아닙니다. 현재 스크립트는 별도의 warmup/반복 측정/통계적 유의성 검정을 하지 않습니다.

Gold evidence는 후속 분석에 사용할 수 있도록 저장합니다. **단어가 겹친다는 이유로 의미적 근거 보존으로 간주하는 점수는 만들지 않습니다.** NotMentioned에도 분류·flip·판단 보존 지표는 동일하게 적용됩니다.

## 검증

```bash
# 모델 의존성 없이 데이터·평가·중단/재개·단계 인계 검증
python -m unittest discover -s tests -v
```

모델 의존성이 설치되어 있으면 작은 랜덤 Qwen3/Llama로 실제 생성·LongLLMLingua 압축·캐시 변환·입력 길이 사전 검증도 실행합니다. 의존성이 없으면 해당 통합 테스트는 skipped로 표시됩니다. 작은 모델의 점수는 연구 성능을 의미하지 않습니다.

이 테스트의 합성 모델 출력은 테스트 전용입니다. CLI는 가짜 모델로 실험 지표를 생성하는 옵션을 제공하지 않습니다. 실제 Qwen/Llama 가중치와 4bit GPU 경로는 위 smoke run으로 별도 확인하세요.

## 출처

- Koreeda & Manning, [ContractNLI](https://aclanthology.org/2021.findings-emnlp.164/) (2021), [공식 데이터 저장소](https://github.com/stanfordnlp/contract-nli).
- Jiang et al., [LongLLMLingua](https://aclanthology.org/2024.acl-long.91/) (2024), [공식 구현](https://github.com/microsoft/LLMLingua).
- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B).

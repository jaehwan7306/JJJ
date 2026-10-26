# ContractNLI 문맥 압축 메인 실험

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

Python 3.11 또는 3.12와 NVIDIA 드라이버가 있는 환경에서 저장소 루트로 이동하세요. 아래 활성화 명령은 Linux/WSL 기준입니다. Windows PowerShell에서는 `\.venv\Scripts\Activate.ps1`로 활성화한 뒤 같은 `python -m ...` 명령을 사용할 수 있습니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# RTX 30/40 시리즈용 예시. 드라이버 호환성은 PyTorch 공식 설치 안내를 확인하세요.
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e '.[inference,quantization,longllmlingua]'
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_properties(0))"
```

두 PC 모두 같은 설치 명령과 소스 버전을 사용하세요. `nvidia-smi`로 실제 GPU 모델·VRAM을 확인하세요. 12GB GPU에 맞춰 4bit를 기본값으로 정했지만 모든 긴 입력의 적재를 보장하지는 않습니다. OOM이 발생하면 먼저 다른 GPU 프로세스를 종료하고 해당 단계를 재실행하세요. 길이 제한을 낮추거나 원문을 자르면 Original 조건의 의미가 바뀌므로 자동으로 수행하지 않습니다.

모델은 Hugging Face에서 최초 다운로드됩니다. 공개 모델 기준으로 API 키는 필수 설정이 아닙니다. 자체 서버/API를 호출하는 코드는 없습니다. 모델 가중치와 캐시를 위한 디스크 공간을 확보하세요.

## 데이터 준비와 첫 실행

```bash
python -m contractnli download
python -m contractnli run --config configs/base.toml --dry-run
# 가중치를 로딩하기 전 토크나이저만으로 전체 입력 길이 확인
python -m contractnli run --config configs/base.toml --preflight
# 먼저 2문서 smoke run: 본 실험과 출력 디렉터리를 구분
python -m contractnli run --config configs/base.toml --max-documents 2 --output-dir outputs/smoke
# 전체 123문서, 2,091문제 × 4조건
python -m contractnli run --config configs/base.toml
```

다운로더는 공식 저장소의 고정 commit ZIP을 받고 SHA256을 검증합니다. JSON과 LICENSE/TERMS만 추출하고 PDF는 추출하지 않습니다. 오프라인에서는 `download --archive /path/to/contract-nli.zip`을 사용할 수 있으며 동일한 체크섬을 검증합니다. 공식 ContractNLI 데이터는 CC BY 4.0이며 원 저자 표시와 데이터 이용 조건을 확인하세요.

전체 실험은 문서 요약 246회, 질문별 압축 2,091회, 판단 8,364회입니다. `--dry-run`은 실제 모델 추론이나 정확도 산출을 하지 않습니다. `--preflight`는 토큰 길이만 확인하며 GPU VRAM 검증은 smoke run이 담당합니다.

## 두 PC에서 단계 나누기

아래 예시는 **동시 실행이 아닌 순차 인계**입니다. 모델 세 개를 같은 GPU에 함께 올리지 않습니다.

```bash
# PC A: 문서별 요약
python -m contractnli run --stage summarize
# PC A 또는 PC B: LongLLMLingua 압축
python -m contractnli run --stage compress
# 그 뒤 PC B: 동일한 네 문맥을 동일 판단 모델로 평가
python -m contractnli run --stage judge
python -m contractnli run --stage report
```

각 단계 완료 후 저장소의 **같은 소스·설정·데이터와 `outputs/base/` 전체**를 다음 PC로 복사하세요. `contexts.jsonl`, `predictions.jsonl`, `run.json`을 함께 옮겨야 합니다. 실행 중에는 복사하지 마세요. 두 PC가 같은 출력 디렉터리에 동시에 기록하거나 결과를 단순 연결하면 안 됩니다. 입력/출력 절대 경로만 달라지는 것은 허용합니다.

다시 같은 명령을 실행하면 완료된 항목을 재사용합니다. 데이터 내용·설정·소스가 달라지면 이전 결과 재사용을 거부합니다. 각 역할별 실제 모델 revision, 패키지 버전, 장치·양자화 설정을 저장하고 변경을 감지합니다. **한 역할을 도중에 다른 GPU/런타임으로 옮겨 이어 붙이는 것은 허용하지 않습니다.** 서로 다른 역할을 PC별로 나누는 것은 가능합니다. 재현 실험에는 `revision = "main"` 대신 Hugging Face의 고정 commit SHA를 설정하는 편이 좋습니다. 최초 실행의 실제 revision은 `run.json`에서 확인할 수 있습니다.

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

`outputs/base/`:

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

시간에는 호출 내부 토큰화와 추론을 포함하며 모델 다운로드·로딩, 체크포인트 I/O는 제외합니다. 별도 GPU의 속도 차이가 처리시간 비교에 영향을 줄 수 있으므로 역할별 장치를 함께 보고하세요. 같은 명령 재실행은 저장된 시간을 재사용하며 새 속도 측정이 아닙니다. 현재 스크립트는 별도의 warmup/반복 측정/통계적 유의성 검정을 하지 않습니다.

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

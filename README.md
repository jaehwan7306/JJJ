# ContractNLI 압축 실험

완료된 RTX 5070 v2 실험은 **[결과 보관본](results/rtx5070_native_v2_2026-10-29/README.md)**에서 확인할 수 있습니다.
본 실험·dev 검사 결과, 실행 출력이 포함된 노트북, 고정 설정과 무결성 검증 기록을 함께 보관합니다.

현재 실행 파일은 **[v2 PyCharm 노트북](notebooks/run_native_v2_rtx5070_pycharm.ipynb)**입니다.
Windows · Python **3.11.9** · RTX 5070 약 12GB · 단일 GPU 기준입니다.

v1의 자유 생성/엄격 파싱 조합은 정답 라벨 뒤의 설명과 Markdown까지 오답으로 집계했습니다.
v2는 같은 Qwen3-8B가 세 라벨 중 하나만 생성하도록 제한하고 성능·형식·시간을 함께 보고합니다. 추가 학습은 하지 않습니다.

|조건|구성|길이 설정|호출 단위|
|---|---|---|---|
|Original|계약 원문|원문 보존, 자동 truncation 없음|문서|
|Naive|Qwen2.5-7B-Instruct 일반 요약|길이 목표 없음, 최대 2,048 생성 토큰|문서당 1회|
|Contextualized|동일 모델, 조건·예외·부정·의무 주체 보존 지시|동일 상한|문서당 1회|
|LongLLMLingua|기본 scorer `NousResearch/Llama-2-7b-hf`와 공식 LongLLMLingua 예제 설정|`rate=0.55`, `target_token` 미지정|문서–가설마다 1회|

**이 메인은 실제 길이가 다른 방법별 기본 동작 비교입니다. 동일 길이 실험은 별도 분석입니다.**
2,048은 사용자가 선택한 요약 상한이며 Qwen의 공식 기본 요약 길이라는 뜻은 아닙니다.
`rate=0.55`는 [공식 README의 LongLLMLingua 예제](https://github.com/microsoft/LLMLingua#-quick-start) 값입니다.
LLMLingua 0.2.2 함수 선언의 `rate` 기본값은 `0.5`이며 두 값을 구분합니다.
4bit·greedy·배치 1·thinking off는 이 실험의 고정 실행 설정입니다.

## 기존 결과 보존

Git이 없다면 [main ZIP](https://github.com/jaehwan7306/JJJ/archive/refs/heads/main.zip)을 받아
**기존 JJJ-main과 다른 새 폴더**에 풉니다. 새 가상환경·Jupyter 커널을 선택하세요.
같은 Windows 계정의 Hugging Face 모델 캐시는 재사용됩니다. 기존 `outputs`는 삭제·이동하지 않습니다.

소스·판단 방식·길이 설정이 바뀌었으므로 v1 체크포인트를 v2 추론으로 이어 쓰지 않습니다.
v2 전체 결과는 `outputs/rtx5070_native_v2`, dev 검사는 `outputs/rtx5070_native_v2_dev_smoke`에 저장합니다.
v1 결과의 읽기 전용 분석은 아래 `audit` 명령으로 가능합니다.

## 설치와 실행

PyCharm 프로젝트와 Jupyter 커널을 **같은 Python 3.11.9 가상환경**으로 설정합니다.
서버 패키지가 필요하면 PyCharm Terminal에서 `python -m pip install jupyter ipykernel`을 실행합니다.
노트북 설치 셀은 각각 실행하세요.

```python
%pip install --upgrade pip
%pip install --upgrade torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
%pip install -e ".[inference,quantization,longllmlingua,rtx5070,analysis]" ipywidgets
```

설치 후 커널을 재시작합니다. 버튼이 없으면 코드 셀 클릭 → **Ctrl+Shift+A → Restart Kernel**을 검색합니다.
1번 환경 셀을 다시 실행하고 설치 셀을 건너뛰어 GPU 검사로 이동합니다.
CUDA 텐서와 bitsandbytes NF4 실제 연산이 모두 PASS여야 진행합니다.

노트북 순서: 환경 → 설치/재시작 → GPU → 모델 revision 고정 → 데이터 → 길이 검사 → dev smoke → 전체 실행 → 결과표.
**7번 dev smoke를 중지하지 말고** `Smoke run completed: True`, Invalid=0을 확인한 뒤
8번 `RUN_FULL_EXPERIMENT`를 True로 바꿉니다. 기본값은 False입니다.
모델은 요약 → 압축 → 판단 순서로 하나씩 적재·해제합니다. 최초 다운로드는 오래 걸릴 수 있습니다.

터미널 실행 예입니다. 셸 명령을 Python 셀에 그대로 입력하지 마세요.

```powershell
python -m contractnli download
python -m contractnli freeze --config configs/rtx5070_v2.toml --destination configs/rtx5070_v2.frozen.json
python -m contractnli run --config configs/rtx5070_v2.frozen.json --preflight
python -m contractnli run --config configs/rtx5070_v2.frozen.json
```

전체 실행 전 dev smoke로 실제 GPU 적재·라벨 출력을 확인하세요.
`freeze`는 가중치 없이 모델 메타데이터만 읽어 세 모델의 immutable HF commit을 고정합니다.
기존 로컬 메타데이터를 우선 사용하며 frozen 파일은 이후에도 유지합니다.
다른 설정으로 기존 frozen 파일을 덮어쓰지 않습니다. 실제 revision·런타임은 `run.json`에 기록합니다.

## 평가와 시간

공식 test **123문서 × 17가설 = 2,091문항 × 4조건**을 평가합니다.
Entailment 968, Contradiction 220, NotMentioned 903개를 전부 유지합니다.
NotMentioned의 부재한 gold evidence를 근거 보존 0점으로 취급하지 않습니다.

- 같은 판단 프롬프트와 라벨 토큰 trie를 사용합니다. 완전한 세 라벨과 EOS만 허용하며 위반 시 즉시 실패합니다.
- 판단 문항은 seed=42의 고정 해시 순서로 섞습니다. 재개에도 같은 순서를 유지합니다.
- 모델 적재마다 고정 합성 예제 1회 워밍업을 합니다. 점수와 처리시간 집계에서는 제외합니다.
- Accuracy, Macro-F1, 클래스별 precision/recall/F1, 혼동행렬, Invalid, 원문 정답 유지율과 정답↔오답 전환을 보고합니다.
- 문맥은 공통 Qwen3 토크나이저로 측정합니다. 평균·중앙값·p10/p90·전체 토큰 합 기준 보존율을 기록합니다.
- 요약/압축·판단·합계 시간, 문항당 시간, 판단 median/p95, 판단 입력/출력 토큰을 기록합니다.
- 모델 준비(다운로드 포함 가능), 워밍업, 체크포인트 I/O는 추론시간에서 제외하고 실행 경과시간을 별도로 기록합니다.
- 단일 실행의 운영 지연시간입니다. 재개는 저장된 시간을 재사용하며 새 반복 측정 벤치마크가 아닙니다.

`metrics.csv`는 시간 열을 포함하는 Excel용 UTF-8 BOM 파일입니다. 노트북 9번은 성능표와 시간표를 함께 표시합니다.
생성 상한에 도달한 요약 수를 보고하며 메인 실험에서 자동 연장·절단·길이 맞춤 재생성은 하지 않습니다.

|출력|내용|
|---|---|
|`run.json`|프로토콜·fingerprint·모델 commit·GPU/런타임·준비/워밍업·실행/실패 이력|
|`contexts.jsonl`|요약/압축문·원문 ID·토큰·시간·길이 설정·종료 사유|
|`predictions.jsonl`|정답·예측·원출력·문맥 key·토큰/시간·gold evidence|
|`metrics.json`, `metrics.csv`|모든 예정 판단이 끝난 뒤 생성하는 집계|

## 기존 결과의 읽기 전용 분석

노트북 10번이나 아래 명령으로 실행합니다. 입력 파일은 기존 프로젝트의 절대 경로도 가능합니다.

```powershell
python -m contractnli audit --predictions outputs/rtx5070_base/predictions.jsonl --metrics outputs/rtx5070_base/metrics.json --output-dir outputs/legacy_audit
```

원본을 바꾸지 않고 `audit.json`, `scores.csv`를 별도 폴더에 저장합니다.
기존 Accuracy·Macro-F1·혼동행렬 재현 검사, strict/첫 줄 파싱 보조 평가, 클래스별·가설별 성능,
정답 유지율, 형식 오류 원출력, 문서 단위 paired bootstrap 20,000회를 포함합니다.
bootstrap은 문서의 모든 가설을 함께 뽑으며 seed=20261027의 탐색적 95% 구간입니다.

실제 보존율 차이가 2%p 이내인 동일 문항의 부분집합 진단도 제공합니다.
선택된 표본의 크기·클래스 분포를 함께 보고하며 **전체 동일 길이 실험으로 해석하지 않습니다.**
동일 토큰량의 비용·시간은 추가 분석으로 분리합니다. 원문·요약을 임의로 잘라 토큰만 맞추지 않습니다.
이미 test 결과를 확인했으므로 이후 수정은 탐색적이며 추가 튜닝은 공식 dev에서 진행합니다.

## 중단·검증

일반 Interrupt 후에는 같은 config·폴더에서 재실행합니다. 완료된 행은 재생성하지 않습니다.
미완성 JSONL·중복 key·잘못된 정답·파싱 불일치·문맥 연결 오류를 조용히 무시하지 않습니다.
강제 종료로 `.run.lock`이 남으면 프로세스 종료를 확인한 뒤 잠금 파일만 제거합니다.
LongLLMLingua prefix 호환 처리는 v2에 통합되어 외부 패치 셀이 필요하지 않습니다.

```powershell
python -m unittest discover -s tests -v
```

합성 backend와 작은 랜덤 Qwen3/Llama의 실제 추론 테스트를 구분합니다. 선택 의존성이 없으면 관련 검사는 skip됩니다.
랜덤 모델 점수는 연구 결과가 아닙니다. 실제 4bit GPU 경로는 사용자 PC의 dev smoke에서 확인해야 합니다.

[프로토콜·수정 내역](docs/EXPERIMENT_PROTOCOL_V2.md) · [이전 v1 안내 보관본](docs/LEGACY_V1.md)

## 출처

- [ContractNLI 논문](https://aclanthology.org/2021.findings-emnlp.164/), [데이터](https://github.com/stanfordnlp/contract-nli): CC BY 4.0, 원 저자 표시와 데이터 이용 조건 유지.
- [LongLLMLingua 논문](https://aclanthology.org/2024.acl-long.91/), [공식 구현·예제](https://github.com/microsoft/LLMLingua).
- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B).

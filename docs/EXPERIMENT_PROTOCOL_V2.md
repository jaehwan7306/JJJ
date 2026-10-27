# v2 실행·평가 프로토콜

## 연구 질문과 해석 범위

메인은 방법별 기본 동작의 정확도·판단 유지·비용을 관찰한다. 실제 압축률이 다르므로 같은 길이에서
방법 자체의 우열을 입증하는 실험이 아니다. 동일 토큰 비용/시간 비교는 추가 분석으로 분리한다.
이미 test 결과를 확인했으므로 v2를 사전 등록된 미관찰 test 검증이라고 주장하지 않는다.
추가 튜닝·학습·예산 조정은 공식 train/dev에서만 수행하고 test에는 적용할 규칙을 고정한다.

## 고정 설정

|항목|설정|
|---|---|
|데이터|공식 ContractNLI test, 123문서, 2,091문항, 세 클래스 유지|
|실행 검사|공식 dev 한 문서, test 결과와 분리|
|요약|Qwen2.5-7B-Instruct, Naive/Contextualized 모두 길이 목표 없음, 최대 2,048 native 생성 토큰|
|압축|NousResearch/Llama-2-7b-hf, LongLLMLingua 공식 README 예제, rate=0.55, target_token 미지정|
|판단|Qwen3-8B, thinking off, greedy, 완전한 세 라벨과 EOS만 허용|
|GPU|RTX 5070 한 대, NF4 4bit, float16, SDPA, 배치 1, 모델 순차 적재|
|revision|세 모델 모두 실제 immutable HF SHA로 고정; 소스·데이터 fingerprint 포함|
|순서|판단 key와 seed=42를 해시해 조건을 섞음; 재개에도 순서 유지|
|워밍업|모델 적재마다 고정 합성 예제 1회, 추론 지표에서 제외|

LongLLMLingua 예제 설정은 `condition_in_question="after_condition"`, `reorder_context="sort"`,
`dynamic_context_compression_ratio=0.3`, `condition_compare=True`, `context_budget="+100"`,
`rank_method="longllmlingua"`이다. 조건·토큰 필터는 켜고 문장 필터는 끈다.
공식 예제의 `rate=0.55`와 함수 시그니처의 기본 `rate=0.5`를 구분한다.
`concate_question=False`는 판단 모델에 가설을 별도 전달하므로 중복 삽입을 피하기 위한 고정 선택이다.
계약서 청킹(최대 512 native 토큰), 4bit, 캐시·prefix 호환 어댑터는 이 실행 환경의 명시적 구현 선택이다.
모든 인자를 생략한 API 기본 호출은 일반 LLMLingua이므로 사용하지 않는다.
rate는 scorer의 질문·청크 토큰을 포함하는 공식 내부 계산식에 적용되므로 Qwen3 문맥 토큰이 정확히 55%가 되는 것은 아니다.

Naive/Contextualized에는 가설·정답·gold evidence를 넣지 않는다. 길이 지시만 제거하고 기존 내용 지시는 유지한다.
2,048은 공식 모델 기본값이 아니라 사용자가 선택한 안전 상한이다. 자동 EOS를 억제해 길이를 채우거나
본문을 자르거나 정답 점수로 생성 결과를 선택하지 않는다. 상한 도달 수는 보고한다.

## 확인된 문제와 조치

|문제|v2 조치|
|---|---|
|라벨 뒤 설명/Markdown이 형식 오류로 오답 처리|세 라벨 토큰 trie와 EOS만 허용. 형식 위반 시 저장 전 실패|
|모델에 포함된 temperature/top_p/top_k로 greedy 경고 발생|새 greedy GenerationConfig 사용, 모델의 sampling/forced-token 설정을 상속하지 않음|
|LLMLingua prefix decode 왕복 assert 실패|기존 경계 계산 우선, 실패한 경우 검증된 fast-tokenizer 원문 offsets 사용. 경계를 넘는 토큰은 추정하지 않음|
|압축 비율·실제 길이·알고리즘 내부 비율 혼동|rate/target_token 모드, 요약 상한, 공통 tokenizer의 실제 길이를 별도 기록|
|성능표에 시간 누락, CSV에 총시간만 포함|성능/시간 두 표, 압축·판단·합계·문항당·median/p95 CSV 열|
|특정 조건이 항상 먼저 판단됨|고정 해시 순서로 조건을 섞음|
|초기 실행 비용과 지연시간 혼동|고정 워밍업 제외, 모델 준비 시간과 실행 전체 wall time 별도 기록|
|중단/재개·다른 역할 간 GPU/환경 혼합|실행 이력, 실패 상태, 같은 host/GPU/패키지 환경 검사|
|체크포인트 내용 손상 가능성|문항·정답·가설·원출력 파싱·문맥 연결 검사; 불일치 시 실패|
|Jupyter에서 소스 수정 후 옛 클래스 정의가 남음|import 시점 소스와 실행 시점 소스를 비교하고 커널 재시작 요구|
|v1/v2 점수 혼합 위험|새 출력 폴더, 변경된 source signature로 기존 추론 결과 재사용 거부|
|보조 파싱으로 기존 결과 덮어쓰기 위험|독립 read-only audit, strict와 first-line 결과를 함께 저장|

## 지표

일차 분류 지표는 세 클래스 Macro-F1과 Accuracy를 함께 보고한다. 클래스별 precision/recall/F1과
혼동행렬, Invalid/출력 상한 도달 수를 반드시 함께 확인한다. 출력 제한은 정답률을 보장하지 않는다.
Constrained greedy는 자유 생성과 다른 디코딩 정책이며, 라벨 토큰화 편향이 완전히 사라진다고 주장하지 않는다.

판단 유지 지표는 원문 정답 유지율, 원문 정답→오답, 원문 오답→정답이다.
기존 decision flip/agreement는 양쪽 유효 라벨만 사용하므로 valid_pair_n과 coverage를 함께 보고한다.
모든 쌍의 출력 변화율도 별도로 제공하며 두 지표의 분모를 혼동하지 않는다.

시간은 호출 내부 토큰화·추론 합이다. Naive/Contextualized 요약은 문서당 한 번만 합산한다.
LongLLMLingua는 가설별 압축 비용을 합산한다. 모델 준비(다운로드 포함 가능), 워밍업, checkpoint I/O는
제외한다. 실행 이력 wall time에는 이런 비용이 포함될 수 있다. 저장된 시간을 재사용하는 재개는 새 측정이 아니다.
median/p95는 서로 다른 문항의 운영 지연 분포로 동일 입력 반복 측정의 오차 범위가 아니다.

보존율은 문맥만 Qwen3 tokenizer로 계산한다. 원문 전체를 자르지 않는다. 평균 비율과 전체 합 비율은
가중치가 달라 둘 다 보고한다. NotMentioned에 gold evidence가 없다는 사실은 클래스를 제외할 근거가 아니다.
단어 중복을 의미적 gold-evidence 보존 점수로 사용하지 않는다.

## 통계와 추가 분석

`audit`은 문서 단위 paired bootstrap 20,000회, seed=20261027, percentile 95% 구간을 사용한다.
같은 문서의 모든 가설을 함께 뽑고 모든 조건에 동일한 문서 표본을 적용한다. 세 클래스 F1에서 분모가 0이면 0이다.
다중비교 보정 없는 탐색적 구간이며 모델 seed, 프롬프트, GPU, 다른 데이터셋의 변동을 포괄하지 않는다.
구간에 0이 포함된다는 이유로 동등성/비열등성이 입증되는 것은 아니다.

첫 줄 파싱은 첫 줄 전체가 허용 라벨 또는 `**라벨**`인 경우만 인정한다. 설명 속 라벨 검색이나
gold를 이용한 선택은 금지한다. 기존 strict 점수는 보존하고 이 분석을 사후 보조 평가로 표시한다.

보존율 2%p 이내의 동일 문항을 고르는 길이 겹침 분석은 gold-blind 진단일 뿐이다.
표본 선택에 따라 문서·클래스 분포가 달라지므로 전체 동일 길이 실험이라고 해석하지 않는다.
추가 동일 토큰량 벤치마크에서는 고정 입력 토큰 수, 반복 횟수, 워밍업, 실행 순서, 압축 포함/제외 비용을
별도로 정해야 한다. 현재 메인은 이를 자동으로 대신하지 않는다.

## 검증 한계

CPU의 작은 랜덤 Qwen3/Llama로 실제 제약 생성, LLMLingua 호출, 캐시 연결을 검사한다.
실제 7B/8B 가중치와 NF4 GPU 경로, Windows 드라이버는 사용자 PC의 dev smoke로 확인한다.
TF32를 끄고 cuDNN benchmark를 끄며 deterministic 설정과 seed를 기록하지만 GPU 비트 단위 완전 재현을 보장하지 않는다.

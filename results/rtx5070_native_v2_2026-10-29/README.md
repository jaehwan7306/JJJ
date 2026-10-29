# RTX 5070 v2 실험 결과 보관본

사용자가 Windows Desktop의 `JJJ-main1`에서 실행하고 제공한 결과입니다.
2026-10-29는 GitHub 보관일이며, 실험 실행일을 의미하지 않습니다.
업로드한 결과·실행 노트북·고정 설정은 원본 바이트를 유지했습니다.
이 업로드 과정에서 모델 추론을 다시 실행하지 않았습니다.

## 파일

|경로|내용|
|---|---|
|[본 실험](outputs/rtx5070_native_v2/)|공식 Test 123문서, 조건별 2,091문항, 총 8,364개 예측|
|[dev smoke](outputs/rtx5070_native_v2_dev_smoke/)|Dev 1문서, 네 조건, 총 68개 예측|
|[Original dev 검증](outputs/rtx5070_v2_judge_dev/)|Dev 61문서, Original 1,037개 예측|
|[실행 노트북](notebooks/run_native_v2_rtx5070_pycharm.ipynb)|사용자 실행 출력과 추가 진단 셀 포함|
|[고정 설정](configs/rtx5070_v2.frozen.json)|본 실험 설정과 세 모델의 immutable revision|
|[검증 기록](verification.json)|체크포인트·소스 signature·집계 재계산 결과|
|[입력 파일 해시](uploaded_sources.json)|첨부 ZIP·노트북·설정의 SHA-256과 크기|
|[SHA256SUMS](SHA256SUMS)|보관 파일의 SHA-256 목록|

각 결과 폴더의 `metrics.json`과 `metrics.csv`는 집계, `predictions.jsonl`은 문항별 예측,
`contexts.jsonl`은 생성한 요약·압축문, `run.json`은 설정·모델·환경·실행 이력을 담습니다.
Original만 평가한 dev 검증에는 `contexts.jsonl`이 없습니다.

## 본 실험 요약

|조건|Accuracy|Macro-F1|전체 문맥 토큰 보존율|요약·압축+판단 시간|
|---|---:|---:|---:|---:|
|Original|66.24%|0.5270|100.00%|33.29분|
|Naive|55.52%|0.4272|12.88%|42.72분|
|Contextualized|64.42%|0.5488|41.64%|109.12분|
|LongLLMLingua|60.02%|0.4437|54.62%|273.54분|

세 정답 클래스와 모든 공식 Test 문항을 유지했습니다. 네 조건 모두 Invalid=0입니다.
요약은 길이 목표 없이 최대 2,048 생성 토큰, LongLLMLingua는 `rate=0.55`입니다.
조건별 실제 길이가 다르므로 동일 토큰 예산에서의 우열을 의미하지 않습니다.
시간은 저장된 단일 실행의 추론 시간 합계이며 모델 준비·워밍업·체크포인트 I/O는 제외합니다.

## 무결성 확인과 사용

소스 기준은 저장소 commit `0e93f90`입니다. 세 실행 모두 소스·설정·데이터 signature와
체크포인트의 문항 metadata·문맥 fingerprint를 확인했습니다.
원본 예측과 문맥으로 `metrics.json` 전체를 다시 계산해 일치함을 확인했습니다.
부동소수점 집계는 서로 다른 Python 런타임의 미세한 차이를 허용하도록 절대 오차 `1e-12`로 비교했습니다.
고정 설정은 본 실험 `run.json`의 설정과 일치합니다.

보관 폴더에서 다음 명령으로 파일 해시를 확인할 수 있습니다.

```bash
sha256sum -c SHA256SUMS
```

노트북은 실행 기록용이며 전체 실행 스위치가 켜져 있습니다.
새 실험은 저장소의 [실행용 노트북](../../notebooks/run_native_v2_rtx5070_pycharm.ipynb)을 사용하고
별도 출력 경로를 지정하세요. 고정 설정·실행 기록 안의 기존 상대 경로와 Windows 절대 경로는
실행 당시 값을 그대로 보관했습니다.

# AI 정제 외관 검사기 (AI Tablet Visual Inspector)

정제·캡슐 외관검사에서 범용 비전 AI(Claude)에게 **어디까지 맡겨도 되는지**를 측정한 사이드 프로젝트.
QC 관점에 맞춰 정확도보다 **불량 누락(유출)** 을 최우선 지표로 두고,
확신이 낮은 판정은 AI가 합격시키지 않고 **사람 재검사**로 넘기는 2단계 구조로 설계했다.

## 결과 (v2, MVTec AD pill + capsule 테스트셋 전수 299장)

| | pill (167) | capsule (132) | 전체 (299) |
|---|---|---|---|
| 정상/불량 판정 정확도 | 93.4% | 93.2% | **93.3%** |
| 불량 유형까지 맞춘 비율 | 64.1% | 75.0% | 68.9% |
| **불량 누락률** (불량→정상) | 7.1% (10/141) | 5.5% (6/109) | **6.4%** (16/250, 95% 상한 10.1%) |
| 정상 오탐율 (정상→불량) | 3.8% (1/26) | 13.0% (3/23) | 8.2% (4/49) |
| 평균 판정 시간 | | | 7.7초/장 (4건 동시 호출) |

**AI 단독으론 부족하다.** 불량 16개 중 1개꼴(6.4%)을 정상으로 넘긴다.
하지만 **누락 16건 전부 모델 신뢰도가 0.7 이하**였다. 그래서 신뢰도 기준 미만은 사람에게 넘기면:

| 재검사 기준 (신뢰도 <) | 사람 재검사 비율 | AI 단독 합격된 불량 (유출) | AI 단독 오탐 |
|---|---|---|---|
| 0.6 | 14.7% | 12/250 (4.8%) | 1 |
| **0.7 (기본값)** | **43.5%** | **3/250 (1.2%)** | 0 |
| 0.75 | 66.2% | 0/250 (0%, 95% 상한 1.5%) | 0 |

→ 유출을 0으로 만들 수는 있지만 대가는 **3장 중 2장을 사람이 다시 보는 것**. 이게 현재 범용 LLM 외관검사의 실력이다.
기준값은 이 테스트셋에서 고른 것이라 낙관적이다 — 실제 운용값은 별도 밸리데이션 셋으로 정하고 확인해야 한다.

약한 불량 유형: pill_type 2/9, contamination 6/21, capsule poke 8/21 (상세: [EVALUATION_REPORT.md](EVALUATION_REPORT.md))

### v1 결과 철회 (정답 누출)

v1(정확도 97.5%, 누락 0%)은 **측정 오류**였다. 프롬프트에 이미지 경로(`.../test/crack/003.png`)를 넣어 폴더명=정답이 모델에 보였다.
같은 79장을 누출 차단 후 재측정하니 **정확도 97.5% → 63.3%, 누락 0/69 → 8/69**.
→ 이미지를 무작위 파일명으로 복사해 전달하도록 수정하고, 재발 방지 테스트를 추가했다. v1 원본은 `results/v1_leaky_*.json`으로 보존.

## 설계 포인트 (GMP 관점)

- **Fail-safe**: 판정 실패·저신뢰도 → 합격이 아니라 `재검사`. 판정 실패도 분모에서 빼지 않고 집계
- **누락률 신뢰구간**: 0건이어도 "0%"가 아니라 95% 상한(Wilson)으로 보고
- **감사추적(ALCOA+ 흉내)**: 데모의 모든 판정을 `audit_trail.jsonl`에 append — 시각, 판정자, 모델, 프롬프트 버전(해시), 이미지 SHA-256, 원출력, 처분
- **누출 방지**: 모델과 브라우저 어디에도 라벨이 든 경로를 넘기지 않음

## 데이터

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) pill/capsule (CC BY-NC-SA 4.0, 비상업 연구용).
실사진 기반 산업용 벤치마크이며 **실제 제약회사 QC 데이터가 아님**.

```bash
pip install kagglehub flask
python -c "import kagglehub; kagglehub.dataset_download('ipythonx/mvtec-ad')"   # 약 5.2GB
```

## 실행

```bash
# Claude Code 구독 인증 필요 (claude auth login). ANTHROPIC_API_KEY는 자동 제외(과금 방지)
python benchmark.py --categories pill capsule --per-type 1000 --workers 4   # 전수
python benchmark.py --categories pill --per-type 2                           # 빠른 확인

python demo_app.py     # http://127.0.0.1:8931
python test_offline.py # 데이터·API 없이 로직 자체검증
```

데모: 랜덤 샘플 → AI 판정 → **합격 / 불합격 / 재검사(사람 확인)** 처분 + 정답 대조 + 감사추적 ID. 배치정보는 데모용 합성.

## 파일

- `benchmark.py` — 벤치마크 하네스 (Claude 구현, GPT/Gemini 스텁)
- `demo_app.py` — Flask 데모
- `test_offline.py` — 누출 방지·채점·데모 라우트 자체검증
- `results/v2_full.json` — v2 전수 결과 원본
- `results/v1_leaky_*.json` — 철회한 v1 결과 (누출 있음)
- `EVALUATION_REPORT.md` — 평가 리포트 / `INTERVIEW.md` — 면접 준비 노트 / `PROJECT_PLAN.md` — 기획서

## 한계와 다음 단계

- 범용 LLM 1종만 측정. 정상품만으로 학습하는 전통 이상탐지(PatchCore 등) 베이스라인 비교 필요
- 결함 위치(바운딩박스/히트맵) 미출력 — 일탈 조사·설명가능성에 필요
- 모델이 말하는 신뢰도는 보정(calibration)되지 않은 값. 0.7에 몰려 있음(65/299)
- 7.7초/장은 라인 전수검사 불가. 현실적 용도는 샘플링 검사 보조·일탈 시 2차 판독·검사원 교육
- 실제 GMP 적용엔 CSV/CSA 밸리데이션, 사람 검사원 대비 동등성 입증, Part 11/Annex 11 수준 통제 필요

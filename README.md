# AI 정제 외관 검사기 (AI Tablet Visual Inspector)

제약 QC 라인에서 육안으로 하던 정제 외관 검사를 AI 비전 모델이 대체할 수 있는지 검증하는 해커톤형 프로젝트.

## 결과 (1차, Claude)

| 카테고리 | 샘플 | 정확도 | 불량 누락률 | 정상 오탐율 |
|---|---|---|---|---|
| capsule | 12 | 100% | 0% | 0% |
| pill | 79 | 97.5% | **0%** | 0% |

불량품을 정상으로 판정한 경우가 단 한 건도 없음 — QC에서 가장 치명적인 지표(불량 유출)에서 만점.

## 데이터

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) pill/capsule (CC BY-NC-SA 4.0).
실사진 기반 산업용 연구 벤치마크 — 실제 제약회사 QC 데이터가 아님을 명시.

```bash
pip install kagglehub
python -c "import kagglehub; kagglehub.dataset_download('ipythonx/mvtec-ad')"
```

## 벤치마크 실행

```bash
# Claude Code 구독 인증 필요 (claude auth login)
python benchmark.py --models claude --categories pill capsule --per-type 10
```

- `--models`: claude (gpt/gemini는 계정 인증 후 구현 예정)
- `--per-type`: 불량 유형당 샘플 수

## 데모 웹앱

```bash
pip install flask
python demo_app.py  # localhost:8931
```

랜덤 정제 샘플 출제 → AI 검사 → 합격/불합격 판정 + GMP 배치정보(합성) 표시.

## 파일

- `benchmark.py` — 벤치마크 하네스
- `demo_app.py` — Flask 데모
- `EVALUATION_REPORT.md` — 심사위원 모드 평가 리포트 (80/100)
- `PROJECT_PLAN.md` — 기획서
- `benchmark_pill.json`, `smoke_capsule.json` — 결과 원본

# AI 정제 외관 검사기 — Project Plan

## 개요
AX 해커톤 스타일 사이드프로젝트. 제약 QC의 육안 외관 검사를 AI가 대체하는 데모 +
3개 모델(Claude / GPT / Gemini) 비전 벤치마크.

## 데이터
- MVTec AD `pill` + `capsule` (CC BY-NC-SA 4.0, 연구·데모용)
- pill: train good 267장 / test 167장(정상 26 + 불량 141), 불량 7종
  (color, contamination, crack, faulty_imprint, pill_type, scratch, combined)
- capsule: crack, faulty_imprint, poke, scratch, squeeze
- 라벨 = test/ 하위 디렉토리명 (정답지)

## 벤치마크 설계
- 태스크: 테스트 이미지 1장 → {정상/불량} + 불량 유형 분류
- 지표: 정확도(accuracy), 불량 유형별 F1, 정상 오탐율(false positive rate)
  - QC 관점에서는 불량 누락(false negative)이 더 치명적 → 별도 집계
- 샘플링: 전체 테스트셋이 크면 층화 샘플링 (유형별 균등, 예: 유형당 최대 10장)
- 프롬프트: 동일 프롬프트를 3모델에 그대로 사용 (공정성)

## 데모 웹앱
- 업로드한 정제 이미지 → 3모델 판정 비교 + 신뢰도 표시
- GMP 맥락 메타데이터 합성 표시 (배치번호, 규격, 판정 로그)
- 스택: 미정 (Streamlit 또는 단일 HTML+JS — 가볍게)

## 평가 루브릭 (내 채점 기준)
1. QC 도메인 정확성 (벤치마크 점수)
2. 조사 워크플로우 정합성 (불량 발견 → 분류 → 조치 제안 흐름)
3. 실무 유용성 (오탐/누락 트레이드오프 설명)
4. 데모 완성도

## 주의
- MVTec 라이선스: 비상업 연구용. 데모·평가 OK, 상용 배포 금지.
- 구독 계정 주간 사용량 제한 → 스프린트형 실행.

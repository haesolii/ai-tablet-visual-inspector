"""v3 분석: results/*.json → PROJECT_PLAN.md §4 합격 기준 A1~A5 판정표 (markdown).
    python analyze.py > results/v3_analysis.md
"""
import json
import random
from pathlib import Path

from benchmark import ERROR, PERTURBATIONS, REVIEW_THRESHOLD, wilson_upper

R = Path(__file__).parent / "results"


def load(name: str) -> dict:
    rs = json.loads((R / f"{name}.json").read_text(encoding="utf-8"))["claude"]["results"]
    return {r["image"]: r for r in rs}


def says_good(r):   # AI 결론이 '정상'인가 (판정 실패는 정상이 아님 → 재검사)
    return r["pred_label"] == "good"


def is_defect(r):
    return r["true_label"] != "good"


def leaked(r):      # AI 단독 합격 처리된 불량 = 유출
    return is_defect(r) and says_good(r) and (r.get("confidence") or 0) >= REVIEW_THRESHOLD


def flip_rate(a: dict, b: dict) -> float:
    ks = a.keys() & b.keys()
    return sum(says_good(a[k]) != says_good(b[k]) for k in ks) / len(ks)


def pct(x):
    return f"{x * 100:.1f}%"


def verdict(ok):
    return "✅ 적합" if ok else "❌ 부적합"


def main():
    v2 = load("v2_full")
    rep1, rep2 = load("v3_rep1"), load("v3_rep2")
    sub = rep1.keys() & rep2.keys()
    base = {k: v2[k] for k in sub}
    out = []

    # A1 특이성: 정상 이미지 (v2 test 49 + train 486)
    goods = [r for r in v2.values() if not is_defect(r)] + list(load("v3_train_good").values())
    fa = sum(r["pred_label"] not in ("good", ERROR) for r in goods)
    err = sum(r["pred_label"] == ERROR for r in goods)
    a1 = fa / len(goods) <= 0.10 and wilson_upper(fa, len(goods)) <= 0.15
    out += ["## A1 특이성 (정상 오탐)", "",
            f"정상 {len(goods)}장 중 불량 판정 {fa}건 = **{pct(fa / len(goods))}** "
            f"(95% 상한 {pct(wilson_upper(fa, len(goods)))}), 판정 실패 {err}건 → 기준 ≤10%·상한 ≤15%: **{verdict(a1)}**", ""]

    # A2 반복성: 같은 70장을 3회 (v2 + rep1 + rep2)
    runs = [base, rep1, rep2]
    same = sum(len({says_good(run[k]) for run in runs}) == 1 for k in sub)
    pair_flips = [flip_rate(base, rep1), flip_rate(base, rep2), flip_rate(rep1, rep2)]
    noise_floor = sum(pair_flips) / 3
    same_type = sum(len({run[k]["pred_label"] for run in runs}) == 1 for k in sub)
    a2 = same / len(sub) >= 0.90
    out += ["## A2 반복성 (같은 이미지 3회 판정)", "",
            f"{len(sub)}장 중 정상/불량 결론 3회 일치 {same}장 = **{pct(same / len(sub))}** → 기준 ≥90%: **{verdict(a2)}**  ",
            f"불량 유형까지 3회 일치: {pct(same_type / len(sub))} / 회차 간 평균 결론 변경률(기준선): {pct(noise_floor)}", ""]

    # A3·A4 강건성
    out += ["## A3·A4 강건성 (열화 조건, 같은 70장)", "",
            "| 조건 | 결론 변경률 (깨끗한 이미지 대비) | 누락 | 오탐 | 재검사 비율 | AI 단독 유출 | A3 | A4 |",
            "|---|---|---|---|---|---|---|---|"]
    a3 = a4 = True
    n_def = sum(is_defect(r) for r in base.values())
    n_good = len(base) - n_def
    for name, rs in [("깨끗함 (v2)", base)] + [(p, load(f"v3_{p}")) for p in PERTURBATIONS]:
        miss = sum(is_defect(r) and says_good(r) for r in rs.values())
        fal = sum(not is_defect(r) and r["pred_label"] not in ("good", ERROR) for r in rs.values())
        rev = sum(r["pred_label"] == ERROR or (r.get("confidence") or 0) < REVIEW_THRESHOLD for r in rs.values())
        leak = sum(leaked(r) for r in rs.values())
        if rs is base:
            out.append(f"| {name} | — | {miss}/{n_def} | {fal}/{n_good} | {pct(rev / len(rs))} | {leak} | | |")
            continue
        fr = flip_rate(base, rs)
        ok3, ok4 = leak == 0, fr <= noise_floor + 0.05
        a3, a4 = a3 and ok3, a4 and ok4
        out.append(f"| {name} | {pct(fr)} | {miss}/{n_def} | {fal}/{n_good} | {pct(rev / len(rs))} | {leak} | "
                   f"{'✅' if ok3 else '❌'} | {'✅' if ok4 else '❌'} |")
    out += ["", f"A3 (모든 조건 유출 0건): **{verdict(a3)}** / "
            f"A4 (변경률 ≤ 기준선 {pct(noise_floor)} + 5%p): **{verdict(a4)}**", ""]

    # A5 기준값 0.7을 무작위 절반(처음 보는 데이터)에서 확인 — 1,000회
    rng = random.Random(0)
    allr = list(v2.values())
    leaks, reviews = [], []
    for _ in range(1000):
        rng.shuffle(allr)
        hold = allr[len(allr) // 2:]
        leaks.append(sum(leaked(r) for r in hold) / sum(is_defect(r) for r in hold))
        reviews.append(sum((r.get("confidence") or 0) < REVIEW_THRESHOLD for r in hold) / len(hold))
    leaks.sort()
    a5 = sum(leaks) / len(leaks) <= 0.02
    out += ["## A5 운용 기준값 0.7 (무작위 절반 1,000회, 사후 분석)", "",
            f"홀드아웃 AI 단독 유출률 평균 **{pct(sum(leaks) / len(leaks))}** "
            f"(95백분위 {pct(leaks[949])}), 재검사 비율 평균 {pct(sum(reviews) / len(reviews))} "
            f"→ 기준 평균 ≤2%: **{verdict(a5)}**", ""]

    summary = {"A1": a1, "A2": a2, "A3": a3, "A4": a4, "A5": a5}
    head = ["# v3 밸리데이션 결과", "",
            "| 기준 | " + " | ".join(summary) + " |", "|---|" + "---|" * len(summary),
            "| 판정 | " + " | ".join("✅" if v else "❌" for v in summary.values()) + " |", ""]
    print("\n".join(head + out))


if __name__ == "__main__":
    main()

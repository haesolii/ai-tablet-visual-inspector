"""밸리데이션 분석: results/*.json → 합격 기준 판정표(markdown) + 대시보드용 results/summary.json
    python analyze.py > results/analysis.md

v3 기준(A1~A5)은 PROJECT_PLAN §4, v4 기준(B1~B5)은 §7. 기준값은 프로토콜에 고정된 값을 쓰고,
운용 설정(benchmark.REVIEW_THRESHOLD)이 나중에 바뀌어도 과거 판정은 바뀌지 않게 한다.
"""
import json
import random
from pathlib import Path

from benchmark import ERROR, EXTERNAL, PERTURBATIONS, REVIEW_THRESHOLD, wilson_upper

R = Path(__file__).parent / "results"
V3_TH = 0.7    # v3 프로토콜의 재검사 기준 (신뢰도 < 0.7)
V4_TH = 0.75   # v4 프로토콜에서 검증하는 기준 (신뢰도 ≤ 0.7, 즉 < 0.75)
COND_LABEL = {"clean": "원본", "noise10": "백색잡음 σ10", "noise25": "백색잡음 σ25",
              "blur2": "블러", "dark60": "조명 60%"}


MAX_ERROR = 0.02  # 판정 실패가 이보다 많은 실험은 무효 (실패를 '누락 아님'으로 세면 지표가 좋아 보인다)
# 실험별 기대 표본 수와 조건. 다른 파일이 섞이거나 일부만 남은 파일을 조용히 채점하지 않기 위함
EXPECTED = {
    "v2_full": (299, {}), "v3_rep1": (70, {}), "v3_rep2": (70, {}), "v3_train_good": (486, {"splits": ["train"]}),
    **{f"v3_{p}": (70, {"perturb": p}) for p in PERTURBATIONS},
    "v4_e1_pharmavision": (640, {"dataset": "pharmavision"}), "v4_e1_pilldefect": (355, {"dataset": "pilldefect"}),
    "v4_e2_ref_test": (299, {"prompt_version": "93b7fafa"}),
    "v4_e2_ref_train": (199, {"prompt_version": "93b7fafa", "splits": ["train"]}),
}


def load(name: str) -> dict:
    run = json.loads((R / f"{name}.json").read_text(encoding="utf-8"))["claude"]
    n, meta = EXPECTED.get(name, (None, {}))
    assert n is None or len(run["results"]) == n, f"{name}: 표본 {len(run['results'])}장, 기대 {n}장"
    for k, v in meta.items():
        assert run["meta"].get(k) == v, f"{name}: meta {k}={run['meta'].get(k)!r}, 기대 {v!r}"
    return {r["image"]: r for r in run["results"]}


def error_rate(run: dict) -> float:
    return sum(r["pred_label"] == ERROR for r in run.values()) / len(run)


def valid(entry: dict, *runs: dict) -> dict:
    """실험(파일)마다 판정 실패율을 따로 본다. 하나라도 MAX_ERROR를 넘으면 기준 판정을 '무효(재시험 필요)'로 덮는다.
    여러 실험을 합쳐서 보면 실패가 많은 실험이 다른 실험에 희석된다 (독립 검토 지적)."""
    e = max(error_rate(run) for run in runs)
    if e > MAX_ERROR:
        entry |= {"ok": False, "invalid": True, "result": f"무효 — 판정 실패 {pct(e)} (재시험 필요)"}
    return entry


def says_good(r):   # AI 결론이 '정상'인가 (판정 실패는 정상이 아님 → 재검사)
    return r["pred_label"] == "good"


def is_defect(r):
    return r["true_label"] != "good"


def conf(r):
    return r.get("confidence") or 0


def review(r, th):
    return r["pred_label"] == ERROR or conf(r) < th


def leaked(r, th=V3_TH):      # AI 단독 합격 처리된 불량 = 유출
    return is_defect(r) and says_good(r) and conf(r) >= th


def false_alarm(r):           # 정상을 불량으로 판정 (판정 실패는 제외 → 재검사)
    return not is_defect(r) and r["pred_label"] not in ("good", ERROR)


def flip_rate(a: dict, b: dict) -> float:
    ks = a.keys() & b.keys()
    return sum(says_good(a[k]) != says_good(b[k]) for k in ks) / len(ks)


def pct(x):
    return f"{x * 100:.1f}%"


def mark(ok):
    return "✅ 적합" if ok else "❌ 부적합"


def v3(md: list, summary: dict):
    v2 = load("v2_full")
    rep1, rep2 = load("v3_rep1"), load("v3_rep2")
    sub = rep1.keys() & rep2.keys()
    base = {k: v2[k] for k in sub}
    crit = summary["criteria"]

    train_good = load("v3_train_good")
    goods = [r for r in v2.values() if not is_defect(r)] + list(train_good.values())
    fa = sum(false_alarm(r) for r in goods)
    ok = fa / len(goods) <= 0.10 and wilson_upper(fa, len(goods)) <= 0.15
    crit.append(valid({"id": "A1", "name": "특이성 (정상 오탐)", "rule": "정상 535장 중 불량 판정 ≤10%, 95% 상한 ≤15%",
                 "result": f"{pct(fa / len(goods))} ({fa}/{len(goods)}), 상한 {pct(wilson_upper(fa, len(goods)))}", "ok": ok},
                      {k: r for k, r in v2.items() if not is_defect(r)}, train_good))
    md += ["## A1 특이성 (정상 오탐)", "", f"정상 {len(goods)}장 중 불량 판정 {fa}건 = **{pct(fa / len(goods))}** "
           f"(95% 상한 {pct(wilson_upper(fa, len(goods)))}) → {mark(ok)}", ""]

    runs = [base, rep1, rep2]
    same = sum(len({says_good(run[k]) for run in runs}) == 1 for k in sub)
    noise_floor = (flip_rate(base, rep1) + flip_rate(base, rep2) + flip_rate(rep1, rep2)) / 3
    ok = same / len(sub) >= 0.90
    crit.append(valid({"id": "A2", "name": "반복성", "rule": "70장 × 3회 정상/불량 결론 일치 ≥90%",
                 "result": f"{pct(same / len(sub))} ({same}/{len(sub)})", "ok": ok}, base, rep1, rep2))
    md += ["## A2 반복성 (같은 이미지 3회 판정)", "", f"결론 3회 일치 {same}/{len(sub)} = **{pct(same / len(sub))}** → {mark(ok)}  ",
           f"회차 간 평균 결론 변경률(기준선): {pct(noise_floor)}", ""]

    md += ["## A3·A4 강건성 (같은 70장)", "", "| 조건 | 결론 변경률 | 누락 | 오탐 | 재검사 | AI 단독 유출 |", "|---|---|---|---|---|---|"]
    a3 = a4 = True
    n_def = sum(is_defect(r) for r in base.values())
    worst_flip, worst_cond = 0, None
    pert_runs = []
    for cond in ["clean", *PERTURBATIONS]:
        rs = base if cond == "clean" else load(f"v3_{cond}")
        pert_runs.append(rs)
        miss = sum(is_defect(r) and says_good(r) for r in rs.values())
        fal = sum(false_alarm(r) for r in rs.values())
        rev = sum(review(r, V3_TH) for r in rs.values()) / len(rs)
        leak = sum(leaked(r) for r in rs.values())
        flip = None if cond == "clean" else flip_rate(base, rs)
        if flip is not None:
            a3, a4 = a3 and leak == 0, a4 and flip <= noise_floor + 0.05
            if flip > worst_flip:
                worst_flip, worst_cond = flip, COND_LABEL[cond]
        summary["robustness"].append({"cond": cond, "label": COND_LABEL[cond], "review": rev, "miss": miss,
                                      "n_def": n_def, "leak": leak, "flip": flip})
        md.append(f"| {COND_LABEL[cond]} | {'—' if flip is None else pct(flip)} | {miss}/{n_def} | {fal}/{len(rs) - n_def} "
                  f"| {pct(rev)} | {leak} |")
    leaks = [x["leak"] for x in summary["robustness"][1:]]
    crit.append(valid({"id": "A3", "name": "강건성 — 유출", "rule": "열화 4조건 모두 AI 단독 유출 0건",
                 "result": f"조건별 {min(leaks)}~{max(leaks)}건", "ok": a3}, *pert_runs))
    crit.append(valid({"id": "A4", "name": "강건성 — 결론 변경", "rule": f"변경률 ≤ 반복성 기준선({pct(noise_floor)}) + 5%p",
                 "result": f"최대 {pct(worst_flip)} ({worst_cond})", "ok": a4}, *pert_runs))
    md += ["", f"A3 {mark(a3)} / A4 {mark(a4)}", ""]

    rng = random.Random(0)
    allr = list(v2.values())
    hold_leaks = []
    for _ in range(1000):
        rng.shuffle(allr)
        hold = allr[len(allr) // 2:]
        hold_leaks.append(sum(leaked(r) for r in hold) / sum(is_defect(r) for r in hold))
    hold_leaks.sort()
    mean = sum(hold_leaks) / len(hold_leaks)
    ok = mean <= 0.02
    crit.append(valid({"id": "A5", "name": "운용 기준값 0.7", "rule": "무작위 절반 1,000회 유출률 평균 ≤2%",
                 "result": f"평균 {pct(mean)} (95백분위 {pct(hold_leaks[949])})", "ok": ok}, v2))
    md += ["## A5 운용 기준값 0.7 (무작위 절반 1,000회)", "", f"홀드아웃 유출률 평균 **{pct(mean)}** (95백분위 {pct(hold_leaks[949])}) → {mark(ok)}", ""]

    # 기준값별 유출·재검사 트레이드오프 (대시보드 차트)
    n_def_all = sum(is_defect(r) for r in v2.values())
    for th in [0.6, 0.65, 0.7, 0.75, 0.8]:
        k = sum(leaked(r, th) for r in v2.values())
        summary["sweep"].append({"th": th, "leak_n": k, "n_def": n_def_all, "leak": k / n_def_all,
                                 "leak_upper": wilson_upper(k, n_def_all),
                                 "review": sum(review(r, th) for r in v2.values()) / len(v2)})
    bin_acc = sum(says_good(r) != is_defect(r) for r in v2.values()) / len(v2)
    miss = sum(is_defect(r) and says_good(r) for r in v2.values())
    summary["tiles"] += [
        {"label": "정상/불량 판정 정확도", "value": pct(bin_acc), "detail": f"MVTec {len(v2)}장 (AI 단독)"},
        {"label": "AI 단독 불량 누락", "value": pct(miss / n_def_all), "detail": f"{miss}/{n_def_all}, 95% 상한 {pct(wilson_upper(miss, n_def_all))}"},
    ]


def v4(md: list, summary: dict):
    crit = summary["criteria"]
    md += ["# v4 CAPA 검증", ""]

    # E1 / B1·B2: 기준 0.75를 처음 쓰는 외부 데이터로
    b1 = b2 = True
    md += ["## E1 외부 데이터 독립 검증 (기준: 신뢰도 < 0.75 재검사)", "",
           "| 데이터셋 | 불량 | 정상 | AI 단독 유출 (95% 상한) | AI 단독 오탐 | 재검사 | 전체 누락 | 전체 오탐 |", "|---|---|---|---|---|---|---|---|"]
    ext_runs = []
    for name in EXTERNAL:
        ext_runs.append(load(f"v4_e1_{name}"))
        rs = list(ext_runs[-1].values())
        n_def = sum(is_defect(r) for r in rs)
        n_good = len(rs) - n_def
        leak = sum(leaked(r, V4_TH) for r in rs)
        auto_fa = sum(false_alarm(r) and conf(r) >= V4_TH for r in rs)
        rev = sum(review(r, V4_TH) for r in rs) / len(rs)
        miss = sum(is_defect(r) and says_good(r) for r in rs)
        fa = sum(false_alarm(r) for r in rs)
        up = wilson_upper(leak, n_def)
        b1, b2 = b1 and up <= 0.03, b2 and auto_fa / n_good <= 0.05
        summary["external"].append({"name": name, "n_def": n_def, "n_good": n_good, "leak_n": leak, "leak_upper": up,
                                    "auto_fa_n": auto_fa, "auto_fa": auto_fa / n_good, "review": rev,
                                    "miss": miss / n_def, "fa": fa / n_good})
        md.append(f"| {name} | {n_def} | {n_good} | {leak} ({pct(up)}) | {auto_fa} ({pct(auto_fa / n_good)}) | {pct(rev)} "
                  f"| {pct(miss / n_def)} | {pct(fa / n_good)} |")
    ext = summary["external"]
    crit.append(valid({"id": "B1", "name": "기준 0.75 — 유출 (외부 데이터)", "rule": "데이터셋별 AI 단독 유출 95% 상한 ≤3%",
                 "result": " / ".join(f"{x['name']} {x['leak_n']}건 (상한 {pct(x['leak_upper'])})" for x in ext), "ok": b1},
                      *ext_runs))
    crit.append(valid({"id": "B2", "name": "기준 0.75 — 오탐 (외부 데이터)", "rule": "데이터셋별 AI 단독 오탐 ≤5%",
                 "result": " / ".join(f"{x['name']} {pct(x['auto_fa'])}" for x in ext), "ok": b2}, *ext_runs))
    md += ["", f"B1 {mark(b1)} / B2 {mark(b2)}", ""]

    # E2 / B3·B4: 레퍼런스 동봉 — 같은 이미지의 기존 결과와 대응 비교
    ref = load("v4_e2_ref_test") | load("v4_e2_ref_train")
    before = load("v2_full") | load("v3_train_good")
    goods = [k for k, r in ref.items() if not is_defect(r)]
    defs = [k for k, r in ref.items() if is_defect(r)]
    fa_after = sum(false_alarm(ref[k]) for k in goods) / len(goods)
    fa_before = sum(false_alarm(before[k]) for k in goods) / len(goods)
    miss_after = sum(says_good(ref[k]) for k in defs) / len(defs)
    miss_before = sum(says_good(before[k]) for k in defs) / len(defs)
    b3, b4 = fa_after <= 0.10, miss_after <= 0.084
    crit.append(valid({"id": "B3", "name": "레퍼런스 동봉 — 오탐", "rule": f"정상 {len(goods)}장 오탐 ≤10%",
                 "result": f"{pct(fa_after)} (동봉 전 {pct(fa_before)})", "ok": b3}, ref))
    crit.append(valid({"id": "B4", "name": "레퍼런스 동봉 — 누락 (부작용)", "rule": f"불량 {len(defs)}장 누락 ≤8.4%",
                 "result": f"{pct(miss_after)} (동봉 전 {pct(miss_before)})", "ok": b4}, ref))
    md += ["## E2 정상품 레퍼런스 동봉 (같은 이미지 대응 비교)", "",
           f"정상 {len(goods)}장 오탐: {pct(fa_before)} → **{pct(fa_after)}** → B3 {mark(b3)}  ",
           f"불량 {len(defs)}장 누락: {pct(miss_before)} → **{pct(miss_after)}** → B4 {mark(b4)}", ""]
    summary["reference"] = {"fa_before": fa_before, "fa_after": fa_after, "miss_before": miss_before,
                            "miss_after": miss_after, "n_good": len(goods), "n_def": len(defs)}

    # E3 / B5: 품질 게이트 (호출 없음)
    g = json.loads((R / "v4_e3_gate.json").read_text(encoding="utf-8"))
    rate = {c: v["blocked"] / v["n"] for c, v in g.items()}
    b5 = rate["noise25"] >= 0.95 and rate["blur2"] >= 0.95 and rate["clean"] <= 0.02
    crit.append({"id": "B5", "name": "이미지 품질 게이트", "rule": "σ25·블러 차단 ≥95%, 원본 오차단 ≤2%",
                 "result": f"σ25 {pct(rate['noise25'])} · 블러 {pct(rate['blur2'])} · 원본 {pct(rate['clean'])}", "ok": b5})
    md += ["## E3 이미지 품질 게이트 (test 299장, train으로만 보정)", "",
           "| 조건 | 차단 |", "|---|---|"] + [f"| {COND_LABEL[c]} | {v['blocked']}/{v['n']} ({pct(rate[c])}) |" for c, v in g.items()] + \
          ["", f"B5 {mark(b5)}", ""]
    summary["gate"] = {c: rate[c] for c in g}


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔(cp949)에서도 표·기호 출력
    summary = {"criteria": [], "robustness": [], "sweep": [], "external": [], "tiles": [],
               "adopted_threshold": REVIEW_THRESHOLD}
    md = ["# 밸리데이션 결과", "", "# v3", ""]
    v3(md, summary)
    if (R / "v4_e2_ref_train.json").exists() and (R / "v4_e3_gate.json").exists():
        v4(md, summary)
    if summary["external"]:  # 운용 기준의 성능은 기준을 고른 데이터가 아니라 독립 데이터로 보여준다
        ext = summary["external"]
        n_def = sum(x["n_def"] for x in ext)
        leak = sum(x["leak_n"] for x in ext)
        summary["tiles"] += [
            {"label": f"독립 데이터 AI 단독 유출 (기준 < {V4_TH})", "value": f"{leak}/{n_def}",
             "detail": f"외부 2개 데이터셋, 95% 상한 {pct(wilson_upper(leak, n_def))}"},
            {"label": "사람 재검사 비율 (독립 데이터)", "value": " / ".join(pct(x["review"]) for x in ext),
             "detail": " / ".join(x["name"] for x in ext)},
        ]
    passed = sum(c["ok"] for c in summary["criteria"])
    summary["conclusion"] = "자동 합격은 맡기지 않는다. 확실한 불량을 먼저 걸러내는 1차 선별기로 쓴다."
    summary["conclusion_sub"] = (f"사전 설정 기준 {len(summary['criteria'])}개 중 {passed}개 적합. "
                                 "부적합은 일탈로 조사해 CAPA를 세우고, CAPA는 처음 보는 데이터로 다시 검증했다.")
    head = ["| 기준 | " + " | ".join(c["id"] for c in summary["criteria"]) + " |",
            "|---|" + "---|" * len(summary["criteria"]),
            "| 판정 | " + " | ".join("⚠️" if c.get("invalid") else "✅" if c["ok"] else "❌"
                                    for c in summary["criteria"]) + " |", ""]
    head += [f"⚠️ **{c['id']} 무효**: {c['result']}  " for c in summary["criteria"] if c.get("invalid")] + [""]
    (R / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(md[:2] + head + md[2:]))


if __name__ == "__main__":
    main()

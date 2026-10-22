"""밸리데이션 분석: results/*.json → 합격 기준 판정표(markdown) + 대시보드용 results/summary.json
    python analyze.py > results/analysis.md

v3 기준(A1~A5)은 PROJECT_PLAN §4, v4 기준(B1~B5)은 §7, v5 기준(C1~C6)은 §8, v6 기준(F1~F6)은 §9. 기준값은 프로토콜에 고정된 값을 쓰고,
운용 설정(benchmark.REVIEW_THRESHOLD)이 나중에 바뀌어도 과거 판정은 바뀌지 않게 한다.
"""
import hashlib
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


def holdout(image: str) -> bool:
    """v5 분할: md5(image_id) 홀수 = 홀드아웃(평가), 짝수 = 보정(기준값 선정)"""
    return int(hashlib.md5(image.encode()).hexdigest(), 16) % 2 == 1


def v5_thresholds(rs: list) -> dict:
    """§8 규칙: 보정 절반만 보고 카테고리별 기준값"""
    th = {}
    for cat in {r["category"] for r in rs}:
        cal = [r for r in rs if r["category"] == cat and not holdout(r["image"])]
        th[cat] = {"pass": 0.95 * min(r["score"] for r in cal if is_defect(r)),
                   "reject": 1.05 * max(r["score"] for r in cal if not is_defect(r))}
    return th


def v5_disposition(r, th) -> str:
    t = th[r["category"]]
    return "pass" if r["score"] < t["pass"] else "reject" if r["score"] > t["reject"] else "review"


def auroc(rs) -> float:
    d = [r["score"] for r in rs if is_defect(r)]
    g = [r["score"] for r in rs if not is_defect(r)]
    return sum((x > y) + 0.5 * (x == y) for x in d for y in g) / (len(d) * len(g))


def v5(md: list, summary: dict):
    runs = [json.loads((R / f"v5_run{i}.json").read_text(encoding="utf-8")) for i in (1, 2, 3)]
    for run in runs:
        assert len(run["results"]) == 299, "v5: 표본 수"
    rs = runs[0]["results"]
    th = v5_thresholds(rs)
    hold = [r for r in rs if holdout(r["image"])]
    disp = {r["image"]: v5_disposition(r, th) for r in hold}
    n_def = sum(is_defect(r) for r in hold)
    n_good = len(hold) - n_def
    leak = sum(is_defect(r) and disp[r["image"]] == "pass" for r in hold)
    rev = sum(d == "review" for d in disp.values()) / len(hold)
    auto_fa = sum(not is_defect(r) and disp[r["image"]] == "reject" for r in hold)
    sec = max(run["meta"]["sec_per_image"] for run in runs)
    same = all([r["score"] for r in run["results"]] == [r["score"] for r in rs] for run in runs[1:])
    offline = all(run["meta"]["offline"] for run in runs[1:])

    # 같은 홀드아웃에서 Claude (v2, 기준 0.75)
    v2 = load("v2_full")
    ch = [v2[r["image"]] for r in hold]
    c_leak = sum(leaked(r, V4_TH) for r in ch)
    c_rev = sum(review(r, V4_TH) for r in ch) / len(ch)
    c_fa = sum(false_alarm(r) and conf(r) >= V4_TH for r in ch)
    # 실제 라인은 대부분 정상품이라, 재검사 부담은 '정상품을 자동 합격시키는 비율'이 좌우한다 (일탈 D8)
    good_pass = sum(not is_defect(r) and disp[r["image"]] == "pass" for r in hold)
    c_good_pass = sum(not is_defect(r) and says_good(r) and conf(r) >= V4_TH for r in ch)

    crit = summary["criteria"]
    crit += [
        {"id": "C1", "name": "로컬 모델 — 유출", "rule": "홀드아웃 모델 단독 합격된 불량 0건",
         "result": f"{leak}/{n_def} (95% 상한 {pct(wilson_upper(leak, n_def))}) · Claude {c_leak}건", "ok": leak == 0},
        {"id": "C2", "name": "로컬 모델 — 재검사 비율", "rule": "≤30%",
         "result": f"{pct(rev)} · Claude {pct(c_rev)}", "ok": rev <= 0.30},
        {"id": "C3", "name": "로컬 모델 — 오탐", "rule": "모델 단독 불합격된 정상 ≤5%",
         "result": f"{auto_fa}/{n_good} ({pct(auto_fa / n_good)}) · Claude {c_fa}건", "ok": auto_fa / n_good <= 0.05},
        {"id": "C4", "name": "로컬 모델 — 속도", "rule": "장당 ≤1초 (CPU)",
         "result": f"{sec:.2f}초 (3회 중 최대) · Claude 7.7초", "ok": sec <= 1.0},
        {"id": "C5", "name": "로컬 모델 — 재현성", "rule": "3회 점수 299장 비트 단위 동일",
         "result": "299장 모두 동일" if same else "불일치 있음", "ok": same},
        {"id": "C6", "name": "망분리 동작", "rule": "네트워크 차단 상태로 전체 검사",
         "result": "차단 상태 2회 완료" if offline else "미확인", "ok": offline},
    ]
    summary["v5"] = {"n_hold": len(hold), "n_def": n_def, "n_good": n_good, "leak": leak, "review": rev,
                     "auto_fa": auto_fa, "sec": sec, "good_pass": good_pass,
                     "claude": {"leak": c_leak, "review": c_rev, "auto_fa": c_fa, "good_pass": c_good_pass},
                     "thresholds": th, "auroc": {c: auroc([r for r in rs if r["category"] == c]) for c in th},
                     "weights_sha256": runs[0]["meta"]["weights_sha256"]}
    md += ["# v5 온프레미스 검사 (PatchCore, 홀드아웃 절반)", "",
           f"홀드아웃 {len(hold)}장 (불량 {n_def}, 정상 {n_good}). 기준값은 보정 절반에서만 선정.", "",
           "| | 로컬 PatchCore | Claude (v2, 기준 0.75) |", "|---|---|---|",
           f"| 단독 유출 | {leak}/{n_def} | {c_leak}/{n_def} |",
           f"| 재검사 비율 | {pct(rev)} | {pct(c_rev)} |",
           f"| 단독 오탐 | {auto_fa}/{n_good} | {c_fa}/{n_good} |",
           f"| 정상품 자동 합격 | {good_pass}/{n_good} | {c_good_pass}/{n_good} |",
           f"| 장당 시간 | {sec:.2f}초 (CPU) | 7.7초 |",
           f"| 반복 재현 | {'100% 동일' if same else '불일치'} | 88.6% (A2) |",
           f"| 이미지 외부 전송 | 없음 (차단 상태 실행) | 있음 |", "",
           "AUROC (참고, 전체 299장): " + ", ".join(f"{c} {v:.3f}" for c, v in summary["v5"]["auroc"].items()), "",
           "기준값: " + ", ".join(f"{c} 합격<{t['pass']:.3f}, 불합격>{t['reject']:.3f}" for c, t in th.items()), "",
           " / ".join(f"{c['id']} {mark(c['ok'])}" for c in crit[-6:]), ""]


def v6_role(r) -> str:
    """§9 분할: test는 v5와 같은 md5 반분, train 정상은 md5(파일명) % 5 (0 보정, 1 평가; 나머지는 정상 기억이라 결과에 없음)"""
    if "/train/" in r["image"]:
        k = int(hashlib.md5(r["image"].rsplit("/", 1)[1].encode()).hexdigest(), 16) % 5
        assert k in (0, 1), r["image"]
        return "calib" if k == 0 else "eval"
    return "eval" if holdout(r["image"]) else "calib"


def v6_thresholds(rs: list) -> dict:
    """v5와 같은 규칙, 보정 정상품에 train 보정분을 더한다"""
    th = {}
    for cat in {r["category"] for r in rs}:
        cal = [r for r in rs if r["category"] == cat and v6_role(r) == "calib"]
        th[cat] = {"pass": 0.95 * min(r["score"] for r in cal if is_defect(r)),
                   "reject": 1.05 * max(r["score"] for r in cal if not is_defect(r))}
    return th


def v6_eval(rs: list) -> dict:
    th = v6_thresholds(rs)
    ev = [r for r in rs if v6_role(r) == "eval"]
    goods = [r for r in ev if not is_defect(r)]
    defs = [r for r in ev if is_defect(r)]
    dp = lambda r: v5_disposition(r, th)  # noqa: E731
    g_pass = sum(dp(r) == "pass" for r in goods) / len(goods)
    g_rej = sum(dp(r) == "reject" for r in goods) / len(goods)
    d_rev = sum(dp(r) == "review" for r in defs) / len(defs)
    return {"th": th, "n_good": len(goods), "n_def": len(defs),
            "leak": sum(dp(r) == "pass" for r in defs), "good_pass": g_pass, "good_reject": g_rej,
            "good_pass_n": sum(dp(r) == "pass" for r in goods), "good_reject_n": sum(dp(r) == "reject" for r in goods),
            "line_review": 0.99 * (1 - g_pass - g_rej) + 0.01 * d_rev,
            "auroc": {c: auroc([r for r in rs if r["category"] == c]) for c in th}}


def v6(md: list, summary: dict):
    runs = [json.loads((R / f"v6_run{i}.json").read_text(encoding="utf-8")) for i in (1, 2, 3)]
    for run in runs:
        assert run["meta"]["protocol"] == "v6" and run["meta"]["method"] == "coreset", "v6: 조건"
        assert len(run["results"]) == len(runs[0]["results"]), "v6: 표본 수"
    rs = runs[0]["results"]
    e = v6_eval(rs)
    rnd = json.loads((R / "v6_random.json").read_text(encoding="utf-8"))
    assert rnd["meta"]["method"] == "random" and len(rnd["results"]) == len(rs)
    er = v6_eval(rnd["results"])
    sec = max(run["meta"]["sec_per_image"] for run in runs)
    same = all([r["score"] for r in run["results"]] == [r["score"] for r in rs] for run in runs[1:])
    offline = all(run["meta"]["offline"] for run in runs[1:])

    # 같은 평가 이미지에서 Claude (v2 test + v3 train 정상, 기준 0.75)
    claude = load("v2_full") | load("v3_train_good")
    ev = [claude[r["image"]] for r in rs if v6_role(r) == "eval"]
    cg = [r for r in ev if not is_defect(r)]
    cd = [r for r in ev if is_defect(r)]
    c_pass = sum(says_good(r) and conf(r) >= V4_TH for r in cg) / len(cg)
    c_rej = sum(false_alarm(r) and conf(r) >= V4_TH for r in cg) / len(cg)
    c = {"leak": sum(leaked(r, V4_TH) for r in cd), "good_pass": c_pass, "good_reject": c_rej,
         "line_review": 0.99 * (1 - c_pass - c_rej) + 0.01 * sum(review(r, V4_TH) for r in cd) / len(cd)}

    crit = summary["criteria"]
    n_def, n_good = e["n_def"], e["n_good"]
    crit += [
        {"id": "F1", "name": "v6 로컬 — 유출", "rule": f"평가 불량 {n_def}장 모델 단독 합격 0건",
         "result": f"{e['leak']}/{n_def} (95% 상한 {pct(wilson_upper(e['leak'], n_def))})", "ok": e["leak"] == 0},
        {"id": "F2", "name": "v6 로컬 — 정상품 자동 합격", "rule": f"평가 정상 {n_good}장 중 ≥50%",
         "result": f"{pct(e['good_pass'])} ({e['good_pass_n']}/{n_good}) · v5 15% · Claude {pct(c_pass)}", "ok": e["good_pass"] >= 0.5},
        {"id": "F3", "name": "v6 로컬 — 정상 오탐", "rule": "모델 단독 불합격된 정상 ≤2%",
         "result": f"{pct(e['good_reject'])} ({e['good_reject_n']}/{n_good})", "ok": e["good_reject"] <= 0.02},
        {"id": "F4", "name": "v6 로컬 — 속도", "rule": "장당 ≤1초 (CPU)", "result": f"{sec:.2f}초 (3회 중 최대)", "ok": sec <= 1.0},
        {"id": "F5", "name": "v6 로컬 — 재현성", "rule": "3회 점수 비트 단위 동일",
         "result": f"{len(rs)}장 모두 동일" if same else "불일치 있음", "ok": same},
        {"id": "F6", "name": "v6 망분리 동작", "rule": "네트워크 차단 상태로 전체 검사",
         "result": "차단 상태 2회 완료" if offline else "미확인", "ok": offline},
    ]
    rows = [
        ["단독 유출 (불량을 혼자 합격)", f"{e['leak']}/{n_def}", f"{er['leak']}/{n_def}", f"{c['leak']}/{n_def}"],
        ["정상품 자동 합격", pct(e["good_pass"]), pct(er["good_pass"]), pct(c["good_pass"])],
        ["정상 단독 불합격 (오탐)", pct(e["good_reject"]), pct(er["good_reject"]), pct(c["good_reject"])],
        ["불량률 1% 라인 가정 재검사", pct(e["line_review"]), pct(er["line_review"]), pct(c["line_review"])],
        ["AUROC (pill / capsule)", " / ".join(f"{e['auroc'][k]:.3f}" for k in ("pill", "capsule")),
         " / ".join(f"{er['auroc'][k]:.3f}" for k in ("pill", "capsule")), "—"],
    ]
    summary["v6"] = {"n_def": n_def, "n_good": n_good, "sec": sec, "cols": ["v6 coreset", "참고: 무작위 보관", "Claude (기준 0.75)"],
                     "rows": rows, "thresholds": e["th"], "weights_sha256": runs[0]["meta"]["weights_sha256"],
                     "good_pass": e["good_pass"], "leak": e["leak"]}
    md += ["# v6 D8 CAPA — 정상품 자동 합격률 (평가: 불량 %d, 정상 %d)" % (n_def, n_good), "",
           "| | " + " | ".join(summary["v6"]["cols"]) + " |", "|---|---|---|---|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    md += ["", "기준값: " + ", ".join(f"{k} 합격<{t['pass']:.3f}, 불합격>{t['reject']:.3f}" for k, t in e["th"].items()), "",
           " / ".join(f"{x['id']} {mark(x['ok'])}" for x in crit[-6:]), ""]


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔(cp949)에서도 표·기호 출력
    summary = {"criteria": [], "robustness": [], "sweep": [], "external": [], "tiles": [],
               "adopted_threshold": REVIEW_THRESHOLD}
    md = ["# 밸리데이션 결과", "", "# v3", ""]
    v3(md, summary)
    if (R / "v4_e2_ref_train.json").exists() and (R / "v4_e3_gate.json").exists():
        v4(md, summary)
    if (R / "v5_run3.json").exists():
        v5(md, summary)
    if (R / "v6_random.json").exists():
        v6(md, summary)
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

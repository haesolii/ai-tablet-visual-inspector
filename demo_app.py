"""AI 정제 외관 검사기 — QC 검사 스테이션 데모 (Flask)

흐름: 샘플 선택(촬영 조건 선택 가능) → 품질 게이트 → AI 판정 → 처분(합격/불합격/재검사/재촬영)
     판정 엔진: Claude(클라우드) 또는 PatchCore(로컬, v5 — 이미지가 PC 밖으로 나가지 않음)
     → 사람 최종 판정 → 해시 체인 감사추적.  두 번째 탭은 밸리데이션 결과 대시보드.
    python demo_app.py   # http://127.0.0.1:8931
"""
import base64
import hashlib
import functools
import io
import json
import random
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file

import quality_gate
from benchmark import (PERTURBATIONS, PROMPT, PROMPT_REF, REVIEW_THRESHOLD, USE_REFERENCE, call_claude,
                       find_category_dir, perturbed_copy, prompt_version, reference_images,
                       resolve_data_dir)

BASE = Path(__file__).parent
AUDIT_LOG = BASE / "audit_trail.jsonl"
SUMMARY = BASE / "results" / "summary.json"
CACHE = Path(tempfile.gettempdir()) / "qc_demo_cache"  # 열화 이미지 캐시 (시드 고정이라 다시 만들어도 같다)
CONDITIONS = {"clean": "원본", "noise10": "백색잡음 σ10", "noise25": "백색잡음 σ25",
              "blur2": "블러 (초점 이탈)", "dark60": "조명 60%"}
assert set(CONDITIONS) == {"clean", *PERTURBATIONS}
ENGINES = {"claude": "Claude (클라우드)", "local": "PatchCore (로컬)"}
app = Flask(__name__)
app.json.sort_keys = False  # 배치 정보 등 화면 표시 순서 유지

# 샘플 이미지 풀: 클라이언트에는 인덱스만 보낸다 (경로에 정답 라벨이 들어 있음)
SAMPLE_POOL = []
REFERENCES = {}


def build_pool():
    if SAMPLE_POOL:
        return SAMPLE_POOL
    data_dir = resolve_data_dir(None)
    for cat in ["pill", "capsule"] if data_dir else []:
        test_dir = find_category_dir(data_dir, cat)
        if test_dir is None:
            continue
        for d in sorted(test_dir.iterdir()):
            if not d.is_dir():
                continue
            for img in sorted(d.glob("*.png")):
                SAMPLE_POOL.append({"category": cat, "true_label": "good" if d.name == "good" else d.name,
                                    "path": str(img), "name": img.name})
    random.Random(0).shuffle(SAMPLE_POOL)  # id 순서로 라벨을 짐작하지 못하게
    if data_dir and USE_REFERENCE:
        REFERENCES.update(reference_images(data_dir, ["pill", "capsule"]))
    return SAMPLE_POOL


def disposition(pred: dict) -> str:
    """AI 판정 → QC 처분. 판정 실패·저신뢰도는 합격시키지 않는다 (fail-safe)."""
    if "error" in pred or (pred.get("confidence") or 0) < REVIEW_THRESHOLD:
        return "review"
    return "pass" if pred.get("verdict") == "good" else "reject"


# ---- 감사추적: append-only + 해시 체인 ----
_audit_lock = threading.Lock()
GENESIS = "GENESIS"


def _record_hash(rec: dict) -> str:
    return hashlib.sha256(json.dumps(rec, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _read_log() -> list[dict]:
    if not AUDIT_LOG.exists():
        return []
    return [json.loads(line) for line in AUDIT_LOG.read_text(encoding="utf-8").splitlines() if line]


def audit(record: dict) -> dict:
    """ALCOA+ 데모: 누가·언제·무엇을(이미지 해시)·어떤 결과로 — 추가만 하는 로그.
    각 기록에 직전 기록의 해시를 넣는 해시 체인: 중간 기록을 고치거나 지우면 verify_chain()이 잡아낸다."""
    # ponytail: 매 기록마다 파일 전체를 읽음 — 데모 규모용. 커지면 마지막 해시를 메모리에 유지
    with _audit_lock:
        log = _read_log()
        record = record | {"prev_hash": log[-1]["hash"] if log else GENESIS}
        record["hash"] = _record_hash(record)
        with AUDIT_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def verify_chain() -> dict:
    """한계: 맨 끝 기록을 지우거나 체인 전체를 다시 계산하면 이것만으로는 못 잡는다.
    실제 시스템이면 마지막 해시를 외부(별도 서버·출력 기록)에 주기적으로 남겨 고정한다."""
    prev = GENESIS
    lines = AUDIT_LOG.read_text(encoding="utf-8").splitlines() if AUDIT_LOG.exists() else []
    for i, line in enumerate(lines, 1):
        try:
            r = json.loads(line)
            h = r.pop("hash")
        except (ValueError, KeyError, AttributeError):
            return {"ok": False, "n": len(lines), "broken_at": i}
        if r.get("prev_hash") != prev or _record_hash(r) != h:
            return {"ok": False, "n": len(lines), "broken_at": i}
        prev = h
    return {"ok": True, "n": len(lines), "broken_at": None}


def now_kst():
    return datetime.now(timezone(timedelta(hours=9)))


def new_audit_id(now) -> str:
    return f"AT-{now.strftime('%y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"


def fake_gmp(category: str):
    """데모용 합성 배치 정보 (실제 제품·배치 아님)"""
    now = now_kst()
    return {
        "배치번호": f"BT-{now.strftime('%y%m%d')}-{random.randint(100, 999)}",
        "제품": "Amoxicillin Cap 250mg (예시)" if category == "capsule" else "Ibuprofen Tab 200mg (예시)",
        "외관 규격": "흠·오염·변색·이물 없을 것 (SOP-QC-041 예시)",
        "라인": f"L-{random.randint(1, 4)}",
    }


# ---- 요청 처리 ----
def sample_or_404():
    try:
        return build_pool()[int(request.values["id"])]
    except (KeyError, ValueError, IndexError):
        abort(404)


def condition_or_400() -> str:
    cond = request.values.get("cond", "clean")
    if cond not in CONDITIONS:
        abort(400)
    return cond


_cache_lock = threading.Lock()


def image_path(s: dict, cond: str) -> str:
    """선택한 촬영 조건의 이미지 경로. 열화본은 캐시한다.
    이미지·게이트 요청이 동시에 와도 한 번만 만들도록 잠근다 (Windows는 열린 파일을 덮어쓰면 실패)."""
    if cond == "clean":
        return s["path"]
    out = CACHE / f"{s['category']}_{hashlib.md5(s['path'].encode()).hexdigest()[:10]}_{cond}.png"
    with _cache_lock:
        if not out.exists():
            CACHE.mkdir(exist_ok=True)
            Path(perturbed_copy(s["path"], cond, str(CACHE))).replace(out)
    return str(out)


_local = {}
_local_lock = threading.Lock()


@functools.cache
def local_decision():
    """§10 채택 규칙(어느 모델·제품별 정책·기본 엔진)을 결과 파일에서 계산. 결과 파일이 없으면 None"""
    import analyze
    try:
        return analyze.local_decision()
    except FileNotFoundError:
        return None


def local_engine():
    """채택된 로컬 모델 (v6 또는 v7). torch가 무거워서 처음 쓸 때 불러온다 (정상 패치 기억은 .cache에 저장해 재사용)"""
    with _local_lock:
        if not _local:
            import analyze
            import patchcore
            dec = local_decision()
            m = patchcore.PatchCore(dec["res"])
            data_dir = resolve_data_dir(None)
            for cat in ["pill", "capsule"]:   # 실험과 같은 정상 기억: train 정상 중 'bank' 몫, greedy coreset
                m.fit(cat, [p for p in patchcore.train_goods(data_dir, cat) if patchcore.train_role(p) == "bank"],
                      keep=dec["keep"])
            _local.update(model=m, th=dec["th"], name=dec["model"], policy=dec["policy"],
                          disposition=lambda r, th: analyze.policy_disposition(r, th, dec["policy"]))
    return _local


def heatmap_data_url(path: str, d, t: dict, tf) -> str:
    """모델이 본 영역 위에, 정상 기준에서 먼 패치일수록 진한 파랑을 겹친다 (빨간 오염과 헷갈리지 않게)"""
    from PIL import Image
    base = tf.transforms[1](tf.transforms[0](Image.open(path).convert("RGB")))
    v = ((d - t["pass"]) / (t["reject"] - t["pass"])).clamp(0, 1)
    mask = Image.fromarray((v * 170).byte().numpy(), "L").resize(base.size, Image.BILINEAR)
    out = Image.composite(Image.new("RGB", base.size, (0, 140, 255)), base, mask)
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def inspect_local(s: dict, path: str) -> tuple[dict, dict]:
    """감사 기록용 필드와, 기록하지 않고 화면에만 보내는 필드(히트맵)"""
    eng = local_engine()
    t0 = time.perf_counter()
    score, d = eng["model"].score(s["category"], path)
    sec = time.perf_counter() - t0
    t = eng["th"][s["category"]]
    rec = {"model": f"PatchCore {eng['name']} {eng['model'].res}px/wide_resnet50_2 sha256:{eng['model'].weights_sha256[:12]}",
           "policy": eng["policy"][s["category"]],
           "score": round(score, 4), "t_pass": round(t["pass"], 4), "t_reject": round(t["reject"], 4),
           "network": "미사용", "disposition": eng["disposition"]({"category": s["category"], "score": score}, eng["th"])}
    return rec, {"heatmap": heatmap_data_url(path, d, t, eng["model"].tf), "sec": round(sec, 3)}


@app.route("/")
def index():
    return render_template("index.html", conditions=CONDITIONS)


@app.route("/api/config")
def api_config():
    return jsonify({"threshold": REVIEW_THRESHOLD, "use_reference": USE_REFERENCE, "engines": ENGINES,
                    "local_ready": local_decision() is not None,
                    "default_engine": "local" if (local_decision() or {}).get("default_local") else "claude",
                    "prompt_version": prompt_version(PROMPT_REF if USE_REFERENCE else PROMPT),
                    "gate": bool(quality_gate.load_limits()), "n_samples": len(build_pool())})


@app.route("/api/sample")
def api_sample():
    pool = build_pool()
    if not pool:
        return jsonify({"error": "데이터셋 없음 — README의 다운로드 안내 참고"}), 503
    i = random.randrange(len(pool))
    return jsonify({"id": i, "category": pool[i]["category"], "name": pool[i]["name"],
                    "gmp": fake_gmp(pool[i]["category"])})


@app.route("/api/image")
def api_image():
    return send_file(image_path(sample_or_404(), condition_or_400()), mimetype="image/png")


@app.route("/api/gate")
def api_gate():
    s, cond = sample_or_404(), condition_or_400()
    limits = quality_gate.load_limits().get(s["category"])
    if not limits:
        return jsonify({"ok": True, "metrics": {}, "failed": [], "limits": None})
    return jsonify(quality_gate.check(image_path(s, cond), limits) | {"limits": limits})


@app.before_request
def require_demo_header():
    """POST는 화면의 fetch만 허용 (사용자 정의 헤더는 다른 출처에서 사전 확인 없이 못 붙인다)"""
    if request.method == "POST" and request.headers.get("X-QC-Demo") != "1":
        abort(403)


@app.post("/api/inspect")
def api_inspect():
    s, cond = sample_or_404(), condition_or_400()
    engine = request.values.get("engine", "claude")
    if engine not in ENGINES:
        abort(400)
    path = image_path(s, cond)
    limits = quality_gate.load_limits().get(s["category"])
    gate = quality_gate.check(path, limits) if limits else {"ok": True, "failed": []}
    now = now_kst()
    rec = {
        "type": "ai_inspection",
        "audit_id": new_audit_id(now),
        "timestamp": now.isoformat(timespec="seconds"),
        "operator": "AI-VISION-01",
        "image_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "image": f"{s['category']}/{s['name']}",
        "condition": cond,
        "engine": engine,
    }
    view = {}
    if not gate["ok"]:  # CAPA D5: 품질 미달 이미지는 AI에 넘기지 않고 재촬영
        rec |= {"disposition": "recapture", "gate_failed": gate["failed"]}
    elif engine == "local":
        local, view = inspect_local(s, path)
        rec |= local
    else:
        rec["threshold"] = REVIEW_THRESHOLD
        ref = REFERENCES.get(s["category"])
        prompt = PROMPT_REF if ref else PROMPT
        try:
            pred = call_claude(path, prompt, reference=ref) if ref else call_claude(path, prompt)
        except Exception as e:
            pred = {"error": str(e)[:200]}
        conf = pred.get("confidence")
        clean = {"error": pred["error"]} if "error" in pred else {
            "verdict": pred.get("verdict") if pred.get("verdict") in ("good", "defective") else None,
            "defect_type": str(pred.get("defect_type"))[:40] if pred.get("defect_type") else None,
            "confidence": conf if isinstance(conf, (int, float)) and 0 <= conf <= 1 else None,
        }
        if "error" not in clean and (clean["verdict"] is None or clean["confidence"] is None):
            clean = {"error": "모델 응답 형식 오류"}
        rec |= {"model": pred.get("_model"), "prompt_version": prompt_version(prompt)}
        rec |= clean | {"disposition": disposition(clean)}
    rec = audit(rec)
    return jsonify(rec | view | {"true_label": s["true_label"]})


@app.post("/api/review")
def api_review():
    """사람 재검사 판정. AI 판정 기록을 고치지 않고 새 기록으로 덧붙인다 (원본 보존)."""
    body = request.get_json(silent=True) or {}
    decision = body.get("decision")
    reviewer = str(body.get("reviewer") or "").strip()[:50]
    reason = str(body.get("reason") or "").strip()[:200]
    if decision not in ("pass", "reject") or not reviewer or not reason:
        return jsonify({"error": "판정(합격/불합격)·검사자·사유는 필수입니다"}), 400
    target = next((r for r in _read_log() if r.get("type") == "ai_inspection"
                   and r.get("audit_id") == body.get("ref")), None)
    if target is None:
        abort(404)
    now = now_kst()
    return jsonify(audit({
        "type": "human_review",
        "audit_id": new_audit_id(now),
        "ref": target["audit_id"],
        "timestamp": now.isoformat(timespec="seconds"),
        "operator": reviewer,
        "ai_disposition": target["disposition"],
        "decision": decision,
        "reason": reason,
    }))


@app.route("/api/audit")
def api_audit():
    return jsonify(verify_chain() | {"records": _read_log()[-10:][::-1]})


@app.route("/api/validation")
def api_validation():
    if not SUMMARY.exists():
        return jsonify({"error": "results/summary.json 없음 — python analyze.py 실행"}), 404
    return jsonify(json.loads(SUMMARY.read_text(encoding="utf-8")))


if __name__ == "__main__":
    build_pool()
    print(f"샘플 {len(SAMPLE_POOL)}장 로드 · 기준 {REVIEW_THRESHOLD} · 레퍼런스 {USE_REFERENCE}")
    app.run(host="127.0.0.1", port=8931)

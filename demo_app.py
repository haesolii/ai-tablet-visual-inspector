"""AI 정제 외관 검사기 — QC 데모 웹앱 (Flask)"""
import hashlib
import json
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path

from flask import Flask, abort, jsonify, render_template_string, request, send_file

from benchmark import (PROMPT, PROMPT_VERSION, REVIEW_THRESHOLD, call_claude,
                       find_category_dir, resolve_data_dir)

BASE = Path(__file__).parent
AUDIT_LOG = BASE / "audit_trail.jsonl"
MODEL_ID = "claude (Claude Code CLI)"
app = Flask(__name__)

# 샘플 이미지 풀: 클라이언트에는 인덱스만 보낸다 (경로에 정답 라벨이 들어 있음)
SAMPLE_POOL = []


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
    return SAMPLE_POOL


def disposition(pred: dict) -> str:
    """AI 판정 → QC 처리 결정. 판정 실패·저신뢰도는 합격시키지 않는다 (fail-safe)."""
    if "error" in pred or (pred.get("confidence") or 0) < REVIEW_THRESHOLD:
        return "review"
    return "pass" if pred.get("verdict") == "good" else "reject"


def audit(record: dict):
    """ALCOA+ 데모: 누가·언제·무엇을(이미지 해시)·어떤 결과로 — 추가만 하는 로그"""
    with AUDIT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def fake_gmp(category: str):
    """데모용 합성 GMP 메타데이터"""
    now = datetime.now(timezone(timedelta(hours=9)))
    return {
        "batch_no": f"BT-{now.strftime('%y%m%d')}-{random.randint(100, 999)}",
        "product": "Amoxicillin Cap 250mg" if category == "capsule" else "Ibuprofen Tab 200mg",
        "spec": "외관: 흠·오염·변색 없을 것 (SOP-QC-041)",
        "line": f"L-{random.randint(1, 4)}",
        "inspector": "AI-VISION-01",
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S KST"),
    }


PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI 정제 외관 검사기 — QC 데모</title>
<style>
body{font-family:-apple-system,'Noto Sans KR',sans-serif;background:#0f172a;color:#e2e8f0;margin:0;padding:24px}
.wrap{max-width:960px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px} .sub{color:#94a3b8;font-size:13px;margin-bottom:20px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.card{background:#1e293b;border-radius:12px;padding:16px}
.card h3{margin:0 0 10px;font-size:14px;color:#94a3b8}
img.sample{width:100%;border-radius:8px;background:#0b1120}
button{background:#38bdf8;border:0;color:#0f172a;font-weight:700;border-radius:8px;padding:10px 16px;cursor:pointer;font-size:14px}
button:disabled{opacity:.5;cursor:wait}
.row{display:flex;gap:10px;margin-top:12px;flex-wrap:wrap}
.verdict{font-size:28px;font-weight:800;margin:8px 0}
.good{color:#4ade80}.bad{color:#f87171}.warn{color:#fbbf24}
.meta{font-size:13px;color:#cbd5e1;line-height:1.8}
.meta b{color:#94a3b8;font-weight:400;display:inline-block;width:90px}
.spin{display:inline-block;animation:sp 1s linear infinite}@keyframes sp{to{transform:rotate(360deg)}}
.conf{height:8px;background:#334155;border-radius:4px;margin-top:6px;overflow:hidden}
.conf>div{height:100%;background:#38bdf8}
.note{font-size:12px;color:#64748b;margin-top:16px}
</style></head><body><div class="wrap">
<h1>💊 AI 정제 외관 검사기</h1>
<div class="sub">제약 QC 외관검사 보조 데모 — AI 1차 스크리닝 → 저신뢰도는 사람 재검사 → 판정 감사추적 기록</div>
<div class="grid">
<div class="card"><h3>검사 대상</h3>
<img id="img" class="sample" src="" alt="sample">
<div class="row">
<button id="btnNew">🎲 다른 샘플</button>
<button id="btnGo">🔍 AI 검사 시작</button>
</div></div>
<div class="card"><h3>판정 결과</h3>
<div id="res"><p class="meta">샘플을 선택하고 [AI 검사 시작]을 누르세요.</p></div>
</div></div>
<div class="card" style="margin-top:16px"><h3>배치 정보 (GMP 합성 데모)</h3>
<div id="gmp" class="meta"></div></div>
<p class="note">* 이미지는 MVTec AD 산업 벤치마크 실사진이며, 실제 제약 QC 데이터가 아닙니다. 배치 정보는 데모용 합성입니다.</p>
</div>
<script>
let cur=null;
async function api(p,m){const r=await fetch(p,{method:m||'GET'});return r.json();}
async function newSample(){cur=await api('/api/sample');document.getElementById('img').src='/api/image?id='+cur.id;
const g=await api('/api/gmp?id='+cur.id);document.getElementById('gmp').innerHTML=Object.entries(g).map(([k,v])=>`<div><b>${k}</b>${v}</div>`).join('');
document.getElementById('res').innerHTML='<p class="meta">샘플을 선택하고 [AI 검사 시작]을 누르세요.</p>';}
document.getElementById('btnNew').onclick=newSample;
document.getElementById('btnGo').onclick=async()=>{
if(!cur)return;const b=document.getElementById('btnGo');b.disabled=true;
document.getElementById('res').innerHTML='<p class="meta"><span class="spin">⏳</span> AI가 외관을 검사 중입니다… (약 15초)</p>';
const r=await api('/api/inspect?id='+cur.id);
b.disabled=false;
const D={pass:['good','✅ 합격'],reject:['bad','❌ 불합격'],review:['warn','⚠️ 재검사 (사람 확인)']}[r.disposition];
const c=r.confidence||0;
document.getElementById('res').innerHTML=`
<div class="verdict ${D[0]}">${D[1]}</div>
<div class="meta"><div><b>AI 판정</b>${r.error?'판정 실패':(r.verdict==='good'?'정상':'불량')}</div>
<div><b>불량 유형</b>${r.defect_type||'—'}</div>
<div><b>신뢰도</b>${(c*100).toFixed(1)}% (기준 ${r.threshold*100}%)<div class="conf"><div style="width:${c*100}%"></div></div></div>
<div><b>실제 정답</b>${r.true_label} ${r.true_label===(r.verdict==='good'?'good':r.defect_type)?'✓':'✗'}</div>
<div><b>감사추적</b>${r.audit_id}</div></div>`;};
newSample();
</script></body></html>"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/api/sample")
def api_sample():
    pool = build_pool()
    if not pool:
        return jsonify({"error": "데이터셋 없음 — README의 다운로드 안내 참고"}), 503
    i = random.randrange(len(pool))
    return jsonify({"id": i, "category": pool[i]["category"], "name": pool[i]["name"]})


def sample_or_404():
    try:
        return build_pool()[int(request.args["id"])]
    except (KeyError, ValueError, IndexError):
        abort(404)


@app.route("/api/image")
def api_image():
    return send_file(sample_or_404()["path"], mimetype="image/png")


@app.route("/api/gmp")
def api_gmp():
    return jsonify(fake_gmp(sample_or_404()["category"]))


@app.route("/api/inspect")
def api_inspect():
    s = sample_or_404()
    try:
        pred = call_claude(s["path"], PROMPT)
    except Exception as e:
        pred = {"error": str(e)[:200]}
    now = datetime.now(timezone(timedelta(hours=9)))
    rec = {
        "audit_id": f"AT-{now.strftime('%y%m%d%H%M%S')}-{random.randint(100, 999)}",
        "timestamp": now.isoformat(timespec="seconds"),
        "operator": "AI-VISION-01",
        "model": MODEL_ID,
        "prompt_version": PROMPT_VERSION,
        "image_sha256": hashlib.sha256(Path(s["path"]).read_bytes()).hexdigest(),
        "image": f"{s['category']}/{s['name']}",
        **pred,
        "disposition": disposition(pred),
        "threshold": REVIEW_THRESHOLD,
    }
    audit(rec)
    return jsonify(rec | {"true_label": s["true_label"]})


if __name__ == "__main__":
    build_pool()
    print(f"샘플 {len(SAMPLE_POOL)}장 로드")
    app.run(host="127.0.0.1", port=8931)

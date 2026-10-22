"""AI 정제 외관 검사기 — QC 데모 웹앱 (Flask)"""
import json
import random
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request

from benchmark import call_claude, PROMPT

BASE = Path(__file__).parent
app = Flask(__name__)

# 샘플 이미지 풀: pill 테스트셋 (카테고리 라벨 포함)
SAMPLE_POOL = []
DATA_ROOT = Path.home() / ".cache/kagglehub/datasets/ipythonx/mvtec-ad/versions/2"


def build_pool():
    global SAMPLE_POOL
    if SAMPLE_POOL:
        return SAMPLE_POOL
    for cat in ["pill", "capsule"]:
        test_dir = DATA_ROOT / cat / "test"
        if not test_dir.is_dir():
            continue
        for d in sorted(test_dir.iterdir()):
            if not d.is_dir():
                continue
            for img in sorted(d.glob("*.png")):
                SAMPLE_POOL.append({"category": cat, "true_label": "good" if d.name == "good" else d.name,
                                    "path": str(img), "name": img.name})
    return SAMPLE_POOL


def fake_gmp():
    """데모용 합성 GMP 메타데이터"""
    now = datetime.now(timezone(timedelta(hours=9)))
    return {
        "batch_no": f"BT-{now.strftime('%y%m%d')}-{random.randint(100, 999)}",
        "product": "Amoxicillin Cap 250mg" if random.random() < 0.5 else "Ibuprofen Tab 200mg",
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
.good{color:#4ade80}.bad{color:#f87171}
.meta{font-size:13px;color:#cbd5e1;line-height:1.8}
.meta b{color:#94a3b8;font-weight:400;display:inline-block;width:90px}
.spin{display:inline-block;animation:sp 1s linear infinite}@keyframes sp{to{transform:rotate(360deg)}}
.conf{height:8px;background:#334155;border-radius:4px;margin-top:6px;overflow:hidden}
.conf>div{height:100%;background:#38bdf8}
.note{font-size:12px;color:#64748b;margin-top:16px}
</style></head><body><div class="wrap">
<h1>💊 AI 정제 외관 검사기</h1>
<div class="sub">제약 QC 라인 데모 — MVTec AD 실사진 + Claude 비전 판정 (Hackathon Demo)</div>
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
async function newSample(){cur=await api('/api/sample');document.getElementById('img').src='/api/image?path='+encodeURIComponent(cur.path);
const g=await api('/api/gmp');document.getElementById('gmp').innerHTML=Object.entries(g).map(([k,v])=>`<div><b>${k}</b>${v}</div>`).join('');
document.getElementById('res').innerHTML='<p class="meta">샘플을 선택하고 [AI 검사 시작]을 누르세요.</p>';}
document.getElementById('btnNew').onclick=newSample;
document.getElementById('btnGo').onclick=async()=>{
if(!cur)return;const b=document.getElementById('btnGo');b.disabled=true;
document.getElementById('res').innerHTML='<p class="meta"><span class="spin">⏳</span> AI가 외관을 검사 중입니다… (약 15초)</p>';
const r=await api('/api/inspect?path='+encodeURIComponent(cur.path));
b.disabled=false;
const ok=r.verdict==='good';
document.getElementById('res').innerHTML=`
<div class="verdict ${ok?'good':'bad'}">${ok?'✅ 합격':'❌ 불합격'}</div>
<div class="meta"><div><b>불량 유형</b>${r.defect_type||'—'}</div>
<div><b>신뢰도</b>${(r.confidence*100).toFixed(1)}%<div class="conf"><div style="width:${r.confidence*100}%"></div></div></div>
<div><b>실제 정답</b>${cur.true_label} ${cur.true_label===(r.defect_type||'good')?'✓':'✗'}</div></div>`;};
newSample();
</script></body></html>"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/api/sample")
def api_sample():
    pool = build_pool()
    s = random.choice(pool)
    return jsonify({k: v for k, v in s.items() if k != "path"} | {"path": s["path"]})


@app.route("/api/image")
def api_image():
    from flask import send_file
    p = Path(request.args["path"])
    # 경로 탈출 방지: 데이터 루트 안에서만
    if DATA_ROOT not in p.resolve().parents:
        return "forbidden", 403
    return send_file(p, mimetype="image/png")


@app.route("/api/gmp")
def api_gmp():
    return jsonify(fake_gmp())


@app.route("/api/inspect")
def api_inspect():
    p = Path(request.args["path"])
    if DATA_ROOT not in p.resolve().parents:
        return "forbidden", 403
    try:
        pred = call_claude(str(p), PROMPT)
    except Exception as e:
        return jsonify({"error": str(e)[:200]}), 500
    return jsonify(pred)


if __name__ == "__main__":
    build_pool()
    print(f"샘플 {len(SAMPLE_POOL)}장 로드")
    app.run(host="0.0.0.0", port=8931)

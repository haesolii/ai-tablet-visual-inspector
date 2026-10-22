"""오프라인 자체검증: 데이터셋·Claude 호출 없이 하네스 로직만 확인.
    python test_offline.py
"""
import os
import re
import tempfile
import threading
from pathlib import Path

tmp = tempfile.mkdtemp()
for t in ["good", "crack", "scratch"]:
    d = Path(tmp) / "pill" / "test" / t
    d.mkdir(parents=True)
    for i in range(3):
        (d / f"00{i}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
os.environ["QC_DATA_DIR"] = tmp

import logging  # noqa: E402

import benchmark as b  # noqa: E402

for h in logging.getLogger().handlers[:]:   # 테스트의 의도적 실패 로그가 실제 benchmark.log에 섞이지 않게
    logging.getLogger().removeHandler(h)
logging.getLogger().addHandler(logging.FileHandler(Path(tmp) / "test.log"))

seen = []
lock = threading.Lock()


def fake_cli(full_prompt, img_dir, timeout):
    paths = re.findall(r"(?:이미지 파일|정상품 사진): (.+)", full_prompt)
    with lock:
        seen.extend(Path(x) for x in paths)
        fail = len(seen) == 2
    for x in paths:
        assert Path(x).exists() and Path(x).parent == Path(img_dir)
        # 라벨 누출 방지: 모델에 넘어가는 경로에 정답 라벨·폴더명이 없어야 한다 (레퍼런스 포함)
        assert not re.search(r"good|crack|scratch|test|train|pill", x), x
    if fail:
        raise RuntimeError("boom")
    return {"verdict": "defective", "defect_type": "crack", "confidence": 0.9}


b._call_claude_cli = fake_cli
samples = b.load_test_set(b.resolve_data_dir(None), ["pill"], 10)
results, errors = b.run_model("claude", b.call_claude, samples, workers=3)
sc = b.score(results)
assert sc["n"] == 9 and errors == 1, sc          # 실패 건도 분모에 남는다
assert sc["error_rate"] == round(1 / 9, 4)
assert sc["miss_rate"] == 0 and sc["false_alarm_rate"] > 0
assert not any(p.exists() for p in seen)          # 임시 복사본 정리됨
assert sc["review_rate"] == round(1 / 9, 4)        # 실패 1건은 재검사행
assert sc["miss_rate_95ci_upper"] == 0.3903        # 누락 0/6이어도 95% 상한은 39%
assert b.wilson_upper(0, 141) < 0.027              # 불량 141장 전수면 상한 2.7%

# 강건성 시험용 열화: 재현 가능(같은 시드)해야 하고 원본과 달라야 한다
from PIL import Image  # noqa: E402

real = Path(tmp) / "real.png"
im = Image.new("RGB", (32, 32), (200, 200, 200))
im.paste((20, 20, 20), (0, 0, 16, 32))  # 경계가 있어야 블러 효과가 보인다
im.save(real)
for name in b.PERTURBATIONS:
    x, y = b.perturbed_copy(str(real), name, tmp), b.perturbed_copy(str(real), name, tmp)
    assert Path(x).read_bytes() == Path(y).read_bytes(), name
    assert Image.open(x).tobytes() != Image.open(real).tobytes(), name
    assert "real" not in Path(x).name

# --resume: 판정 실패 건만 다시 돌리고, 성공한 건은 그대로 두며, 원래 파일은 보존한다
import json  # noqa: E402
import sys  # noqa: E402

out = Path(tmp) / "run.json"
out.write_text(json.dumps({"claude": {"results": results}}), encoding="utf-8")
seen.clear()
sys.argv = ["benchmark.py", "--categories", "pill", "--out", str(out), "--resume"]
b.main()
merged = json.loads(out.read_text(encoding="utf-8"))["claude"]["results"]
assert len(seen) == 1 and len(merged) == 9                       # 실패했던 1건만 재호출
assert sum(r["pred_label"] == b.ERROR for r in merged) == 0
assert (Path(tmp) / "run_before_resume.json").exists()

# 레퍼런스 동봉(CAPA D3): 레퍼런스 경로도 익명화되어 모델에 넘어간다
seen.clear()
ref = Path(tmp) / "pill" / "test" / "good" / "000.png"
b.run_model("claude", b.call_claude, samples[:2], references={"pill": str(ref)})
assert len(seen) == 4, seen                       # 판정 이미지 2 + 레퍼런스 2

# 품질 게이트(CAPA D5): 정상 범위로 보정한 뒤 열화 이미지를 걸러낸다
import quality_gate as q  # noqa: E402

limits = q.calibrate([real])
assert q.check(real, limits)["ok"]
for name in ["noise25", "blur2", "dark60"]:
    res = q.check(b.perturbed_copy(str(real), name, tmp), limits)
    assert not res["ok"], (name, res)

import demo_app as d  # noqa: E402

assert d.disposition({"error": "x"}) == "review"
assert d.disposition({"verdict": "good", "confidence": 0.5}) == "review"
assert d.disposition({"verdict": "good", "confidence": 0.9}) == "pass"
assert d.disposition({"verdict": "defective", "confidence": 0.9}) == "reject"

d.call_claude = lambda p, pr, **kw: {"verdict": "good", "defect_type": None, "confidence": 0.95,
                                     "operator": "hacker", "type": "human_review",  # 감사 필드 덮어쓰기 시도
                                     "_model": "fake", "_cost_usd": 0.01}
d.AUDIT_LOG = Path(tmp) / "audit.jsonl"
d.CACHE = Path(tmp) / "cache"
gate_limits = q.calibrate([real])
q.load_limits = lambda: {"pill": gate_limits}     # 진짜 보정값 대신 테스트 이미지 기준
for img in (Path(tmp) / "pill" / "test").rglob("*.png"):
    Image.open(real).save(img)                    # 데모는 실제로 이미지를 열어 게이트를 잰다
c = d.app.test_client()
H = {"X-QC-Demo": "1"}
s = c.get("/api/sample").get_json()
assert "path" not in s, s                         # 브라우저에 라벨 든 경로 노출 금지
assert c.get(f"/api/image?id={s['id']}").status_code == 200
assert c.get(f"/api/image?id={s['id']}&cond=noise25").status_code == 200
assert c.get(f"/api/image?id={s['id']}&cond=../x").status_code == 400
assert c.get("/api/image?id=999").status_code == 404
assert c.get("/api/image?id=../etc").status_code == 404
assert c.get(f"/api/gate?id={s['id']}").get_json()["ok"]
# 품질 미달(강한 노이즈)은 AI에 넘기지 않고 재촬영 처분 (CAPA D5)
assert c.post(f"/api/inspect?id={s['id']}").status_code == 403    # 헤더 없는 POST(다른 사이트 폼) 거부
g = c.post(f"/api/inspect?id={s['id']}&cond=noise25", headers=H).get_json()
assert g["disposition"] == "recapture" and "noise" in g["gate_failed"] and "model" not in g
d.AUDIT_LOG.unlink()
r = c.post(f"/api/inspect?id={s['id']}", headers=H).get_json()
assert r["operator"] == "AI-VISION-01" and r["type"] == "ai_inspection"   # 모델 출력이 감사 필드를 못 덮는다
assert r["disposition"] == "pass" and len(r["image_sha256"]) == 64 and r["model"] == "fake"
assert "_cost_usd" not in r and "_model" not in r
assert d.AUDIT_LOG.read_text(encoding="utf-8").count("\n") == 1
assert c.get("/").status_code == 200
assert c.get("/api/config").get_json()["threshold"] == b.REVIEW_THRESHOLD

# 사람 재검사: 필수값 검증, 원 기록은 고치지 않고 덧붙임
assert c.post("/api/review", headers=H, json={"ref": r["audit_id"], "decision": "pass", "reviewer": "kim"}).status_code == 400
assert c.post("/api/review", headers=H, json={"ref": "nope", "decision": "pass", "reviewer": "kim", "reason": "x"}).status_code == 404
rv = c.post("/api/review", headers=H, json={"ref": r["audit_id"], "decision": "reject", "reviewer": "kim", "reason": "모서리 결손"})
assert rv.status_code == 200 and rv.get_json()["ai_disposition"] == "pass"
a = c.get("/api/audit").get_json()
assert a["ok"] and a["n"] == 2 and a["records"][0]["type"] == "human_review"

# 해시 체인: 첫 기록을 몰래 고치면 탐지
lines = d.AUDIT_LOG.read_text(encoding="utf-8").splitlines()
d.AUDIT_LOG.write_text("\n".join([lines[0].replace('"pass"', '"reject"', 1)] + lines[1:]) + "\n", encoding="utf-8")
assert d.verify_chain() == {"ok": False, "n": 2, "broken_at": 1}
d.AUDIT_LOG.write_text(lines[0] + "\n{깨진 줄\n", encoding="utf-8")    # 형식이 깨진 줄도 500 없이 손상으로 보고
assert d.verify_chain() == {"ok": False, "n": 2, "broken_at": 2}

# 형식이 깨진 모델 응답(verdict 없음)은 '불량 검출'이 아니라 판정 실패
res, _ = b.run_model("x", lambda p, pr: {"confidence": 0.9}, samples[:1])
assert res[0]["pred_label"] == b.ERROR

# 분석: 기준 경계값과 실험 유효성 (일탈 D7 회귀 방지)
import analyze as an  # noqa: E402

edge = {"true_label": "crack", "pred_label": "good", "confidence": 0.7}
assert an.leaked(edge, an.V3_TH) and not an.leaked(edge, an.V4_TH)   # v3는 '< 0.7'이라 0.7은 유출, v4는 아님
ok_run = {f"i{i}": {"pred_label": "good"} for i in range(100)}
bad_run = {f"j{i}": {"pred_label": b.ERROR if i < 5 else "good"} for i in range(100)}
assert not an.valid({"ok": True, "result": ""}, ok_run).get("invalid")
v = an.valid({"ok": True, "result": ""}, ok_run, ok_run, ok_run, bad_run)   # 합치면 1.25%지만 bad_run 단독 5%
assert v["invalid"] and not v["ok"]

# v5 로컬 엔진: 기준값은 보정 절반에서만 정한다 (홀드아웃 점수가 바뀌어도 기준값 불변)
names = [f"pill/test/{lab}/{i:03d}.png" for lab in ("good", "crack") for i in range(40)]
rs5 = [{"image": n, "category": "pill", "true_label": n.split("/")[2], "score": 1.0 if "good" in n else 3.0} for n in names]
th5 = an.v5_thresholds(rs5)
assert th5 == {"pill": {"pass": 0.95 * 3.0, "reject": 1.05 * 1.0}}
for r in rs5:
    if an.holdout(r["image"]):
        r["score"] = 99.0
assert an.v5_thresholds(rs5) == th5
assert [an.v5_disposition({"category": "pill", "score": x}, {"pill": {"pass": 1, "reject": 2}}) for x in (0.5, 1.5, 2.5)] == ["pass", "review", "reject"]

import torch  # noqa: E402

fake = type("M", (), {"weights_sha256": "0" * 64, "score": lambda self, c, p: (1.5, torch.full((28, 28), 1.5))})()
d.AUDIT_LOG.unlink()                              # 위에서 일부러 깨뜨린 로그 정리
d.local_engine = lambda: {"model": fake, "th": {"pill": {"pass": 1.0, "reject": 2.0}}, "disposition": an.v5_disposition}
assert c.post(f"/api/inspect?id={s['id']}&engine=x", headers=H).status_code == 400
r = c.post(f"/api/inspect?id={s['id']}&engine=local", headers=H).get_json()
assert r["disposition"] == "review" and r["network"] == "미사용" and r["heatmap"].startswith("data:image/png")
assert "heatmap" not in d.AUDIT_LOG.read_text(encoding="utf-8").splitlines()[-1]   # 화면용 이미지는 감사기록에 안 넣음

# 망분리 모드: 차단 후 외부 연결 시도는 실패해야 한다 (전역 소켓을 바꾸므로 맨 마지막)
import socket  # noqa: E402

import patchcore  # noqa: E402

patchcore.block_network()
try:
    socket.create_connection(("example.com", 80), timeout=1)
    raise AssertionError("차단 실패")
except ConnectionError:
    pass
print("OK")

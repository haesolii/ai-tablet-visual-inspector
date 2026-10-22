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

import benchmark as b  # noqa: E402

seen = []
lock = threading.Lock()


def fake_cli(img, prompt, timeout):
    with lock:
        seen.append(img)
        fail = len(seen) == 2
    assert img.exists()
    # 라벨 누출 방지: 모델에 넘어가는 경로·프롬프트에 정답 라벨이 없어야 한다
    assert not re.search(r"good|crack|scratch|test", prompt.format(image_path=str(img)).split("이미지 파일:")[1].split("\n")[0])
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

import demo_app as d  # noqa: E402

assert d.disposition({"error": "x"}) == "review"
assert d.disposition({"verdict": "good", "confidence": 0.5}) == "review"
assert d.disposition({"verdict": "good", "confidence": 0.9}) == "pass"
assert d.disposition({"verdict": "defective", "confidence": 0.9}) == "reject"

d.call_claude = lambda p, pr: {"verdict": "good", "defect_type": None, "confidence": 0.95}
d.AUDIT_LOG = Path(tmp) / "audit.jsonl"
c = d.app.test_client()
s = c.get("/api/sample").get_json()
assert "path" not in s, s                         # 브라우저에 라벨 든 경로 노출 금지
assert c.get(f"/api/image?id={s['id']}").status_code == 200
assert c.get("/api/image?id=999").status_code == 404
assert c.get("/api/image?id=../etc").status_code == 404
r = c.get(f"/api/inspect?id={s['id']}").get_json()
assert r["disposition"] == "pass" and len(r["image_sha256"]) == 64
assert d.AUDIT_LOG.read_text(encoding="utf-8").count("\n") == 1
assert c.get("/").status_code == 200
print("OK")

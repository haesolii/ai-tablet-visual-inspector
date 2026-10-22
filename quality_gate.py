"""이미지 품질 게이트 (CAPA D5): AI 판정 전에 노이즈·블러·조명 이상을 걸러 재촬영으로 돌린다.

지표 3개 (흑백 변환 후, Pillow만 사용):
- noise : 원본과 3x3 중앙값 필터의 평균 차이. 센서 노이즈면 커지고, 블러면 지나치게 작아진다
- sharp : 중앙값 필터 후 윤곽선 강도의 표준편차
- bright: 평균 밝기
허용 범위 = 카테고리별 정상 학습 이미지(train/good)의 최소~최대 ± MARGIN. test 이미지는 보정에 쓰지 않는다.

    python quality_gate.py              # MVTec train/good으로 보정 → results/quality_gate_limits.json
    python quality_gate.py --evaluate   # test 원본·열화본 차단율 → results/v4_e3_gate.json
"""
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter, ImageStat

LIMITS_FILE = Path(__file__).parent / "results" / "quality_gate_limits.json"
MARGIN = 0.25  # 카메라·조명 고정 라인 가정. 실제 라인에선 설비 적격성 평가(OQ/PQ) 데이터로 다시 정한다
METRICS = ("noise", "sharp", "bright")


def measure(path) -> dict:
    g = Image.open(path).convert("L")
    med = g.filter(ImageFilter.MedianFilter(3))
    return {
        "noise": ImageStat.Stat(ImageChops.difference(g, med)).mean[0],
        "sharp": ImageStat.Stat(med.filter(ImageFilter.FIND_EDGES)).stddev[0],
        "bright": ImageStat.Stat(g).mean[0],
    }


def calibrate(paths) -> dict:
    ms = [measure(p) for p in paths]
    return {k: [min(m[k] for m in ms) * (1 - MARGIN), max(m[k] for m in ms) * (1 + MARGIN)] for k in METRICS}


def check(path, limits: dict) -> dict:
    """반환: {"ok": bool, "metrics": {...}, "failed": ["noise", ...]}"""
    m = measure(path)
    failed = [k for k in METRICS if not limits[k][0] <= m[k] <= limits[k][1]]
    return {"ok": not failed, "metrics": {k: round(v, 3) for k, v in m.items()}, "failed": failed}


def load_limits() -> dict:
    return json.loads(LIMITS_FILE.read_text(encoding="utf-8")) if LIMITS_FILE.exists() else {}


def evaluate(data, limits: dict) -> dict:
    """v4 E3: test 이미지(보정에 안 쓴 것)의 원본·열화본 차단율"""
    import tempfile
    from benchmark import PERTURBATIONS, find_category_dir, perturbed_copy

    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for cat in limits:
            for img in sorted(find_category_dir(data, cat).rglob("*.png")):
                for cond in ["clean", *PERTURBATIONS]:
                    path = img if cond == "clean" else perturbed_copy(str(img), cond, tmp)
                    out.setdefault(cond, []).append(not check(path, limits[cat])["ok"])
                    if cond != "clean":
                        Path(path).unlink()
    return {c: {"blocked": sum(v), "n": len(v)} for c, v in out.items()}


if __name__ == "__main__":
    import sys
    from benchmark import find_category_dir, resolve_data_dir

    data = resolve_data_dir(None)
    if "--evaluate" in sys.argv:
        res = evaluate(data, load_limits())
        (LIMITS_FILE.parent / "v4_e3_gate.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(res)
        sys.exit()
    limits = {}
    for cat in ["pill", "capsule"]:
        train = sorted((find_category_dir(data, cat).parent / "train" / "good").glob("*.png"))
        limits[cat] = calibrate(train)
        print(cat, len(train), {k: [round(x, 2) for x in v] for k, v in limits[cat].items()})
    LIMITS_FILE.write_text(json.dumps(limits, indent=2), encoding="utf-8")
    print(f"저장: {LIMITS_FILE}")

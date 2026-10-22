"""
정제 외관검사 비전 AI 벤치마크 하네스
- 데이터: MVTec AD pill/capsule (유형 분류), 외부 2분류 데이터셋 (독립 검증)
- 시험: 기본 / 강건성(--perturb) / 특이성 확대(--splits train) / 레퍼런스 동봉(--reference)
- 모델: claude (구독 인증 CLI). gpt/gemini는 인터페이스만 있는 스텁
- 정답 누출 방지: 모델에는 무작위 파일명 복사본만 넘긴다 (v1 일탈의 CAPA)

사용법 (README 참고):
    python benchmark.py --categories pill --per-type 2
    python benchmark.py --per-type 1000 --workers 4 --out results/run.json
"""
import argparse
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = Path(__file__).parent
LOG_FILE = BASE / "benchmark.log"

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("benchmark")

PROMPT = """당신은 제약 품질관리(QC) 외관 검사 담당자입니다.
아래 정제 이미지를 보고 판정하세요.

이미지 파일: {image_path}

반드시 JSON 객체 하나만 답하세요. 코드블록이나 설명은 금지합니다:
{{"verdict": "good" 또는 "defective", "defect_type": "<유형명 또는 null>", "confidence": 0.0~1.0}}

불량 유형은 아래 후보 안에서만 고르세요:
color, contamination, crack, faulty_imprint, pill_type, scratch, combined, poke, squeeze
정상이면 verdict는 "good", defect_type은 null."""
# CAPA D3: 정상품 레퍼런스 동봉 프롬프트. MVTec pill은 정상품에도 붉은 반점이 있어 기준 없이 보면 오염으로 오인한다
PROMPT_REF = PROMPT.replace("이미지 파일: {image_path}\n", """이미지 파일: {image_path}

비교 기준: 아래는 같은 제품의 '정상품' 사진입니다. 정상품에도 있는 무늬·반점·인쇄·색조는 불량이 아닙니다.
정상품과 달라진 부분만 불량으로 판단하세요.
정상품 사진: {reference_path}
""")
assert PROMPT_REF != PROMPT


def prompt_version(prompt: str) -> str:
    """감사추적·변경관리용: 프롬프트가 한 글자라도 바뀌면 값이 바뀐다"""
    return hashlib.sha256(prompt.encode()).hexdigest()[:8]


PROMPT_VERSION = prompt_version(PROMPT)


def resolve_data_dir(cli_data: str | None) -> Path | None:
    """--data → QC_DATA_DIR → ./data → kagglehub 캐시 순으로 탐색"""
    candidates = []
    if cli_data:
        candidates.append(Path(cli_data))
    env_dir = os.environ.get("QC_DATA_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(BASE / "data")
    cache = Path.home() / ".cache/kagglehub/datasets/ipythonx/mvtec-ad"
    if cache.exists():
        candidates.extend(sorted(cache.glob("versions/*"), reverse=True))
        candidates.append(cache)  # 압축 해제 전 구조 대비
    for c in candidates:
        if c and c.exists():
            return c
    return None


def find_category_dir(data_dir: Path, category: str) -> Path | None:
    """<root>/pill/test 또는 <root>/*/pill/test 구조를 유연하게 탐색"""
    for depth_root in [data_dir, *data_dir.iterdir()]:
        if not depth_root.is_dir():
            continue
        cand = depth_root / category / "test"
        if cand.is_dir():
            return cand
        # kaggle 데이터셋이 'mvtec_ad/pill/...' 같은 중간 폴더를 가질 때
        for sub in depth_root.iterdir():
            if sub.is_dir():
                cand2 = sub / category / "test"
                if cand2.is_dir():
                    return cand2
    return None


def load_test_set(data_dir: Path, categories: list[str], per_type_max: int = 10,
                   seed: int = 42, splits: tuple[str, ...] = ("test",)) -> list[dict]:
    """층화 샘플링. 라벨=디렉토리명. 학습을 안 하므로 train/(정상만 있음)도 평가에 쓸 수 있다."""
    rng = random.Random(seed)
    samples = []
    for cat in categories:
        test_dir = find_category_dir(data_dir, cat)
        if test_dir is None:
            print(f"[skip] {cat}: test 디렉토리 없음")
            continue
        for split in splits:
            for defect_dir in sorted((test_dir.parent / split).iterdir()):
                if not defect_dir.is_dir():
                    continue
                label = "good" if defect_dir.name == "good" else defect_dir.name
                imgs = sorted(defect_dir.glob("*.png"))
                picked = rng.sample(imgs, min(per_type_max, len(imgs)))
                for img in picked:
                    samples.append({"category": cat, "label": label, "path": str(img), "split": split})
    rng.shuffle(samples)
    return samples


def _white_noise(im, sigma: float, rng: random.Random):
    """가우시안 백색잡음 (0~255 스케일 표준편차 sigma): 센서 노이즈·저가 카메라 모사"""
    from PIL import Image, ImageChops
    n = im.size[0] * im.size[1] * 3
    noise = bytes(min(255, max(0, int(128 + rng.gauss(0, sigma)))) for _ in range(n))  # 평균 128
    return ImageChops.add(im, Image.frombytes("RGB", im.size, noise), offset=-128)


def _perturbations(rng: random.Random):
    from PIL import ImageEnhance, ImageFilter
    return {
        "noise10": lambda im: _white_noise(im, 10, rng),   # 약한 센서 노이즈
        "noise25": lambda im: _white_noise(im, 25, rng),   # 강한 센서 노이즈
        "blur2": lambda im: im.filter(ImageFilter.GaussianBlur(2)),  # 초점 이탈·진동
        "dark60": lambda im: ImageEnhance.Brightness(im).enhance(0.6),  # 조명 열화
    }


PERTURBATIONS = ["noise10", "noise25", "blur2", "dark60"]


def perturbed_copy(path: str, name: str, out_dir: str) -> str:
    """강건성 시험용 열화 이미지. 재현성을 위해 이미지 경로로 시드를 고정한다."""
    # ponytail: 시드가 절대 경로 기준이라 다른 PC에선 노이즈 패턴이 달라진다(통계적으로는 동등).
    # 기기 간 완전 재현이 필요하면 이미지 바이트 해시로 바꾸되, 그러면 v3 결과와 패턴이 달라진다
    from PIL import Image
    rng = random.Random(hashlib.md5(f"{Path(path).as_posix()}{name}".encode()).hexdigest())
    im = Image.open(path).convert("RGB")
    out = Path(out_dir) / f"{uuid.uuid4().hex}.png"
    _perturbations(rng)[name](im).save(out)
    return str(out)


IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}
# 정상/불량 2분류 외부 데이터셋 (kagglehub 캐시). 학습을 안 하므로 모든 split을 평가에 쓴다.
# 값: (kaggle 핸들, {폴더명: 라벨})
EXTERNAL = {
    "pharmavision": ("tahirlee/pharmaceutical-tablets-dataset", {"normal": "good", "defective": "defective"}),
    "pilldefect": ("pudpawat/pill-defect-dataset", {"normal": "good", "defect all": "defective"}),
}


def load_external(name: str, per_type_max: int, seed: int) -> list[dict]:
    """외부 데이터셋 층화 샘플링. category=데이터셋 이름, split=라벨 폴더의 상위 폴더명"""
    import kagglehub
    handle, labels = EXTERNAL[name]
    root = Path(kagglehub.dataset_download(handle))
    rng = random.Random(seed)
    samples = []
    for d in sorted(p for p in root.rglob("*") if p.is_dir() and p.name in labels):
        imgs = sorted(f for f in d.iterdir() if f.suffix.lower() in IMG_EXT)
        for img in rng.sample(imgs, min(per_type_max, len(imgs))):
            samples.append({"category": name, "label": labels[d.name], "path": str(img),
                            "split": d.parent.name.replace(" ", "_")})
    rng.shuffle(samples)
    return samples


def reference_images(data_dir: Path, categories: list[str]) -> dict:
    """카테고리별 정상품 레퍼런스: train/good 첫 장 (평가 표본에서는 제외한다)"""
    return {c: str(sorted((find_category_dir(data_dir, c).parent / "train" / "good").glob("*.png"))[0])
            for c in categories}


CLAUDE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["good", "defective"]},
        "defect_type": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["verdict", "defect_type", "confidence"],
}


def extract_json(text: str) -> dict:
    """텍스트에서 JSON 객체 추출 (코드블록·잡담 혼합 대응)"""
    # 코드블록 안 JSON 우선
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    # 가장 바깥 중괄호 블록
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError("JSON을 찾을 수 없음")


def claude_bin() -> str:
    """Windows npm 설치는 claude.cmd 셸 스크립트라 여러 줄 프롬프트 인자가 cmd.exe에서 잘린다.
    옆에 있는 실제 claude.exe를 직접 호출한다."""
    found = shutil.which("claude") or "claude"
    exe = Path(found).parent / "node_modules/@anthropic-ai/claude-code/bin/claude.exe"
    return str(exe) if found.lower().endswith(".cmd") and exe.exists() else found


def call_claude(image_path: str, prompt: str, reference: str | None = None, timeout: int = 180) -> dict:
    """구독 인증된 claude CLI로 비전 판정. JSON dict 반환 (+ _model, _cost_usd)."""
    # 라벨 누출 방지: MVTec 경로(.../test/crack/003.png)에 정답이 들어 있으므로
    # 무작위 이름의 임시 파일로 복사해 모델이 경로를 보고 답할 수 없게 한다 (레퍼런스도 동일)
    with tempfile.TemporaryDirectory() as tmp:
        def anon(src):
            dst = Path(tmp) / f"{uuid.uuid4().hex}{Path(src).suffix}"
            shutil.copy(src, dst)
            return str(dst)
        full_prompt = prompt.format(image_path=anon(image_path),
                                    reference_path=anon(reference) if reference else "")
        return _call_claude_cli(full_prompt, tmp, timeout)


def _call_claude_cli(full_prompt: str, img_dir: str, timeout: int) -> dict:
    cmd = [
        claude_bin(), "-p", full_prompt,
        "--output-format", "json",
        "--json-schema", json.dumps(CLAUDE_SCHEMA),
        "--tools", "Read",
        "--allowedTools", "Read",
        "--add-dir", img_dir,
        "--permission-mode", "dontAsk",
        "--no-session-persistence",
        "--strict-mcp-config",
    ]
    # API 키를 제거해 구독(OAuth) 인증만 사용 — 과금 방지
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          env=env, stdin=subprocess.DEVNULL, cwd=img_dir)  # 프로젝트 폴더(결과 파일)를 작업 경로로 주지 않는다
    if proc.returncode != 0:  # 사용량 한도 등은 stderr가 비고 stdout JSON에만 나온다
        raise RuntimeError(f"claude CLI 실패 ({proc.returncode}): {(proc.stderr or proc.stdout)[-500:]}")
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"CLI 출력 파싱 실패: {proc.stdout[-300:]}")
    # structured_output 우선, 없으면 result 텍스트에서 JSON 추출
    pred = out["structured_output"] if isinstance(out.get("structured_output"), dict) \
        else extract_json(out.get("result", ""))
    # 감사추적: 실제로 응답한 모델 ID와 API 환산 비용 (구독 인증이라 실제 과금은 없음)
    return pred | {"_model": ",".join(out.get("modelUsage") or {}) or None,
                   "_cost_usd": out.get("total_cost_usd")}


def call_gpt(image_path: str, prompt: str, reference: str | None = None) -> dict:
    """TODO: Codex 계정 인증 후 구현"""
    raise NotImplementedError


def call_gemini(image_path: str, prompt: str, reference: str | None = None) -> dict:
    """TODO: Gemini 계정 인증 후 구현"""
    raise NotImplementedError


MODELS = {
    "claude": call_claude,
    "gpt": call_gpt,
    "gemini": call_gemini,
}


ERROR = "error"  # 판정 실패 → 합격 처리하지 않고 재검사로 보낸다 (fail-safe)
# ---- 운용 설정: 바꾸면 변경관리 대상 (PROJECT_PLAN §4·§7 기준으로 승인된 값만) ----
# 이 신뢰도 미만이면 AI가 확정하지 않고 사람 재검사로 넘긴다.
# 변경 이력: 0.7 → 0.75 (CAPA D1. v3 유출 14건 전부 신뢰도 0.7 경계값.
#            v4 B1·B2 적합 — 처음 쓰는 외부 데이터 995장에서 AI 단독 유출 0, AI 단독 오탐 0)
REVIEW_THRESHOLD = 0.75
# 정상품 레퍼런스 동봉 프롬프트 사용 여부 (CAPA D3).
# v4 B3 부적합(정상 오탐 10.9% → 53.6%)으로 채택하지 않음. 일탈 D6 참고
USE_REFERENCE = False


def wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    """비율 k/n의 95% 신뢰구간 상한. 0/n이어도 '0%'가 아니라 '최대 x%'로 말하기 위함."""
    if n == 0:
        return 0.0
    p = k / n
    return (p + z * z / (2 * n) + z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def needs_review(r: dict) -> bool:
    return r["pred_label"] == ERROR or (r.get("confidence") or 0) < REVIEW_THRESHOLD


def score(results: list[dict]) -> dict:
    """정확도 / 불량 누락률 / 정상 오탐율 / 판정 실패율 / 유형별 정답률 집계.
    판정 실패(ERROR)도 분모에 남겨 숨기지 않는다: 정확도에선 오답, 누락에선 제외(재검사행)."""
    total = len(results)
    correct = sum(1 for r in results if r["pred_label"] == r["true_label"])
    missed = sum(1 for r in results if r["true_label"] != "good" and r["pred_label"] == "good")
    false_alarm = sum(1 for r in results
                      if r["true_label"] == "good" and r["pred_label"] not in ("good", ERROR))
    errors = sum(1 for r in results if r["pred_label"] == ERROR)
    n_defective = sum(1 for r in results if r["true_label"] != "good")
    n_good = total - n_defective
    per_type = {}
    for r in results:
        t = per_type.setdefault(r["true_label"], [0, 0])
        t[0] += r["pred_label"] == r["true_label"]
        t[1] += 1
    # 2단계 판정: 저신뢰도·실패는 사람 재검사 → AI 단독 합격 처리된 불량만이 '유출'
    reviewed = sum(1 for r in results if needs_review(r))
    auto_missed = sum(1 for r in results
                      if r["true_label"] != "good" and r["pred_label"] == "good" and not needs_review(r))
    lat = [r["latency_s"] for r in results if r.get("latency_s") is not None]
    return {
        "n": total,
        "accuracy": round(correct / total, 4) if total else 0,
        "miss_rate": round(missed / n_defective, 4) if n_defective else 0,
        "miss_rate_95ci_upper": round(wilson_upper(missed, n_defective), 4),
        "review_threshold": REVIEW_THRESHOLD,
        "review_rate": round(reviewed / total, 4) if total else 0,
        "auto_pass_miss_rate": round(auto_missed / n_defective, 4) if n_defective else 0,
        "avg_latency_s": round(sum(lat) / len(lat), 1) if lat else None,
        "false_alarm_rate": round(false_alarm / n_good, 4) if n_good else 0,
        "error_rate": round(errors / total, 4) if total else 0,
        "per_type": {k: f"{c}/{n}" for k, (c, n) in sorted(per_type.items())},
    }


def image_id(s: dict) -> str:
    return f"{s['category']}/{s.get('split', 'test')}/{s['label']}/{Path(s['path']).name}"


def run_model(name: str, fn, samples: list[dict], workers: int = 1,
              perturb: str | None = None, references: dict | None = None) -> tuple[list[dict], int]:
    """판정 실패는 ERROR로 결과에 남긴다. 반환: (결과, 실패 건수)"""
    def one(i_s):
        i, s = i_s
        print(f"  [{i}/{len(samples)}] {s['category']}/{s['label']}/{Path(s['path']).name}")
        t0 = time.monotonic()
        try:
            ref = (references or {}).get(s["category"])
            kw = {"reference": ref} if ref else {}
            prompt = PROMPT_REF if ref else PROMPT
            if perturb:
                with tempfile.TemporaryDirectory() as tmp:
                    pred = fn(perturbed_copy(s["path"], perturb, tmp), prompt, **kw)
            else:
                pred = fn(s["path"], prompt, **kw)
            verdict = pred.get("verdict")
            if verdict not in ("good", "defective"):  # 형식이 깨진 응답을 '불량 검출'로 세지 않는다
                raise ValueError(f"verdict 없음/형식 오류: {verdict!r}")
            pred_label = "good" if verdict == "good" else (pred.get("defect_type") or "defective")
        except NotImplementedError:
            raise
        except Exception as e:
            log.warning("ERROR %s (%s/%s): %s", s["path"], s["category"], s["label"], e)
            pred, pred_label = {}, ERROR
        return {"true_label": s["label"], "pred_label": pred_label,
                "category": s["category"], "image": image_id(s),
                "confidence": pred.get("confidence"),
                "model": pred.get("_model"), "cost_usd": pred.get("_cost_usd"),
                "latency_s": round(time.monotonic() - t0, 1)}

    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(one, enumerate(samples, 1)))
    return results, sum(r["pred_label"] == ERROR for r in results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["claude"], choices=list(MODELS))
    ap.add_argument("--categories", nargs="+", default=["pill", "capsule"])
    ap.add_argument("--per-type", type=int, default=10)
    ap.add_argument("--data", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=1, help="동시 호출 수 (구독 사용량 한도 주의)")
    ap.add_argument("--splits", nargs="+", default=["test"], choices=["test", "train"],
                    help="train = MVTec 학습용 정상 이미지 (특이성 표본 확대용)")
    ap.add_argument("--perturb", choices=PERTURBATIONS, default=None, help="강건성 시험용 이미지 열화")
    ap.add_argument("--dataset", choices=["mvtec", *EXTERNAL], default="mvtec",
                    help="외부 데이터셋은 정상/불량 2분류로만 채점 (독립 검증용)")
    ap.add_argument("--reference", action="store_true", help="정상품 레퍼런스 이미지 동봉 (MVTec만)")
    ap.add_argument("--out", default=str(BASE / "benchmark_results.json"))
    ap.add_argument("--resume", action="store_true",
                    help="--out 파일의 판정 실패(error) 건만 다시 돌려 합친다 (일시적 실패 재시험용)")
    args = ap.parse_args()

    data_dir = resolve_data_dir(args.data)
    if data_dir is None:
        print("[!] MVTec AD 데이터를 찾지 못했습니다. README의 다운로드 안내를 따르거나 --data로 경로를 지정하세요.")
        return

    if args.dataset == "mvtec":
        samples = load_test_set(data_dir, args.categories, args.per_type, args.seed, tuple(args.splits))
    else:
        samples = load_external(args.dataset, args.per_type, args.seed)
    refs = reference_images(data_dir, args.categories) if args.reference else None
    if refs:  # 레퍼런스 이미지 자체는 평가하지 않는다
        samples = [s for s in samples if s["path"] not in refs.values()]
    prev = {}
    if args.resume:  # 성공한 판정은 그대로 두고 실패 건만 재시험
        out = Path(args.out)
        old = json.loads(out.read_text(encoding="utf-8"))["claude"]
        for k, v in {"dataset": args.dataset, "perturb": args.perturb,
                     "prompt_version": prompt_version(PROMPT_REF if refs else PROMPT)}.items():
            if k in old.get("meta", {}) and old["meta"][k] != v:
                raise SystemExit(f"[!] --resume 조건 불일치: 기존 {k}={old['meta'][k]!r}, 지금 {v!r}")
        prev = {r["image"]: r for r in old["results"]}
        samples = [s for s in samples if prev.get(image_id(s), {}).get("pred_label", ERROR) == ERROR]
        backup = out.with_name(out.stem + "_before_resume.json")
        if samples and not backup.exists():  # 최초 실패 기록(일탈 증거)은 덮어쓰지 않는다
            backup.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"데이터: {data_dir} / 샘플 {len(samples)}장")
    if not samples:
        return

    all_results = {}
    for name in args.models:
        print(f"=== {name} ===")
        fn = MODELS[name]
        try:
            results, skipped = run_model(name, fn, samples, args.workers, args.perturb, refs)
            if prev:
                results = list((prev | {r["image"]: r for r in results}).values())
                skipped = sum(r["pred_label"] == ERROR for r in results)
        except NotImplementedError:
            print(f"[{name}] 미구현 — 계정 인증 후 실행")
            continue
        sc = score(results)
        sc["skipped"] = skipped
        all_results[name] = {"meta": {"prompt_version": prompt_version(PROMPT_REF if refs else PROMPT),
                                      "seed": args.seed, "per_type": args.per_type, "dataset": args.dataset,
                                      "splits": args.splits if args.dataset == "mvtec" else None,
                                      "perturb": args.perturb,
                                      "reference": refs and {c: Path(p).name for c, p in refs.items()}},
                             "score": sc, "results": results}
        print(f"[{name}] {sc} (판정 실패 {skipped})")

    out = Path(args.out)
    out.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()

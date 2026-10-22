"""
3-모델 비전 벤치마크 하네스
- MVTec AD pill/capsule 테스트셋 로드 → 모델별 분류 → 채점
- 모델 호출: claude(구현됨, 구독 인증 CLI), gpt/gemini(스텁, 계정 인증 후 구현)

사용법:
    python benchmark.py --models claude --categories pill --per-type 2
    python benchmark.py --data /path/to/mvtec --per-type 10
"""
import argparse
import json
import logging
import os
import random
import re
import subprocess
from pathlib import Path

BASE = Path(__file__).parent
LOG_FILE = BASE / "benchmark.log"

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("benchmark")

CATEGORIES = ["pill", "capsule"]

PROMPT = """당신은 제약 품질관리(QC) 외관 검사 담당자입니다.
아래 정제 이미지를 보고 판정하세요.

이미지 파일: {image_path}

반드시 JSON 객체 하나만 답하세요. 코드블록이나 설명은 금지합니다:
{{"verdict": "good" 또는 "defective", "defect_type": "<유형명 또는 null>", "confidence": 0.0~1.0}}

불량 유형은 아래 후보 안에서만 고르세요:
color, contamination, crack, faulty_imprint, pill_type, scratch, combined, poke, squeeze
정상이면 verdict는 "good", defect_type은 null."""


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
                   seed: int = 42) -> list[dict]:
    """test/ 디렉토리에서 층화 샘플링. 라벨=디렉토리명."""
    rng = random.Random(seed)
    samples = []
    for cat in categories:
        test_dir = find_category_dir(data_dir, cat)
        if test_dir is None:
            print(f"[skip] {cat}: test 디렉토리 없음")
            continue
        for defect_dir in sorted(test_dir.iterdir()):
            if not defect_dir.is_dir():
                continue
            label = "good" if defect_dir.name == "good" else defect_dir.name
            imgs = sorted(defect_dir.glob("*.png"))
            picked = rng.sample(imgs, min(per_type_max, len(imgs)))
            for img in picked:
                samples.append({"category": cat, "label": label, "path": str(img)})
    rng.shuffle(samples)
    return samples


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


def call_claude(image_path: str, prompt: str, timeout: int = 180) -> dict:
    """구독 인증된 claude CLI로 비전 판정. JSON dict 반환."""
    img = Path(image_path).resolve()
    full_prompt = prompt.format(image_path=str(img))
    cmd = [
        "claude", "-p", full_prompt,
        "--output-format", "json",
        "--json-schema", json.dumps(CLAUDE_SCHEMA),
        "--tools", "Read",
        "--allowedTools", "Read",
        "--add-dir", str(img.parent),
        "--permission-mode", "dontAsk",
        "--no-session-persistence",
        "--strict-mcp-config",
    ]
    # API 키를 제거해 구독(OAuth) 인증만 사용 — 과금 방지
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          env=env, stdin=subprocess.DEVNULL)
    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI 실패: {proc.stderr[-500:]}")
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"CLI 출력 파싱 실패: {proc.stdout[-300:]}")
    # structured_output 우선, 없으면 result 텍스트에서 JSON 추출
    if isinstance(out.get("structured_output"), dict):
        return out["structured_output"]
    return extract_json(out.get("result", ""))


def call_gpt(image_path: str, prompt: str) -> dict:
    """TODO: Codex 계정 인증 후 구현"""
    raise NotImplementedError


def call_gemini(image_path: str, prompt: str) -> dict:
    """TODO: Gemini 계정 인증 후 구현"""
    raise NotImplementedError


MODELS = {
    "claude": call_claude,
    "gpt": call_gpt,
    "gemini": call_gemini,
}


def score(results: list[dict]) -> dict:
    """정확도 / 불량 누락률 / 정상 오탐율 집계"""
    total = len(results)
    correct = sum(1 for r in results if r["pred_label"] == r["true_label"])
    missed = sum(1 for r in results if r["true_label"] != "good" and r["pred_label"] == "good")
    false_alarm = sum(1 for r in results if r["true_label"] == "good" and r["pred_label"] != "good")
    n_defective = sum(1 for r in results if r["true_label"] != "good")
    n_good = total - n_defective
    return {
        "n": total,
        "accuracy": round(correct / total, 4) if total else 0,
        "miss_rate": round(missed / n_defective, 4) if n_defective else 0,
        "false_alarm_rate": round(false_alarm / n_good, 4) if n_good else 0,
    }


def run_model(name: str, fn, samples: list[dict]) -> tuple[list[dict], int]:
    results, skipped = [], 0
    for i, s in enumerate(samples, 1):
        print(f"  [{i}/{len(samples)}] {s['category']}/{s['label']}/{Path(s['path']).name}")
        try:
            pred = fn(s["path"], PROMPT)
            verdict = pred.get("verdict")
            pred_label = "good" if verdict == "good" else (pred.get("defect_type") or "defective")
        except NotImplementedError:
            raise
        except Exception as e:
            log.warning("SKIP %s (%s/%s): %s", s["path"], s["category"], s["label"], e)
            skipped += 1
            continue
        results.append({"true_label": s["label"], "pred_label": pred_label,
                        "category": s["category"], "path": s["path"],
                        "confidence": pred.get("confidence")})
    return results, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["claude"], choices=list(MODELS))
    ap.add_argument("--categories", nargs="+", default=["pill", "capsule"])
    ap.add_argument("--per-type", type=int, default=10)
    ap.add_argument("--data", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(BASE / "benchmark_results.json"))
    args = ap.parse_args()

    data_dir = resolve_data_dir(args.data)
    if data_dir is None:
        print("[!] 데이터 디렉토리를 찾지 못했습니다.")
        print("    다운로드가 끝나면 다음 중 하나로 연결하세요:")
        print("      python benchmark.py --data ~/.cache/kagglehub/datasets/ipythonx/mvtec-ad/versions/2")
        print("    또는: ln -s <캐시>/versions/2 ~/workspace/sideproject-qc-vision/data")
        return

    samples = load_test_set(data_dir, args.categories, args.per_type, args.seed)
    print(f"데이터: {data_dir} / 샘플 {len(samples)}장")
    if not samples:
        return

    all_results = {}
    for name in args.models:
        print(f"=== {name} ===")
        fn = MODELS[name]
        try:
            results, skipped = run_model(name, fn, samples)
        except NotImplementedError:
            print(f"[{name}] 미구현 — 계정 인증 후 실행")
            continue
        sc = score(results)
        sc["skipped"] = skipped
        all_results[name] = {"score": sc, "results": results}
        print(f"[{name}] {sc} (건너뜀 {skipped})")

    out = Path(args.out)
    out.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    print(f"저장: {out}")


if __name__ == "__main__":
    main()

"""온프레미스 판정: PatchCore 방식 이상탐지 (PROJECT_PLAN §8 v5, §9 v6)
이미지가 PC 밖으로 나가지 않는다. 학습 없이 ImageNet 특징 + 정상품 패치 기억으로 점수를 매긴다.

    python patchcore.py                                         # v6: coreset, train 정상 분할 → results/v6_run1.json
    python patchcore.py --offline --out results/v6_run2.json    # 네트워크 차단 상태 재실행 (F5·F6)
    python patchcore.py --method random --out results/v6_random.json   # 참고: v5 보관 방식을 같은 분할로
    python patchcore.py --protocol v5 --method random --out results/v5_run1.json   # v5 재현
"""
import argparse
import hashlib
import json
import socket
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

import benchmark as b

torch.use_deterministic_algorithms(True)
torch.set_num_threads(4)   # 스레드 수가 바뀌면 부동소수 합산 순서가 바뀔 수 있어 고정 (C5)
KEEP = 0.1                 # 정상 패치 10% 보관 (v5: 무작위, v6: greedy coreset)
CACHE = Path(__file__).parent / ".cache"
TF = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def block_network():
    """C6: 이후 모든 외부 연결 시도를 예외로 만든다 (기밀 이미지가 밖으로 못 나감을 코드로 확인)"""
    def deny(*a, **kw):
        raise ConnectionError("네트워크 차단됨 (온프레미스 모드)")
    socket.socket.connect = deny
    socket.create_connection = deny


class PatchCore:
    def __init__(self):
        w = Wide_ResNet50_2_Weights.IMAGENET1K_V1
        self.net = wide_resnet50_2(weights=w).eval()
        ckpt = Path(torch.hub.get_dir()) / "checkpoints" / Path(w.url).name
        self.weights_sha256 = hashlib.sha256(ckpt.read_bytes()).hexdigest()   # 모델 버전 고정 증거
        self.banks = {}

    @torch.no_grad()
    def features(self, paths: list[str]) -> torch.Tensor:
        """layer2·layer3 특징을 28×28 격자로 맞춰 이어붙임 → (B, 1536, 28, 28)"""
        n = self.net
        x = torch.stack([TF(Image.open(p).convert("RGB")) for p in paths])
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        l2 = n.layer2(n.layer1(x))
        l3 = n.layer3(l2)
        l3 = F.interpolate(F.avg_pool2d(l3, 3, 1, 1), size=l2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([F.avg_pool2d(l2, 3, 1, 1), l3], 1)

    def fit(self, category: str, good_paths: list[str], use_cache: bool = True, method: str = "coreset"):
        """정상품 패치 특징을 기억. 캐시는 데모 기동용 — 실험(main)은 매번 새로 만들어 재현성까지 확인한다"""
        cached = CACHE / f"patchcore_{category}_{len(good_paths)}_{method}.pt"
        if use_cache and cached.exists():
            self.banks[category] = torch.load(cached)
            return
        g = torch.Generator().manual_seed(0)
        feats = [self.features(good_paths[i:i + 8]).permute(0, 2, 3, 1).reshape(-1, 1536)
                 for i in range(0, len(good_paths), 8)]
        if method == "random":
            self.banks[category] = torch.cat([f[torch.randperm(len(f), generator=g)[:int(len(f) * KEEP)]] for f in feats])
        else:
            x = torch.cat(feats)
            self.banks[category] = x[greedy_coreset(x, int(len(x) * KEEP), g)]
        CACHE.mkdir(exist_ok=True)
        torch.save(self.banks[category], cached)

    @torch.no_grad()
    def score(self, category: str, path: str) -> tuple[float, torch.Tensor]:
        """이미지 점수(가장 이상한 패치의 정상 기준까지 거리)와 28×28 히트맵"""
        f = self.features([path])[0].permute(1, 2, 0).reshape(-1, 1536)
        d = torch.cdist(f, self.banks[category]).min(1).values.view(28, 28)
        return float(d.max()), d


@torch.no_grad()
def greedy_coreset(x: torch.Tensor, k: int, g: torch.Generator) -> list[int]:
    """k-center greedy: 지금까지 고른 점에서 가장 먼 점을 계속 고른다 → 드문 정상 패턴도 기억에 남는다.
    거리는 128차원 무작위 투영 공간에서 계산 (원 논문 방식, 속도용)"""
    z = x @ (torch.randn(x.shape[1], 128, generator=g) / 128 ** 0.5)
    sel = [0]
    d = ((z - z[0]) ** 2).sum(1)
    for _ in range(k - 1):
        i = int(d.argmax())
        sel.append(i)
        d = torch.minimum(d, ((z - z[i]) ** 2).sum(1))
    return sel


def train_role(path: str) -> str:
    """v6 train/good 분할 (§9): md5(파일명) % 5 — 0 보정, 1 평가, 나머지 정상 기억"""
    r = int(hashlib.md5(Path(path).name.encode()).hexdigest(), 16) % 5
    return "calib" if r == 0 else "eval" if r == 1 else "bank"


def train_goods(data_dir: Path, category: str) -> list[str]:
    return [str(p) for p in sorted((b.find_category_dir(data_dir, category).parent / "train" / "good").glob("*.png"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/v6_run1.json")
    ap.add_argument("--protocol", choices=["v5", "v6"], default="v6")
    ap.add_argument("--method", choices=["coreset", "random"], default="coreset")
    ap.add_argument("--offline", action="store_true", help="가중치를 내려받은 뒤 네트워크 차단 상태로 실행")
    args = ap.parse_args()

    data_dir = b.resolve_data_dir(None)
    model = PatchCore()           # 가중치 로딩(최초 1회 다운로드)까지는 차단 전
    if args.offline:
        block_network()
        try:
            socket.create_connection(("example.com", 80), timeout=3)
            raise SystemExit("차단 실패")
        except ConnectionError:
            pass
    samples = b.load_test_set(data_dir, ["pill", "capsule"], per_type_max=10**6)
    results, times = [], []
    for cat in ["pill", "capsule"]:
        goods = train_goods(data_dir, cat)
        if args.protocol == "v6":   # 정상 기억에 쓴 이미지는 보정·평가에 쓰지 않는다
            samples += [{"category": cat, "label": "good", "path": p, "split": "train"} for p in goods if train_role(p) != "bank"]
            goods = [p for p in goods if train_role(p) == "bank"]
        model.fit(cat, goods, use_cache=False, method=args.method)
    for s in samples:
        t = time.perf_counter()
        sc, _ = model.score(s["category"], s["path"])
        times.append(time.perf_counter() - t)
        results.append({"image": b.image_id(s), "category": s["category"], "true_label": s["label"], "score": sc})
    out = {"meta": {"weights_sha256": model.weights_sha256, "keep": KEEP, "torch": torch.__version__,
                    "protocol": args.protocol, "method": args.method,
                    "threads": torch.get_num_threads(), "offline": args.offline,
                    "bank": {c: len(v) for c, v in model.banks.items()},
                    "sec_per_image": sum(times) / len(times)},
           "results": sorted(results, key=lambda r: r["image"])}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(results)}장, 장당 {out['meta']['sec_per_image']:.3f}초 → {args.out}")


if __name__ == "__main__":
    main()

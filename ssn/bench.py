"""Latency benchmark. Run this on the actual flight computer, not on a laptop.

The frame budget is what the mission allows, not what the marketing deck says.
At 3 Hz the budget is 333 ms per frame for proposer + verifier + tracking, on
the cores left over after navigation. Gate B requires the verifier to finish
5 candidates inside 120 ms.
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import List, Optional

import numpy as np
import torch

from .model import CROP, SSN, T_STEPS


def timeit(fn, iters: int, warmup: int = 5) -> List[float]:
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def report(name: str, ms: List[float]) -> float:
    med = statistics.median(ms)
    p95 = sorted(ms)[int(0.95 * (len(ms) - 1))]
    print(f"{name:<34} median {med:7.1f} ms   p95 {p95:7.1f} ms")
    return med


def main() -> None:
    ap = argparse.ArgumentParser(description="SSN / proposer latency benchmark")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--threads", type=int, default=3, help="cores left to perception")
    ap.add_argument("--ssn", default=None, help="trained checkpoint; random weights if omitted")
    ap.add_argument("--proposer", default=None, help="e.g. yolov8n.pt; skipped if omitted")
    ap.add_argument("--imgsz", type=int, default=416)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    print(f"torch threads {torch.get_num_threads()}   T={T_STEPS}   crop={CROP}\n")

    model = SSN.load(args.ssn) if args.ssn else SSN().eval()
    print(f"SSN parameters: {model.num_params()}\n")
    total: Optional[float] = None
    with torch.no_grad():
        for batch in (1, 3, 5):
            x = torch.randn(batch, T_STEPS, 2, CROP, CROP)
            med = report(f"SSN verifier, {batch} candidate(s)", timeit(lambda: model(x), args.iters))
            if batch == 5:
                total = med
                print(f"{'':34} Gate B limit 120 ms -> {'PASS' if med <= 120 else 'FAIL'}")

    if args.proposer:
        from .perception import Proposer

        prop = Proposer(args.proposer, imgsz=args.imgsz)
        frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
        med = report(f"YOLOv8n proposer @ {args.imgsz}", timeit(lambda: prop(frame), args.iters))
        total = (total or 0) + med

    if total is not None:
        print(f"\npipeline (proposer + 5 crops)     {total:7.1f} ms   budget 333 ms @ 3 Hz")


if __name__ == "__main__":
    main()

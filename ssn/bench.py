"""Latency harness: does the perception stack fit the frame budget?

Measures the three costs that actually move on the target - the per-frame
ego-motion warp, the proposer, and the SSN vote - against the mission budget.
The verifier is batched over candidates, so its cost is reported per batch size:
what matters is the whole batch finishing inside one frame, not the per-item mean.

These numbers are only ever true for the machine that produced them. A desktop
x86 result says nothing about a Raspberry Pi 5 beyond a lower bound; rerun this
on the flight hardware, with the proposer exported to NCNN, before believing any
margin.
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import torch

from .model import CROP, T_STEPS, SSN
from .perception import Proposer, align

FRAME_BUDGET_MS = 120.0  # Gate B: one full verification cycle
PROPOSER_BUDGET_MS = 25.0


def time_ms(fn, repeats: int = 20, warmup: int = 3) -> dict:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return {
        "min": samples[0],  # least scheduler interference - the stable comparator
        "median": statistics.median(samples),
        "p95": samples[min(len(samples) - 1, int(0.95 * len(samples)))],
        "max": samples[-1],
    }


def bench_model(model: SSN, batches=(1, 3, 5), repeats: int = 20) -> dict:
    torch.set_grad_enabled(False)
    model.eval()
    out = {}
    for batch in batches:
        x = torch.randn(batch, T_STEPS, 2, CROP, CROP)
        out[batch] = time_ms(lambda: model(x), repeats)
    return out


def bench_align(width: int = 640, height: int = 480, repeats: int = 20) -> dict:
    rng = np.random.default_rng(0)
    prev = rng.integers(0, 255, (height, width), dtype=np.uint8)
    cur = np.roll(prev, 3, axis=1)
    return time_ms(lambda: align(prev, cur), repeats)


def bench_proposer(width: int = 640, height: int = 480, repeats: int = 10) -> dict:
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    proposer = Proposer()
    proposer(frame)  # force the weight load out of the timed region
    return time_ms(lambda: proposer(frame), repeats, warmup=1)


def _row(name: str, stats: dict, budget: float | None = None) -> str:
    line = f"{name:<24} {stats['median']:7.1f} {stats['p95']:8.1f} {stats['max']:7.1f}"
    if budget is not None:
        verdict = "OK" if stats["p95"] <= budget else "OVER"
        line += f"   {budget:6.0f}  {verdict}"
    return line


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposer", action="store_true",
                        help="also time YOLOv8n (downloads weights on first run)")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args(argv)

    model = SSN()
    print(f"SSN {model.num_params()} params, T={T_STEPS}, crop={CROP}, "
          f"threads={torch.get_num_threads()}")
    print(f"{'stage':<24} {'median':>7} {'p95':>8} {'max':>7}   {'budget':>6}")

    warp = bench_align(args.width, args.height, args.repeats)
    print(_row("ego-motion warp", warp))

    total_p95 = warp["p95"]
    for batch, stats in bench_model(model, repeats=args.repeats).items():
        print(_row(f"ssn verify (batch {batch})", stats))
        if batch == 5:
            total_p95 += stats["p95"]

    if args.proposer:
        prop = bench_proposer(args.width, args.height, max(4, args.repeats // 2))
        print(_row("proposer (yolov8n)", prop, PROPOSER_BUDGET_MS))
        total_p95 += prop["p95"]
    else:
        print(f"{'proposer (yolov8n)':<24} {'skipped - pass --proposer':>25}")

    headroom = FRAME_BUDGET_MS / total_p95 if total_p95 > 0 else float("inf")
    verdict = "OK" if total_p95 <= FRAME_BUDGET_MS else "OVER BUDGET"
    print(f"\ncycle p95 {total_p95:.1f} ms vs {FRAME_BUDGET_MS:.0f} ms budget "
          f"-> {headroom:.1f}x margin, {verdict}")
    print("Measured on this machine, not on flight hardware. Rerun on the Pi 5.")
    return 0 if total_p95 <= FRAME_BUDGET_MS else 1


def _demo():
    stats = time_ms(lambda: time.sleep(0.005), repeats=5, warmup=1)
    assert 4.0 < stats["median"] < 40.0, stats
    assert stats["max"] >= stats["median"]

    model = SSN()
    result = bench_model(model, batches=(1, 4), repeats=8)
    assert set(result) == {1, 4}
    assert all(s["median"] > 0 for s in result.values())
    # Batching must amortise: four candidates cost well under four separate runs.
    # Compared on best-of-N, not the median - a median is at the mercy of whatever
    # else the machine is doing, and this check should fail for real reasons only.
    ratio = result[4]["min"] / result[1]["min"]
    assert ratio < 3.0, ratio
    assert bench_align(160, 120, repeats=3)["median"] > 0
    print(f"ok: batch4 costs {ratio:.1f}x batch1 (4 separate runs would be 4.0x)")


if __name__ == "__main__":
    raise SystemExit(main())

"""Train the SSN verifier on harvested crop sequences.

The metric that matters is not accuracy. The verifier exists to cut the
proposer's false positives without giving up recall, so the final report
compares both at *matched recall*: threshold the raw proposer confidence until
its recall equals the SSN's, then count the false positives each one lets
through. That number is the competition claim, and it is the Gate B decision.
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import split
from .model import SSN


def metrics(scores: np.ndarray, labels: np.ndarray, thresh: float) -> Dict[str, float]:
    pred = scores >= thresh
    tp = float(np.sum(pred & (labels == 1)))
    fp = float(np.sum(pred & (labels == 0)))
    fn = float(np.sum(~pred & (labels == 1)))
    recall = tp / max(1.0, tp + fn)
    precision = tp / max(1.0, tp + fp)
    return {
        "recall": recall,
        "precision": precision,
        "f1": 2 * precision * recall / max(1e-9, precision + recall),
        "fp": fp,
    }


def fp_at_recall(scores: np.ndarray, labels: np.ndarray, target_recall: float) -> float:
    """Lowest false-positive count achievable at >= target_recall."""
    best = float(np.sum(labels == 0))
    for t in np.unique(scores):
        m = metrics(scores, labels, t)
        if m["recall"] >= target_recall:
            best = min(best, m["fp"])
    return best


@torch.no_grad()
def evaluate(model: SSN, loader: DataLoader) -> tuple:
    model.eval()
    scores: List[float] = []
    labels: List[float] = []
    for x, y in loader:
        scores += torch.sigmoid(model(x)).tolist()
        labels += y.tolist()
    return np.array(scores), np.array(labels)


def train(
    data: str,
    out: str = "ssn.pt",
    epochs: int = 30,
    batch: int = 32,
    lr: float = 3e-3,
    val_frac: float = 0.2,
    thresh: float = 0.6,
    seed: int = 0,
) -> Dict[str, float]:
    torch.manual_seed(seed)
    train_set, val_set = split(data, val_frac, seed)
    train_ld = DataLoader(train_set, batch_size=batch, shuffle=True, drop_last=False)
    val_ld = DataLoader(val_set, batch_size=batch)

    labels = train_set.dataset.labels()[list(train_set.indices)]
    pos = max(1, int(labels.sum()))
    pos_weight = torch.tensor(float(len(labels) - pos) / pos)
    print(f"train {len(train_set)} / val {len(val_set)} sequences, pos_weight {pos_weight:.2f}")

    model = SSN()
    print(f"SSN parameters: {model.num_params()}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        total, n, t0 = 0.0, 0, time.perf_counter()
        for x, y in train_ld:
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss) * len(y)
            n += len(y)
        sched.step()
        scores, ys = evaluate(model, val_ld)
        m = metrics(scores, ys, thresh)
        print(
            f"epoch {epoch:3d}  loss {total / max(1, n):.4f}  "
            f"val recall {m['recall']:.3f}  precision {m['precision']:.3f}  "
            f"f1 {m['f1']:.3f}  spike_rate {model.last_spike_rate:.3f}  "
            f"{time.perf_counter() - t0:.1f}s"
        )
        if m["f1"] > best:
            best = m["f1"]
            model.save(out)

    # Gate B report: SSN vs raw proposer confidence at matched recall.
    model = SSN.load(out)
    scores, ys = evaluate(model, val_ld)
    ssn_m = metrics(scores, ys, thresh)
    prop = np.array(
        [float(np.load(val_set.dataset.files[i])["proposer_conf"]) for i in val_set.indices]
    )
    base_fp = fp_at_recall(prop, ys, ssn_m["recall"])
    cut = 100.0 * (1 - ssn_m["fp"] / max(1.0, base_fp))
    print(
        f"\nGate B  recall {ssn_m['recall']:.3f}  "
        f"false positives: proposer {base_fp:.0f} -> SSN {ssn_m['fp']:.0f}  "
        f"({cut:.0f}% cut; gate is >= 30%)"
    )
    return {"f1": best, "recall": ssn_m["recall"], "fp_cut_pct": cut}


def main() -> None:
    ap = argparse.ArgumentParser(description="train the SSN survivor verifier")
    ap.add_argument("--data", default="data/seqs")
    ap.add_argument("--out", default="ssn.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--thresh", type=float, default=0.6)
    args = ap.parse_args()
    train(args.data, args.out, args.epochs, args.batch, args.lr, thresh=args.thresh)


if __name__ == "__main__":
    main()

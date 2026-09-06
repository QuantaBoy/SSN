"""Train the SSN verifier and score it the way the mission is scored.

Accuracy is the wrong objective here and would be actively misleading: the
harvested windows are mostly negatives, so a network that answers "no survivor"
to everything scores well and is worthless. What the mission cares about is a
different trade: the proposer already found every survivor it is going to find,
so the verifier's job is to throw away false positives *without* throwing away
survivors.

So the reported metric is false-positive reduction at matched recall. Pick the
threshold where the network still keeps `target_recall` of the true survivors,
then ask how many of the proposer's false positives it removed at that setting.
Gate B asks for at least a 30% reduction. Recall is held near 1.0 rather than
traded off, because a missed survivor in a collapsed building is not a metric.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split

from .data import SequenceDataset, synthetic
from .model import CROP, SSN, T_STEPS

TARGET_RECALL = 0.98
GATE_B_REDUCTION = 0.30


def matched_recall(scores, labels, target_recall: float = TARGET_RECALL) -> dict:
    """False positives removed at the threshold that preserves target recall.

    The baseline is the proposer alone, which accepts every candidate: its recall
    is 1.0 and its false positives are every negative in the set. Sweeping the
    threshold over observed scores avoids assuming any particular calibration -
    an untrained network's scores sit in a narrow band and a fixed 0.5 would
    report nonsense.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    positives, negatives = labels.sum(), (1.0 - labels).sum()
    if positives == 0 or negatives == 0:
        return {"threshold": 0.0, "recall": float("nan"), "fp_reduction": 0.0,
                "fp_kept": int(negatives), "fp_baseline": int(negatives)}

    # highest threshold that still keeps target_recall of the survivors
    best = {"threshold": float(scores.min()), "recall": 1.0, "fp_reduction": 0.0,
            "fp_kept": int(negatives), "fp_baseline": int(negatives)}
    for threshold in np.unique(scores):
        kept = scores >= threshold
        recall = float((kept & (labels > 0)).sum() / positives)
        if recall < target_recall:
            continue
        fp_kept = int((kept & (labels == 0)).sum())
        candidate = {"threshold": float(threshold), "recall": recall,
                     "fp_reduction": 1.0 - fp_kept / negatives,
                     "fp_kept": fp_kept, "fp_baseline": int(negatives)}
        if candidate["fp_reduction"] >= best["fp_reduction"]:
            best = candidate
    return best


@torch.no_grad()
def evaluate(model: SSN, loader) -> dict:
    model.eval()
    scores, labels = [], []
    for x, y in loader:
        scores.append(torch.sigmoid(model(x.float())).cpu().numpy())
        labels.append(y.cpu().numpy())
    if not scores:
        return matched_recall([], [])
    return matched_recall(np.concatenate(scores), np.concatenate(labels))


def train(model: SSN, train_loader, val_loader, epochs: int = 10,
          lr: float = 2e-3, pos_weight: float | None = None,
          checkpoint: str | None = None, verbose: bool = True) -> dict:
    """Returns the best validation metrics seen, and writes the best checkpoint."""
    weight = (torch.tensor([float(pos_weight)]) if pos_weight else None)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history, best = [], {"fp_reduction": -1.0}
    for epoch in range(epochs):
        model.train()
        total, seen = 0.0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x.float()), y.float())
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(y)
            seen += len(y)

        metrics = evaluate(model, val_loader)
        metrics["loss"] = total / max(seen, 1)
        metrics["epoch"] = epoch
        metrics["spike_rate"] = model.last_spike_rate
        history.append(metrics)
        if metrics["fp_reduction"] > best["fp_reduction"]:
            best = metrics
            if checkpoint:
                model.save(checkpoint)
        if verbose:
            print(f"epoch {epoch:3d}  loss {metrics['loss']:.4f}  "
                  f"recall {metrics['recall']:.3f}  "
                  f"fp-reduction {metrics['fp_reduction'] * 100:5.1f}%  "
                  f"spikes {metrics['spike_rate']:.3f}")
    best["history"] = history
    return best


def _time_split(paths, val_fraction: float = 0.2):
    """Indices of (train, val), where val is the tail of each clip in time.

    Consecutive windows of one track are the same person a fifth of a second
    apart - near-duplicates. A random split therefore puts almost-identical
    windows on both sides, and the score that comes back measures memorisation,
    not detection. Splitting each clip at a point in time is what makes the
    number mean anything: the model is scored on footage it has not seen.

    Filenames are `{stem}_{frame:06d}_{track:04d}.npz`, so the clip and the
    frame index are recoverable without opening anything.
    """
    keyed = []
    for i, path in enumerate(paths):
        stem, frame, _track = str(getattr(path, "stem", path)).rsplit("_", 2)
        keyed.append((stem, int(frame), i))

    train_idx, val_idx = [], []
    for stem in sorted({k[0] for k in keyed}):
        rows = sorted(k for k in keyed if k[0] == stem)
        cut = max(1, int(len(rows) * (1.0 - val_fraction)))
        train_idx += [r[2] for r in rows[:cut]]
        val_idx += [r[2] for r in rows[cut:]]
    return sorted(train_idx), sorted(val_idx)


@torch.no_grad()
def by_clip(model: SSN, dataset, indices) -> dict:
    """The same metric, computed per source clip as well as over everything.

    Positives and negatives are not evenly spread across clips - one clip can
    supply nearly all the people and another nearly all the clutter. A model
    that learned only which clip a window came from would post the same overall
    number as one that learned to recognise a person, and the difference only
    shows up when each clip is scored on its own. A clip holding both classes is
    the honest number, because clip identity cannot separate anything there.
    """
    model.eval()
    rows = {}
    for i in indices:
        x, y = dataset[i]
        score = float(torch.sigmoid(model(torch.from_numpy(np.asarray(x))[None]
                                          .float()))[0])
        clip = Path(dataset.paths[i]).stem.rsplit("_", 2)[0]
        rows.setdefault(clip, []).append((float(y), score))

    out = {}
    for clip, pairs in sorted(rows.items()):
        labels = np.array([p[0] for p in pairs])
        scores = np.array([p[1] for p in pairs])
        out[clip] = matched_recall(scores, labels)
        out[clip]["pos"] = int(labels.sum())
        out[clip]["neg"] = int((1.0 - labels).sum())
    return out


def _loaders(dataset, batch_size: int, val_fraction: float = 0.2, seed: int = 0):
    """Time-split loaders for harvested clips, random split for tensors.

    The validation half is served by a second, un-augmented view of the same
    directory. Augmenting validation would score the model on blurred and
    occluded windows it will never be handed in flight.
    """
    if hasattr(dataset, "paths"):
        train_idx, val_idx = _time_split(dataset.paths, val_fraction)
        clean = SequenceDataset(dataset.root, train=False)
        return (DataLoader(Subset(dataset, train_idx), batch_size=batch_size,
                           shuffle=True),
                DataLoader(Subset(clean, val_idx), batch_size=batch_size))

    val_size = max(1, int(len(dataset) * val_fraction))
    train_set, val_set = random_split(
        dataset, [len(dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(seed))
    return (DataLoader(train_set, batch_size=batch_size, shuffle=True),
            DataLoader(val_set, batch_size=batch_size))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="train the SSN verifier")
    parser.add_argument("--data", default=None,
                        help="directory of harvested .npz windows")
    parser.add_argument("--synthetic", type=int, default=0,
                        help="train on N synthetic windows instead (smoke test only)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--out", default="ssn.pt")
    args = parser.parse_args(argv)

    if args.synthetic:
        x, y = synthetic(args.synthetic)
        dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
        print(f"synthetic run on {args.synthetic} windows - proves the loop "
              f"trains, says nothing about mission performance")
    elif args.data:
        dataset = SequenceDataset(args.data)
        labels = dataset.labels()
        pos_weight = float((labels == 0).sum() / max((labels == 1).sum(), 1))
        print(f"{len(dataset)} windows, {int(labels.sum())} positive, "
              f"pos_weight {pos_weight:.1f}")
    else:
        parser.error("pass --data DIR (harvested windows) or --synthetic N")

    train_loader, val_loader = _loaders(dataset, args.batch_size)
    model = SSN()
    best = train(model, train_loader, val_loader, args.epochs, args.lr,
                 pos_weight, checkpoint=args.out)

    reduction = best["fp_reduction"]
    print(f"\nbest: {reduction * 100:.1f}% false positives removed at "
          f"{best['recall'] * 100:.1f}% recall (threshold {best['threshold']:.3f})")
    print(f"kept {best['fp_kept']} of {best['fp_baseline']} proposer false positives")
    if not args.synthetic and args.data:
        print(f"\n{'clip':<34} {'pos':>4} {'neg':>4} {'fp_red':>8} {'recall':>7}")
        _, val_idx = _time_split(dataset.paths, 0.2)
        clean = SequenceDataset(args.data, train=False)
        for clip, m in by_clip(SSN.load(args.out), clean, val_idx).items():
            print(f"{clip:<34} {m['pos']:>4} {m['neg']:>4} "
                  f"{m['fp_reduction'] * 100:>7.1f}% {m['recall']:>7.2f}")
        print("A clip with both classes is the honest column - clip identity "
              "cannot separate anything inside one clip.")

    if args.synthetic:
        print("Gate B is not assessable on synthetic data - needs flight footage.")
    elif reduction >= GATE_B_REDUCTION:
        print(f"Gate B PASS (>= {GATE_B_REDUCTION * 100:.0f}%). Checkpoint: {args.out}")
    else:
        print(f"Gate B FAIL (< {GATE_B_REDUCTION * 100:.0f}%). More or harder data needed.")
    return 0


def _demo():
    # the metric must be right before any training result means anything
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=float)
    perfect = np.array([0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1])
    result = matched_recall(perfect, labels)
    assert result["recall"] >= TARGET_RECALL
    assert result["fp_reduction"] == 1.0, result

    useless = np.array([0.5] * 8)  # cannot separate: no false positive is removable
    assert matched_recall(useless, labels)["fp_reduction"] == 0.0

    # half the negatives separable -> exactly half removed, recall untouched
    partial = np.array([0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.9, 0.9])
    half = matched_recall(partial, labels)
    assert abs(half["fp_reduction"] - 0.5) < 1e-9, half
    assert half["recall"] == 1.0

    # a threshold that would gain FP reduction by dropping survivors is refused
    greedy = np.array([0.1, 0.9, 0.9, 0.9, 0.2, 0.2, 0.2, 0.2])
    assert matched_recall(greedy, labels)["recall"] >= TARGET_RECALL
    assert matched_recall(greedy, labels)["fp_reduction"] == 0.0

    # the split must not leak: a random split of harvested windows puts the same
    # person, frames apart, on both sides and the metric stops meaning anything
    names = [f"clipA_{f:06d}_0001" for f in range(0, 100, 10)]
    names += [f"clipB_{f:06d}_0002" for f in range(0, 50, 10)]
    train_idx, val_idx = _time_split(names, 0.2)
    assert not set(train_idx) & set(val_idx)
    assert len(train_idx) + len(val_idx) == len(names)
    for stem in ("clipA", "clipB"):
        frames = lambda rows: [int(names[i].rsplit("_", 2)[1])  # noqa: E731
                               for i in rows if names[i].startswith(stem)]
        assert max(frames(train_idx)) < min(frames(val_idx)), stem
    # both clips must contribute to validation, not just the longest one
    assert {names[i].rsplit("_", 2)[0] for i in val_idx} == {"clipA", "clipB"}

    # a clip breakdown must not be foldable back into one number: two clips with
    # opposite class balance have to be reported apart, or a model that only
    # learned clip identity reads as a model that learned to detect people
    class _Fake:
        paths = [Path(f"clipA_{f:06d}_0001.npz") for f in range(4)]
        paths += [Path(f"clipB_{f:06d}_0002.npz") for f in range(4)]

        def __getitem__(self, i):
            #  clipA is all survivors, clipB all clutter - the confound itself
            return np.zeros((T_STEPS, 2, CROP, CROP), np.float32), float(i < 4)

    split = by_clip(SSN(), _Fake(), range(8))
    assert set(split) == {"clipA", "clipB"}
    assert (split["clipA"]["pos"], split["clipA"]["neg"]) == (4, 0)
    assert (split["clipB"]["pos"], split["clipB"]["neg"]) == (0, 4)
    # neither clip holds both classes, so neither can claim any FP reduction
    assert all(m["fp_reduction"] == 0.0 for m in split.values()), split

    # and the loop itself must actually learn the persistent-vs-flash task
    torch.manual_seed(0)
    x, y = synthetic(96, np.random.default_rng(0))
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    train_loader, val_loader = _loaders(dataset, batch_size=16)
    model = SSN()
    before = evaluate(model, val_loader)["fp_reduction"]
    best = train(model, train_loader, val_loader, epochs=4, verbose=False)
    assert best["history"][-1]["loss"] < best["history"][0]["loss"], "loss did not fall"
    assert best["fp_reduction"] > before, (before, best["fp_reduction"])
    print(f"ok: metric verified, fp-reduction {before:.2f} -> "
          f"{best['fp_reduction']:.2f} after 4 epochs")


if __name__ == "__main__":
    raise SystemExit(main())

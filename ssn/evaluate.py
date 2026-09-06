"""Score a trained checkpoint and write the results to disk.

Separate from training on purpose. Training reports one mission number as it
goes; this reports the whole picture for a checkpoint that already exists,
without spending forty epochs to see it again, and it leaves files behind so a
result can be read after the terminal is closed.

Everything lands in `output/`:

    output/metrics.json   every number below, machine-readable
    output/report.txt     the same thing formatted, plus per-clip and per-class
    output/curves.png     ROC and precision-recall, with the operating point marked

Two thresholds matter and both are reported. 0.5 is what a classifier defaults
to and what `sklearn` assumes; it is *not* what flies. The mission threshold is
the one from `train.matched_recall` - the highest cut that still keeps
`TARGET_RECALL` of the survivors - because a missed survivor costs more than a
false alarm the operator dismisses in a second. The threshold-free numbers
(ROC-AUC, average precision) are the ones to compare runs on, since they do not
depend on where that cut lands.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .data import SequenceDataset
from .model import SSN
from .train import TARGET_RECALL, _time_split, matched_recall

OUT_DIR = "output"
LABELS = ("clutter", "survivor")


@torch.no_grad()
def score(model: SSN, dataset, indices) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Sigmoid score, label and source clip for each window in `indices`."""
    model.eval()
    labels, scores, clips = [], [], []
    for i in indices:
        x, y = dataset[i]
        logit = model(torch.from_numpy(np.asarray(x))[None].float())
        scores.append(float(torch.sigmoid(logit)[0]))
        labels.append(float(y))
        clips.append(Path(dataset.paths[i]).stem.rsplit("_", 2)[0])
    return np.asarray(labels), np.asarray(scores), clips


def metrics(y_true, y_score, threshold: float) -> dict:
    """Standard classification metrics at one threshold, plus the rank metrics.

    ROC-AUC and average precision are undefined with only one class present -
    which happens per clip, where one clip can hold nothing but clutter - so
    they come back as None rather than raising or, worse, as a made-up 0.5.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    both = len(np.unique(y_true)) == 2

    # a clip of pure clutter is a legitimate input here, not a mistake worth a
    # warning on every run - `labels` already forces the 2x2 shape sklearn asks for
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        balanced = float(balanced_accuracy_score(y_true, y_pred))
    return {
        "threshold": float(threshold),
        "support": {"survivor": int(y_true.sum()), "clutter": int((1 - y_true).sum())},
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": balanced,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if both else None,
        "average_precision": (float(average_precision_score(y_true, y_score))
                              if both else None),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        # what the mission actually pays for: a survivor the verifier threw away
        "survivors_lost": int(fn),
        "false_alarms_kept": int(fp),
    }


def curves(y_true, y_score, path, operating_point: float | None = None) -> bool:
    """ROC and precision-recall to one PNG. False if only one class is present."""
    if len(np.unique(np.asarray(y_true, dtype=int))) < 2:
        return False
    import matplotlib

    matplotlib.use("Agg")  # no display on the flight machine or in CI
    import matplotlib.pyplot as plt

    fpr, tpr, roc_th = roc_curve(y_true, y_score)
    prec, rec, pr_th = precision_recall_curve(y_true, y_score)

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6))
    left.plot(fpr, tpr, lw=2)
    left.plot([0, 1], [0, 1], "--", lw=1, color="0.6")
    left.set(xlabel="false positive rate", ylabel="true positive rate",
             title=f"ROC (AUC {roc_auc_score(y_true, y_score):.3f})")
    right.plot(rec, prec, lw=2)
    right.set(xlabel="recall", ylabel="precision",
              title=f"Precision-recall (AP {average_precision_score(y_true, y_score):.3f})")

    if operating_point is not None:
        i = int(np.argmin(np.abs(roc_th - operating_point)))
        left.plot(fpr[i], tpr[i], "o", ms=8, color="crimson")
        j = int(np.argmin(np.abs(pr_th - operating_point)))
        right.plot(rec[j], prec[j], "o", ms=8, color="crimson",
                   label=f"mission threshold {operating_point:.3f}")
        right.legend(loc="lower left")

    for axis in (left, right):
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def _text(result: dict, y_true, y_score) -> str:
    mission = result["mission"]
    lines = [
        "SSN verifier evaluation",
        f"checkpoint          {result['checkpoint']}",
        f"dataset             {result['dataset']}",
        f"validation windows  {len(y_true)} "
        f"({int(np.sum(y_true))} survivor, {int(np.sum(1 - y_true))} clutter)",
        "split               temporal tail of each clip, un-augmented",
        "",
        "-- mission operating point "
        f"(threshold {mission['threshold']:.3f}, chosen to keep "
        f"{TARGET_RECALL * 100:.0f}% recall) --",
    ]
    at = result["at_mission_threshold"]
    default = result["at_0.5"]
    for name, block in (("mission threshold", at), ("threshold 0.5", default)):
        lines += [
            f"{name:<20} precision {block['precision']:.3f}  "
            f"recall {block['recall']:.3f}  f1 {block['f1']:.3f}  "
            f"balanced-acc {block['balanced_accuracy']:.3f}",
            f"{'':<20} tp {block['confusion']['tp']:<4} fp {block['confusion']['fp']:<4} "
            f"fn {block['confusion']['fn']:<4} tn {block['confusion']['tn']:<4} "
            f"-> {block['survivors_lost']} survivors lost, "
            f"{block['false_alarms_kept']} false alarms kept",
        ]
    auc = at["roc_auc"]
    ap = at["average_precision"]
    lines += [
        "",
        f"threshold-free      roc_auc {auc:.4f}  average_precision {ap:.4f}"
        if auc is not None else "threshold-free      undefined (one class only)",
        f"mission metric      {mission['fp_reduction'] * 100:.1f}% of the proposer's "
        f"false positives removed at {mission['recall'] * 100:.1f}% recall",
        "",
        classification_report(np.asarray(y_true, dtype=int),
                              (np.asarray(y_score) >= mission["threshold"]).astype(int),
                              labels=[0, 1], target_names=list(LABELS),
                              zero_division=0),
        "-- per clip (a clip holding both classes is the honest row: clip identity",
        "   cannot separate anything inside one clip) --",
        f"{'clip':<34} {'surv':>5} {'clut':>5} {'f1':>6} {'auc':>7} {'fp_red':>8}",
    ]
    for clip, block in result["per_clip"].items():
        auc = "     -" if block["roc_auc"] is None else f"{block['roc_auc']:7.3f}"
        lines.append(f"{clip:<34} {block['support']['survivor']:>5} "
                     f"{block['support']['clutter']:>5} {block['f1']:>6.3f} {auc} "
                     f"{block['fp_reduction'] * 100:>7.1f}%")
    return "\n".join(lines) + "\n"


def evaluate(data: str, weights: str, out_dir: str = OUT_DIR,
             val_fraction: float = 0.2) -> dict:
    """Score `weights` on the held-out tail of `data`, write the artefacts."""
    dataset = SequenceDataset(data, train=False)  # never augment what you score
    _train_idx, val_idx = _time_split(dataset.paths, val_fraction)
    model = SSN.load(weights)
    y_true, y_score, clips = score(model, dataset, val_idx)

    mission = matched_recall(y_score, y_true)
    result = {
        "checkpoint": weights,
        "dataset": data,
        "windows": len(y_true),
        "mission": mission,
        "at_mission_threshold": metrics(y_true, y_score, mission["threshold"]),
        "at_0.5": metrics(y_true, y_score, 0.5),
        "per_clip": {},
    }
    for clip in sorted(set(clips)):
        keep = np.array([c == clip for c in clips])
        block = metrics(y_true[keep], y_score[keep], mission["threshold"])
        block["fp_reduction"] = matched_recall(y_score[keep], y_true[keep])["fp_reduction"]
        result["per_clip"][clip] = block

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result["curves"] = curves(y_true, y_score, out / "curves.png",
                              mission["threshold"])
    (out / "report.txt").write_text(_text(result, y_true, y_score), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez(out / "scores.npz", y_true=y_true, y_score=y_score,
             clips=np.array(clips))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="evaluate a trained SSN checkpoint")
    parser.add_argument("--data", default="dataset",
                        help="directory of harvested .npz windows")
    parser.add_argument("--weights", default="ssn.pt")
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args(argv)

    result = evaluate(args.data, args.weights, args.out)
    print(Path(args.out, "report.txt").read_text(encoding="utf-8"))
    written = ["metrics.json", "report.txt", "scores.npz"]
    if result["curves"]:
        written.append("curves.png")
    print(f"wrote {', '.join(written)} to {Path(args.out).resolve()}")
    return 0


def _demo():
    import tempfile

    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])

    # perfect separation: every standard metric must agree it is perfect
    perfect = metrics(y, np.array([0.9, 0.8, 0.95, 0.85, 0.1, 0.2, 0.05, 0.15]), 0.5)
    assert perfect["f1"] == 1.0 and perfect["roc_auc"] == 1.0
    assert perfect["confusion"] == {"tn": 4, "fp": 0, "fn": 0, "tp": 4}
    assert perfect["survivors_lost"] == 0

    # the metric must be reported at the threshold given, not at a convenient one:
    # the same scores cut at 0.99 keep nothing and lose every survivor
    strict = metrics(y, np.array([0.9, 0.8, 0.95, 0.85, 0.1, 0.2, 0.05, 0.15]), 0.99)
    assert strict["recall"] == 0.0 and strict["survivors_lost"] == 4
    assert strict["roc_auc"] == 1.0, "ranking is threshold-free and must not move"

    # one class only - undefined rather than invented, and no curve to draw
    single = metrics(np.ones(4), np.array([0.9, 0.8, 0.7, 0.6]), 0.5)
    assert single["roc_auc"] is None and single["average_precision"] is None
    tmp = Path(tempfile.mkdtemp())
    assert curves(np.ones(4), np.array([0.9, 0.8, 0.7, 0.6]), tmp / "x.png") is False
    assert not (tmp / "x.png").exists()

    # a real curve does get written, and is a readable PNG
    assert curves(y, np.array([0.9, 0.6, 0.95, 0.4, 0.3, 0.2, 0.05, 0.7]),
                  tmp / "c.png", 0.5) is True
    assert (tmp / "c.png").stat().st_size > 1000
    print(f"ok: f1 {perfect['f1']:.2f}, auc {perfect['roc_auc']:.2f}, "
          f"curve {(tmp / 'c.png').stat().st_size // 1024} KB")


if __name__ == "__main__":
    raise SystemExit(main())

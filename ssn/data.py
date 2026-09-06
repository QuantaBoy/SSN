"""Training data: harvest crop sequences from flight video, auto-labelled.

There is no public dataset of survivors lying in a 1 m indoor corridor filmed
from a drone at 1.3 m altitude, so the training set is your own footage. Manual
labelling of sequences is the bottleneck, so it is skipped: a large offline
teacher detector (YOLOv8l/x, running on a laptop with no latency budget) labels
what the small onboard proposer found.

That gives exactly the supervision the verifier needs - the negatives are the
onboard proposer's own false positives, which is the distribution it will face.
Its ceiling is that the teacher's own misses become false negatives, so spot
check a sample of label 0 sequences before training on them.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .model import CROP, T_STEPS
from .perception import Proposer, Tracker, iou, seq_to_tensor


def harvest(
    video: str,
    out_dir: str,
    teacher: str = "yolov8l.pt",
    proposer: str = "yolov8n.pt",
    teacher_conf: float = 0.70,
    every: int = 3,
    limit: int = 0,
) -> Tuple[int, int]:
    """Write one .npz per sampled crop sequence. Returns (positives, negatives)."""
    from ultralytics import YOLO

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")

    prop = Proposer(proposer, conf=0.15)
    teach = YOLO(teacher)
    tracker = Tracker()
    stem = Path(video).stem
    pos = neg = frame_no = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        dets = prop(frame)
        tracks = tracker.update(frame, dets)
        if frame_no % every:
            continue

        tboxes = [
            (tuple(float(v) for v in b.xyxy[0].tolist()), float(b.conf[0]))
            for b in teach.predict(frame, conf=teacher_conf, classes=[0], verbose=False)[0].boxes
        ]
        for track in tracks:
            if not track.ready:
                continue
            label = int(any(iou(track.box, tb) >= 0.5 for tb, _ in tboxes))
            np.savez_compressed(
                out / f"{stem}_f{frame_no:06d}_t{track.id:03d}.npz",
                seq=np.stack(track.crops).astype(np.uint8),
                label=np.int64(label),
                box=np.array(track.box, np.float32),
                proposer_conf=np.float32(track.conf),
            )
            pos, neg = pos + label, neg + (1 - label)
        if limit and pos + neg >= limit:
            break

    cap.release()
    return pos, neg


def augment(seq: np.ndarray, rng: random.Random) -> np.ndarray:
    """Augment a (T, 64, 64) uint8 sequence. Every op is applied to all frames
    identically except the frame dropout, which must break time coherence."""
    seq = seq.copy()
    if rng.random() < 0.5:
        seq = seq[:, :, ::-1]
    if rng.random() < 0.8:  # brightness / contrast - indoor lighting varies hard
        gain = rng.uniform(0.7, 1.3)
        bias = rng.uniform(-30, 30)
        seq = np.clip(seq.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:  # sensor noise at high ISO in a dark corridor
        noise = np.random.normal(0, rng.uniform(2, 12), seq.shape)
        seq = np.clip(seq.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:  # dropped/duplicated frame: the tracker does miss
        i = rng.randrange(1, len(seq))
        seq[i] = seq[i - 1]
    if rng.random() < 0.5:  # residual alignment error the phase correlation left
        dx, dy = rng.randint(-3, 3), rng.randint(-3, 3)
        seq = np.roll(seq, (dy, dx), axis=(1, 2))
    return np.ascontiguousarray(seq)


class SeqDataset(Dataset):
    """Crop sequences on disk -> (T, 2, 64, 64) tensors."""

    def __init__(self, root: str, train: bool = True, seed: int = 0):
        self.files: List[Path] = sorted(Path(root).glob("*.npz"))
        if not self.files:
            raise SystemExit(f"no .npz sequences in {root} - run `python -m ssn.data harvest` first")
        self.train = train
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.files)

    def labels(self) -> np.ndarray:
        return np.array([int(np.load(f)["label"]) for f in self.files])

    def __getitem__(self, i: int):
        rec = np.load(self.files[i])
        seq = rec["seq"]
        if len(seq) < T_STEPS:  # pad short windows by repeating the first frame
            seq = np.concatenate([np.repeat(seq[:1], T_STEPS - len(seq), 0), seq])
        seq = seq[-T_STEPS:]
        if self.train:
            seq = augment(seq, self.rng)
        return seq_to_tensor(seq), torch.tensor(float(rec["label"]))


def split(root: str, val_frac: float = 0.2, seed: int = 0):
    """Deterministic train/val split over the same directory."""
    train, val = SeqDataset(root, True, seed), SeqDataset(root, False, seed)
    idx = list(range(len(train)))
    random.Random(seed).shuffle(idx)
    cut = int(len(idx) * (1 - val_frac))
    return torch.utils.data.Subset(train, idx[:cut]), torch.utils.data.Subset(val, idx[cut:])


def main() -> None:
    ap = argparse.ArgumentParser(description="harvest SSN training sequences from flight video")
    ap.add_argument("cmd", choices=["harvest"])
    ap.add_argument("video")
    ap.add_argument("--out", default="data/seqs")
    ap.add_argument("--teacher", default="yolov8l.pt")
    ap.add_argument("--proposer", default="yolov8n.pt")
    ap.add_argument("--every", type=int, default=3, help="save every Nth frame per track")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    pos, neg = harvest(
        args.video, args.out, args.teacher, args.proposer, every=args.every, limit=args.limit
    )
    print(f"{pos} positive, {neg} negative sequences -> {args.out}")


if __name__ == "__main__":
    main()

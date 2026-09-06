"""Training data: harvest candidate sequences from video, label them with a
teacher, and augment them without destroying the signal the network reads.

The labelling is teacher-student. The student proposer (YOLOv8n at low
confidence) decides *what gets looked at*, exactly as it will in flight, and a
heavier teacher (YOLOv8l at high confidence) decides *what was really there*.
Training on the student's own candidate distribution is the point: a set of
tidy person crops would teach the network nothing about the blurred debris and
mannequin limbs it actually has to reject.

Augmentation rule, and the easiest thing in this file to get wrong: a transform
caused by the camera is coherent across the window, and only sensor noise is
per-frame. Motion blur, occluding debris and exposure apply identically to every
frame of a sample. Re-rolling them per frame would inject exactly the kind of
one-frame flicker the SSN is trained to treat as evidence of an artefact, and it
would learn that flicker is normal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .model import CROP, T_STEPS
from .perception import Proposer, Tracker, align, crop_pair, iou, to_gray

TEACHER_WEIGHTS = "yolov8l.pt"
TEACHER_CONF = 0.5
MATCH_IOU = 0.5


def teacher_boxes(frame, model=None, conf: float = TEACHER_CONF) -> list[tuple]:
    """High-confidence person boxes from the heavy model - the label source."""
    if model is None:
        from ultralytics import YOLO

        model = teacher_boxes.cache = getattr(teacher_boxes, "cache", None) or YOLO(
            TEACHER_WEIGHTS
        )
    result = model.predict(frame, conf=conf, classes=[0], verbose=False)[0]
    return [tuple(map(float, b)) for b in result.boxes.xyxy.tolist()]


def harvest(video: str, out_dir: str, proposer: Proposer | None = None,
            teacher=None, max_frames: int = 0, stride: int = 1,
            modality: str = "rgb") -> dict:
    """Write one .npz per completed candidate window. Returns a label tally.

    A window is labelled positive when the student's tracked box still overlaps a
    teacher box at the moment the window closes. Tracks that the teacher never
    confirms are the hard negatives, and they are the majority - that imbalance
    is real and is handled at training time, not by throwing them away here.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    proposer = proposer or Proposer()
    tracker = Tracker()

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open video: {video}")

    prev_gray, bounds, index, saved = None, None, 0, {"pos": 0, "neg": 0}
    stem = Path(video).stem
    while True:
        ok, frame = capture.read()
        if not ok or (max_frames and index >= max_frames):
            break
        if index % stride:
            index += 1
            continue

        # same conversion the runtime uses - a mismatch here trains the network
        # on an input distribution it will never see in flight
        gray, bounds = to_gray(frame, modality, bounds)
        delta = (align(prev_gray, gray) if prev_gray is not None
                 else np.zeros_like(gray, dtype=np.float32))
        prev_gray = gray

        truth = teacher_boxes(frame, teacher)
        for track in tracker.update(proposer(frame)):
            track.samples.append(crop_pair(gray, delta, track.box))
            if not track.ready:
                continue
            label = int(any(iou(track.box, t) >= MATCH_IOU for t in truth))
            np.savez_compressed(
                out / f"{stem}_{index:06d}_{track.id:04d}.npz",
                x=np.stack(list(track.samples)).astype(np.float32),
                y=np.float32(label),
            )
            saved["pos" if label else "neg"] += 1
            track.samples.clear()  # non-overlapping windows: no leakage
        index += 1

    capture.release()
    return saved


def _directional_blur(image: np.ndarray, length: int, angle: float) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    matrix = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5),
                                     np.degrees(angle), 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (length, length))
    total = kernel.sum()
    return cv2.filter2D(image, -1, kernel / total if total > 0 else kernel)


def augment(sample: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Physics-based augmentation of one (T, 2, CROP, CROP) window.

    Every transform below is drawn once and applied to the whole window, because
    every one of them has a cause that persists for longer than a frame - except
    sensor noise, which is genuinely independent per frame.
    """
    out = sample.copy()

    if rng.random() < 0.5:  # airframe roll / heading: whole window flips together
        out = out[:, :, :, ::-1].copy()

    if rng.random() < 0.4:  # ego-motion smear, one direction for the whole pass
        length = int(rng.integers(3, 8))
        angle = rng.uniform(0.0, np.pi)
        for t in range(out.shape[0]):
            out[t, 0] = _directional_blur(out[t, 0], length, angle)

    if rng.random() < 0.5:  # rubble interior: dim, low contrast, fixed exposure
        gain = rng.uniform(0.45, 1.15)
        bias = rng.uniform(-0.25, 0.1)
        out[:, 0] = np.clip(out[:, 0] * gain + bias, -1.0, 1.0)

    if rng.random() < 0.35:  # debris occludes the same place across the window
        h = int(rng.integers(CROP // 6, CROP // 2))
        w = int(rng.integers(CROP // 6, CROP // 2))
        y = int(rng.integers(0, CROP - h))
        x = int(rng.integers(0, CROP - w))
        drift = rng.integers(-1, 2, size=2)
        for t in range(out.shape[0]):
            dy, dx = int(drift[0] * t), int(drift[1] * t)
            y0, x0 = np.clip(y + dy, 0, CROP - h), np.clip(x + dx, 0, CROP - w)
            out[t, :, y0:y0 + h, x0:x0 + w] = rng.uniform(-1.0, -0.4)

    # sensor noise is the one genuinely per-frame effect
    out += rng.normal(0.0, rng.uniform(0.0, 0.06), size=out.shape).astype(np.float32)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


class SequenceDataset:
    """Loads harvested .npz windows. Indexable, so torch's DataLoader accepts it."""

    def __init__(self, root: str, train: bool = True, seed: int = 0):
        self.paths = sorted(Path(root).glob("*.npz"))
        if not self.paths:
            raise SystemExit(f"no .npz windows in {root} - run harvest first")
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        record = np.load(self.paths[index])
        x, y = record["x"].astype(np.float32), float(record["y"])
        if self.train:
            x = augment(x, self.rng)
        return x, np.float32(y)

    def labels(self) -> np.ndarray:
        return np.array([float(np.load(p)["y"]) for p in self.paths])


def synthetic(count: int, rng: np.random.Generator | None = None):
    """Stand-in windows for exercising the training loop with no footage.

    Positives hold a persistent bright blob; negatives flash one for a single
    frame at the same total energy. That is a caricature of the real task, and it
    proves the loop learns *something*, never that the network is mission-ready.
    """
    rng = rng or np.random.default_rng(0)
    x = rng.normal(0.0, 0.25, (count, T_STEPS, 2, CROP, CROP)).astype(np.float32)
    y = (rng.random(count) < 0.5).astype(np.float32)
    for i in range(count):
        cy, cx = rng.integers(16, 48, size=2)
        if y[i]:
            x[i, :, 0, cy - 8:cy + 8, cx - 6:cx + 6] += 0.9
        else:
            x[i, rng.integers(0, T_STEPS), 0, cy - 8:cy + 8, cx - 6:cx + 6] += 0.9 * T_STEPS
    return np.clip(x, -1.0, 1.0), y


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="harvest SSN training windows")
    parser.add_argument("video")
    parser.add_argument("--out", default="dataset")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--modality", choices=("rgb", "thermal"), default="rgb")
    args = parser.parse_args(argv)
    tally = harvest(args.video, args.out, max_frames=args.max_frames,
                    stride=args.stride, modality=args.modality)
    total = tally["pos"] + tally["neg"]
    print(f"{total} windows -> {args.out} "
          f"({tally['pos']} positive, {tally['neg']} negative)")
    return 0 if total else 1


def _demo():
    rng = np.random.default_rng(0)
    # real windows are temporally smooth: one scene, small per-frame sensor
    # noise. Building the base from independent noise per frame would saturate
    # the flicker measure below and make the check unable to fail.
    scene = rng.normal(0, 0.2, (2, CROP, CROP)).astype(np.float32)
    scene[0, 20:40, 20:40] += 0.8
    sample = np.stack([scene] * T_STEPS)
    sample += rng.normal(0, 0.01, sample.shape).astype(np.float32)
    sample = np.clip(sample, -1.0, 1.0).astype(np.float32)

    out = augment(sample, np.random.default_rng(1))
    assert out.shape == sample.shape and out.dtype == np.float32
    assert -1.0 <= out.min() and out.max() <= 1.0

    # the load-bearing property: augmentation must not add per-frame flicker.
    # Frame-to-frame variation may grow a little from noise, but nothing like
    # the jump a per-frame re-rolled transform would produce.
    def flicker(seq):
        return float(np.abs(np.diff(seq[:, 0], axis=0)).mean())

    # Compared against the same transforms re-rolled per frame - the mistake this
    # design exists to avoid. Both paths carry identical per-frame sensor noise,
    # so the gap between them is purely the coherence of the structural
    # transforms. Absolute flicker would just measure the noise term.
    coherent = np.median([flicker(augment(sample, np.random.default_rng(s)))
                          for s in range(12)])
    per_frame = np.median([
        flicker(np.stack([augment(sample, np.random.default_rng(s * T_STEPS + t))[t]
                          for t in range(T_STEPS)]))
        for s in range(12)
    ])
    assert per_frame > 3.0 * coherent, (per_frame, coherent)

    x, y = synthetic(8, rng)
    assert x.shape == (8, T_STEPS, 2, CROP, CROP)
    assert set(np.unique(y)) <= {0.0, 1.0}
    print(f"ok: flicker {coherent:.4f} coherent vs {per_frame:.4f} per-frame "
          f"({per_frame / coherent:.1f}x)")


if __name__ == "__main__":
    raise SystemExit(main())

"""Perception front-end: propose -> track -> verify.

The proposer runs at low confidence on purpose (recall over precision), so most
of what comes out is not a survivor. The tracker gives each candidate a stable
identity across frames, and only once a track has accumulated T_STEPS frames does
the SSN vote on it. Nothing is reported from a single frame.

Ego-motion compensation happens once per frame on the whole image, not once per
candidate: one affine warp of the previous frame onto the current one, then a
signed difference. On a moving drone an unwarped difference is dominated by
camera translation, and a lying survivor - who barely moves - vanishes into it.
One warp per frame rather than one per crop is what keeps this affordable on a
Pi 5 CPU.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from .model import CROP, T_STEPS, SSN

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels

THERMAL_PCT = (1.0, 99.0)  # robust stretch bounds, ignores dead pixels and sun glint
THERMAL_ALPHA = 0.1  # how fast those bounds may move per frame


def to_gray(frame: np.ndarray, modality: str = "rgb", bounds=None):
    """Single-channel uint8 from an RGB or a thermal frame, plus carried state.

    Thermal cores return radiometric counts on an arbitrary scale, so they need a
    stretch that RGB does not. The bounds are carried across frames and allowed to
    move only slowly: a per-frame stretch would make the normalisation itself
    flicker, and `align` would read that flicker as scene motion - the same
    coherence trap the augmentation avoids, arriving through the sensor instead.
    """
    if modality == "rgb":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        return gray.astype(np.uint8), bounds

    if frame.ndim == 3:
        frame = frame[..., 0]
    lo, hi = np.percentile(frame, THERMAL_PCT)
    if bounds is not None:
        lo = bounds[0] + THERMAL_ALPHA * (lo - bounds[0])
        hi = bounds[1] + THERMAL_ALPHA * (hi - bounds[1])
    stretched = (frame.astype(np.float32) - lo) / max(float(hi - lo), 1e-6)
    return (np.clip(stretched, 0.0, 1.0) * 255.0).astype(np.uint8), (lo, hi)


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def align(prev_gray: np.ndarray, cur_gray: np.ndarray) -> np.ndarray:
    """Signed ego-motion-compensated difference, cur - warp(prev), in [-255, 255].

    Sparse LK flow on corners gives a partial affine (rotation, translation,
    uniform scale). That model fits drone motion over one frame interval well
    enough; a full homography needs more correspondences than a low-texture
    indoor maze reliably provides, and fails loudly when it does not get them.
    """
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01,
                                  minDistance=8)
    warped = prev_gray
    if pts is not None and len(pts) >= 6:
        nxt, ok, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, pts, None)
        ok = ok.ravel().astype(bool)
        if ok.sum() >= 6:
            matrix, _ = cv2.estimateAffinePartial2D(pts[ok], nxt[ok])
            if matrix is not None:
                h, w = cur_gray.shape[:2]
                warped = cv2.warpAffine(prev_gray, matrix, (w, h),
                                        borderMode=cv2.BORDER_REPLICATE)
    return cur_gray.astype(np.float32) - warped.astype(np.float32)


def crop_pair(gray: np.ndarray, delta: np.ndarray, box: Box) -> np.ndarray:
    """Two-channel CROPxCROP sample in [-1, 1]: luma and ego-compensated delta."""
    h, w = gray.shape[:2]
    x1 = int(max(0, min(box[0], w - 1)))
    y1 = int(max(0, min(box[1], h - 1)))
    x2 = int(max(x1 + 1, min(box[2], w)))
    y2 = int(max(y1 + 1, min(box[3], h)))
    size = (CROP, CROP)
    luma = cv2.resize(gray[y1:y2, x1:x2], size).astype(np.float32) / 127.5 - 1.0
    diff = cv2.resize(delta[y1:y2, x1:x2], size) / 127.5
    return np.stack([luma, np.clip(diff, -1.0, 1.0)])


@dataclass
class Track:
    id: int
    box: Box
    misses: int = 0
    hits: int = 1
    samples: deque = field(default_factory=lambda: deque(maxlen=T_STEPS))
    confidence: float = 0.0

    @property
    def ready(self) -> bool:
        return len(self.samples) == T_STEPS


class Tracker:
    """Greedy IoU association. No motion model.

    Frame-to-frame displacement indoors at survey speed is small next to a person
    -sized box, so IoU alone associates correctly; a Kalman filter would buy
    nothing here but state to get wrong.
    """

    def __init__(self, iou_thresh: float = 0.3, max_misses: int = 3):
        self.iou_thresh = iou_thresh
        self.max_misses = max_misses
        self.tracks: list[Track] = []
        self._next_id = 0

    def update(self, boxes: list[Box]) -> list[Track]:
        pairs = sorted(
            ((iou(t.box, b), ti, bi)
             for ti, t in enumerate(self.tracks)
             for bi, b in enumerate(boxes)),
            reverse=True,
        )
        used_t: set[int] = set()
        used_b: set[int] = set()
        for score, ti, bi in pairs:
            if score < self.iou_thresh or ti in used_t or bi in used_b:
                continue
            track = self.tracks[ti]
            track.box, track.misses, track.hits = boxes[bi], 0, track.hits + 1
            used_t.add(ti)
            used_b.add(bi)

        for ti, track in enumerate(self.tracks):
            if ti not in used_t:
                track.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        for bi, box in enumerate(boxes):
            if bi not in used_b:
                self.tracks.append(Track(self._next_id, box))
                self._next_id += 1
        return self.tracks


class Proposer:
    """YOLOv8n person detector, tuned for recall. Weights load on first call."""

    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.15,
                 imgsz: int = 480):
        self.weights, self.conf, self.imgsz = weights, conf, imgsz
        self._model = None

    def __call__(self, frame: np.ndarray) -> list[Box]:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.weights)
        result = self._model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                                     classes=[0], verbose=False)[0]
        return [tuple(map(float, b)) for b in result.boxes.xyxy.tolist()]


class Verifier:
    """Runs the SSN over every track that has a full T_STEPS window."""

    def __init__(self, model: SSN | None = None, threshold: float = 0.5,
                 max_batch: int = 5):
        self.model = (model or SSN()).eval()
        self.threshold = threshold
        self.max_batch = max_batch

    @torch.no_grad()
    def __call__(self, tracks: list[Track]) -> list[Track]:
        ready = [t for t in tracks if t.ready][: self.max_batch]
        if not ready:
            return []
        batch = torch.from_numpy(
            np.stack([np.stack(list(t.samples)) for t in ready])
        ).float()
        for track, conf in zip(ready, self.model.confidence(batch).tolist()):
            track.confidence = conf
        return [t for t in ready if t.confidence >= self.threshold]


class Pipeline:
    """Per-frame glue: one warp, one proposer call, one batched verification."""

    def __init__(self, proposer: Proposer | None = None,
                 tracker: Tracker | None = None,
                 verifier: Verifier | None = None,
                 modality: str = "rgb"):
        self.proposer = proposer or Proposer()
        self.tracker = tracker or Tracker()
        self.verifier = verifier or Verifier()
        self.modality = modality
        self._prev_gray: np.ndarray | None = None
        self._bounds = None

    def step(self, frame: np.ndarray) -> list[Track]:
        gray, self._bounds = to_gray(frame, self.modality, self._bounds)
        delta = (align(self._prev_gray, gray) if self._prev_gray is not None
                 else np.zeros_like(gray, dtype=np.float32))
        self._prev_gray = gray

        tracks = self.tracker.update(self.proposer(frame))
        for track in tracks:
            track.samples.append(crop_pair(gray, delta, track.box))
        return self.verifier(tracks)


def _demo():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (240, 320), dtype=np.uint8)

    # ego-motion: a pure pan must cancel, leaving a near-zero delta. An unwarped
    # difference of the same pair is large - that gap is the whole point.
    shifted = np.roll(base, 4, axis=1)
    compensated = np.abs(align(base, shifted)).mean()
    raw = np.abs(shifted.astype(np.float32) - base.astype(np.float32)).mean()
    assert compensated < raw / 2, (compensated, raw)

    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    # identity survives jitter, and a track only dies after max_misses frames
    tracker = Tracker()
    first = tracker.update([(10, 10, 50, 90)])[0].id
    for step in range(3):
        moved = (10 + step, 12 + step, 50 + step, 92 + step)
        assert tracker.update([moved])[0].id == first
    for _ in range(tracker.max_misses):
        assert len(tracker.update([])) == 1
    assert tracker.update([]) == []

    # a distant box is a new identity, not the old one drifting
    tracker = Tracker()
    tracker.update([(10, 10, 50, 90)])
    assert len(tracker.update([(200, 10, 240, 90)])) == 2

    sample = crop_pair(base, align(base, shifted), (10, 10, 50, 90))
    assert sample.shape == (2, CROP, CROP)
    assert -1.0 <= sample.min() and sample.max() <= 1.0

    # verifier stays silent until a track holds a full window
    tracker, verifier = Tracker(), Verifier()
    for _ in range(T_STEPS):
        tracks = tracker.update([(10, 10, 50, 90)])
        assert verifier(tracks) == []
        for track in tracks:
            track.samples.append(sample)
    assert tracker.tracks[0].ready
    verifier.threshold = 0.0
    assert len(verifier(tracker.tracks)) == 1
    print(f"ok: delta {compensated:.1f} vs raw {raw:.1f}, "
          f"conf {tracker.tracks[0].confidence:.3f}")


if __name__ == "__main__":
    _demo()

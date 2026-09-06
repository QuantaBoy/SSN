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

RGB only. A visible-light camera is the whole payload, so everything the
verifier gets has to be recovered from luma and motion, and the two places that
costs something are handled here: contrast equalisation per crop, so a candidate
in an unlit room reads like the same candidate in a lit corridor, and the batch
priority in `Verifier`, because an RGB proposer in rubble over-proposes harder
than a thermal one would.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from .model import CROP, T_STEPS, SSN

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels

FLOW_SCALE = 0.5  # ego-motion is measured at this scale and applied at full res

NOISE_FLOOR = 8.0  # grey counts; below this a crop is flat and only holds noise


def to_gray(frame: np.ndarray) -> np.ndarray:
    """Single-channel uint8 luma from a BGR frame."""
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame.astype(np.uint8)


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

    Corners and flow run at FLOW_SCALE, which is where the cost of this function
    sits: a quarter of the pixels for the same affine, since one frame of drone
    motion is many pixels wide and does not need full resolution to be measured.
    Under that rescale only the translation column of a partial affine changes.
    """
    small_prev = cv2.resize(prev_gray, None, fx=FLOW_SCALE, fy=FLOW_SCALE)
    small_cur = cv2.resize(cur_gray, None, fx=FLOW_SCALE, fy=FLOW_SCALE)
    pts = cv2.goodFeaturesToTrack(small_prev, maxCorners=200, qualityLevel=0.01,
                                  minDistance=4)
    warped = prev_gray
    if pts is not None and len(pts) >= 6:
        nxt, ok, _ = cv2.calcOpticalFlowPyrLK(small_prev, small_cur, pts, None)
        ok = ok.ravel().astype(bool)
        if ok.sum() >= 6:
            matrix, _ = cv2.estimateAffinePartial2D(pts[ok], nxt[ok])
            if matrix is not None:
                matrix[:, 2] /= FLOW_SCALE  # measured in half-res pixels
                h, w = cur_gray.shape[:2]
                warped = cv2.warpAffine(prev_gray, matrix, (w, h),
                                        borderMode=cv2.BORDER_REPLICATE)
    return cur_gray.astype(np.float32) - warped.astype(np.float32)


def crop_pair(gray: np.ndarray, delta: np.ndarray, box: Box) -> np.ndarray:
    """Two-channel CROPxCROP sample in [-1, 1]: luma and ego-compensated delta.

    The luma crop is standardised per crop. Auto-exposure and auto-gain change
    the luma by an affine map, and subtracting the crop mean and dividing by its
    spread cancels exactly that map: the same candidate in an unlit room and in a
    window-lit corridor arrives at the network as the same tensor. Dividing by
    twice the spread keeps roughly 95% of the crop inside [-1, 1] instead of
    clipping most of it flat, and the NOISE_FLOOR stops a featureless crop from
    being amplified into pure sensor noise.

    The delta crop is deliberately left alone: a difference carries no exposure
    to remove, and rescaling it would destroy the very magnitude that separates a
    moving survivor from noise.
    """
    h, w = gray.shape[:2]
    x1 = int(max(0, min(box[0], w - 1)))
    y1 = int(max(0, min(box[1], h - 1)))
    x2 = int(max(x1 + 1, min(box[2], w)))
    y2 = int(max(y1 + 1, min(box[3], h)))
    size = (CROP, CROP)
    luma = cv2.resize(gray[y1:y2, x1:x2], size).astype(np.float32)
    luma = np.clip((luma - luma.mean()) / (2.0 * max(float(luma.std()),
                                                     NOISE_FLOOR)), -1.0, 1.0)
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

    @torch.inference_mode()
    def __call__(self, tracks: list[Track]) -> list[Track]:
        # Most-established first. Past max_batch the batch has to drop someone,
        # and dropping by list order drops whichever candidate the proposer
        # happened to emit last. A track with more hits has survived more frames
        # of association and is the better use of a fixed budget - which matters
        # more on RGB, where a low-confidence proposer in rubble routinely
        # returns more candidates than one batch can hold.
        ready = sorted((t for t in tracks if t.ready), key=lambda t: -t.hits)
        ready = ready[: self.max_batch]
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
                 verifier: Verifier | None = None):
        self.proposer = proposer or Proposer()
        self.tracker = tracker or Tracker()
        self.verifier = verifier or Verifier()
        self._prev_gray: np.ndarray | None = None

    def step(self, frame: np.ndarray) -> list[Track]:
        gray = to_gray(frame)
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

    # the same scene under a different exposure must reach the network as the
    # same luma tensor - that invariance is what standardising the crop buys
    lit = np.full((90, 40), 200, np.uint8)
    lit[20:70, 10:30] = 60
    dim = (lit.astype(np.float32) * 0.5 + 20).astype(np.uint8)
    flat = np.zeros(lit.shape, dtype=np.float32)
    box = (0, 0, 40, 90)
    assert np.abs(crop_pair(lit, flat, box)[0]
                  - crop_pair(dim, flat, box)[0]).max() < 0.02

    # a featureless crop must stay featureless, not be amplified into noise
    empty = np.full((90, 40), 40, np.uint8)
    empty[0, 0] = 41  # one count of dither, still far below NOISE_FLOOR
    assert np.abs(crop_pair(empty, flat, box)[0]).max() < 0.2

    # and none of that may reach the delta channel: twice the motion signal has
    # to stay twice as large, not be normalised back to the same crop
    faint = crop_pair(lit, np.full_like(flat, 20.0), box)
    strong = crop_pair(lit, np.full_like(flat, 40.0), box)
    assert abs(strong[1].mean() - 2.0 * faint[1].mean()) < 1e-6

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

    # over budget, the batch keeps the established track and drops the fresh one
    established = Track(0, (10, 10, 50, 90), hits=9)
    fresh = Track(1, (60, 10, 100, 90), hits=1)
    for track in (established, fresh):
        for _ in range(T_STEPS):
            track.samples.append(sample)
    small = Verifier(max_batch=1, threshold=0.0)
    assert [t.id for t in small([fresh, established])] == [0]

    print(f"ok: delta {compensated:.1f} vs raw {raw:.1f}, "
          f"conf {tracker.tracks[0].confidence:.3f}")


if __name__ == "__main__":
    _demo()

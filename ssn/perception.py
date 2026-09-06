"""Runtime perception: proposer -> tracker -> ego-motion-aligned crop buffer -> SSN.

One frame in, a list of tracks out, each carrying a temporal confidence from the
SSN verifier. Nothing in here talks to the flight controller or the map; the
caller wires those.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from .model import CROP, SSN, T_STEPS

Box = Tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


# --------------------------------------------------------------------------- #
# Crop preparation - shared by harvesting, training and runtime so there is no
# train/serve skew in how a sequence is built.
# --------------------------------------------------------------------------- #

_HANN = cv2.createHanningWindow((CROP, CROP), cv2.CV_32F)


def crop_patch(frame: np.ndarray, box: Box, pad: float = 0.15) -> np.ndarray:
    """Grayscale CROPxCROP patch around box, padded by `pad` of the box size."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 = int(max(0, x1 - pad * bw))
    y1 = int(max(0, y1 - pad * bh))
    x2 = int(min(w, x2 + pad * bw))
    y2 = int(min(h, y2 + pad * bh))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return np.zeros((CROP, CROP), np.uint8)
    patch = frame[y1:y2, x1:x2]
    if patch.ndim == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return cv2.resize(patch, (CROP, CROP), interpolation=cv2.INTER_AREA)


def align(patch: np.ndarray, ref: Optional[np.ndarray]) -> np.ndarray:
    """Shift `patch` onto `ref` by phase correlation (ego-motion compensation).

    The tracker box already removes most of the apparent target motion; this
    removes the residual sub-box drift, so the frame delta fed to channel 1
    carries target motion rather than camera motion.
    """
    if ref is None:
        return patch
    (dx, dy), _ = cv2.phaseCorrelate(
        ref.astype(np.float32), patch.astype(np.float32), _HANN
    )
    if abs(dx) < 0.5 and abs(dy) < 0.5:
        return patch
    dx = int(np.clip(round(dx), -CROP // 4, CROP // 4))
    dy = int(np.clip(round(dy), -CROP // 4, CROP // 4))
    return np.roll(patch, (-dy, -dx), axis=(0, 1))


def seq_to_tensor(seq: Sequence[np.ndarray]) -> torch.Tensor:
    """(T, 64, 64) uint8 crops -> (T, 2, 64, 64) float tensor in [-1, 1].

    Channel 0 is luma, channel 1 is the frame delta (zero at t = 0).
    """
    arr = np.stack(seq).astype(np.float32) / 127.5 - 1.0
    delta = np.zeros_like(arr)
    delta[1:] = arr[1:] - arr[:-1]
    return torch.from_numpy(np.stack([arr, delta], axis=1))


# --------------------------------------------------------------------------- #
# Proposer
# --------------------------------------------------------------------------- #


class Proposer:
    """YOLOv8n person detector, tuned for recall. Let it over-propose."""

    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.15, imgsz: int = 416):
        from ultralytics import YOLO  # lazy import: tests run without weights

        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz

    def __call__(self, frame: np.ndarray) -> List[Tuple[Box, float]]:
        res = self.model.predict(
            frame, imgsz=self.imgsz, conf=self.conf, classes=[0], verbose=False
        )[0]
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            out.append(((x1, y1, x2, y2), float(b.conf[0])))
        return out


# --------------------------------------------------------------------------- #
# Tracking
# --------------------------------------------------------------------------- #


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


@dataclass
class Track:
    id: int
    box: Box
    conf: float
    crops: List[np.ndarray] = field(default_factory=list)
    hits: int = 1
    misses: int = 0
    ssn_conf: float = 0.0
    confirmations: int = 0

    @property
    def ready(self) -> bool:
        return len(self.crops) >= T_STEPS

    def push(self, frame: np.ndarray, box: Box, conf: float) -> None:
        self.box, self.conf = box, conf
        ref = self.crops[-1] if self.crops else None
        self.crops.append(align(crop_patch(frame, box), ref))
        if len(self.crops) > T_STEPS:
            self.crops.pop(0)


class Tracker:
    """Greedy IoU association.

    Indoor corridors present few simultaneous targets, so the O(n*m) match is
    free and a motion model buys nothing over box IoU at 3 Hz.
    """

    # ponytail: greedy IoU, no Kalman. Swap in ByteTrack only if targets start
    # crossing each other within one frame - in a 1 m corridor they do not.
    def __init__(self, iou_thres: float = 0.3, max_misses: int = 4):
        self.iou_thres = iou_thres
        self.max_misses = max_misses
        self.tracks: List[Track] = []
        self._ids = count(1)

    def update(self, frame: np.ndarray, dets: Sequence[Tuple[Box, float]]) -> List[Track]:
        unmatched = list(range(len(dets)))
        for track in self.tracks:
            best, best_iou = None, self.iou_thres
            for i in unmatched:
                score = iou(track.box, dets[i][0])
                if score >= best_iou:
                    best, best_iou = i, score
            if best is None:
                track.misses += 1
            else:
                unmatched.remove(best)
                track.hits += 1
                track.misses = 0
                track.push(frame, *dets[best])
        for i in unmatched:
            track = Track(next(self._ids), dets[i][0], dets[i][1])
            track.push(frame, *dets[i])
            self.tracks.append(track)
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return self.tracks


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #


class Verifier:
    """Batched SSN inference over every track that has a full crop window."""

    def __init__(
        self,
        weights: Optional[str] = None,
        thresh: float = 0.6,
        need: int = 2,
        max_batch: int = 5,
    ):
        self.model = SSN.load(weights) if weights else SSN().eval()
        self.trained = weights is not None
        self.thresh = thresh
        self.need = need  # consecutive frames above threshold before confirming
        self.max_batch = max_batch
        # Leave one core for navigation / SLAM.
        torch.set_num_threads(max(1, (torch.get_num_threads() or 4) - 1))

    @torch.no_grad()
    def __call__(self, tracks: Sequence[Track]) -> List[Track]:
        """Score ready tracks; return those newly confirmed on this frame."""
        ready = [t for t in tracks if t.ready]
        ready.sort(key=lambda t: -t.conf)
        ready = ready[: self.max_batch]
        if not ready:
            return []
        batch = torch.stack([seq_to_tensor(t.crops) for t in ready])
        probs = torch.sigmoid(self.model(batch)).tolist()
        confirmed = []
        for track, p in zip(ready, probs):
            track.ssn_conf = p
            if p >= self.thresh:
                track.confirmations += 1
                if track.confirmations == self.need:
                    confirmed.append(track)
            else:
                track.confirmations = 0
        return confirmed

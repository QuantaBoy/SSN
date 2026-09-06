"""Runnable checks for the SSN stack. No framework needed:  python tests/test_ssn.py

The one that matters is test_temporal_evidence: it trains the network on a
synthetic set where positives and negatives carry *identical total energy* and
differ only in how that energy is spread over time. If the network can separate
them, the temporal integration is doing real work; if it cannot, the whole
justification for a spiking verifier is gone and the honest move is to ship the
tracker alone.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ssn.localize import (  # noqa: E402
    Arena,
    Camera,
    Pose,
    SurvivorRegistry,
    grid_cell,
    locate,
    ray_to_floor,
)
from ssn.model import CROP, LIF, SSN, T_STEPS  # noqa: E402
from ssn.perception import Tracker, align, crop_patch, iou, seq_to_tensor  # noqa: E402


@torch.no_grad()
def test_lif_integrates_and_leaks():
    lif = LIF(1, tau=2.0)
    v = torch.zeros(1, 1)
    # A sub-threshold input sustained over time must eventually fire.
    fired = [float(lif(torch.full((1, 1), 0.4), v)[0]) for _ in range(6)]
    assert sum(fired) == 0.0, "0.4 with decay 0.5 saturates below 1.0 and must never fire"
    v = torch.zeros(1, 1)
    fired = []
    for _ in range(6):
        s, v = lif(torch.full((1, 1), 0.7), v)
        fired.append(float(s))
    assert sum(fired) > 0, "sustained 0.7 accumulates past threshold and must fire"
    # A single pulse then silence must decay away.
    v = torch.zeros(1, 1)
    s, v = lif(torch.full((1, 1), 0.9), v)
    for _ in range(8):
        s, v = lif(torch.zeros(1, 1), v)
    assert float(v) < 0.05, f"membrane must leak to rest, got {float(v)}"


def test_model_shapes_and_gradients():
    model = SSN()
    x = torch.randn(4, T_STEPS, 2, CROP, CROP)
    y = model(x)
    assert y.shape == (4,), y.shape
    y.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads), "surrogate gradient is dead"
    assert 0.0 < model.last_spike_rate < 1.0, "network is silent or saturated"


def _synthetic(n=96, seed=0):
    """Positives: faint blob in every frame. Negatives: the same total energy in
    one frame. Both share mean intensity, so only the time axis separates them."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:CROP, 0:CROP]
    blob = np.exp(-(((xx - 32) ** 2 + (yy - 32) ** 2) / (2 * 9.0**2))).astype(np.float32)
    xs, ys = [], []
    for i in range(n):
        base = rng.normal(110, 8, (T_STEPS, CROP, CROP)).astype(np.float32)
        label = i % 2
        if label:
            base += blob * 22.0
        else:
            base[rng.integers(T_STEPS)] += blob * 22.0 * T_STEPS
        seq = np.clip(base, 0, 255).astype(np.uint8)
        xs.append(seq_to_tensor(seq))
        ys.append(float(label))
    return torch.stack(xs), torch.tensor(ys)


def test_temporal_evidence():
    torch.manual_seed(0)
    x, y = _synthetic(96, seed=0)
    xv, yv = _synthetic(48, seed=1)
    model = SSN()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(25):
        model.train()
        for i in range(0, len(x), 16):
            opt.zero_grad(set_to_none=True)
            lossf(model(x[i : i + 16]), y[i : i + 16]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        acc = float(((torch.sigmoid(model(xv)) >= 0.5).float() == yv).float().mean())
    assert acc >= 0.75, f"temporal separation failed at {acc:.2f} - the SNN earns nothing"
    print(f"    temporal separation accuracy {acc:.2f}")


def test_tracker_keeps_identity():
    frame = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
    tracker = Tracker()
    for i in range(T_STEPS + 2):
        tracks = tracker.update(frame, [((50 + i, 60, 110 + i, 180), 0.4)])
    assert len(tracks) == 1, f"one target must not split into {len(tracks)} tracks"
    assert tracks[0].id == 1 and tracks[0].hits == T_STEPS + 2
    assert len(tracks[0].crops) == T_STEPS, "crop buffer must hold exactly T frames"
    # A box far away must open a second track, not steal the first.
    tracks = tracker.update(frame, [((10, 10, 40, 50), 0.4)])
    assert len(tracks) == 2 and tracks[1].id == 2


def test_iou_and_crops():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-6
    frame = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
    patch = crop_patch(frame, (100, 100, 160, 200))
    assert patch.shape == (CROP, CROP) and patch.dtype == np.uint8
    assert crop_patch(frame, (10, 10, 10, 10)).shape == (CROP, CROP), "degenerate box must not crash"


def test_alignment_cancels_translation():
    rng = np.random.default_rng(0)
    ref = (rng.random((CROP, CROP)) * 255).astype(np.uint8)
    shifted = np.roll(ref, (0, 4), axis=(0, 1))  # camera slid 4 px right
    fixed = align(shifted, ref)
    assert np.abs(fixed.astype(int) - ref.astype(int)).mean() < np.abs(
        shifted.astype(int) - ref.astype(int)
    ).mean(), "phase correlation must reduce, not increase, residual motion"


def test_seq_tensor_delta():
    seq = np.zeros((T_STEPS, CROP, CROP), np.uint8)
    seq[3:] = 255
    t = seq_to_tensor(seq)
    assert t.shape == (T_STEPS, 2, CROP, CROP)
    assert float(t[0, 1].abs().max()) == 0.0, "delta at t=0 must be zero"
    assert float(t[3, 1].mean()) > 1.9, "delta must fire on the step change"
    assert float(t[4, 1].abs().max()) == 0.0, "delta must be zero on a static frame"


def test_floor_projection():
    cam = Camera(640, 480, 70.0, mount_pitch_deg=20.0)
    pose = Pose(0.0, 0.0, 1.3, 0.0)
    x, y = ray_to_floor(cam, pose, 320, 240)
    expect = 1.3 / math.tan(math.radians(20.0))
    assert abs(x - expect) < 1e-3 and abs(y) < 1e-6, (x, y, expect)
    # Yaw must rotate the projection into the world frame.
    x2, y2 = ray_to_floor(cam, Pose(2.0, 3.0, 1.3, math.pi / 2), 320, 240)
    assert abs(x2 - 2.0) < 1e-6 and abs(y2 - (3.0 + expect)) < 1e-3, (x2, y2)
    # A ray above the horizon has no floor intersection.
    assert ray_to_floor(cam, Pose(0, 0, 1.3, 0.0), 320, 0) is None
    # Bottom-centre of the box is what gets projected.
    assert locate(cam, pose, (300, 100, 340, 240)) == (x, y)


def test_grid_labels():
    assert grid_cell(0.0, 0.0) == "A1"
    assert grid_cell(3.57, 0.0) == "D1"
    assert grid_cell(2.4, 5.9) == "C6"
    assert grid_cell(99.0, 99.0) == "O15", "must clamp to the 15 m arena"
    assert grid_cell(3.5, 1.2, Arena(cell=2.0)) == "B1", "room-sized cells relabel"


def test_registry_merges_and_caps():
    reg = SurvivorRegistry(merge_radius=1.0, min_hits=2, max_count=6)
    assert reg.add(3.5, 1.2, 0.8) is None, "one sighting must not commit"
    event = reg.add(3.6, 1.3, 0.9)
    assert event and event["cell"] == "D2" and event["survivor_id"] == 1
    assert reg.add(3.55, 1.25, 0.7) is None, "an already-committed survivor re-fires no event"
    assert len(reg.survivors) == 1, "three sightings of one person are one survivor"
    for i in range(2, 9):  # six committed is the cap from the mission brief
        reg.add(float(i) * 2, 9.0, 0.9)
        reg.add(float(i) * 2, 9.0, 0.9)
    assert len(reg.committed()) == 6, len(reg.committed())


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok   {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

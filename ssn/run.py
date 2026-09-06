"""Mission runtime: video in, survivor grid events out.

Wires the three stages together - propose/track/verify, then project the verified
box onto the floor and merge it into the registry - and writes one JSON document
of what was found and where.

Pose is injected, not solved here. There is no SLAM in this package; the flight
build supplies a callable that answers "where was the camera at time t" and the
default is a fixed hover pose, which is only honest for bench footage from a
stationary camera. Feeding real flight video through the default pose will
produce confident, wrong grid cells.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

import cv2

from .localize import Arena, Camera, Pose, grid_cell, locate
from .perception import Pipeline, Proposer, Tracker, Verifier
from .model import SSN


@dataclass
class Mission:
    camera: Camera
    arena: Arena
    pipeline: Pipeline
    pose_at: object  # callable: seconds -> Pose, fed by FC telemetry / VIO

    def step(self, frame, timestamp: float) -> list[dict]:
        """One geotagged detection record per verified track. No merging.

        Deduplicating sightings of the same person across viewpoints is map
        building, and the SLAM mapping layer downstream owns that - it has the
        full trajectory and a better pose than anything available here. Merging
        at this layer too would give two components a claim on the same map.
        `SurvivorRegistry` in ssn.localize is that merge, for whoever runs it.
        """
        records = []
        pose = self.pose_at(timestamp)
        for track in self.pipeline.step(frame):
            # floor contact is at the bottom edge of the box, not its centre -
            # projecting the centre puts a standing person half a body too far.
            u = (track.box[0] + track.box[2]) / 2.0
            point = locate(self.camera, pose, u, track.box[3])
            if point is None:
                continue
            records.append({
                "t": round(timestamp, 3),
                "track": track.id,
                "box": [round(float(c), 1) for c in track.box],
                "confidence": round(track.confidence, 3),
                "position": [round(float(point[0]), 3), round(float(point[1]), 3)],
                "cell": grid_cell(self.arena, point),
                "pose": [round(float(c), 3) for c in pose.position],
            })
        return records


def build(width: int, height: int, hfov: float = 70.0, altitude: float = 3.0,
          weights: str | None = None, threshold: float = 0.5,
          modality: str = "rgb", pose_at=None) -> Mission:
    """Assemble the onboard stage. `pose_at` is the FC telemetry / VIO hook.

    The default fixed hover pose is a stand-in for bench footage only. In flight
    this must be the live pose stream, or every geotag is wrong by however far
    the aircraft has moved.
    """
    model = SSN.load(weights) if weights else SSN()
    arena = Arena()
    hover = Pose.from_euler(arena.size / 2, arena.size / 2, altitude)
    return Mission(
        camera=Camera.from_fov(width, height, hfov),
        arena=arena,
        pipeline=Pipeline(Proposer(), Tracker(), Verifier(model, threshold),
                          modality=modality),
        pose_at=pose_at or (lambda _t: hover),
    )


def run(source, out_path: str | None = None, max_frames: int = 0,
        mission: Mission | None = None, show: bool = False) -> dict:
    capture = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    if not capture.isOpened():
        raise SystemExit(f"cannot open video source: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    mission = mission or build(width, height)

    detections, index = [], 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or (max_frames and index >= max_frames):
                break
            detections.extend(mission.step(frame, index / fps))
            index += 1
            if show:
                cv2.imshow("ssn", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        capture.release()
        if show:
            # destroyAllWindows raises on headless builds with no GUI backend
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    # handed to the SLAM mapping layer, which merges these into survivor positions
    report = {"frames": index, "fps": round(float(fps), 3),
              "modality": mission.pipeline.modality, "detections": detections}
    if out_path:
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="run the SSN mission pipeline")
    parser.add_argument("source", help="video file path, or a webcam index")
    parser.add_argument("--out", default="detections.json")
    parser.add_argument("--weights", default=None, help="trained SSN checkpoint")
    parser.add_argument("--modality", choices=("rgb", "thermal"), default="rgb")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--altitude", type=float, default=3.0)
    parser.add_argument("--hfov", type=float, default=70.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)

    if args.weights is None:
        print("warning: untrained SSN - verification output is meaningless",
              file=sys.stderr)
    capture = cv2.VideoCapture(int(args.source) if args.source.isdigit()
                               else args.source)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    capture.release()

    mission = build(width, height, args.hfov, args.altitude, args.weights,
                    args.threshold, args.modality)
    report = run(args.source, args.out, args.max_frames, mission)
    cells = {d["cell"] for d in report["detections"] if d["cell"]}
    print(f"{report['frames']} frames, {len(report['detections'])} geotagged "
          f"detections in {len(cells)} cells -> {args.out}")
    print("Merging these into survivor positions is the SLAM mapping layer's job.")
    return 0


def _demo():
    import tempfile
    from pathlib import Path

    import numpy as np

    from .model import T_STEPS
    from .perception import to_gray

    tmp = Path(tempfile.mkdtemp())
    path = tmp / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0,
                             (320, 240))
    rng = np.random.default_rng(0)
    for step in range(T_STEPS + 4):
        frame = rng.integers(0, 60, (240, 320, 3), dtype=np.uint8)
        cv2.rectangle(frame, (140 + step, 100), (180 + step, 190), (220, 220, 220), -1)
        writer.write(frame)
    writer.release()

    # a stub proposer keeps the check offline - YOLO weights are not the subject
    box_seen = [(140.0, 100.0, 180.0, 190.0)]
    stub = type("Stub", (), {"__call__": lambda self, frame: box_seen})()
    mission = build(320, 240)
    mission.pipeline.proposer = stub
    mission.pipeline.verifier.threshold = 0.0  # untrained net, accept everything

    report = run(path, out_path=str(tmp / "out.json"), mission=mission)
    assert report["frames"] == T_STEPS + 4, report["frames"]
    assert report["detections"], "a fully tracked box should yield detections"
    # nothing may be reported before the verifier has a full window
    assert min(d["t"] for d in report["detections"]) >= (T_STEPS - 1) / 30.0

    # every record must be independently usable by the mapping layer: its own
    # position, its own pose, no reference to a merged survivor identity
    for record in report["detections"]:
        assert set(record) == {"t", "track", "box", "confidence", "position",
                               "cell", "pose"}, record.keys()
        assert record["cell"] is not None
    saved = json.loads((tmp / "out.json").read_text())
    assert saved["detections"] == report["detections"]

    # thermal frames are 16-bit on an arbitrary scale; the converter must map
    # them onto the same 0-255 grey the RGB path produces
    thermal = (rng.integers(2800, 3400, (240, 320)).astype(np.uint16))
    thermal[100:190, 140:180] += 900
    gray, bounds = to_gray(thermal, "thermal")
    assert gray.dtype == np.uint8 and gray.max() > gray.min()
    assert bounds is not None
    # and the bounds must move only slowly, or the delta channel reads the
    # normalisation as motion
    shifted, moved = to_gray(thermal + 500, "thermal", bounds)
    assert abs(moved[0] - bounds[0]) < 500, (bounds, moved)
    print(f"ok: {report['frames']} frames, {len(report['detections'])} detections, "
          f"cell {report['detections'][0]['cell']}, thermal path ok")


if __name__ == "__main__":
    raise SystemExit(main())

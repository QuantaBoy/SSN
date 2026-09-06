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

from .localize import Arena, Camera, Pose, SurvivorRegistry, locate
from .perception import Pipeline, Proposer, Tracker, Verifier
from .model import SSN


@dataclass
class Mission:
    camera: Camera
    arena: Arena
    pipeline: Pipeline
    registry: SurvivorRegistry
    pose_at: object  # callable: seconds -> Pose

    def step(self, frame, timestamp: float) -> list[dict]:
        events = []
        pose = self.pose_at(timestamp)
        for track in self.pipeline.step(frame):
            # floor contact is at the bottom edge of the box, not its centre -
            # projecting the centre puts a standing person half a body too far.
            u = (track.box[0] + track.box[2]) / 2.0
            point = locate(self.camera, pose, u, track.box[3])
            if point is None:
                continue
            survivor = self.registry.add(point, track.confidence)
            if survivor is None:
                continue
            events.append({"t": round(timestamp, 3), "track": track.id,
                           "survivor": survivor["id"], "cell": survivor["cell"],
                           "confidence": round(track.confidence, 3)})
        return events


def build(width: int, height: int, hfov: float = 70.0, altitude: float = 3.0,
          weights: str | None = None, threshold: float = 0.5) -> Mission:
    model = SSN.load(weights) if weights else SSN()
    arena = Arena()
    hover = Pose.from_euler(arena.size / 2, arena.size / 2, altitude)
    return Mission(
        camera=Camera.from_fov(width, height, hfov),
        arena=arena,
        pipeline=Pipeline(Proposer(), Tracker(), Verifier(model, threshold)),
        registry=SurvivorRegistry(arena),
        pose_at=lambda _t: hover,
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

    events, index = [], 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or (max_frames and index >= max_frames):
                break
            events.extend(mission.step(frame, index / fps))
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

    report = {"frames": index, "events": events,
              "survivors": mission.registry.report()}
    if out_path:
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="run the SSN mission pipeline")
    parser.add_argument("source", help="video file path, or a webcam index")
    parser.add_argument("--out", default="survivors.json")
    parser.add_argument("--weights", default=None, help="trained SSN checkpoint")
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
                    args.threshold)
    report = run(args.source, args.out, args.max_frames, mission)
    print(f"{report['frames']} frames, {len(report['events'])} events, "
          f"{len(report['survivors'])} survivors -> {args.out}")
    for survivor in report["survivors"]:
        print(f"  survivor {survivor['id']}: cell {survivor['cell']} "
              f"conf {survivor['confidence']} ({survivor['sightings']} sightings)")
    return 0


def _demo():
    import tempfile
    from pathlib import Path

    import numpy as np

    from .model import T_STEPS

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
    assert report["events"], "a fully tracked box should yield events"
    assert len(report["survivors"]) == 1, report["survivors"]
    assert report["survivors"][0]["cell"] is not None
    # nothing may be reported before the verifier has a full window
    assert min(e["t"] for e in report["events"]) >= (T_STEPS - 1) / 30.0
    saved = json.loads((tmp / "out.json").read_text())
    assert saved["survivors"] == report["survivors"]
    print(f"ok: {report['frames']} frames, {len(report['events'])} events, "
          f"cell {report['survivors'][0]['cell']}")


if __name__ == "__main__":
    raise SystemExit(main())

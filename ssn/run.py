"""Mission runtime: video in, survivor grid tags out.

    python -m ssn.run --source 0 --ssn ssn.pt --pose-file pose.jsonl --out events.jsonl

Emits one JSON line per newly committed survivor, on stdout and to --out, for the
Ground Control Station to draw on the map. Everything here runs onboard; the GCS
is a consumer of this stream, not a participant in the decision.
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import cv2

from .localize import Arena, Camera, Pose, SurvivorRegistry, grid_cell, locate
from .perception import Proposer, Tracker, Verifier


class PoseSource:
    """Drone pose, interpolated to the frame time.

    Reads a JSONL log for bench replay. Onboard, replace `at()` with a MAVLink
    LOCAL_POSITION_NED + ATTITUDE read from the flight controller - the rest of
    the pipeline does not care where the pose came from, only that it is
    timestamped against the frame.
    """

    def __init__(self, path: Optional[str], default_z: float = 1.3):
        self.default_z = default_z
        self.times: List[float] = []
        self.poses: List[Pose] = []
        if path:
            for line in Path(path).read_text().splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                self.times.append(float(d["t"]))
                self.poses.append(
                    Pose(
                        float(d["x"]),
                        float(d["y"]),
                        float(d.get("z", default_z)),
                        float(d["yaw"]),
                        float(d.get("pitch", 0.0)),
                        float(d["t"]),
                    )
                )

    def at(self, t: float) -> Pose:
        if not self.poses:
            return Pose(0.0, 0.0, self.default_z, 0.0, 0.0, t)
        i = min(len(self.poses) - 1, bisect.bisect_left(self.times, t))
        if i and abs(self.times[i - 1] - t) < abs(self.times[i] - t):
            i -= 1
        return self.poses[i]


def draw(frame, tracks, cam, pose, arena):
    for tr in tracks:
        x1, y1, x2, y2 = (int(v) for v in tr.box)
        hot = tr.ssn_conf >= 0.6
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0) if hot else (60, 60, 200), 2)
        p = locate(cam, pose, tr.box)
        cell = grid_cell(*p, arena) if p else "--"
        cv2.putText(
            frame,
            f"#{tr.id} ssn {tr.ssn_conf:.2f} {cell}",
            (x1, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description="SSN indoor survivor detection runtime")
    ap.add_argument("--source", default="0", help="camera index or video path")
    ap.add_argument("--weights", default="yolov8n.pt", help="proposer weights")
    ap.add_argument("--ssn", default=None, help="trained SSN checkpoint")
    ap.add_argument("--pose-file", default=None, help="JSONL pose log for replay")
    ap.add_argument("--out", default=None, help="write survivor events here")
    ap.add_argument("--record", default=None, help="save the flight video for harvesting")
    ap.add_argument("--hz", type=float, default=3.0, help="perception rate")
    ap.add_argument("--thresh", type=float, default=0.6, help="SSN confirm threshold")
    ap.add_argument("--conf", type=float, default=0.15, help="proposer confidence")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--cell", type=float, default=1.0, help="grid box side, metres")
    ap.add_argument("--hfov", type=float, default=70.0)
    ap.add_argument("--mount-pitch", type=float, default=20.0, help="camera down-tilt, degrees")
    ap.add_argument("--max-survivors", type=int, default=6)
    ap.add_argument("--show", action="store_true", help="dev preview window")
    ap.add_argument("--status", action="store_true", help="emit a status line per frame")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source {args.source}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    cam = Camera(w, h, args.hfov, args.mount_pitch)
    arena = Arena(cell=args.cell)
    poses = PoseSource(args.pose_file)
    prop = Proposer(args.weights, conf=args.conf, imgsz=args.imgsz)
    tracker = Tracker()
    verifier = Verifier(args.ssn, thresh=args.thresh)
    registry = SurvivorRegistry(arena=arena, max_count=args.max_survivors)
    if not verifier.trained:
        print("WARNING: SSN is untrained (random weights) - verification is meaningless",
              file=sys.stderr)

    writer = None
    if args.record:
        writer = cv2.VideoWriter(
            args.record, cv2.VideoWriter_fourcc(*"mp4v"), args.hz, (w, h)
        )
    events = open(args.out, "a", buffering=1) if args.out else None
    period = 1.0 / max(0.1, args.hz)
    start = time.perf_counter()
    frames = 0

    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1
            now = t0 - start
            pose = poses.at(now)

            tracks = tracker.update(frame, prop(frame))
            for track in verifier(tracks):
                point = locate(cam, pose, track.box)
                if point is None:
                    continue
                event = registry.add(*point, track.ssn_conf)
                if event:
                    event["t"] = round(now, 2)
                    line = json.dumps(event)
                    print(line, flush=True)
                    if events:
                        events.write(line + "\n")

            if args.status:
                print(
                    json.dumps(
                        {
                            "status": "flying",
                            "t": round(now, 2),
                            "pose": {k: round(v, 2) for k, v in asdict(pose).items()},
                            "cell": grid_cell(pose.x, pose.y, arena),
                            "tracks": len(tracks),
                            "survivors": len(registry.committed()),
                            "ms": round((time.perf_counter() - t0) * 1000, 1),
                        }
                    ),
                    flush=True,
                )
            if writer:
                writer.write(frame)
            if args.show:
                cv2.imshow("ssn", draw(frame, tracks, cam, pose, arena))
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            lag = period - (time.perf_counter() - t0)
            if lag > 0:
                time.sleep(lag)
    finally:
        cap.release()
        if writer:
            writer.release()
        if events:
            events.close()
        if args.show:
            cv2.destroyAllWindows()  # headless OpenCV builds raise here

    print(
        json.dumps(
            {
                "status": "complete",
                "frames": frames,
                "seconds": round(time.perf_counter() - start, 1),
                "survivors": registry.committed(),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

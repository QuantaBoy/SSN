"""Pixel -> floor point -> arena grid cell, plus survivor de-duplication.

The mission is scored on the grid box a survivor is in, not on a bounding box, so
this file is where detection turns into points. Indoors every survivor is on the
floor, which makes the localisation a ray/plane intersection rather than a depth
estimate: cast the ray through the bottom-centre pixel of the detection and
intersect it with the floor plane at the drone's known altitude.

Frames:
  world  - X right, Y forward, Z up. Origin at the arena corner nearest the entry
           point, axes along the arena walls. Metres.
  yaw    - radians, CCW from +X.
  pitch  - radians, nose/camera down positive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Camera:
    width: int = 640
    height: int = 480
    hfov_deg: float = 70.0
    mount_pitch_deg: float = 20.0  # fixed downward tilt of the camera on the airframe

    @property
    def fx(self) -> float:
        return (self.width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)

    @property
    def fy(self) -> float:
        return self.fx  # square pixels; re-derive from a checkerboard calibration


@dataclass
class Pose:
    """Drone pose from the VIO / SLAM front-end, interpolated to the frame timestamp."""

    x: float
    y: float
    z: float  # height above the floor plane, metres
    yaw: float  # radians
    pitch: float = 0.0  # radians, nose-down positive
    t: float = 0.0  # frame timestamp, seconds


@dataclass
class Arena:
    size: float = 15.0  # arena is at most 15 m x 15 m
    cell: float = 1.0  # grid box side; set to 2.0 to label by room instead


def ray_to_floor(
    cam: Camera, pose: Pose, u: float, v: float, floor_z: float = 0.0
) -> Optional[Tuple[float, float]]:
    """Intersect the ray through pixel (u, v) with the floor plane.

    Returns world (x, y) in metres, or None if the ray does not point downward
    (target above the horizon - a detection on a wall, or a bad pose).
    """
    xr = (u - cam.width / 2.0) / cam.fx
    yd = (v - cam.height / 2.0) / cam.fy
    theta = math.radians(cam.mount_pitch_deg) + pose.pitch
    ct, st = math.cos(theta), math.sin(theta)
    cy, sy = math.cos(pose.yaw), math.sin(pose.yaw)

    # optical axis (forward, tilted down by theta) and the camera's down axis
    fwd = (cy * ct, sy * ct, -st)
    down = (-cy * st, -sy * st, -ct)
    right = (sy, -cy, 0.0)

    ray = tuple(xr * right[i] + yd * down[i] + fwd[i] for i in range(3))
    if ray[2] >= -1e-6:
        return None
    scale = (pose.z - floor_z) / -ray[2]
    return pose.x + scale * ray[0], pose.y + scale * ray[1]


def locate(
    cam: Camera, pose: Pose, box: Tuple[float, float, float, float]
) -> Optional[Tuple[float, float]]:
    """World floor position of a detection, from the bottom-centre of its box."""
    x1, y1, x2, y2 = box
    return ray_to_floor(cam, pose, (x1 + x2) / 2.0, y2)


def grid_cell(x: float, y: float, arena: Arena = Arena()) -> str:
    """World metres -> grid label, e.g. 'D4'. Columns are letters along X."""
    n = max(1, int(round(arena.size / arena.cell)))
    col = min(n - 1, max(0, int(x // arena.cell)))
    row = min(n - 1, max(0, int(y // arena.cell)))
    return f"{chr(ord('A') + col)}{row + 1}"


@dataclass
class Survivor:
    id: int
    x: float
    y: float
    conf: float
    hits: int = 1
    committed: bool = False


@dataclass
class SurvivorRegistry:
    """Merges repeated sightings of one survivor and caps the reported count.

    A survivor seen from three viewpoints must be reported once, at one grid box.
    Merging happens in world coordinates, not image space, because the same
    person seen from a different heading has no image-space overlap at all.
    """

    merge_radius: float = 1.0  # metres; roughly one grid cell
    min_hits: int = 2  # sightings before a survivor is committed to the map
    max_count: int = 6  # the brief places at most 6 survivors
    arena: Arena = field(default_factory=Arena)
    survivors: List[Survivor] = field(default_factory=list)
    _next_id: int = 1

    # ponytail: first-come commit, no eviction. A false positive that reaches
    # min_hits holds its slot. Add confidence-ranked eviction only if field
    # testing shows the registry filling with clutter before the real targets.
    def add(self, x: float, y: float, conf: float) -> Optional[Dict]:
        """Record a sighting. Returns an event dict when a survivor is newly committed."""
        near, best = None, self.merge_radius
        for s in self.survivors:
            d = math.hypot(s.x - x, s.y - y)
            if d <= best:
                near, best = s, d
        if near is None:
            near = Survivor(self._next_id, x, y, conf)
            self._next_id += 1
            self.survivors.append(near)
        else:
            w = 1.0 / (near.hits + 1)
            near.x += (x - near.x) * w
            near.y += (y - near.y) * w
            near.conf = max(near.conf, conf)
            near.hits += 1

        committed = sum(1 for s in self.survivors if s.committed)
        if not near.committed and near.hits >= self.min_hits and committed < self.max_count:
            near.committed = True
            return self.event(near)
        return None

    def event(self, s: Survivor) -> Dict:
        return {
            "survivor_id": s.id,
            "x": round(s.x, 2),
            "y": round(s.y, 2),
            "cell": grid_cell(s.x, s.y, self.arena),
            "confidence": round(s.conf, 3),
            "sightings": s.hits,
        }

    def committed(self) -> List[Dict]:
        return [self.event(s) for s in self.survivors if s.committed]

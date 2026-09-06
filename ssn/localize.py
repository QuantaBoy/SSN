"""Turn a verified detection into an arena grid cell.

The mission reports *where* a survivor is, not just that one exists, and it does
so without GPS. What is known instead is the drone pose from the SLAM front-end
and the camera intrinsics, and one strong fact about the scene: survivors are on
the floor. That fact is what makes the problem solvable from a single camera -
the pixel gives a ray, the floor plane gives the depth the ray is missing.

World frame: x east, y north, z up, floor at z = 0, origin at the arena corner.
Camera frame: x right, y down, z along the optical axis (standard pinhole).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# camera->world rotation for a camera pointing straight down with zero yaw:
# optical axis to world -z, image x to world +x, image y (down) to world -y.
NADIR = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


@dataclass(frozen=True)
class Camera:
    """Pinhole intrinsics in pixels."""

    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float) -> "Camera":
        fx = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
        return cls(fx, fx, width / 2.0, height / 2.0)

    def ray(self, u: float, v: float) -> np.ndarray:
        """Unit direction in camera frame through pixel (u, v)."""
        d = np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0])
        return d / np.linalg.norm(d)


@dataclass(frozen=True)
class Pose:
    """Camera position in world metres and camera->world rotation."""

    position: np.ndarray
    rotation: np.ndarray = field(default_factory=lambda: NADIR.copy())

    @classmethod
    def from_euler(cls, x: float, y: float, z: float, yaw: float = 0.0,
                   pitch: float = 0.0, roll: float = 0.0) -> "Pose":
        """Angles in radians. pitch tilts the camera up from straight down."""
        cy_, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
        rz = np.array([[cy_, -sy, 0.0], [sy, cy_, 0.0], [0.0, 0.0, 1.0]])
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
        ry = np.array([[cr, 0.0, sr], [0.0, 1.0, 0.0], [-sr, 0.0, cr]])
        return cls(np.asarray([x, y, z], dtype=float), rz @ NADIR @ rx @ ry)


def locate(camera: Camera, pose: Pose, u: float, v: float) -> np.ndarray | None:
    """Floor point (x, y, 0) seen at pixel (u, v), or None if the ray misses it.

    A ray that points level or upward never meets the floor, and one that meets
    it far behind the drone is a numerically meaningless grazing hit; both are
    rejected rather than reported as a position.
    """
    direction = pose.rotation @ camera.ray(u, v)
    if direction[2] > -1e-3 or pose.position[2] <= 0.0:
        return None
    distance = -pose.position[2] / direction[2]
    return pose.position + distance * direction


@dataclass(frozen=True)
class Arena:
    """Square arena split into a letter x number grid. A1 is the origin corner."""

    size: float = 15.0
    divisions: int = 5

    @property
    def cell(self) -> float:
        return self.size / self.divisions

    def contains(self, point) -> bool:
        return 0.0 <= point[0] <= self.size and 0.0 <= point[1] <= self.size


def grid_cell(arena: Arena, point) -> str | None:
    """'B3' for a floor point, or None if it fell outside the arena."""
    if not arena.contains(point):
        return None
    col = min(int(point[0] / arena.cell), arena.divisions - 1)
    row = min(int(point[1] / arena.cell), arena.divisions - 1)
    return f"{chr(ord('A') + col)}{row + 1}"


class SurvivorRegistry:
    """Deduplicates sightings into at most `capacity` survivor positions.

    The same person is seen from several headings on a survey pass, so raw
    detections must be merged by proximity or the count inflates. Merging keeps a
    running mean of the position - later sightings from closer range pull the
    estimate in without discarding the earlier evidence.
    """

    def __init__(self, arena: Arena | None = None, radius: float = 1.5,
                 capacity: int = 6):
        self.arena = arena or Arena()
        self.radius = radius
        self.capacity = capacity
        self.survivors: list[dict] = []

    def add(self, point, confidence: float = 1.0) -> dict | None:
        point = np.asarray(point, dtype=float)[:2]
        if not self.arena.contains(point):
            return None
        for survivor in self.survivors:
            if np.linalg.norm(survivor["position"] - point) <= self.radius:
                n = survivor["sightings"] + 1
                survivor["position"] += (point - survivor["position"]) / n
                survivor["sightings"] = n
                survivor["confidence"] = max(survivor["confidence"], confidence)
                survivor["cell"] = grid_cell(self.arena, survivor["position"])
                return survivor

        if len(self.survivors) >= self.capacity:
            weakest = min(self.survivors, key=lambda s: s["confidence"])
            if weakest["confidence"] >= confidence:
                return None
            self.survivors.remove(weakest)

        survivor = {"id": len(self.survivors), "position": point, "sightings": 1,
                    "confidence": confidence, "cell": grid_cell(self.arena, point)}
        self.survivors.append(survivor)
        return survivor

    def report(self) -> list[dict]:
        return [{"id": s["id"], "cell": s["cell"], "sightings": s["sightings"],
                 "confidence": round(s["confidence"], 3),
                 "position": [round(float(c), 2) for c in s["position"]]}
                for s in sorted(self.survivors, key=lambda s: -s["confidence"])]


def _demo():
    cam = Camera.from_fov(640, 480, 70.0)
    arena = Arena()

    # nadir at 4 m: the principal point must land directly under the drone
    pose = Pose.from_euler(7.5, 7.5, 4.0)
    point = locate(cam, pose, 320, 240)
    assert np.allclose(point, [7.5, 7.5, 0.0], atol=1e-6), point
    assert grid_cell(arena, point) == "C3"

    # a pixel right of centre must land east of the drone, and the offset must
    # scale with altitude - that is the projection working, not a constant.
    right = locate(cam, pose, 480, 240)
    assert right[0] > point[0] and abs(right[1] - point[1]) < 1e-6
    high = locate(cam, Pose.from_euler(7.5, 7.5, 8.0), 480, 240)
    assert abs((high[0] - 7.5) - 2 * (right[0] - 7.5)) < 1e-6

    # yaw 90 deg turns that eastward offset into a northward one
    turned = locate(cam, Pose.from_euler(7.5, 7.5, 4.0, yaw=math.pi / 2), 480, 240)
    assert abs(turned[0] - 7.5) < 1e-6 and turned[1] > 7.5

    # rays that cannot meet the floor are refused, not extrapolated
    assert locate(cam, Pose.from_euler(7.5, 7.5, 4.0, pitch=math.pi / 2), 320, 240) is None
    assert grid_cell(arena, [16.0, 2.0]) is None
    assert grid_cell(arena, [0.0, 0.0]) == "A1"
    assert grid_cell(arena, [14.9, 14.9]) == "E5"

    # one survivor seen twice from 0.4 m apart is one survivor, not two
    reg = SurvivorRegistry(arena)
    reg.add([3.0, 3.0], 0.7)
    reg.add([3.4, 3.0], 0.9)
    assert len(reg.survivors) == 1
    assert reg.survivors[0]["sightings"] == 2
    assert reg.survivors[0]["confidence"] == 0.9
    assert abs(reg.survivors[0]["position"][0] - 3.2) < 1e-6
    reg.add([12.0, 12.0], 0.8)
    assert len(reg.survivors) == 2

    # the cap holds, and only a stronger sighting may displace a weaker one
    reg = SurvivorRegistry(arena, radius=0.5, capacity=2)
    reg.add([1.0, 1.0], 0.9)
    reg.add([5.0, 5.0], 0.4)
    assert reg.add([9.0, 9.0], 0.3) is None
    assert reg.add([9.0, 9.0], 0.8) is not None
    assert len(reg.survivors) == 2
    assert {s["cell"] for s in reg.survivors} == {"A1", "D4"}
    print(f"ok: nadir {grid_cell(arena, point)}, report {reg.report()[0]['cell']}")


if __name__ == "__main__":
    _demo()

# SSN — Spiking Survivor Network

Onboard human detection for **NIDAR Track 1 · AirMouse**: an autonomous drone entering a covered
15 m × 15 m maze with no GPS, finding up to six survivors, and tagging the grid box each one is in
on a 2D map that is generated during flight.

This repository is the **perception and localisation** half of that mission. Video and drone pose go
in; JSON survivor events with grid coordinates come out, live, on the flight computer. SLAM/VIO,
path planning, flight control and the Ground Control Station are separate subsystems that consume
this stream — see [Interfaces](#interfaces).

```
frame ──► YOLOv8n proposer ──► IoU tracker ──► aligned crop buffer ──► SSN verifier ──► floor
          (recall-tuned)       (identity)      (T = 6 frames)          (temporal vote)   projection
                                                                                            │
                                             {"survivor_id": 3, "cell": "D7", ...} ◄──── registry
                                                                                        (dedupe, cap 6)
```

## Why a spiking network here

The honest claim, and the only one that survives a judge who knows SNNs:

> **Temporal evidence accumulation for false-positive rejection.**

Indoors the survivors are 0.5–4 m from the camera, so they are large in frame and a single-frame
detector already finds them. The problem is not sensitivity, it is **everything else the proposer
also calls a person**: debris silhouettes, a mannequin arm, motion-blurred smears as the drone yaws
through a junction, a jacket over a chair. Each false positive puts a wrong marker on the rescue
team's map.

A per-frame detector throws away the previous frame's evidence. The LIF membranes in this network do
not: a real survivor is *coherent over time* and accumulates past threshold, while blur artefacts and
one-frame clutter leak away. `tests/test_ssn.py::test_temporal_evidence` proves the mechanism on
synthetic data where positives and negatives carry **identical total energy** and differ only in how
that energy is distributed over time.

What this network does **not** buy, and what you should never claim:

| Claim | Verdict |
|---|---|
| Lower energy | **No.** A CPU executes a zero at the same cost as a one. Energy wins need neuromorphic silicon (Loihi 2, Akida, Speck), not an ARM core. |
| Lower latency | **No.** T timesteps cost roughly T forward passes. It is affordable only because it runs on 64×64 crops of at most 5 candidates. |
| Higher per-frame accuracy | **No.** Directly-trained SNNs sit at or slightly below equivalent ANNs on single frames. |
| Fewer false positives at equal recall | **Yes.** This is the whole case. `ssn/train.py` reports exactly this number. |

## Install

```bash
pip install -r requirements.txt
```

On the flight computer use `opencv-python-headless`, and export the proposer to NCNN or ONNX
(`yolo export model=yolov8n.pt format=ncnn`) — the PyTorch proposer is for the bench, not the drone.
The SSN itself is 24k parameters and runs in PyTorch on CPU inside the budget.

## Workflow

**1 — Fly and record.** The training set is your own footage; no public dataset shows a survivor
lying in a 1 m indoor corridor filmed from 1.3 m. Fly the maze with the mission camera, at mission
speed, with the survivors staged the way the organizers will stage them.

```bash
python -m ssn.run --source 0 --record flight01.mp4 --hz 3
```

**2 — Harvest labelled sequences.** A large offline teacher labels what the small onboard proposer
found. No manual sequence labelling. The negatives are the onboard proposer's own false positives —
exactly the distribution the verifier will face in the maze.

```bash
python -m ssn.data harvest flight01.mp4 --out data/seqs --teacher yolov8l.pt
```

**3 — Train.** The final line is the Gate B decision, not the accuracy.

```bash
python -m ssn.train --data data/seqs --out ssn.pt --epochs 30
# ...
# Gate B  recall 0.94  false positives: proposer 61 -> SSN 34  (44% cut; gate is >= 30%)
```

**4 — Benchmark on the flight computer.** Not on a laptop.

```bash
python -m ssn.bench --ssn ssn.pt --proposer yolov8n.pt --threads 3
```

**5 — Fly the mission.**

```bash
python -m ssn.run --source 0 --ssn ssn.pt --out events.jsonl --hz 3 --status
{"survivor_id": 1, "x": 3.55, "y": 1.25, "cell": "D2", "confidence": 0.91, "sightings": 2, "t": 41.2}
```

## Interfaces

| Boundary | Contract |
|---|---|
| **Pose in** | `ssn.run.PoseSource.at(t)` returns a `Pose`. For replay it reads a JSONL log; onboard, replace that one method with a MAVLink `LOCAL_POSITION_NED` + `ATTITUDE` read from the flight controller. Interpolate to the frame timestamp — a 250 ms pose lag at 1 m/s is a 0.25 m tagging error. |
| **Survivors out** | One JSON line per newly committed survivor, on stdout and `--out`. `cell` is the grid box the GCS marks. Emitted during flight; nothing is deferred to after landing, as the brief requires. |
| **Status out** | With `--status`, one JSON line per frame: pose, current grid cell, live track count, committed survivor count, per-frame latency. This is what the GCS mission-status panel reads. |
| **Map** | Not in this repo. The occupancy map comes from the SLAM subsystem; this stream supplies the markers to draw on it, in the same world frame. |

World frame: X right, Y forward, Z up, origin at the arena corner nearest the entry point, axes along
the arena walls, metres. Yaw in radians CCW from +X. The SLAM front-end must publish in this frame,
or grid labels will be wrong even when detection is perfect.

## Calibration

Real hardware never matches the datasheet. These are the knobs to tune on the actual airframe, and
every one of them moves the reported grid cell:

| Flag | Default | What it is |
|---|---|---|
| `--hfov` | 70° | Camera horizontal field of view. Measure it from a checkerboard calibration, do not trust the spec sheet. |
| `--mount-pitch` | 20° | Fixed downward tilt of the camera on the frame. **The single largest source of tagging error.** Measure it on the bench with the drone level; 5° of error at 3 m range is 0.5 m on the floor. |
| `--cell` | 1.0 m | Grid box side. Set to 2.0 to label by room instead of by metre. |
| `--thresh` | 0.6 | SSN confirm threshold. Lower it to favour recall, raise it to conserve markers. |
| `--conf` | 0.15 | Proposer confidence. Deliberately low — the verifier is what removes the junk. |
| `--imgsz` | 416 | Proposer input size. Targets are large indoors; 640 buys little and costs a third of the frame budget. |

`Pose.z` is height above the **floor plane**, from the downward rangefinder, not a barometer.

## Gates

| Gate | Condition | Where it is measured |
|---|---|---|
| **A** | The plain YOLOv8n proposer + tracker + geotagging flies the maze end to end and tags survivors, before any SSN work continues. | `python -m ssn.run` with `--thresh 0.0` |
| **B** | SSN cuts false positives **≥ 30 %** at equal recall, **and** scores 5 candidates in **≤ 120 ms** on the flight computer. | last line of `ssn/train.py`; `ssn/bench.py` |

Miss Gate B and the correct move is to ship the proposer plus tracker and keep the SSN as a written
contribution. The gate exists so a missed target costs a feature, never the mission.

Do not set an 80 % mAP target. That number belongs to a different problem regime and would have you
scrapping a working system.

## Layout

| File | Contents |
|---|---|
| `ssn/model.py` | LIF neuron with learnable time constant, surrogate-gradient spiking, the SSN network. 24k parameters. |
| `ssn/perception.py` | Proposer, IoU tracker, phase-correlation crop alignment, batched verifier. |
| `ssn/localize.py` | Pixel → floor-plane ray cast → world point → grid label; survivor dedupe and the 6-survivor cap. |
| `ssn/data.py` | Teacher-labelled sequence harvesting, augmentation, dataset. |
| `ssn/train.py` | Training loop and the Gate B report. |
| `ssn/bench.py` | Latency percentiles against the frame budget. |
| `ssn/run.py` | Mission runtime and the JSON event stream. |
| `tests/test_ssn.py` | `python tests/test_ssn.py` — 10 assertions, ~10 s, no framework. |

## Known ceilings

Deliberate simplifications, each marked with a `ponytail:` comment at its site:

- **Greedy IoU tracking, no motion model.** Fine while targets do not cross within one frame, which
  in a 1 m corridor they do not. Swap in ByteTrack if that stops being true.
- **First-come survivor commit, no eviction.** A false positive that reaches `min_hits` holds its
  slot against a later real survivor. Add confidence-ranked eviction only if field testing shows the
  registry filling with clutter before the real targets are found.
- **Teacher-labelled training data.** The teacher's own misses become false negatives in the training
  set. Spot-check a sample of label-0 sequences before training on them.

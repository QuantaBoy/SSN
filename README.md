# SSN — Spiking Survivor Network

Onboard survivor detection for the NIDAR AirMouse mission: an autonomous drone
searching a 15×15 m indoor maze with no GPS, reporting which grid cell each
survivor is in.

## The problem this solves

A single-frame person detector cannot be trusted indoors after a collapse. Run
YOLOv8n at a confidence low enough to catch a survivor who is prone, partly
buried and badly lit, and it also returns debris, mannequin limbs, reflections
and motion-blurred clutter. Raise the confidence to remove those and it starts
missing the people it exists to find.

So the proposer is left at high recall, and a second stage decides what was real.
The insight it exploits is temporal: **a survivor is present in every frame; an
artefact is not.** A spiking network integrates evidence across a short window in
its membrane potentials, so weak-but-persistent evidence accumulates past
threshold while a one-frame artefact leaks away before it can fire.

Full function-level architecture, call graph and design rationale:
[`docs/architecture.html`](docs/architecture.html) — open it in a browser.

## Pipeline

```
ONBOARD (this package)
  frame ──► ego-motion warp ──► YOLOv8n proposer ──► IoU tracker
  RGB or thermal                                        │
                                6-frame window per track ▼
                                                SSN spiking verifier
                                                        │
                        pose from FC telemetry / VIO ──► geotag
                                                        │
                                   geotagged detections ▼
─────────────────────────────────────────────────────────────────
DOWNSTREAM
                                          SLAM mapping layer
                                   (merges sightings into survivors)
```

This package ends at geotagged detections. It deliberately does **not** merge
sightings of the same person into a survivor list — the mapping layer owns that,
because it has the full trajectory and a better pose. `SurvivorRegistry` in
`ssn.localize` implements the merge for whoever runs it downstream.

| stage | module | cost (this machine, p95) |
|---|---|---|
| ego-motion warp | `ssn.perception.align` | 15.5 ms |
| proposer | `ssn.perception.Proposer` | run `--proposer` to measure |
| verifier, 5 candidates | `ssn.model.SSN` | 9.8 ms |
| floor projection | `ssn.localize` | negligible |

The warp costs more than the network. If the Pi 5 runs tight, cut the flow
estimate to quarter resolution before touching the SSN.

## Install

```bash
pip install -r requirements.txt
```

On the aircraft use `opencv-python-headless`, and export the proposer to NCNN
(`yolo export model=yolov8n.pt format=ncnn`) before flying.

## Use

```bash
# 1. harvest labelled windows from footage (YOLOv8l teacher labels the student's candidates)
python -m ssn.data flight.mp4 --out dataset --modality rgb

# 2. train, scored on false positives removed at matched recall
python -m ssn.train --data dataset --epochs 40 --out ssn.pt

# 3. measure latency on the hardware you will actually fly
python -m ssn.bench --proposer

# 4. run the onboard stage
python -m ssn.run flight.mp4 --weights ssn.pt --modality rgb --out detections.json
```

`--modality thermal` switches the input path: thermal cores return radiometric
counts on an arbitrary scale, so they get a percentile stretch whose bounds move
only slowly between frames. A per-frame stretch would make the normalisation
itself flicker and the ego-motion channel would read that as scene movement.
Harvest and run must use the same modality or the network trains on an input
distribution it never meets in flight.

Without `--weights` the verifier has random weights and its output is noise.
`run.py` says so on stderr rather than quietly emitting confident nonsense.

## How it is scored

Accuracy is the wrong number — harvested windows are mostly negatives, so
"no survivor" to everything scores well and is useless. The metric is
**false-positive reduction at matched recall**: fix the threshold where the
network still keeps 98% of real survivors, then count how many of the proposer's
false positives it removed. Gate B asks for ≥30%.

Recall is held near 1.0 rather than traded away. A missed survivor in a collapsed
building is not a metric.

## Design decisions worth knowing

- **Ego-motion compensation is per frame, not per candidate.** One affine warp of
  the whole previous frame, then a signed difference. On a moving drone an
  unwarped difference is dominated by camera translation and a barely-moving
  survivor disappears into it. One warp per frame instead of one per crop is what
  makes this affordable on a Pi.
- **Conv and BatchNorm are batched over `T*B` in one call**; only the neuron
  recurrence loops over time. That fusion is the difference between usable and
  unusable inference on ARM.
- **Augmentation is temporally coherent.** Motion blur, occluding debris and
  exposure are drawn once per window, because each has a cause that outlasts a
  frame. Only sensor noise is per-frame. Re-rolling the others per frame would
  teach the network that one-frame flicker is normal — destroying the exact
  signal it classifies on.
- **Floor contact is the bottom edge of the box, not its centre.** Projecting the
  centre puts a standing person half a body too far away.
- **Pose is injected, not solved.** There is no SLAM here — `build(pose_at=…)`
  takes the live pose stream from FC telemetry or VIO. The default fixed hover
  pose is a stand-in for bench footage; real flight video through it yields
  confident, wrong geotags.
- **Mapping is downstream.** The onboard stage emits one record per verified
  detection with its own position and pose, and stops. Merging those into
  survivor positions belongs to the SLAM mapping layer; doing it here as well
  would give two components a claim on the same map.

## Tests

```bash
python tests/test_ssn.py     # or: pytest tests/
```

Each module owns its checks in `_demo()`, next to the code they constrain. They
assert behaviour, not shapes — that persistent weak drive fires while a single
pulse does not, that a pan cancels, that projected offset scales with altitude,
that augmentation stays coherent, that the matched-recall metric refuses to buy
false-positive reduction by dropping survivors.

## Status

Everything above runs and is checked. The model is **untrained** — that needs
maze footage, which does not exist yet. Until then the pipeline is verified
mechanically (synthetic video, synthetic windows) and the Gate B number is not
assessable.

Next: capture footage → `ssn.data` → `ssn.train` → confirm ≥30% at 98% recall →
rerun `ssn.bench` on the Pi 5 → fly it.

# SSN Operating Guidelines

How to capture data for the SSN, train it, validate it, and put it on the aircraft —
and the mistakes that quietly produce a model that scores well and fails in flight.

For what the code *is*, see [`architecture.html`](architecture.html). This document is
what to *do*.

---

## 0. Scope

This package is the **onboard stage**: video in, geotagged detections out.

```
camera ──► SSN onboard stage ──► geotagged detections ──► SLAM mapping layer
 RGB                             (this package ends here)   (merges into survivors)
```

It does **not** decide how many survivors there are. Merging repeat sightings of the
same person is the mapping layer's job — it has the full trajectory and a better pose.
If you find yourself adding deduplication here, you are building the map twice.

---

## 1. Footage capture

This is the highest-leverage step and the easiest to get wrong. A week of careful
capture beats a month of model work.

### Shoot this

| What | Why |
|---|---|
| Person lying on the floor, partly occluded | The actual target. Standing people are the easy case and not the mission. |
| Poor and uneven lighting | A collapsed interior is not evenly lit. Train for it. |
| Clutter with **no** person — bags, coats, cushions, boxes, debris | These become the negatives. Without them the network never learns to reject. |
| Varied range, angle and height | The verifier must work at the distances you will actually fly. |
| Continuous camera motion | The delta channel is half the input. |

### Non-negotiables

- **Keep the camera moving.** A tripod shot teaches the ego-motion channel nothing.
  Walk, circle, approach, pass by.
- **Record at the frame rate you will fly at.** The six-frame window spans wall-clock
  time, so a window at 60 fps covers half the motion of one at 30 fps. Train and fly at
  the same rate.
- **Do not use `--stride` to speed up harvesting.** It widens the gap between frames in
  training but not at runtime, so the network learns an inter-frame motion magnitude the
  mission never produces. If harvest is too slow, shoot shorter clips or drop the capture
  frame rate — do not skip frames.
- **Shoot the lighting you will fly in, not better lighting.** `crop_pair` standardises
  each luma crop, so a uniform exposure offset costs nothing — but a blown-out doorway or
  a crushed-black corner is not an affine change and cannot be undone. Those frames have
  to be in the dataset, not avoided during capture.

### Start small

Shoot **one 30-second clip** and take it all the way through harvest before capturing
hours of footage. The tally in step 2 will tell you whether your lighting, range and
framing actually work — and that is much cheaper to learn now.

---

## 2. Harvesting

```bash
python -m ssn.data clip.mp4 --out dataset
```

The first run downloads YOLOv8n and YOLOv8l (~90 MB). YOLOv8l runs on **every frame** and
is slow on CPU — time the short clip and extrapolate before committing to long footage.

### Read the tally

The printed `N windows (P positive, Q negative)` is the whole diagnosis.

| Symptom | Cause | Fix |
|---|---|---|
| `0 positive` | The teacher never confirmed anyone. Person too small, too dark, too occluded for YOLOv8l. | Fix the footage: closer, brighter, less occluded. Not a code problem. |
| `0 negative` | No clutter was proposed. Scenes too clean. | Shoot messier environments. |
| Very few windows total | Tracks are breaking before six frames. | Smoother camera motion; check the person stays in frame. |
| Positives ≈ negatives | Suspiciously balanced — usually means clutter is being labelled positive. | Inspect a few windows. Check `MATCH_IOU`. |

Expect **mostly negatives**. That imbalance is real and is handled at training time by
`pos_weight`, not by discarding data.

### Target volume

Aim for a few thousand windows with **at least several hundred positive**. Below roughly
200 positives the matched-recall metric gets too coarse to trust — at 98% recall you are
measuring the threshold against a handful of samples.

---

## 3. Training

```bash
python -m ssn.train --data dataset --epochs 40 --out ssn.pt
```

### Read the right number

Ignore loss and never look at accuracy. The harvested set is mostly negatives, so
answering "no survivor" to everything scores well and is worthless.

The number that matters is **false positives removed at matched recall**: the threshold is
fixed where the network still keeps 98% of real survivors, and the metric is how many of
the proposer's false positives it dropped at that setting. Gate B asks for **≥ 30%**.

Recall is held near 1.0 rather than traded off. A missed survivor is not a metric.

### When to stop

The checkpoint is written whenever FP-reduction improves, so the file on disk is the best
epoch, not the last. Stop when FP-reduction has been flat for ~10 epochs. If it never
climbs above ~10%, the problem is almost always the data, not the hyperparameters —
go back and look at actual harvested windows.

### Sanity check before trusting a run

```bash
python -m ssn.train --synthetic 256 --epochs 6
```

Should reach high FP-reduction within a few epochs on the synthetic persistent-vs-flash
task. If that fails, something is broken in the loop itself and no amount of real data
will help.

---

## 4. Validation before flight

Do all four. Skipping any one of them is how a system that passes on a laptop fails on
the aircraft.

1. **Gate B on held-out real data.** Not synthetic. Not the training split.
   ```bash
   python -m ssn.evaluate --data dataset --weights ssn.pt
   ```
   Reads `output/report.txt` back to the terminal and leaves `metrics.json`,
   `curves.png` and `scores.npz` beside it. Check the **per-clip table**, not just the
   total: a clip that holds only one class proves nothing, and a total driven by such a
   clip is a model that learned which footage a window came from.
2. **Latency on the actual flight computer.**
   ```bash
   python -m ssn.bench --proposer
   ```
   Numbers from a development machine are a lower bound and nothing more. Re-run on the
   Pi 5 or Orin with the proposer exported to NCNN.
3. **Camera calibration.** `Camera.from_fov` assumes an ideal pinhole at a *stated* field
   of view. That is a placeholder. Calibrate the real lens and supply true intrinsics, or
   every geotag inherits the error.
4. **Pose wiring.** Confirm `build(pose_at=…)` is fed by live FC telemetry or VIO. The
   default fixed hover pose is for bench footage only — flying with it produces confident,
   precisely wrong positions.

---

## 5. Deployment

```bash
python -m ssn.run flight.mp4 --weights ssn.pt --out detections.json
```

- Install `opencv-python-headless` on the aircraft, not `opencv-python`.
- Export the proposer: `yolo export model=yolov8n.pt format=ncnn`.
- Without `--weights` the model has random weights. It warns on stderr and its output is
  noise — never interpret a run that printed that warning.
- Checkpoints (`*.pt`), datasets and footage are gitignored on purpose. Share a trained
  checkpoint as a release asset or via LFS, not a plain commit.

---

## 6. Invariants — do not break these

Each of these exists because breaking it produces a model that looks fine and is not.

- **Augmentation is coherent across a window.** Motion blur and occlusion are drawn once
  per window because each has a cause that outlasts a frame; only sensor noise is
  per-frame. Re-rolling the rest per frame teaches the network that one-frame flicker is
  normal — destroying the exact signal it classifies on. Asserted in `data._demo`.
- **Normalization is per crop, never per frame.** `crop_pair` standardises the luma crop
  so exposure cancels. Doing it on the whole frame instead would remap between frames and
  `align` would read that remapping as scene motion. Asserted in `perception._demo`.
- **The delta channel is never normalized.** It is a difference, so it carries no exposure
  to remove, and rescaling it destroys the magnitude that separates a moving survivor from
  noise. Asserted in `perception._demo`.
- **One ego-motion warp per frame, not per candidate.** Per-candidate warping is the
  difference between fitting the frame budget and not.
- **Harvest and run use the same frame rate.**
- **The onboard stage does not merge detections.** That is the mapping layer's decision.
- **Never `torch.set_grad_enabled(False)` at module scope.** It leaks out of the function
  and silently breaks training later in the same process. Use a scoped `torch.no_grad()`.
  (This was a real bug, caught by test ordering.)

Run `python tests/test_ssn.py` after any change. Eight checks, under ten seconds. They
assert behaviour, not shapes — if one fails, something real broke.

---

## 7. Known gaps

Honest status, so nobody builds on a false assumption.

| Gap | Consequence | Blocked on |
|---|---|---|
| Model is untrained | Detection output is noise | Footage |
| `harvest` never run with real YOLO weights | Detector integration unproven; plumbing is proven with stubs | First real clip |
| No SLAM / pose source wired | Geotags wrong in flight | The mapping/VIO layer |
| Camera intrinsics are a guessed FOV | Position error in every geotag | Calibration session |
| Never run on Pi 5 / Orin | Latency margin unverified on target | Hardware access |
| Non-affine exposure damage unmodelled | Blown highlights / crushed blacks are not cancelled by crop standardisation | Footage shot against bright doorways |

### The question that gates several of these

`organizer-query.md` question 1 asks whether survivors are live volunteers, heated
mannequins, or unheated dummies. This is not a detail:

- If **unheated mannequins**, YOLOv8l will not label them as people. The entire
  teacher-student scheme produces zero positives and needs replacing with hand-labelling
  or a different teacher.
- If **live volunteers**, proxy footage of real people is exactly right and can be shot
  immediately.

The sensor half of that question is already settled: the payload is RGB only, so nothing
in this package waits on a thermal answer. What still gates capture is whether YOLOv8l
will label the targets at all.

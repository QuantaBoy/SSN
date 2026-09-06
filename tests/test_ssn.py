"""Runs every module's self-check. Works under pytest or as a plain script.

Each module owns its own assertions in `_demo()`, next to the code they
constrain, so this file only has to find them and run them.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ssn import bench, data, localize, model, perception, run, train  # noqa: E402

MODULES = [model, perception, localize, data, train, bench, run]


def test_model():
    model._demo()


def test_perception():
    perception._demo()


def test_localize():
    localize._demo()


def test_data():
    data._demo()


def test_train():
    train._demo()


def test_bench():
    bench._demo()


def test_run():
    run._demo()


def test_harvest_chain():
    """harvest -> npz -> SequenceDataset -> train, with stub detectors.

    Spans four modules, so it lives here rather than in any one _demo. Stubs stand
    in for YOLO: the point is the plumbing between the stages, and pulling 90 MB of
    weights would make this untestable offline.
    """
    import tempfile
    from pathlib import Path

    import cv2
    import numpy as np
    import torch

    from ssn.data import SequenceDataset, harvest
    from ssn.model import CROP, SSN, T_STEPS
    from ssn.train import _loaders, train as run_train

    person = (140.0, 100.0, 180.0, 190.0)
    clutter = (30.0, 40.0, 70.0, 110.0)

    tmp = Path(tempfile.mkdtemp())
    clip = tmp / "clip.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
    rng = np.random.default_rng(0)
    for i in range(40):
        frame = rng.integers(0, 70, (240, 320, 3), dtype=np.uint8)
        cv2.rectangle(frame, (140 + i % 5, 100), (180 + i % 5, 190), (215,) * 3, -1)
        cv2.rectangle(frame, (30, 40), (70, 110), (140,) * 3, -1)
        writer.write(frame)
    writer.release()

    boxes = type("B", (), {})
    result = lambda b: type("R", (), {"boxes": type("X", (), {"xyxy": torch.tensor(b)})})()
    teacher = type("T", (), {"predict": lambda self, f, **k: [result([list(person)])]})()
    proposer = type("P", (), {"__call__": lambda self, f: [person, clutter]})()

    out = tmp / "dataset"
    tally = harvest(str(clip), str(out), proposer=proposer, teacher=teacher)
    # the teacher confirms only the person, so both classes must appear
    assert tally["pos"] > 0, "teacher matching produced no positives"
    assert tally["neg"] > 0, "clutter was not harvested as negative"

    dataset = SequenceDataset(str(out))
    assert len(dataset) == tally["pos"] + tally["neg"]
    x, y = dataset[0]
    assert x.shape == (T_STEPS, 2, CROP, CROP), x.shape
    assert float(y) in (0.0, 1.0)

    train_loader, val_loader = _loaders(dataset, batch_size=4)
    best = run_train(SSN(), train_loader, val_loader, epochs=1, verbose=False)
    assert "fp_reduction" in best


if __name__ == "__main__":
    checks = [(m.__name__, m._demo) for m in MODULES]
    checks.append(("harvest chain", test_harvest_chain))

    failed = 0
    for name, check in checks:
        try:
            check()
        except Exception as error:  # noqa: BLE001 - report all, fail at the end
            failed += 1
            print(f"FAIL {name}: {type(error).__name__}: {error}")
        else:
            print(f"PASS {name}")
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    raise SystemExit(1 if failed else 0)

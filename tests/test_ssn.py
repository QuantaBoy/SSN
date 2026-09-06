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


if __name__ == "__main__":
    failed = 0
    for module in MODULES:
        name = module.__name__
        try:
            module._demo()
        except Exception as error:  # noqa: BLE001 - report all, fail at the end
            failed += 1
            print(f"FAIL {name}: {type(error).__name__}: {error}")
        else:
            print(f"PASS {name}")
    print(f"\n{len(MODULES) - failed}/{len(MODULES)} modules passed")
    raise SystemExit(1 if failed else 0)

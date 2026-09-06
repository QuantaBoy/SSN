"""SSN - Spiking Survivor Network: onboard human detection for the NIDAR
AirMouse indoor GPS-denied search mission."""

from .data import SequenceDataset, augment, harvest
from .localize import Arena, Camera, Pose, SurvivorRegistry, grid_cell, locate
from .model import CROP, LIF, SSN, T_STEPS
from .perception import Pipeline, Proposer, Track, Tracker, Verifier
from .run import Mission
# the train *function* is deliberately not re-exported here: it would shadow the
# ssn.train module for anyone doing `from ssn import train`.
from .train import matched_recall

__all__ = [
    "SSN",
    "LIF",
    "CROP",
    "T_STEPS",
    "Proposer",
    "Tracker",
    "Track",
    "Verifier",
    "Pipeline",
    "Mission",
    "Camera",
    "Pose",
    "Arena",
    "SurvivorRegistry",
    "locate",
    "grid_cell",
    "harvest",
    "augment",
    "SequenceDataset",
    "matched_recall",
]

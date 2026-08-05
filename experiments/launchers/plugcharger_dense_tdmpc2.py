#!/usr/bin/env python3
"""Launcher wrapper for PlugChargerDense-v1.

Imports the custom env module (registers PlugChargerDense-v1 with gymnasium)
before delegating to ManiSkill's stock TD-MPC2 train.py via exec.
The custom env lives at experiments/envs/plugcharger_dense.py and defines
PlugChargerDense-v1 with a staged dense reward (reach→touch→grasp→insert + angular alignment).
"""
import importlib
import sys
from pathlib import Path

# Register PlugChargerDense-v1 before train.py calls gym.make()
importlib.import_module("experiments.envs.plugcharger_dense")

# Now exec train.py with the same argv (minus this script's path)
train_py = str(Path(__file__).resolve().parent.parent / "benchmarks" / "ManiSkill" / "examples" / "baselines" / "tdmpc2" / "train.py")
sys.argv = [train_py] + sys.argv[1:]
exec(Path(train_py).read_text(), {"__name__": "__main__", "__file__": train_py})

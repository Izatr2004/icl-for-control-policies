"""Run ManiSkill's unchanged PPO baseline with PushTVarGoal-v1 registered.

Usage: python state_rl/run_ppo.py /path/to/ManiSkill/examples/baselines/ppo/ppo.py --env_id=PushTVarGoal-v1 [ppo.py args]

Optimizer state: ppo.py saves only the agent weights, so resuming from a checkpoint restarts Adam, and
the fresh optimizer's oversized first steps wrecked a trained policy (continuation run, 23 Nov: 67% -> 0%
within 1M steps). This wrapper therefore
  * saves Adam's state next to every checkpoint ppo.py writes, as <checkpoint>.optim
    (not *.pt, so the eval tool's checkpoint search ignores it), and
  * when ppo.py loads --checkpoint, also loads <checkpoint>.optim into the optimizer if it exists.
Training math is unchanged; ppo.py is run as is.
"""
import os
import runpy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pusht_vargoal  # noqa: E402,F401  registers PushTVarGoal-v1

_OPTIMIZERS = []
_adam_init = torch.optim.Adam.__init__


def _recording_adam_init(self, *args, **kwargs):
    _adam_init(self, *args, **kwargs)
    _OPTIMIZERS.append(self)


torch.optim.Adam.__init__ = _recording_adam_init

_save = torch.save
_load = torch.load


def _save_with_optimizer(obj, f, *args, **kwargs):
    _save(obj, f, *args, **kwargs)
    if isinstance(f, (str, os.PathLike)) and str(f).endswith(".pt") and _OPTIMIZERS:
        _save(_OPTIMIZERS[0].state_dict(), str(f) + ".optim")



def _load_with_optimizer(f, *args, **kwargs):
    obj = _load(f, *args, **kwargs)
    if isinstance(f, (str, os.PathLike)) and str(f).endswith(".pt"):
        optim_file = str(f) + ".optim"
        if os.path.exists(optim_file) and _OPTIMIZERS:
            _OPTIMIZERS[0].load_state_dict(_load(optim_file))
            print(f"[run_ppo] loaded optimizer state from {optim_file}", flush=True)
        elif _OPTIMIZERS:
            print(f"[run_ppo] WARNING: no optimizer state at {optim_file}; Adam starts fresh", flush=True)
    return obj


torch.save = _save_with_optimizer
torch.load = _load_with_optimizer

ppo_path = sys.argv[1]
sys.argv = [ppo_path] + sys.argv[2:]
runpy.run_path(ppo_path, run_name="__main__")

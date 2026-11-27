"""Profile approaches: steps taken vs distance travelled, lifted or not, and per-step tip speed by sub-phase."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gymnasium as gym
import numpy as np
import torch

import pusht_vargoal  # noqa: F401
from planner_pusher import PlannerParams, PlannerPusher

env = gym.make("PushTVarGoal-v1", obs_mode="state_dict", control_mode="pd_joint_delta_pos", sim_backend="physx_cpu", num_envs=1)
pol = PlannerPusher(PlannerParams(), seed=0, randomize=False)
apps = []
for ep in range(8):
    obs, _ = env.reset(seed=3000 + ep)
    pol.reset(obs)
    cur = None
    for t in range(100):
        a = pol.act(obs)
        s = pol._parse(obs)
        if pol.phase == "approach":
            if cur is None:
                cur = dict(steps=0, start=s["tip"].copy(), lifted=len(pol.waypoints) == 3, zs=[], xy_moves=[], amax=[])
            cur["steps"] += 1
            cur["zs"].append(s["tip"][2])
            cur["amax"].append(np.abs(a).max())
        elif cur is not None:
            cur["end"] = s["tip"].copy()
            apps.append(cur)
            cur = None
        obs, *_ = env.step(torch.as_tensor(a, dtype=torch.float32)[None])
for kind in (True, False):
    sel = [a for a in apps if a["lifted"] == kind]
    if not sel:
        continue
    steps = np.array([a["steps"] for a in sel])
    dist = np.array([np.linalg.norm(a["end"][:2] - a["start"][:2]) for a in sel])
    print(f"{'lifted' if kind else 'planar'}: n={len(sel)} steps median {np.median(steps):.0f} max {steps.max()}  "
          f"xy distance median {np.median(dist):.3f}  -> {np.median(dist / steps):.4f} m/step;"
          f"  fraction of steps with |a|max=1: {np.mean(np.concatenate([a['amax'] for a in sel]) > 0.99):.2f}")
ex = [a for a in apps if a["lifted"]][:2]
for a in ex:
    print("lifted approach z trace:", np.round(a["zs"], 3))
ex = [a for a in apps if not a["lifted"]][:2]
for a in ex:
    print("planar approach z trace:", np.round(a["zs"], 3))

"""Per-episode diagnostics: final position/angle error, closest approach, and steps per phase."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gymnasium as gym
import numpy as np
import torch

import pusht_vargoal  # noqa: F401
from scripted_pusher import PusherParams, ScriptedPusher

q = int(sys.argv[1])
n = int(sys.argv[2])
params = PusherParams(**json.loads(sys.argv[3])) if len(sys.argv) > 3 else PusherParams()
env = gym.make("PushTVarGoal-v1", obs_mode="state_dict", control_mode="pd_joint_delta_pos", sim_backend="physx_cpu",
               num_envs=1, goal_quadrants=(q,))
pusher = ScriptedPusher(params, seed=q, randomize=True)
rows = []
for i in range(n):
    obs, _ = env.reset(seed=1001 + 1000 * q + i)
    pusher.reset(obs)
    s0 = pusher._parse(obs)
    e0, th0 = pusher._errors(s0)
    phases, best, succ = Counter(), (9, 9), []
    for t in range(100):
        a = pusher.act(obs)
        phases[pusher.phase] += 1
        obs, r, term, trunc, info = env.step(torch.as_tensor(a, dtype=torch.float32)[None])
        succ.append(bool(info["success"][0]))
        s = pusher._parse(obs)
        e, th = pusher._errors(s)
        if np.linalg.norm(e) + 0.06 * abs(th) < best[0] + 0.06 * abs(best[1]):
            best = (np.linalg.norm(e), th)
    rows.append((i, np.linalg.norm(e0), th0, np.linalg.norm(e), th, best[0], best[1], any(succ), succ[-1], dict(phases)))
    print(f"ep{i}: start |e| {np.linalg.norm(e0):.3f} th {th0:+.2f} -> end |e| {np.linalg.norm(e):.3f} th {th:+.2f}"
          f"  best |e| {best[0]:.3f} th {best[1]:+.2f}  once {any(succ)} end {succ[-1]}  {dict(phases)}", flush=True)

"""Measure how the T actually responds to pushes, to calibrate the quasi-static model.

For every step where the tip is touching the T, record the contact lever arm r (contact - COM), the push direction d
(tip motion direction), and the T's actual translation and rotation. Fit the limit-surface length c in
  dtheta = (r x d) / c^2 * k,   dc = k d,   k = L / (1 + (r x d)^2 / c^2)
by comparing actual rotation per unit of translation-along-d with (r x d) / c^2.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gymnasium as gym
import numpy as np
import torch

import pusht_vargoal  # noqa: F401
from scripted_pusher import PusherParams, ScriptedPusher, dist_to_poly, world_poly, wrap

env = gym.make("PushTVarGoal-v1", obs_mode="state_dict", control_mode="pd_joint_delta_pos", sim_backend="physx_cpu",
               num_envs=1)
pusher = ScriptedPusher(PusherParams(), seed=0, randomize=True)
rows = []
for ep in range(12):
    obs, _ = env.reset(seed=500 + ep)
    pusher.reset(obs)
    s = pusher._parse(obs)
    for t in range(100):
        a = pusher.act(obs)
        obs, *_ = env.step(torch.as_tensor(a, dtype=torch.float32)[None])
        s2 = pusher._parse(obs)
        dc = s2["t_xy"] - s["t_xy"]
        dth = wrap(s2["t_th"] - s["t_th"])
        tipmove = s2["tip"][:2] - s["tip"][:2]
        near = dist_to_poly(s["tip"][:2], world_poly(s["t_xy"], s["t_th"])) < 0.012 and s["tip"][2] < 0.035
        if near and np.linalg.norm(dc) > 0.002 and np.linalg.norm(tipmove) > 0.002:
            d = tipmove / np.linalg.norm(tipmove)
            r = s["tip"][:2] - s["t_xy"]
            m = r[0] * d[1] - r[1] * d[0]
            rows.append((m, np.dot(dc, d), dth, np.linalg.norm(dc), np.linalg.norm(tipmove)))
        s = s2
rows = np.array(rows)
print("contact steps:", len(rows))
m, along, dth, dcn, tipn = rows.T
ok = along > 0.001
ratio = dth[ok] / along[ok]           # actual rotation per metre translated along d
for c in (0.03, 0.04, 0.05, 0.055, 0.065, 0.08, 0.1):
    pred = m[ok] / c ** 2
    print(f"c={c:.3f}: median |pred - actual| rotation per metre = {np.median(np.abs(pred - ratio)):.2f} rad/m;"
          f" sign agreement {np.mean(np.sign(pred) == np.sign(ratio)):.0%}")
print("T translation / tip motion (median):", np.median(dcn / tipn).round(3))
best = min((0.02 + 0.005 * i for i in range(30)), key=lambda c: np.median(np.abs(m[ok] / c ** 2 - ratio)))
print("best c:", round(best, 3))

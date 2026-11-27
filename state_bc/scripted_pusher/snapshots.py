"""Render a grid of top-down snapshots of one episode (every few steps) to a PNG, for debugging."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

import pusht_vargoal  # noqa: F401
from run_pusher import run_episode
from scripted_pusher import GOAL_THETA, PusherParams, ScriptedPusher, world_poly

q, idx = int(sys.argv[1]), int(sys.argv[2])
out = sys.argv[3]
params = PusherParams(**json.loads(sys.argv[4])) if len(sys.argv) > 4 else PusherParams()
env = gym.make("PushTVarGoal-v1", obs_mode="state_dict", control_mode="pd_joint_delta_pos", sim_backend="physx_cpu",
               num_envs=1, goal_quadrants=(q,))
pusher = ScriptedPusher(params, seed=q, randomize=True)
for i in range(idx + 1):            # replay the RNG stream exactly as diag.py does
    ep = run_episode(env, pusher, seed=1001 + 1000 * q + i)
fr = ep["frames"]
steps = list(range(0, 100, 7))[:15]
fig, axes = plt.subplots(3, 5, figsize=(14, 8.6))
for ax, t in zip(axes.flat, steps):
    f = fr[t]
    g = np.array(f["goal"])
    ax.add_patch(Polygon(world_poly(g, GOAL_THETA), closed=True, fc="#dddddd", ec="#777", ls="--"))
    ax.add_patch(Polygon(world_poly(np.array(f["t_xy"]), f["t_th"]), closed=True, fc="#c8453a", alpha=0.85, ec="#333"))
    tr = np.array([x["tip"][:2] for x in fr[max(0, t - 7):t + 1]])
    ax.plot(tr[:, 0], tr[:, 1], color="#1f6feb", lw=1)
    ax.plot(*f["tip"][:2], "o", color="#1f6feb", mfc="white" if f["tip"][2] > 0.05 else "#1f6feb")
    ax.set_xlim(-0.45, 0.15), ax.set_ylim(-0.38, 0.38), ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(f"t={t} {f['phase']} z={f['tip'][2]:.3f}", fontsize=8)
fig.tight_layout()
fig.savefig(out, dpi=70)
print("once", ep["success_once"], "end", ep["success_at_end"])

"""Print a step-by-step trace of one episode: phase, tip position, errors, and how far the tip moved."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gymnasium as gym
import numpy as np
import torch

import pusht_vargoal  # noqa: F401
from scripted_pusher import PusherParams, ScriptedPusher

q = int(sys.argv[1]) if len(sys.argv) > 1 else 1
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 2001
params = PusherParams(**json.loads(sys.argv[3])) if len(sys.argv) > 3 else PusherParams()
env = gym.make("PushTVarGoal-v1", obs_mode="state_dict", control_mode="pd_joint_delta_pos", sim_backend="physx_cpu",
               num_envs=1, goal_quadrants=(q,))
pusher = ScriptedPusher(params, seed=0, randomize=False)
obs, _ = env.reset(seed=seed)
pusher.reset(obs)
prev = None
for t in range(100):
    a = pusher.act(obs)
    s = pusher._parse(obs)
    e, dth = pusher._errors(s)
    moved = 0 if prev is None else np.linalg.norm(s["tip"] - prev) / 0.05
    prev = s["tip"].copy()
    obs, r, term, trunc, info = env.step(torch.as_tensor(a, dtype=torch.float32)[None])
    if t % 3 == 0 or t < 12:
        tgt = pusher.waypoints[0] if pusher.phase == "approach" and pusher.waypoints else None
        print(f"{t:3d} {pusher.phase:8s} tip {s['tip'].round(3)} tipspeed {moved:.2f}m/s  |e| {np.linalg.norm(e):.3f} "
              f"dth {dth:+.2f}  |a|max {np.abs(a).max():.2f}  succ {bool(info['success'][0])}"
              f"{'  wp ' + str(np.round(tgt, 3)) if tgt is not None else ''}")

"""How close does each episode get? Closest pose error over the episode, error at step 50 and 100, pushes executed."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gymnasium as gym
import numpy as np
import torch

import pusht_vargoal  # noqa: F401
from planner_pusher import PlannerParams, PlannerPusher
from scripted_pusher import GOAL_THETA, wrap

params = PlannerParams(**json.loads(sys.argv[1])) if len(sys.argv) > 1 else PlannerParams()
horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 100
env = gym.make("PushTVarGoal-v1", obs_mode="state_dict", control_mode="pd_joint_delta_pos", sim_backend="physx_cpu",
               num_envs=1, max_episode_steps=horizon)
pol = PlannerPusher(params, seed=0, randomize=False)
rows = []
for ep in range(16):
    obs, _ = env.reset(seed=4000 + ep)
    pol.reset(obs)
    best, pushes, prev, at50, succ = (9, 9), 0, None, None, False
    for t in range(horizon):
        a = pol.act(obs)
        if pol.phase == "push" and prev != "push":
            pushes += 1
        prev = pol.phase
        obs, r, term, trunc, info = env.step(torch.as_tensor(a, dtype=torch.float32)[None])
        succ |= bool(info["success"][0])
        s = pol._parse(obs)
        e = np.linalg.norm(s["goal"] - s["t_xy"])
        th = abs(wrap(GOAL_THETA - s["t_th"]))
        if e + 0.06 * th < best[0] + 0.06 * best[1]:
            best = (e, th)
        if t == 49:
            at50 = (e, th)
    rows.append((best[0], best[1], at50[0], at50[1], e, th, pushes, succ))
    print(f"ep{ep}: closest {best[0]*100:4.1f} cm {np.degrees(best[1]):5.1f} deg | step50 {at50[0]*100:4.1f} cm "
          f"{np.degrees(at50[1]):5.1f} deg | end {e*100:4.1f} cm {np.degrees(th):5.1f} deg | pushes {pushes} | success {succ}")
r = np.array(rows, dtype=float)
print(f"closest: median {np.median(r[:,0])*100:.1f} cm / {np.degrees(np.median(r[:,1])):.1f} deg; within 1.5cm & 6deg: "
      f"{np.mean((r[:,0] < 0.015) & (r[:,1] < np.radians(6))):.0%}; success {np.mean(r[:,7]):.0%}")

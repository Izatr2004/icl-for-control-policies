"""Run the MPPI expert on a few episodes, report success and timing, and render top-down videos."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gymnasium as gym
import numpy as np
import torch

import pusht_vargoal  # noqa: F401
from mppi_pusher import MPPIParams, MPPIPusher
from run_pusher import render_video
from scripted_pusher import GOAL_THETA, quat_to_yaw, wrap

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=2)
ap.add_argument("--quadrants", type=int, nargs="+", default=[1])
ap.add_argument("--params", default="{}")
ap.add_argument("--videos", type=int, default=99)
ap.add_argument("--seed-base", type=int, default=1001)
ap.add_argument("--out", default="mppi")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
params = MPPIParams(**json.loads(args.params))
rows = []
for q in args.quadrants:
    env = gym.make("PushTVarGoal-v1", obs_mode="state_dict", control_mode="pd_joint_delta_pos", sim_backend="physx_cpu",
                   num_envs=1, goal_quadrants=(q,))
    pol = MPPIPusher(env, params, seed=q)
    for i in range(args.episodes):
        obs, _ = env.reset(seed=args.seed_base + 1000 * q + i)
        pol.reset(obs)
        frames, succ, t0 = [], [], time.time()
        for t in range(100):
            a = pol.act(obs)
            obs, r, term, trunc, info = env.step(torch.as_tensor(a, dtype=torch.float32)[None])
            succ.append(bool(info["success"][0]))
            tee = obs["extra"]["obj_pose"][0].numpy()
            frames.append(dict(tip=obs["extra"]["tcp_pose"][0, :3].tolist(), t_xy=tee[:2].tolist(), t_th=quat_to_yaw(tee[3:]),
                               goal=obs["extra"]["goal_pos"][0, :2].tolist(), phase=f"MPPI ({params.samples}x{params.horizon})",
                               success=succ[-1]))
        e = np.linalg.norm(np.array(frames[-1]["goal"]) - np.array(frames[-1]["t_xy"]))
        th = abs(wrap(GOAL_THETA - frames[-1]["t_th"]))
        row = dict(q=q, i=i, success_once=any(succ), success_at_end=succ[-1], end_cm=round(e * 100, 1),
                   end_deg=round(float(np.degrees(th)), 1), sec_per_step=round(float(np.mean(pol.timing)), 3),
                   episode_sec=round(time.time() - t0))
        rows.append(row)
        print(json.dumps(row), flush=True)
        if i < args.videos:
            tag = "success" if row["success_at_end"] else ("reached" if row["success_once"] else "fail")
            render_video(dict(frames=frames), out / f"q{q}_ep{i}_{tag}.mp4", f"MPPI Q{q} episode {i}")
    env.close()
print("success once", np.mean([r["success_once"] for r in rows]), "at end", np.mean([r["success_at_end"] for r in rows]))

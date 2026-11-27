"""Evaluate PlannerPusher per quadrant (validation seeds by default), optional videos and parameter overrides.

python eval_planner.py --episodes 10 --quadrants 1 2 3 4 --params '{"beam": 24}' --videos 1 --out pl1
"""
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
from planner_pusher import PlannerParams, PlannerPusher
from run_pusher import render_video


def episode(env, pol, seed, horizon=100):
    obs, _ = env.reset(seed=seed)
    pol.reset(obs)
    frames, succ, acts = [], [], []
    for t in range(horizon):
        a = pol.act(obs)
        acts.append(a)
        obs, r, term, trunc, info = env.step(torch.as_tensor(a, dtype=torch.float32)[None])
        succ.append(bool(info["success"][0]))
        s = pol._parse(obs)
        frames.append(dict(tip=s["tip"].tolist(), t_xy=s["t_xy"].tolist(), t_th=s["t_th"], goal=s["goal"].tolist(),
                           phase=pol.phase, success=succ[-1]))
    return dict(seed=seed, success_once=any(succ), success_at_end=succ[-1],
                first=next((i + 1 for i, x in enumerate(succ) if x), 0), actions=np.array(acts)), frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--quadrants", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seed-base", type=int, default=1001)
    ap.add_argument("--params", default="{}")
    ap.add_argument("--no-random", action="store_true")
    ap.add_argument("--videos", type=int, default=0)
    ap.add_argument("--video-successes", type=int, default=0, help="max success videos per quadrant")
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--save-actions", action="store_true")
    ap.add_argument("--out", default="pl")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    params = PlannerParams(**json.loads(args.params))
    res, t0, all_actions = {}, time.time(), []
    for q in args.quadrants:
        env = gym.make("PushTVarGoal-v1", obs_mode="state_dict", control_mode="pd_joint_delta_pos",
                       sim_backend="physx_cpu", num_envs=1, goal_quadrants=(q,), max_episode_steps=args.horizon)
        pol = PlannerPusher(params, seed=1000 * q, randomize=not args.no_random)
        eps, nsv = [], 0
        for i in range(args.episodes):
            ep, frames = episode(env, pol, args.seed_base + 1000 * q + i, args.horizon)
            eps.append(ep)
            all_actions.append(ep["actions"])
            want = i < args.videos or (ep["success_once"] and nsv < args.video_successes)
            if want:
                nsv += ep["success_once"] and i >= args.videos
                tag = "success" if ep["success_at_end"] else ("reached" if ep["success_once"] else "fail")
                render_video(dict(frames=frames), out / f"q{q}_ep{i}_{tag}.mp4", f"Q{q} episode {i} (planner)")
        env.close()
        so = np.mean([e["success_once"] for e in eps])
        se = np.mean([e["success_at_end"] for e in eps])
        fs = [e["first"] for e in eps if e["first"]]
        res[f"Q{q}"] = dict(success_once=float(so), success_at_end=float(se),
                            mean_first_success_step=float(np.mean(fs)) if fs else None)
        print(f"Q{q}: once {so:.0%}  end {se:.0%}  first-success step {np.mean(fs) if fs else float('nan'):.0f}"
              f"   ({time.time() - t0:.0f}s)", flush=True)
    res["mean"] = dict(success_once=float(np.mean([v["success_once"] for v in res.values()])),
                       success_at_end=float(np.mean([v["success_at_end"] for k, v in res.items() if k != "mean"])))
    (out / "summary.json").write_text(json.dumps(dict(params=params.__dict__, seed_base=args.seed_base,
                                                      episodes_per_quadrant=args.episodes, results=res), indent=1, default=str))
    if args.save_actions:
        np.save(out / "actions.npy", np.concatenate(all_actions))
    print(json.dumps(res["mean"]))


if __name__ == "__main__":
    main()

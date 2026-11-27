"""Evaluate the scripted pusher per goal quadrant and render top-down videos from the simulator state.

python run_pusher.py --episodes 10 --quadrants 1 2 3 4 --videos 2 --out results
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gymnasium as gym
import numpy as np
import torch

import pusht_vargoal  # noqa: F401
from scripted_pusher import GOAL_THETA, PusherParams, ScriptedPusher, world_poly


def run_episode(env, pusher, seed, max_steps=100):
    obs, _ = env.reset(seed=seed)
    pusher.reset(obs)
    frames, succ, actions = [], [], []
    for t in range(max_steps):
        a = pusher.act(obs)
        actions.append(a)
        obs, r, term, trunc, info = env.step(torch.as_tensor(a, dtype=torch.float32)[None])
        succ.append(bool(info["success"][0]))
        s = pusher._parse(obs)
        frames.append(dict(tip=s["tip"].tolist(), t_xy=s["t_xy"].tolist(), t_th=s["t_th"], goal=s["goal"].tolist(),
                           phase=pusher.phase, success=succ[-1]))
    first = next((i + 1 for i, x in enumerate(succ) if x), 0)
    return dict(seed=seed, success_once=any(succ), success_at_end=succ[-1], first_success_step=first,
                frames=frames, max_abs_action=float(np.abs(np.array(actions)).max()))


def render_video(ep, path, title):
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    frames = ep["frames"]
    goal = np.array(frames[0]["goal"])
    trail = []
    writer = imageio.get_writer(path, fps=20, codec="libx264", quality=7, macro_block_size=1)
    for t, f in enumerate(frames):
        fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=110)
        ax.set_facecolor("#f6f6f2")
        ax.add_patch(Polygon(world_poly(goal, GOAL_THETA), closed=True, fc="#cfcfcf", ec="#7a7a7a", lw=1.2, ls="--"))
        col = "#2e9e4f" if f["success"] else "#c8453a"
        ax.add_patch(Polygon(world_poly(np.array(f["t_xy"]), f["t_th"]), closed=True, fc=col, ec="#333333", lw=1.0,
                             alpha=0.9))
        trail.append(f["tip"][:2])
        tr = np.array(trail[-40:])
        ax.plot(tr[:, 0], tr[:, 1], color="#1f6feb", lw=1.0, alpha=0.6)
        lifted = f["tip"][2] > 0.05
        ax.plot(*f["tip"][:2], "o", ms=7, mfc="white" if lifted else "#1f6feb", mec="#1f6feb", mew=1.5)
        ax.set_xlim(-0.45, 0.15)
        ax.set_ylim(-0.38, 0.38)
        ax.set_aspect("equal")
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f"{title}   step {t + 1:3d}/100\nphase: {f['phase']}"
                     f"{'   SUCCESS' if f['success'] else ''}", fontsize=8, loc="left")
        ax.text(0.14, -0.37, "robot base is to the left; hollow dot = tip lifted", fontsize=5.5, ha="right",
                color="#666666")
        fig.tight_layout()
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        writer.append_data(img)
        plt.close(fig)
    writer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--quadrants", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seed-base", type=int, default=1001)
    ap.add_argument("--videos", type=int, default=0, help="videos per quadrant (first episodes)")
    ap.add_argument("--video-successes", action="store_true", help="also render every episode that succeeds")
    ap.add_argument("--no-random", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--params", default="{}", help="JSON overrides for PusherParams")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    params = PusherParams(**json.loads(args.params))
    summary = {}
    t0 = time.time()
    for q in args.quadrants:
        env = gym.make("PushTVarGoal-v1", obs_mode="state_dict", control_mode="pd_joint_delta_pos",
                       sim_backend="physx_cpu", num_envs=1, goal_quadrants=(q,))
        pusher = ScriptedPusher(params, seed=1000 * q, randomize=not args.no_random)
        eps = []
        for i in range(args.episodes):
            ep = run_episode(env, pusher, seed=args.seed_base + 1000 * q + i)
            eps.append(ep)
            if i < args.videos or (ep["success_once"] and args.video_successes):
                tag = "success" if ep["success_at_end"] else ("reached" if ep["success_once"] else "fail")
                render_video(ep, out / f"q{q}_ep{i}_{tag}.mp4", f"Q{q} episode {i}")
        env.close()
        so = np.mean([e["success_once"] for e in eps])
        se = np.mean([e["success_at_end"] for e in eps])
        steps = [e["first_success_step"] for e in eps if e["success_once"]]
        summary[f"Q{q}"] = dict(episodes=len(eps), success_once=float(so), success_at_end=float(se),
                                mean_first_success_step=float(np.mean(steps)) if steps else None)
        print(f"Q{q}: success-once {so:.0%}  at-end {se:.0%}  first-success step "
              f"{np.mean(steps) if steps else float('nan'):.1f}   ({time.time() - t0:.0f}s)", flush=True)
        (out / f"episodes_q{q}.json").write_text(json.dumps(
            [{k: v for k, v in e.items() if k != "frames"} for e in eps], indent=1))
    summary["mean"] = dict(success_once=float(np.mean([v["success_once"] for v in summary.values()])),
                           success_at_end=float(np.mean([v["success_at_end"] for k, v in summary.items() if k != "mean"])))
    (out / "summary.json").write_text(json.dumps(dict(params=params.__dict__, randomize=not args.no_random,
                                                      seed_base=args.seed_base, results=summary), indent=1, default=str))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

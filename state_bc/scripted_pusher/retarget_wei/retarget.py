"""Re-target Wei et al.'s human Push-T demos (Drake, 10 Hz, metres) into PushTVarGoal-v1 and replay them.

For each demo:
  1. pick a goal in one of our quadrants;
  2. map the demo rigidly so its goal pose lands on ours (their T angle = ours - pi; see geometry_check.py);
  3. place our T and the stick tip where the demo started (T pose set directly, arm placed by IK);
  4. replay the human pusher path through damped-least-squares IK as native joint-delta actions, time-compressed to
     fit 100 steps: pauses are skipped and the path is traversed at a steady tip speed.
Two replay modes:
  world   - follow the mapped pusher path in world coordinates;
  object  - follow the pusher path relative to the T (demo pusher position in the demo T's frame, placed on our T's
            current pose, with their slightly smaller T scaled up), so contacts stay right when our T drifts.

python retarget.py --demos 40 --mode object --videos 4 --out rt_object
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
from pusht_vargoal import sample_goals
from run_pusher import render_video
from scripted_pusher import (DT, GOAL_THETA, JOINT_DELTA_SCALE, ROBOT_BASE, ArmIK, quat_to_yaw, rot, rot_error, wrap)

THETA_OFFSET = -math.pi          # our T angle = Wei T angle - pi
LOCAL_SCALE = np.array([1.1, 1.11])   # their T is ~10% smaller; scale pusher-in-T-frame offsets (our local x, y)
LOCAL_SHIFT = np.array([0.0, 0.008])
Z_PUSH = 0.015


def load_demos(path):
    d = np.load(path)
    ends = d["ends"]
    starts = np.r_[0, ends[:-1]]
    return [dict(pusher=d["state"][a:b, :2], slider=d["slider"][a:b], target=d["target"][a]) for a, b in zip(starts, ends)]


def to_ours(xy_th):
    """Wei slider pose -> our convention (angle shifted)."""
    return np.array([xy_th[0], xy_th[1], wrap(xy_th[2] + THETA_OFFSET)])


def rigid_map(demo, goal_xy):
    """Rotation phi and translation mapping the demo's goal pose onto (goal_xy, GOAL_THETA)."""
    tgt = to_ours(demo["target"])
    phi = wrap(GOAL_THETA - tgt[2])
    R = rot(phi)
    t = goal_xy - R @ tgt[:2]
    return phi, R, t


def demo_in_world(demo, goal_xy):
    phi, R, t = rigid_map(demo, goal_xy)
    pusher = demo["pusher"] @ R.T + t
    sl = np.array([to_ours(s) for s in demo["slider"]])
    slider = np.c_[sl[:, :2] @ R.T + t, [wrap(a + phi) for a in sl[:, 2]]]
    return pusher, slider


def local_offsets(demo):
    """Pusher position in the demo T's frame, converted to our T's local frame (rotation + size scaling)."""
    out = []
    for p, s in zip(demo["pusher"], demo["slider"]):
        th = s[2] + THETA_OFFSET                     # our-convention angle of their T
        c, sn = math.cos(-th), math.sin(-th)
        r = p - s[:2]
        out.append([c * r[0] - sn * r[1], sn * r[0] + c * r[1]])
    return np.array(out) * LOCAL_SCALE + LOCAL_SHIFT


def time_index(pusher, step_len):
    """Demo indices visited when the pusher path is traversed at `step_len` metres per control step (pauses skipped)."""
    seg = np.r_[0, np.linalg.norm(np.diff(pusher, axis=0), axis=1)]
    s = np.cumsum(seg)
    targets = np.arange(0, s[-1] + step_len, step_len)
    return np.searchsorted(s, targets).clip(0, len(pusher) - 1)


def ik_place(ik, env, tip_xy, q_init, q_ref, iters=60):
    """Joint configuration that puts the stick tip at (tip_xy, Z_PUSH) with the reset orientation."""
    q = q_init.copy()
    target = np.array([tip_xy[0], tip_xy[1], Z_PUSH])
    for _ in range(iters):
        tf = ik.chain.forward_kinematics(torch.as_tensor(q)[None]).get_matrix()[0].numpy()
        p = tf[:3, 3] + ROBOT_BASE
        from scipy.spatial.transform import Rotation
        qx = Rotation.from_matrix(tf[:3, :3]).as_quat()            # x, y, z, w
        q_cur = np.array([qx[3], qx[0], qx[1], qx[2]])
        err = target - p
        if np.linalg.norm(err) < 1e-3:
            break
        q = q + ik.joint_delta(q, np.clip(err, -0.05, 0.05), 0.5 * rot_error(q_cur, q_ref))
    return q, float(np.linalg.norm(err))


def parse(obs):
    g = lambda k: obs["extra"][k][0].cpu().numpy().astype(np.float64)  # noqa: E731
    tcp, tee = g("tcp_pose"), g("obj_pose")
    return dict(qpos=obs["agent"]["qpos"][0].cpu().numpy().astype(np.float64), tip=tcp[:3], tcp_q=tcp[3:],
                t_xy=tee[:2], t_th=quat_to_yaw(tee[3:]), goal=g("goal_pos")[:2])


def replay(env, ik, demo, goal_xy, mode, tip_speed, gain=3.0, lead=0.03, max_steps=100):
    from mani_skill.utils.structs.pose import Pose
    u = env.unwrapped
    pusher_w, slider_w = demo_in_world(demo, goal_xy)
    obs, _ = env.reset(options=dict(goal_xy=torch.as_tensor(goal_xy, dtype=torch.float32)[None]))
    s = parse(obs)
    q_ref = s["tcp_q"].copy()
    # place the T where the demo started
    t0 = slider_w[0]
    p = torch.tensor([[t0[0], t0[1], 0.04 / 2 + 1e-3]], dtype=torch.float32)
    qz = torch.tensor([[math.cos(t0[2] / 2), 0, 0, math.sin(t0[2] / 2)]], dtype=torch.float32)
    u.tee.set_pose(Pose.create_from_pq(p=p, q=qz))
    # place the stick tip where the demo's pusher started
    q_arm, ik_err = ik_place(ik, env, pusher_w[0], s["qpos"], q_ref)
    u.agent.reset(torch.as_tensor(q_arm, dtype=torch.float32)[None])
    if hasattr(u.scene, "_gpu_apply_all"):
        pass
    obs = u.get_obs()
    offsets = local_offsets(demo) if mode == "object" else None
    idx = time_index(pusher_w, tip_speed * DT)
    frames, succ = [], []
    for t in range(max_steps):
        s = parse(obs)
        i = idx[min(t, len(idx) - 1)]
        if mode == "world":
            tgt = pusher_w[i]
        else:
            tgt = s["t_xy"] + rot(s["t_th"]) @ offsets[i]
        d = tgt - s["tip"][:2]
        n = np.linalg.norm(d)
        if n > lead:
            d *= lead / n
        dz = float(np.clip(1.5 * (Z_PUSH - s["tip"][2]), -0.04, 0.04))
        dq = ik.joint_delta(s["qpos"], np.array([d[0], d[1], dz]), 0.5 * rot_error(s["tcp_q"], q_ref))
        a = np.clip(gain * dq / JOINT_DELTA_SCALE, -1, 1)
        obs, r, term, trunc, info = env.step(torch.as_tensor(a, dtype=torch.float32)[None])
        succ.append(bool(info["success"][0]))
        s2 = parse(obs)
        frames.append(dict(tip=s2["tip"].tolist(), t_xy=s2["t_xy"].tolist(), t_th=s2["t_th"], goal=list(goal_xy),
                           phase=f"replay ({mode}) demo step {i}/{len(pusher_w) - 1}", success=succ[-1]))
    e = np.linalg.norm(s2["t_xy"] - goal_xy)
    th = abs(wrap(GOAL_THETA - s2["t_th"]))
    return dict(success_once=any(succ), success_at_end=succ[-1], first=next((k + 1 for k, x in enumerate(succ) if x), 0),
                end_pos_err=float(e), end_ang_err=float(th), ik_err=ik_err, demo_len=len(pusher_w),
                path_m=float(np.sum(np.linalg.norm(np.diff(pusher_w, axis=0), axis=1))),
                replay_covers=float(min(1.0, max_steps / max(1, len(idx))))), frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="wei_lowdim.npz")
    ap.add_argument("--demos", type=int, default=40)
    ap.add_argument("--mode", choices=["world", "object"], default="object")
    ap.add_argument("--tip-speed", type=float, default=0.3, help="m/s along the human's pusher path")
    ap.add_argument("--videos", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="rt")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    demos = load_demos(args.data)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(demos))[:args.demos]
    env = gym.make("PushTVarGoal-v1", obs_mode="state_dict", control_mode="pd_joint_delta_pos",
                   sim_backend="physx_cpu", num_envs=1)
    ik = ArmIK()
    torch.manual_seed(args.seed)
    rows, t0 = [], time.time()
    for n, di in enumerate(order):
        q = int(rng.integers(1, 5))
        goal = sample_goals(1, (q,))[0].numpy().astype(np.float64)
        res, frames = replay(env, ik, demos[di], goal, args.mode, args.tip_speed)
        res.update(demo=int(di), quadrant=q)
        rows.append(res)
        if n < args.videos or (res["success_once"] and sum(r["success_once"] for r in rows) <= 3):
            tag = "success" if res["success_at_end"] else ("reached" if res["success_once"] else "fail")
            render_video(dict(frames=frames), out / f"demo{di}_q{q}_{tag}.mp4", f"Wei demo {di} -> Q{q} ({args.mode})")
    so = np.mean([r["success_once"] for r in rows])
    se = np.mean([r["success_at_end"] for r in rows])
    fits = np.mean([r["replay_covers"] >= 1.0 for r in rows])
    summary = dict(mode=args.mode, tip_speed=args.tip_speed, demos=len(rows), success_once=float(so), success_at_end=float(se),
                   fits_in_100_steps=float(fits),
                   median_end_pos_err=float(np.median([r["end_pos_err"] for r in rows])),
                   median_end_ang_err=float(np.median([r["end_ang_err"] for r in rows])),
                   max_ik_err=float(max(r["ik_err"] for r in rows)), seconds=round(time.time() - t0))
    (out / "summary.json").write_text(json.dumps(dict(summary=summary, episodes=rows), indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

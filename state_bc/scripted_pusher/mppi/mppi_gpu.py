"""Batched sampling-based MPC (MPPI) expert for PushTVarGoal-v1, for the GPU simulator.

E episodes run in lockstep inside one ManiSkill environment with E*N parallel copies: copy e*N is episode e's "leader"
(the real episode), copies e*N+1 .. e*N+N-1 are its scratch copies for rollouts. Every control step:
  1. copy each leader's full simulator state into its N copies;
  2. roll out N candidate tip-velocity plans per episode for H steps (all E*N in parallel), converting tip velocities to
     joint-delta actions with batched damped-least-squares IK;
  3. cost = T pose error to the goal at the end (+ small tip-to-T distance and effort terms): a plain geometric cost, not
     PPO's reward; MPPI-weight the N candidates of each episode into a new plan;
  4. restore every copy to its leader's state and apply the plan's first action to all copies of the group (so all copies,
     including the leader, take the real step), then shift the plan.
The executed joint-delta action is recorded as the BC label, with the 31-number state in PushT-v1's `state` layout.
Optional action noise on the executed step (DART-style: record the clean action, execute a noisy one).

Usage (from the pilot notebook):
    python mppi_gpu.py --episodes-per-batch 8 --samples 256 --horizon 10 --quadrants 1 2 3 4 --out OUT
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

GOAL_THETA = 5.0 / 3.0 * math.pi
JOINT_DELTA_SCALE = 0.1
DT = 1.0 / 20.0
T_POLY = torch.tensor([(-0.1, -0.0625), (0.1, -0.0625), (0.1, -0.0125), (0.025, -0.0125),
                       (0.025, 0.1375), (-0.025, 0.1375), (-0.025, -0.0125), (-0.1, -0.0125)])


@dataclass
class MPPIConfig:
    samples: int = 256
    horizon: int = 10
    temperature: float = 0.004
    sigma: float = 0.25          # m/s
    smooth: float = 0.6
    vmax: float = 0.5            # m/s
    rho: float = 0.06            # metres per radian in the pose cost
    w_tip: float = 0.15
    w_effort: float = 0.002
    z_push: float = 0.015
    gain: float = 3.0
    exec_noise: float = 0.0      # std of noise added to the executed joint-delta action (labels stay clean)


# ---------------------------------------------------------------- batched math
def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw(q):
    """(B, 4) wxyz -> (B,) yaw."""
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rot_error(q_cur, q_ref):
    """Small-angle rotation vectors (B, 3) taking q_cur to q_ref (both wxyz)."""
    w1, x1, y1, z1 = q_cur.unbind(-1)
    w2, x2, y2, z2 = q_ref.unbind(-1)
    w = w2 * w1 + x2 * x1 + y2 * y1 + z2 * z1
    x = -w2 * x1 + x2 * w1 - y2 * z1 + z2 * y1
    y = -w2 * y1 + x2 * z1 + y2 * w1 - z2 * x1
    z = -w2 * z1 - x2 * y1 + y2 * x1 + z2 * w1
    s = torch.where(w < 0, -1.0, 1.0)
    return 2.0 * torch.stack([x * s, y * s, z * s], -1)


def tip_to_t_distance(tip, t_xy, t_th):
    """Distance from tip (B, 2) to the T outline at pose (t_xy, t_th)."""
    poly = T_POLY.to(tip)
    c, s = torch.cos(t_th), torch.sin(t_th)
    px = t_xy[:, None, 0] + c[:, None] * poly[None, :, 0] - s[:, None] * poly[None, :, 1]
    py = t_xy[:, None, 1] + s[:, None] * poly[None, :, 0] + c[:, None] * poly[None, :, 1]
    ax, ay = px, py
    bx, by = px.roll(-1, 1), py.roll(-1, 1)
    abx, aby = bx - ax, by - ay
    t = (((tip[:, None, 0] - ax) * abx + (tip[:, None, 1] - ay) * aby) / (abx ** 2 + aby ** 2)).clamp(0, 1)
    dx = tip[:, None, 0] - (ax + t * abx)
    dy = tip[:, None, 1] - (ay + t * aby)
    return torch.sqrt(dx ** 2 + dy ** 2).min(1).values


class BatchedIK:
    def __init__(self, device):
        import mani_skill
        import pytorch_kinematics as pk
        urdf = Path(mani_skill.__file__).parent / "assets/robots/panda/panda_stick.urdf"
        self.chain = pk.build_serial_chain_from_urdf(urdf.read_bytes(), "panda_hand_tcp").to(dtype=torch.float32,
                                                                                             device=device)

    def joint_delta(self, qpos, dpos, rot_err, damping=0.02):
        J = self.chain.jacobian(qpos)                               # (B, 6, 7)
        target = torch.cat([dpos, rot_err], -1)[..., None]          # (B, 6, 1)
        JJt = J @ J.transpose(1, 2) + (damping ** 2) * torch.eye(6, device=J.device)
        return (J.transpose(1, 2) @ torch.linalg.solve(JJt, target))[..., 0]


# ---------------------------------------------------------------- the expert
class MPPIExpert:
    def __init__(self, env, episodes: int, cfg: MPPIConfig, seed: int = 0):
        self.env, self.u, self.E, self.cfg = env, env.unwrapped, episodes, cfg
        self.N = cfg.samples
        assert self.u.num_envs == self.E * self.N, (self.u.num_envs, self.E, self.N)
        self.device = self.u.device
        self.ik = BatchedIK(self.device)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.leaders = torch.arange(self.E, device=self.device) * self.N

    # ------------------------------------------------------------ state helpers
    def _snap(self):
        u = self.u
        tcp = u.agent.tcp.pose.raw_pose
        tee = u.tee.pose.raw_pose
        return dict(qpos=u.agent.robot.get_qpos(), qvel=u.agent.robot.get_qvel(), tcp=tcp, tip=tcp[:, :3], tcp_q=tcp[:, 3:],
                    tee=tee, t_xy=tee[:, :2], t_th=yaw(tee[:, 3:]), goal=u.goal_tee.pose.p)

    def _repeat_leaders(self, x):
        if isinstance(x, dict):
            return {k: self._repeat_leaders(v) for k, v in x.items()}
        return x[self.leaders].repeat_interleave(self.N, 0)

    def sync_to_leaders(self):
        """Copy every leader's full simulator state into all copies of its group."""
        state = self.u.get_state_dict()
        self.u.set_state_dict(self._repeat_leaders(state))

    def max_group_divergence(self):
        """Largest difference between any copy and its leader (for the self-check)."""
        s = self._snap()
        diff = (s["tee"] - s["tee"][self.leaders].repeat_interleave(self.N, 0)).abs().max()
        diff_q = (s["qpos"] - s["qpos"][self.leaders].repeat_interleave(self.N, 0)).abs().max()
        return float(diff), float(diff_q)

    # ------------------------------------------------------------ episode
    def reset(self, goals_xy, seed):
        """goals_xy: (E, 2). All copies of an episode get the same goal and the leader's start state."""
        goal_all = torch.as_tensor(goals_xy, dtype=torch.float32, device=self.device).repeat_interleave(self.N, 0)
        self.env.reset(seed=seed, options=dict(goal_xy=goal_all))
        self.sync_to_leaders()
        s = self._snap()
        self.q_ref = s["tcp_q"].clone()
        self.plan = torch.zeros(self.E, self.cfg.horizon, 2, device=self.device)

    def _actions(self, s, vel):
        """Tip velocities (B, 2) -> joint-delta actions (B, 7)."""
        dz = 1.5 * (self.cfg.z_push - s["tip"][:, 2:3])
        d = torch.cat([vel * DT, dz], -1)
        dq = self.ik.joint_delta(s["qpos"], d, 0.5 * rot_error(s["tcp_q"], self.q_ref))
        return (self.cfg.gain * dq / JOINT_DELTA_SCALE).clamp(-1, 1)

    def _noise(self):
        c = self.cfg
        eps = torch.randn(self.E, self.N, c.horizon, 2, device=self.device, generator=self.gen)
        for t in range(1, c.horizon):
            eps[:, :, t] = c.smooth * eps[:, :, t - 1] + math.sqrt(1 - c.smooth ** 2) * eps[:, :, t]
        return eps * c.sigma

    def _cost(self, s):
        c = self.cfg
        e = (s["goal"][:, :2] - s["t_xy"]).norm(dim=-1)
        th = wrap(GOAL_THETA - s["t_th"]).abs()
        pose = torch.sqrt(e ** 2 + (c.rho * th) ** 2)
        return pose + c.w_tip * tip_to_t_distance(s["tip"][:, :2], s["t_xy"], s["t_th"])

    @torch.no_grad()
    def step(self):
        """One real control step for all E episodes. Returns (leader obs (E, 31), clean action (E, 7), success (E,))."""
        c, u, E, N = self.cfg, self.u, self.E, self.N
        steps_before = u._elapsed_steps.clone()
        self.sync_to_leaders()
        saved = u.get_state_dict()
        s0 = self._snap()
        obs31 = torch.cat([s0["qpos"], s0["qvel"], s0["tcp"], s0["goal"], s0["tee"]], -1)[self.leaders]
        cand = (self.plan[:, None] + self._noise()).clamp(-c.vmax, c.vmax)       # (E, N, H, 2)
        cand[:, 0] = self.plan
        flat = cand.reshape(E * N, c.horizon, 2)
        for t in range(c.horizon):
            s = self._snap()
            u.step(self._actions(s, flat[:, t]))
        cost = self._cost(self._snap()).reshape(E, N) + c.w_effort * (cand ** 2).sum((2, 3))
        w = torch.softmax(-(cost - cost.min(1, keepdim=True).values) / c.temperature, dim=1)
        self.plan = (w[:, :, None, None] * cand).sum(1)
        u.set_state_dict(saved)
        u._elapsed_steps = steps_before
        s = self._snap()
        vel = self.plan[:, 0].repeat_interleave(N, 0)
        act = self._actions(s, vel)
        clean = act[self.leaders].clone()
        if c.exec_noise > 0:
            noise = torch.randn(E, 7, device=self.device, generator=self.gen) * c.exec_noise
            act = (act + noise.repeat_interleave(N, 0)).clamp(-1, 1)
        _, _, _, _, info = u.step(act)
        self.plan = torch.cat([self.plan[:, 1:], self.plan[:, -1:]], 1)
        return obs31, clean, info["success"][self.leaders].clone()


# ---------------------------------------------------------------- pilot runner
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-per-batch", type=int, default=8)
    ap.add_argument("--batches-per-quadrant", type=int, default=1)
    ap.add_argument("--quadrants", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--config", default="{}", help="JSON overrides for MPPIConfig")
    ap.add_argument("--seed-base", type=int, default=1001)
    ap.add_argument("--sim-backend", default="physx_cuda")
    ap.add_argument("--self-check", action="store_true", help="verify group syncing and exit")
    ap.add_argument("--frames-episodes", type=int, default=2, help="episodes per quadrant to keep frames for (videos)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import gymnasium as gym
    import pusht_vargoal  # noqa: F401
    from pusht_vargoal import sample_goals

    cfg = MPPIConfig(samples=args.samples, horizon=args.horizon, **json.loads(args.config))
    E = args.episodes_per_batch
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = gym.make("PushTVarGoal-v1", obs_mode="state_dict", control_mode="pd_joint_delta_pos",
                   sim_backend=args.sim_backend, num_envs=E * cfg.samples, max_episode_steps=10 ** 9)
    expert = MPPIExpert(env, E, cfg, seed=0)

    if args.self_check:
        goals = sample_goals(E, (1, 2, 3, 4)).numpy()
        expert.reset(goals, seed=0)
        print("after reset: max copy-vs-leader difference (T pose, qpos):", expert.max_group_divergence())
        for _ in range(3):
            expert.step()
        expert.sync_to_leaders()
        print("after 3 steps + sync:", expert.max_group_divergence())
        g = expert._snap()["goal"][:, :2]
        print("goals match within groups:", bool(torch.allclose(g, g[expert.leaders].repeat_interleave(cfg.samples, 0))))
        return

    rows, demo_obs, demo_act, demo_succ, frames_all = [], [], [], [], {}
    t_start = time.time()
    for q in args.quadrants:
        for b in range(args.batches_per_quadrant):
            seed = args.seed_base + 1000 * q + 100 * b
            torch.manual_seed(seed)
            goals = sample_goals(E, (q,)).numpy()
            expert.reset(goals, seed=seed)
            succ, obs_l, act_l, frames = [], [], [], []
            t0 = time.time()
            for t in range(100):
                obs31, act, success = expert.step()
                obs_l.append(obs31.cpu().numpy())
                act_l.append(act.cpu().numpy())
                succ.append(success.cpu().numpy())
                if b == 0:
                    s = expert._snap()
                    L = expert.leaders[:args.frames_episodes]
                    frames.append(dict(tip=s["tip"][L].cpu().numpy(), t_xy=s["t_xy"][L].cpu().numpy(),
                                       t_th=s["t_th"][L].cpu().numpy(), goal=s["goal"][L, :2].cpu().numpy(),
                                       success=success[:args.frames_episodes].cpu().numpy()))
            secs = time.time() - t0
            succ = np.stack(succ, 1)                       # (E, 100)
            for e in range(E):
                rows.append(dict(quadrant=q, batch=b, episode=e, success_once=bool(succ[e].any()),
                                 success_at_end=bool(succ[e, -1]),
                                 first_success_step=int(np.argmax(succ[e]) + 1) if succ[e].any() else 0))
            demo_obs.append(np.stack(obs_l, 1))
            demo_act.append(np.stack(act_l, 1))
            demo_succ.append(succ)
            if b == 0:
                frames_all[q] = frames
            print(f"Q{q} batch {b}: success-once {succ.any(1).mean():.0%}  at-end {succ[:, -1].mean():.0%}  "
                  f"{secs / 100:.3f} s/step  ({secs:.0f} s for {E} episodes)", flush=True)
    total = time.time() - t_start
    res = {}
    for q in args.quadrants:
        r = [x for x in rows if x["quadrant"] == q]
        res[f"Q{q}"] = dict(episodes=len(r), success_once=float(np.mean([x["success_once"] for x in r])),
                            success_at_end=float(np.mean([x["success_at_end"] for x in r])))
    n_eps = len(rows)
    summary = dict(config=asdict(cfg), episodes_per_batch=E, num_envs=E * cfg.samples, episodes=n_eps,
                   seconds_total=round(total), seconds_per_episode=round(total / n_eps, 2),
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", results=res,
                   mean_success_once=float(np.mean([v["success_once"] for v in res.values()])),
                   mean_success_at_end=float(np.mean([v["success_at_end"] for v in res.values()])))
    (out / "summary.json").write_text(json.dumps(dict(summary=summary, episodes=rows), indent=1))
    np.savez_compressed(out / "demos.npz", obs=np.concatenate(demo_obs), actions=np.concatenate(demo_act),
                        success=np.concatenate(demo_succ))
    np.save(out / "frames.npy", np.array(frames_all, dtype=object), allow_pickle=True)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

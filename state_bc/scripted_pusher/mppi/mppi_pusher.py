"""Sampling-based MPC (MPPI) Push-T expert, using the simulator itself as the model. CPU prototype.

Every control step, from the current simulator state:
  1. sample N smooth perturbations of the current tip-velocity plan over H steps;
  2. roll each out in the simulator (state saved and restored), converting tip velocities to joint-delta actions with IK;
  3. cost = T pose error to the goal at the end of the rollout (+ small terms for tip-to-T distance and effort);
     a plain geometric cost, not PPO's reward;
  4. exponentially weight the samples by cost, average them into a new plan, execute its first step, shift the plan.
The executed action is the real joint-delta action, so it can be recorded directly as a BC label.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from scripted_pusher import (DT, GOAL_THETA, JOINT_DELTA_SCALE, ArmIK, quat_to_yaw, rot_error, world_poly, wrap,
                             dist_to_poly)


@dataclass
class MPPIParams:
    samples: int = 16
    horizon: int = 6
    temperature: float = 0.004      # cost scale for the exponential weights (metres)
    sigma: float = 0.25             # m/s, std of velocity perturbations
    smooth: float = 0.6             # temporal correlation of perturbations
    vmax: float = 0.5               # m/s tip speed limit
    rho: float = 0.06               # metres per radian in the pose cost
    w_tip: float = 0.15             # weight on tip-to-T distance (approach shaping)
    w_effort: float = 0.002
    z_push: float = 0.015
    gain: float = 3.0


class MPPIPusher:
    def __init__(self, env, params: MPPIParams | None = None, seed: int = 0):
        self.env = env
        self.u = env.unwrapped
        self.p = params or MPPIParams()
        self.ik = ArmIK()
        self.rng = np.random.default_rng(seed)

    def reset(self, obs):
        self.plan = np.zeros((self.p.horizon, 2))
        self.q_ref = obs["extra"]["tcp_pose"][0, 3:].numpy().astype(np.float64)
        self.timing = []

    # ------------------------------------------------------------ helpers
    def _state(self):
        u = self.u
        tcp = u.agent.tcp.pose.raw_pose[0].numpy().astype(np.float64)
        tee = u.tee.pose.raw_pose[0].numpy().astype(np.float64)
        return dict(qpos=u.agent.robot.get_qpos()[0].numpy().astype(np.float64), tip=tcp[:3], tcp_q=tcp[3:],
                    t_xy=tee[:2], t_th=quat_to_yaw(tee[3:]), goal=u.goal_tee.pose.p[0, :2].numpy().astype(np.float64))

    def _action(self, s, v):
        """Tip velocity (2,) -> joint-delta action (7,)."""
        d = np.array([v[0] * DT, v[1] * DT, 1.5 * (self.p.z_push - s["tip"][2])])
        dq = self.ik.joint_delta(s["qpos"], d, 0.5 * rot_error(s["tcp_q"], self.q_ref))
        return np.clip(self.p.gain * dq / JOINT_DELTA_SCALE, -1.0, 1.0)

    def _cost(self, s):
        e = np.linalg.norm(s["goal"] - s["t_xy"])
        th = abs(wrap(GOAL_THETA - s["t_th"]))
        pose = math.sqrt(e * e + (self.p.rho * th) ** 2)
        tip_gap = dist_to_poly(s["tip"][:2], world_poly(s["t_xy"], s["t_th"]))
        return pose + self.p.w_tip * tip_gap

    def _noise(self):
        p = self.p
        eps = np.zeros((p.samples, p.horizon, 2))
        eps[:, 0] = self.rng.standard_normal((p.samples, 2))
        for t in range(1, p.horizon):
            eps[:, t] = p.smooth * eps[:, t - 1] + math.sqrt(1 - p.smooth ** 2) * self.rng.standard_normal((p.samples, 2))
        return eps * p.sigma

    # ------------------------------------------------------------ main step
    def act(self, obs):
        t0 = time.time()
        p, u = self.p, self.u
        saved = u.get_state_dict()
        saved_steps = u._elapsed_steps.clone()
        cand = np.clip(self.plan[None] + self._noise(), -p.vmax, p.vmax)
        cand[0] = self.plan                                       # always evaluate the current plan as is
        costs = np.zeros(p.samples)
        for k in range(p.samples):
            u.set_state_dict(saved)
            for t in range(p.horizon):
                s = self._state()
                u.step(torch.as_tensor(self._action(s, cand[k, t]), dtype=torch.float32)[None])
            costs[k] = self._cost(self._state()) + p.w_effort * np.sum(cand[k] ** 2)
        u.set_state_dict(saved)
        u._elapsed_steps = saved_steps
        w = np.exp(-(costs - costs.min()) / p.temperature)
        w /= w.sum()
        self.plan = np.tensordot(w, cand, axes=1)
        v = self.plan[0].copy()
        self.plan = np.r_[self.plan[1:], self.plan[-1:]]          # shift (warm start)
        a = self._action(self._state(), v)
        self.timing.append(time.time() - t0)
        self.last_cost = float(costs.min())
        return a

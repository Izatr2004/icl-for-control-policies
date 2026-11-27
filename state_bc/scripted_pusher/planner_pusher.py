"""Scripted Push-T pusher with multi-step (receding-horizon) push planning, for PushTVarGoal-v1.

Each replan searches sequences of pushes with a calibrated quasi-static model:
  depth 1: every contact point on the T outline x every push length -> predicted T pose after the push;
  depth 2: from the best B of those, every second push -> predicted pose after two pushes.
A sequence's score = remaining pose error after it + a time cost (approach travel, lifts over the T, push length).
Only the first push of the best sequence is executed; then the pusher replans from the new state.

Execution reuses the joint-delta IK, approach / lift-over routing, lead-target pushing and hold logic of
scripted_pusher.py. Demo-like variety: random choice among the top sequences, per-episode speed and stand-off
jitter, smooth tip jitter faded out near the goal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from scripted_pusher import (CONTACTS, DT, GOAL_THETA, JOINT_DELTA_SCALE, T_HEIGHT, ArmIK, dist_to_poly, point_in_poly,
                             quat_to_yaw, rot, rot_error, segment_hits_poly, world_poly, wrap)

R_LOC = np.array([c[0] for c in CONTACTS])          # (N, 2) contact points, T frame
N_LOC = np.array([c[1] for c in CONTACTS])          # (N, 2) outward normals, T frame


@dataclass
class PlannerParams:
    c_gyr: float = 0.08            # limit-surface length (calibrate.py)
    rho: float = 0.2               # metres of pose error per radian (planning)
    lengths: tuple = (0.003, 0.006, 0.012, 0.025, 0.045, 0.07, 0.10, 0.14, 0.18)
    beam: int = 24
    step_cost: float = 0.0009      # metres of pose error that one control step is worth
    lift_steps: int = 8            # extra steps for lifting over the T (measured)
    move_step: float = 0.024       # metres per step in free space (measured at move_speed 1.0)
    push_step: float = 0.013       # metres per step while pushing (planning estimate)
    top_k: int = 3
    # execution
    standoff: float = 0.022
    tip_margin: float = 0.018
    move_speed: float = 1.0        # requested; PD lag delivers about 0.5 m/s
    push_lead: float = 0.03
    retreat: float = 0.006
    z_push: float = 0.015
    z_lift: float = 0.062
    z_step: float = 0.03           # metres per step for vertical moves
    gain: float = 3.0
    max_approach_steps: int = 25
    pos_tol: float = 0.0025         # success needs about 3 mm (overlap 0.85 at 5 mm)
    ang_tol: float = 0.05           # and about 3-4 degrees
    fine_lead: float = 0.012        # smaller, gentler push target for fine pushes
    progress_stop: float = 0.92     # end a push once this fraction of the predicted T motion happened (>5 disables)
    # variety
    noise_std: float = 0.004
    noise_tau: float = 0.4
    noise_fade: float = 0.06
    speed_jitter: float = 0.15
    standoff_jitter: float = 0.006


def _rot_batch(th):
    c, s = np.cos(th), np.sin(th)
    return c, s


def predict(pose, L, p: PlannerParams):
    """pose (3,) -> for every contact x length: new poses (N, nL, 3), stand-off points (N, 2), push dirs (N, 2)."""
    x, y, th = pose
    c, s = math.cos(th), math.sin(th)
    r = np.stack([c * R_LOC[:, 0] - s * R_LOC[:, 1], s * R_LOC[:, 0] + c * R_LOC[:, 1]], 1)
    n = np.stack([c * N_LOC[:, 0] - s * N_LOC[:, 1], s * N_LOC[:, 0] + c * N_LOC[:, 1]], 1)
    d = -n
    m = r[:, 0] * d[:, 1] - r[:, 1] * d[:, 0]                      # (N,)
    c2 = p.c_gyr ** 2
    k = L[None, :] / (1 + (m ** 2)[:, None] / c2)                 # (N, nL)
    new = np.empty((len(R_LOC), len(L), 3))
    new[..., 0] = x + k * d[:, 0:1]
    new[..., 1] = y + k * d[:, 1:2]
    new[..., 2] = th + k * m[:, None] / c2
    stand = np.array([x, y]) + r + n * p.standoff
    contact = np.array([x, y]) + r
    return new, stand, contact, d


def pose_err(poses, goal, p):
    e = poses[..., :2] - goal
    dth = (GOAL_THETA - poses[..., 2] + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt(np.sum(e ** 2, -1) + (p.rho * dth) ** 2)


def _inside_or_near(pts, poly, margin):
    """Vectorized: is each point (M, 2) inside the polygon or within `margin` of its outline?"""
    a = poly
    b = np.roll(poly, -1, axis=0)
    px, py = pts[:, 0:1], pts[:, 1:2]
    ax, ay, bx, by = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
    cross = ((ay > py) != (by > py)) & (px < (bx - ax) * (py - ay) / (by - ay + 1e-12) + ax)
    inside = np.sum(cross, axis=1) % 2 == 1
    ab = b - a
    t = np.clip(((px - ax) * ab[:, 0] + (py - ay) * ab[:, 1]) / np.sum(ab ** 2, 1), 0, 1)
    dx = px - (ax + t * ab[:, 0])
    dy = py - (ay + t * ab[:, 1])
    near = np.min(dx ** 2 + dy ** 2, axis=1) < margin ** 2
    return inside | near


def reachable(points):
    """Stick-tip reach at table height, from pusht_vargoal's measured workspace (x in [-0.40, 0.05], |y| <= 0.45),
    shrunk by a margin."""
    return (points[:, 0] > -0.385) & (points[:, 0] < 0.035) & (np.abs(points[:, 1]) < 0.42)


def needs_lift(tip, stands, pose, p):
    """For each stand-off point: does the straight tip path cross the (inflated) T?"""
    poly = world_poly(pose[:2], pose[2])
    ts = np.linspace(0.15, 0.85, 8)
    pts = tip[None, None, :] + ts[None, :, None] * (stands[:, None, :] - tip[None, None, :])     # (N, 8, 2)
    hit = _inside_or_near(pts.reshape(-1, 2), poly, p.tip_margin).reshape(len(stands), len(ts))
    return hit.any(axis=1)


class PlannerPusher:
    def __init__(self, params: PlannerParams | None = None, seed: int = 0, randomize: bool = True):
        self.p = params or PlannerParams()
        self.ik = ArmIK()
        self.rng = np.random.default_rng(seed)
        self.randomize = randomize
        self.L = np.array(self.p.lengths)

    # ------------------------------------------------------------ planning
    def plan(self, s):
        p, L = self.p, self.L
        pose = np.array([*s["t_xy"], s["t_th"]])
        goal = s["goal"]
        tip = s["tip"][:2]
        new1, stand1, contact1, d1 = predict(pose, L, p)
        lift1 = needs_lift(tip, stand1, pose, p)
        bad1 = ~reachable(stand1)
        travel1 = np.linalg.norm(stand1 - tip, axis=1) / p.move_step + lift1 * p.lift_steps       # (N,)
        steps1 = travel1[:, None] + (p.standoff + L[None, :]) / p.push_step                    # (N, nL)
        err1 = pose_err(new1, goal, p)
        score1 = err1 + p.step_cost * steps1
        score1[bad1] = np.inf
        # beam: best first pushes
        flat = np.argsort(score1, axis=None)[:p.beam]
        best = []
        for f in flat:
            i, j = np.unravel_index(f, score1.shape)
            pose2 = new1[i, j]
            tip2 = contact1[i] + d1[i] * L[j]                      # tip ends where the push ended
            new2, stand2, _, _ = predict(pose2, L, p)
            lift2 = needs_lift(tip2, stand2, pose2, p)
            travel2 = np.linalg.norm(stand2 - tip2, axis=1) / p.move_step + lift2 * p.lift_steps
            steps2 = travel2[:, None] + (p.standoff + L[None, :]) / p.push_step
            score2 = pose_err(new2, goal, p) + p.step_cost * (steps1[i, j] + steps2)
            score2[~reachable(stand2)] = np.inf
            total = min(score2.min(), score1[i, j])                # a single push may already be best
            best.append((total, i, j))
        best.sort()
        k = p.top_k if self.randomize else 1
        _, i, j = best[int(self.rng.integers(min(k, len(best))))]
        return i, float(L[j])

    # ------------------------------------------------------------ episode
    def reset(self, obs):
        s = self._parse(obs)
        self.q_ref = s["tcp_q"].copy()
        j = self.p.speed_jitter if self.randomize else 0
        self.speed = float(self.rng.uniform(1 - j, 1 + j))
        k = self.p.standoff_jitter if self.randomize else 0
        self.standoff = self.p.standoff + float(self.rng.uniform(-k, k))
        self.noise = np.zeros(2)
        self.phase = "plan"
        self.waypoints = []
        self.log = []

    @staticmethod
    def _parse(obs):
        g = lambda k: obs["extra"][k][0].detach().cpu().numpy().astype(np.float64)  # noqa: E731
        tcp, tee, goal = g("tcp_pose"), g("obj_pose"), g("goal_pos")
        return dict(qpos=obs["agent"]["qpos"][0].detach().cpu().numpy().astype(np.float64),
                    tip=tcp[:3], tcp_q=tcp[3:], t_xy=tee[:2], t_th=quat_to_yaw(tee[3:]), goal=goal[:2])

    def _errors(self, s):
        return s["goal"] - s["t_xy"], wrap(GOAL_THETA - s["t_th"])

    def _drive(self, s, target, speed, lead=None):
        if self.randomize and self.p.noise_std > 0:
            e, dth = self._errors(s)
            scale = min(1.0, (np.linalg.norm(e) + self.p.rho * abs(dth)) / self.p.noise_fade)
            a = DT / self.p.noise_tau
            self.noise = (1 - a) * self.noise + math.sqrt(2 * a) * self.p.noise_std * self.rng.standard_normal(2)
            target = target.copy()
            target[:2] += scale * self.noise
        dxy = target[:2] - s["tip"][:2]
        dist = np.linalg.norm(dxy)
        cap = lead if lead is not None else speed * self.speed * DT
        if dist > cap:
            dxy = dxy * (cap / dist)
        dz_err = target[2] - s["tip"][2]
        dz = float(np.clip(dz_err * 2.0, -self.p.z_step, self.p.z_step)) if abs(dz_err) > 0.004 else dz_err
        dq = self.ik.joint_delta(s["qpos"], np.array([dxy[0], dxy[1], dz]), 0.5 * rot_error(s["tcp_q"], self.q_ref))
        return np.clip(self.p.gain * dq / JOINT_DELTA_SCALE, -1.0, 1.0)

    def _route(self, s, stand):
        poly = world_poly(s["t_xy"], s["t_th"])
        tip = s["tip"][:2]
        if segment_hits_poly(tip, stand, poly, self.p.tip_margin):
            rise = tip + 0.25 * (stand - tip)                    # climb diagonally, not straight up
            return [np.array([*rise, self.p.z_lift]), np.array([*stand, self.p.z_lift]), np.array([*stand, self.p.z_push])]
        return [np.array([*stand, self.p.z_push])]

    def _start_plan(self, s):
        i, Lp = self.plan(s)
        self.ci, self.length = i, Lp
        pose = np.array([*s["t_xy"], s["t_th"]])
        new, *_ = predict(pose, np.array([Lp]), self.p)
        self.pred_delta = new[i, 0] - pose                       # predicted T motion (x, y, theta)
        R = rot(s["t_th"])
        stand = s["t_xy"] + R @ R_LOC[i] + (R @ N_LOC[i]) * self.standoff
        self.waypoints = self._route(s, stand)
        self.phase = "approach"
        self.approach_steps = 0

    def act(self, obs):
        s = self._parse(obs)
        e, dth = self._errors(s)
        R = rot(s["t_th"])
        if np.linalg.norm(e) < self.p.pos_tol and abs(dth) < self.p.ang_tol:
            self.phase = "hold"
        elif self.phase == "hold" and (np.linalg.norm(e) > 2 * self.p.pos_tol or abs(dth) > 2 * self.p.ang_tol):
            self.phase = "plan"

        if self.phase == "hold":
            poly = world_poly(s["t_xy"], s["t_th"])
            tip = s["tip"][:2]
            if dist_to_poly(tip, poly) < 0.04:
                away = (tip - s["t_xy"]) / (np.linalg.norm(tip - s["t_xy"]) + 1e-9)
                return self._drive(s, np.array([*(tip + 0.02 * away), self.p.z_lift]), self.p.move_speed)
            return np.zeros(7)

        if self.phase == "push" and s["tip"][2] > T_HEIGHT - 0.003:
            self.phase = "plan"                                   # rode onto the T: start over
        if self.phase == "plan":
            self._start_plan(s)

        if self.phase == "approach":
            self.approach_steps += 1
            stand = s["t_xy"] + R @ R_LOC[self.ci] + (R @ N_LOC[self.ci]) * self.standoff
            for w in self.waypoints[-2:]:
                w[:2] = stand
            if self.approach_steps > self.p.max_approach_steps:
                self._start_plan(s)
            wp = self.waypoints[0]
            lifted = wp[2] > self.p.z_push + 0.01
            reached = (np.linalg.norm(wp[:2] - s["tip"][:2]) < 0.025 and s["tip"][2] > self.p.z_lift - 0.025) if lifted \
                else (np.linalg.norm(wp[:2] - s["tip"][:2]) < 0.012 and s["tip"][2] < self.p.z_push + 0.009)
            if reached:
                self.waypoints.pop(0)
                if not self.waypoints:
                    self.phase = "push"
                    self.push_start = s["tip"][:2].copy()
                    self.push_dir = -(R @ N_LOC[self.ci])
                    self.push_steps = 0
                    self.t_hist = []
                    self.push_pose0 = np.array([*s["t_xy"], s["t_th"]])
            if self.phase == "approach":
                return self._drive(s, self.waypoints[0], self.p.move_speed)

        if self.phase == "push":
            self.push_steps += 1
            travelled = float(np.dot(s["tip"][:2] - self.push_start, self.push_dir))
            goal_dist = self.standoff + self.length
            self.t_hist.append(np.array([*s["t_xy"], s["t_th"]]))
            stalled = len(self.t_hist) > 6 and travelled > self.standoff + 0.005 and \
                np.linalg.norm(self.t_hist[-1][:2] - self.t_hist[-6][:2]) < 0.002 and \
                abs(wrap(self.t_hist[-1][2] - self.t_hist[-6][2])) < 0.01
            # progress = how much of the predicted T motion has happened (pose space, angle weighted)
            w = np.array([1.0, 1.0, self.p.rho])
            act_d = np.array([*(s["t_xy"] - self.push_pose0[:2]), wrap(s["t_th"] - self.push_pose0[2])]) * w
            pred_d = self.pred_delta * w
            progress = float(np.dot(act_d, pred_d) / max(np.dot(pred_d, pred_d), 1e-9))
            done = progress >= self.p.progress_stop or travelled >= goal_dist + (0.015 if self.p.progress_stop < 5 else -0.002)
            if done or stalled or self.push_steps > 30:
                self.phase = "retreat"
                self.retreat_from = s["tip"][:2].copy()
            else:
                fine = self.length < 0.02
                lead = self.p.fine_lead if fine else self.p.push_lead
                along = max(travelled, 0.0) + lead
                target = np.array([*(self.push_start + self.push_dir * along), self.p.z_push])
                return self._drive(s, target, None, lead=lead)

        if self.phase == "retreat":
            back = self.retreat_from - self.push_dir * self.p.retreat
            if np.linalg.norm(s["tip"][:2] - back) < 0.004:
                self.phase = "plan"
                return self.act(obs)
            return self._drive(s, np.array([*back, self.p.z_push]), self.p.move_speed)

        self.phase = "plan"
        return np.zeros(7)

"""Scripted, demo-like Push-T pusher for PushTVarGoal-v1 (panda_stick, pd_joint_delta_pos).

Plan -> act loop, in the spirit of OCBench's scripted policies (smooth, randomized, history-dependent):
  1. PLAN: score candidate pushes (contact point on the T outline x push length) with a quasi-static
     ellipsoidal limit-surface model and pick one of the best (randomly among the top few).
  2. APPROACH: move the stick tip to a stand-off point behind the contact, lifting over the T if the
     straight path would hit it.
  3. PUSH: drive the tip along the push direction; then RETREAT a little and replan.
  4. HOLD: once the T is within tolerance of the goal, back away from it and keep still.
The desired tip motion is turned into the robot's native joint-delta actions with damped least squares
on the arm's Jacobian (pytorch_kinematics on ManiSkill's panda_stick.urdf), so recorded actions can be
used directly as BC labels.

Only the simulator state is used: tcp pose, T pose and goal position (state_dict observation).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------- T geometry (mani_skill push_t.py create_tee)
# Local frame, centre of mass at the origin. Bar: x in [-0.1, 0.1], y in [-0.0625, -0.0125];
# stem: x in [-0.025, 0.025], y in [-0.0125, 0.1375]. Counterclockwise outline.
T_POLY = np.array([(-0.1, -0.0625), (0.1, -0.0625), (0.1, -0.0125), (0.025, -0.0125),
                   (0.025, 0.1375), (-0.025, 0.1375), (-0.025, -0.0125), (-0.1, -0.0125)])
T_HEIGHT = 0.04
GOAL_THETA = 5.0 / 3.0 * math.pi          # PushT-v1's fixed goal orientation
ROBOT_BASE = np.array([-0.615, 0.0, 0.0])
DT = 1.0 / 20.0                            # control period
JOINT_DELTA_SCALE = 0.1                    # pd_joint_delta_pos: action 1.0 = 0.1 rad target change


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def rot(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def quat_to_yaw(q):
    """ManiSkill quaternions are (w, x, y, z); the T only rotates about z."""
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def world_poly(xy, theta):
    return T_POLY @ rot(theta).T + xy


def _edges():
    """Contact candidates in the T frame: points along each edge with the edge's outward normal."""
    out = []
    n = len(T_POLY)
    for i in range(n):
        a, b = T_POLY[i], T_POLY[(i + 1) % n]
        e = b - a
        length = np.linalg.norm(e)
        normal = np.array([e[1], -e[0]]) / length          # outward for a CCW polygon
        k = max(1, int(round((length - 0.02) / 0.01)))
        for t in np.linspace(0.01 / length, 1 - 0.01 / length, k) if length > 0.03 else [0.5]:
            out.append((a + t * e, normal))
    return out


INNER_CORNERS = np.array([(0.025, -0.0125), (-0.025, -0.0125)])   # where the stem meets the bar (concave)
# Contacts near the inside corners wedge the tip against two faces at once, so leave them out.
CONTACTS = [(r, n) for r, n in _edges() if np.min(np.linalg.norm(INNER_CORNERS - r, axis=1)) > 0.03]


def point_in_poly(p, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        if (a[1] > p[1]) != (b[1] > p[1]):
            x = a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if x > p[0]:
                inside = not inside
    return inside


def dist_to_poly(p, poly):
    """Distance from point p to the polygon outline (0 if inside)."""
    if point_in_poly(p, poly):
        return 0.0
    best = 1e9
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
        best = min(best, np.linalg.norm(p - (a + t * ab)))
    return best


def segment_hits_poly(p0, p1, poly, margin):
    for t in np.linspace(0, 1, 25):
        if dist_to_poly(p0 + t * (p1 - p0), poly) < margin:
            return True
    return False


# ---------------------------------------------------------------- kinematics
class ArmIK:
    """Damped least-squares velocity IK for panda_stick's tcp (position + keep the stick pointing down)."""

    def __init__(self):
        import mani_skill
        import pytorch_kinematics as pk
        urdf = Path(mani_skill.__file__).parent / "assets/robots/panda/panda_stick.urdf"
        self.chain = pk.build_serial_chain_from_urdf(urdf.read_bytes(), "panda_hand_tcp").to(dtype=torch.float64)

    def joint_delta(self, qpos, dpos, rot_err, damping=0.02):
        q = torch.as_tensor(qpos, dtype=torch.float64)[None]
        J = self.chain.jacobian(q)[0].numpy()                       # 6 x 7 (linear; angular), base frame
        target = np.concatenate([dpos, rot_err])
        JJt = J @ J.T + (damping ** 2) * np.eye(6)
        return J.T @ np.linalg.solve(JJt, target)


def rot_error(q_cur, q_ref):
    """Small-angle rotation vector taking q_cur to q_ref (both w, x, y, z)."""
    w1, x1, y1, z1 = q_cur
    w2, x2, y2, z2 = q_ref
    # q_err = q_ref * conj(q_cur)
    w = w2 * w1 + x2 * x1 + y2 * y1 + z2 * z1
    x = -w2 * x1 + x2 * w1 - y2 * z1 + z2 * y1
    y = -w2 * y1 + x2 * z1 + y2 * w1 - z2 * x1
    z = -w2 * z1 - x2 * y1 + y2 * x1 + z2 * w1
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
    return 2.0 * np.array([x, y, z])


# ---------------------------------------------------------------- the pusher
@dataclass
class PusherParams:
    # quasi-static model and planning
    c_gyr: float = 0.08           # limit-surface characteristic length (m), fitted by calibrate.py
    rho: float = 0.06             # metres of error per radian of angle error in the cost
    push_lengths: tuple = (0.012, 0.025, 0.045, 0.075, 0.11)
    top_k: int = 3                # sample among the best k pushes (demo-like variety)
    # execution
    standoff: float = 0.025       # stand-off distance behind the contact point
    tip_margin: float = 0.018     # treat the T as this much larger when checking tip paths
    move_speed: float = 0.45      # m/s in free space
    push_speed: float = 0.22      # m/s while pushing
    retreat: float = 0.007        # back-off after a push
    push_lead: float = 0.03       # keep the push target this far ahead of the tip (sets the push force)
    max_push_steps: int = 40      # cap on one continuous push
    max_approach_steps: int = 25  # replan if an approach takes longer than this
    steer_radius: float = 0.035   # contacts this close to the tip count as "nearby" while pushing
    steer_gain_min: float = 0.985 # keep pushing only while a nearby push still reduces the cost by >1.5%
    lift_penalty: float = 0.004   # extra cost for contacts that need lifting over the T
    travel_weight: float = 0.05   # cost per squared metre of tip travel to the stand-off point
    z_push: float = 0.015         # tcp height while pushing (T is 0.04 tall)
    z_lift: float = 0.075         # tcp height when passing over the T
    gain: float = 3.0             # joint-delta gain (PD lag compensation)
    # termination
    pos_tol: float = 0.007
    ang_tol: float = 0.05
    # demo-like randomness
    noise_std: float = 0.004      # smooth (Ornstein-Uhlenbeck) jitter on the tip target, metres
    noise_tau: float = 0.4        # seconds
    noise_fade: float = 0.06      # jitter scales down linearly once the pose error is below this
    speed_jitter: float = 0.15    # per-episode speed scale in [1-j, 1+j]
    standoff_jitter: float = 0.008


@dataclass
class _Plan:
    contact: np.ndarray = None    # T frame
    normal: np.ndarray = None     # T frame, outward
    length: float = 0.0


class ScriptedPusher:
    def __init__(self, params: PusherParams | None = None, seed: int = 0, randomize: bool = True):
        self.p = params or PusherParams()
        self.ik = ArmIK()
        self.rng = np.random.default_rng(seed)
        self.randomize = randomize

    # ------------------------------------------------------------ episode setup
    def reset(self, obs):
        s = self._parse(obs)
        self.q_ref = s["tcp_q"].copy()                       # keep the stick orientation from reset
        j = self.p.speed_jitter if self.randomize else 0.0
        self.speed = float(self.rng.uniform(1 - j, 1 + j))
        k = self.p.standoff_jitter if self.randomize else 0.0
        self.standoff = self.p.standoff + float(self.rng.uniform(-k, k))
        self.noise = np.zeros(2)
        self.phase = "plan"
        self.waypoints = []
        self.plan = None
        self.push_start = None
        self.log = []

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _parse(obs):
        g = lambda k: obs["extra"][k][0].detach().cpu().numpy().astype(np.float64)  # noqa: E731
        tcp, tee, goal = g("tcp_pose"), g("obj_pose"), g("goal_pos")
        return dict(qpos=obs["agent"]["qpos"][0].detach().cpu().numpy().astype(np.float64),
                    tip=tcp[:3], tcp_q=tcp[3:], t_xy=tee[:2], t_th=quat_to_yaw(tee[3:]), goal=goal[:2])

    def _errors(self, s):
        return s["goal"] - s["t_xy"], wrap(GOAL_THETA - s["t_th"])

    def _cost(self, e, dth):
        return float(np.sum(e ** 2) + (self.p.rho * dth) ** 2)

    def _predict(self, s, r_loc, n_loc, L):
        """Quasi-static (ellipsoidal limit surface, sticking contact) prediction of a push of length L."""
        e, dth = self._errors(s)
        R = rot(s["t_th"])
        r, n = R @ r_loc, R @ n_loc
        d = -n
        m = r[0] * d[1] - r[1] * d[0]
        c2 = self.p.c_gyr ** 2
        k = L / (1 + m * m / c2)
        return self._cost(e - k * d, wrap(dth - k * m / c2))

    def _choose_push(self, s):
        R = rot(s["t_th"])
        poly = world_poly(s["t_xy"], s["t_th"])
        tip = s["tip"][:2]
        scored = []
        for r_loc, n_loc in CONTACTS:
            stand = s["t_xy"] + R @ r_loc + (R @ n_loc) * self.standoff
            travel = np.linalg.norm(stand - tip)
            if dist_to_poly(stand, poly) < 0.6 * self.standoff:      # stand-off point inside a concave notch
                continue
            extra = self.p.travel_weight * travel ** 2
            if segment_hits_poly(tip, stand, poly, self.p.tip_margin):
                extra += self.p.lift_penalty
            best = min(self._predict(s, r_loc, n_loc, L) for L in self.p.push_lengths)
            scored.append((best + extra, r_loc, n_loc))
        scored.sort(key=lambda x: x[0])
        k = self.p.top_k if self.randomize else 1
        pick = scored[int(self.rng.integers(min(k, len(scored))))]
        return _Plan(contact=pick[1], normal=pick[2], length=0.0)

    def _steer(self, s):
        """While in contact: best nearby contact (sliding along the edge), or None if no nearby push still helps."""
        R = rot(s["t_th"])
        tip = s["tip"][:2]
        e, dth = self._errors(s)
        now = self._cost(e, dth)
        best = None
        for r_loc, n_loc in CONTACTS:
            cw = s["t_xy"] + R @ r_loc
            if np.linalg.norm(cw - tip) > self.p.steer_radius:
                continue
            c = min(self._predict(s, r_loc, n_loc, L) for L in (0.01, 0.02, 0.04))
            if best is None or c < best[0]:
                best = (c, r_loc, n_loc)
        if best is None or best[0] > now * self.p.steer_gain_min:
            return None
        return best

    def _route(self, s, stand_xy):
        """Waypoints (x, y, z) from the tip to the stand-off point, lifting over the T if needed."""
        poly = world_poly(s["t_xy"], s["t_th"])
        tip = s["tip"][:2]
        if segment_hits_poly(tip, stand_xy, poly, self.p.tip_margin):
            return [np.array([*tip, self.p.z_lift]), np.array([*stand_xy, self.p.z_lift]),
                    np.array([*stand_xy, self.p.z_push])]
        return [np.array([*stand_xy, self.p.z_push])]

    def _drive(self, s, target, speed, lead=None):
        """Joint-delta action that moves the tip toward target (x, y, z).

        Free motion: the xy step is capped at `speed`. Pushing (`lead` given): the xy target is kept `lead` metres
        ahead of the tip, because the joint PD controller only applies force in proportion to its tracking error.
        Height is corrected separately so the tip neither sinks into the table nor floats over the T."""
        if self.randomize and self.p.noise_std > 0:
            # smooth jitter, faded out near the goal so variety does not cost final precision
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
        dz = float(np.clip(1.5 * (target[2] - s["tip"][2]), -0.04, 0.04))
        delta = np.array([dxy[0], dxy[1], dz])
        dq = self.ik.joint_delta(s["qpos"], delta, 0.5 * rot_error(s["tcp_q"], self.q_ref))
        return np.clip(self.p.gain * dq / JOINT_DELTA_SCALE, -1.0, 1.0)

    # ------------------------------------------------------------ main step
    def act(self, obs):
        s = self._parse(obs)
        e, dth = self._errors(s)
        R = rot(s["t_th"])
        done_now = np.linalg.norm(e) < self.p.pos_tol and abs(dth) < self.p.ang_tol

        if done_now and self.phase not in ("hold",):
            self.phase = "hold"
        elif self.phase == "hold" and (np.linalg.norm(e) > 2 * self.p.pos_tol or abs(dth) > 2 * self.p.ang_tol):
            self.phase = "plan"

        if self.phase == "hold":
            poly = world_poly(s["t_xy"], s["t_th"])
            tip = s["tip"][:2]
            if dist_to_poly(tip, poly) < 0.04:                  # back away from the T, then stay still
                away = tip - s["t_xy"]
                away = away / (np.linalg.norm(away) + 1e-9)
                act = self._drive(s, np.array([*(tip + 0.02 * away), self.p.z_lift]), self.p.move_speed)
            else:
                act = np.zeros(7)
            self.log.append(dict(s, phase="hold"))
            return act

        if self.phase == "plan":
            self.plan = self._choose_push(s)
            contact_w = s["t_xy"] + R @ self.plan.contact
            stand = contact_w + (R @ self.plan.normal) * self.standoff
            self.waypoints = self._route(s, stand)
            self.phase = "approach"
            self.approach_steps = 0

        if self.phase == "approach":
            self.approach_steps += 1
            if self.approach_steps > self.p.max_approach_steps:
                self.phase = "plan"
                self.approach_steps = 0
                self.plan = self._choose_push(s)
                stand = s["t_xy"] + R @ self.plan.contact + (R @ self.plan.normal) * self.standoff
                self.waypoints = self._route(s, stand)
            # re-aim at the stand-off point as the T may have drifted
            contact_w = s["t_xy"] + R @ self.plan.contact
            stand = contact_w + (R @ self.plan.normal) * self.standoff
            self.waypoints[-1][:2] = stand
            if len(self.waypoints) >= 2:
                self.waypoints[-2][:2] = stand
            wp = self.waypoints[0]
            lifted_wp = wp[2] > self.p.z_push + 0.01
            reached = (np.linalg.norm(wp[:2] - s["tip"][:2]) < 0.025 and s["tip"][2] > self.p.z_lift - 0.025) if lifted_wp \
                else (np.linalg.norm(wp[:2] - s["tip"][:2]) < 0.012 and s["tip"][2] < self.p.z_push + 0.006)
            if reached:
                self.waypoints.pop(0)
                if not self.waypoints:
                    self.phase = "push"
                    self.push_start = s["tip"][:2].copy()
                    self.push_dir = -(R @ self.plan.normal)
                    self.push_steps = 0
                    self.t_hist = []
            if self.phase == "approach":
                act = self._drive(s, self.waypoints[0], self.p.move_speed)
                self.log.append(dict(s, phase="approach"))
                return act

        if self.phase == "push" and s["tip"][2] > T_HEIGHT - 0.008:
            # The tip rode up onto the T: lift off and plan again instead of rubbing across its top.
            if point_in_poly(s["tip"][:2], world_poly(s["t_xy"], s["t_th"])) or s["tip"][2] > T_HEIGHT - 0.003:
                self.phase = "plan"
                return self.act(obs)

        if self.phase == "push":
            # Closing the stand-off gap: aim at the planned contact. In contact: steer among nearby contacts.
            self.push_steps += 1
            contact_w = s["t_xy"] + R @ self.plan.contact
            in_contact = np.linalg.norm(contact_w - s["tip"][:2]) < self.p.steer_radius
            if in_contact and self.push_steps > 2:
                best = self._steer(s)
                self.t_hist.append(s["t_xy"].copy())
                stalled = len(self.t_hist) > 5 and np.linalg.norm(self.t_hist[-1] - self.t_hist[-6]) < 0.004
                if best is None or stalled or self.push_steps > self.p.max_push_steps:
                    self.phase = "retreat"
                    self.retreat_from = s["tip"][:2].copy()
                    self.push_dir = -(R @ self.plan.normal)
                else:
                    self.plan = _Plan(contact=best[1], normal=best[2])
                    contact_w = s["t_xy"] + R @ self.plan.contact
            if self.phase == "push":
                d = -(R @ self.plan.normal)
                self.push_dir = d
                target = np.array([*(contact_w + d * self.p.push_lead), self.p.z_push])
                act = self._drive(s, target, self.p.push_speed, lead=self.p.push_lead)
                self.log.append(dict(s, phase="push"))
                return act

        if self.phase == "retreat":
            back = self.retreat_from - self.push_dir * self.p.retreat
            if np.linalg.norm(s["tip"][:2] - back) < 0.004:
                self.phase = "plan"
                return self.act(obs)
            act = self._drive(s, np.array([*back, self.p.z_push]), self.p.move_speed)
            self.log.append(dict(s, phase="retreat"))
            return act

        self.phase = "plan"
        return np.zeros(7)

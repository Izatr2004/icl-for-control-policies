"""PushT-VariableGoal-v1: ManiSkill PushT-v1 with a random goal and random stick-tip start per episode.

Changes from PushT-v1 (ManiSkill v3.0.1):
- Goal: the goal T's position is sampled every episode within +-GOAL_HALF_WIDTH (x and y) of
  PushT-v1's fixed goal (the "centre"). Its rotation stays fixed. Quadrants around the centre, world
  frame: Q1 (+x, +y), Q2 (-x, +y), Q3 (-x, -y), Q4 (+x, -y). `goal_quadrants` limits which are
  sampled (all four for teachers; Q1-Q3 for transformer training).
- Success check: uses a per-environment goal transform. PushT-v1 computes one fixed transform at load
  time, which would score every episode against the original goal.
- Stick-tip start: sampled within +-TIP_HALF_WIDTH of the centre, in a quadrant allowed by
  `start_quadrants` (default all four), at least TIP_CLEARANCE from the movable T's centre. The arm is
  placed there by damped-least-squares inverse kinematics on the GPU, keeping PushT-v1's rest height
  and stick orientation. If IK misses by more than IK_TOLERANCE, that episode keeps PushT-v1's start.
  `random_tip_start=False` restores PushT-v1's fixed tip start.
- Movable T: PushT-v1's original absolute spawn region and random rotation (unchanged).
- Starts are sampled independently of the goal, so a start never reveals the goal.
- Observations are unchanged: state mode still includes `extra/goal_pos`, which now varies.

Per-episode records on the env: goal_xy, goal_quadrant, tip_start_xy, tip_start_quadrant,
tee_start_quadrant, tip_ik_error, tip_ik_fallback.

Importing this module registers the environment.
"""

import numpy as np
import torch

ENV_ID = "PushT-VariableGoal-v1"
GOAL_HALF_WIDTH = 0.10   # metres; goals within +-10 cm of the centre in x and y
TIP_HALF_WIDTH = 0.15    # metres; stick-tip starts within +-15 cm of the centre in x and y
TIP_CLEARANCE = 0.15     # metres from the T's centre of mass; the T reaches about 0.14 m from it
TIP_CANDIDATES = 32      # rejection-sampling candidates per episode
IK_ITERATIONS = 60
IK_TOLERANCE = 0.01      # metres; larger position error falls back to PushT-v1's start
ROBOT_BASE_XYZ = (-0.615, 0.0, 0.0)  # PushT-v1 places the robot base here, unrotated
QUADRANT_SIGNS = {1: (1.0, 1.0), 2: (-1.0, 1.0), 3: (-1.0, -1.0), 4: (1.0, -1.0)}


def quadrant_of(offset_xy):
    """Quadrant (1-4) of offsets [n, 2] from the centre. Zero counts as positive."""
    pos_x, pos_y = offset_xy[:, 0] >= 0, offset_xy[:, 1] >= 0
    q = torch.full((offset_xy.shape[0],), 3, dtype=torch.long, device=offset_xy.device)
    q[pos_x & pos_y] = 1
    q[~pos_x & pos_y] = 2
    q[pos_x & ~pos_y] = 4
    return q


def sample_goal_offsets(n, quadrants, half_width=GOAL_HALF_WIDTH, generator=None, device=None):
    """Uniform over the allowed quadrants, then uniform within the chosen quadrant's square."""
    allowed = torch.as_tensor(sorted(quadrants), device=device)
    pick = allowed[torch.randint(len(allowed), (n,), generator=generator, device=device)]
    signs = torch.as_tensor([QUADRANT_SIGNS[int(q)] for q in range(1, 5)], device=device)[pick - 1]
    mags = torch.rand((n, 2), generator=generator, device=device) * half_width
    return signs * mags, pick


def sample_tip_offsets(tee_offset_xy, quadrants, half_width=TIP_HALF_WIDTH, clearance=TIP_CLEARANCE,
                       candidates=TIP_CANDIDATES, generator=None):
    """Tip offsets from the centre [n, 2] in an allowed quadrant, at least `clearance` from the T.

    Draws `candidates` points per episode and keeps the first that clears the T; if none does,
    keeps the candidate farthest from it. Returns offsets, quadrants and the clearance achieved.
    """
    n, device = tee_offset_xy.shape[0], tee_offset_xy.device
    offs, quads = sample_goal_offsets(n * candidates, quadrants, half_width, generator, device)
    offs, quads = offs.view(n, candidates, 2), quads.view(n, candidates)
    dist = torch.linalg.norm(offs - tee_offset_xy.unsqueeze(1), dim=-1)
    ok = dist >= clearance
    first_ok = torch.where(ok.any(1), ok.float().argmax(1), dist.argmax(1))
    rows = torch.arange(n, device=device)
    return offs[rows, first_ok], quads[rows, first_ok], dist[rows, first_ok]


def orientation_error(r_current, r_target):
    """Small-angle rotation error [n, 3] taking r_current to r_target (both [n, 3, 3])."""
    return 0.5 * torch.cross(r_current, r_target, dim=1).sum(-1)


def dls_ik(fk, jacobian, q0, target_pos, target_rot, lower, upper, iterations=IK_ITERATIONS, damping=0.05,
           max_step=0.2):
    """Batched damped-least-squares IK for position and orientation.

    fk(q) -> (pos [n, 3], rot [n, 3, 3]); jacobian(q) -> [n, 6, dof] (linear rows first).
    Returns joint positions and the final position error in metres.
    """
    q = q0.clone()
    eye = torch.eye(6, device=q.device, dtype=q.dtype)
    for _ in range(iterations):
        pos, rot = fk(q)
        err = torch.cat([target_pos - pos, orientation_error(rot, target_rot)], dim=1)
        J = jacobian(q)
        JJt = J @ J.transpose(1, 2) + (damping ** 2) * eye
        dq = (J.transpose(1, 2) @ torch.linalg.solve(JJt, err.unsqueeze(-1))).squeeze(-1)
        dq = dq.clamp(-max_step, max_step)
        q = torch.maximum(torch.minimum(q + dq, upper), lower)
    pos, _ = fk(q)
    return q, torch.linalg.norm(target_pos - pos, dim=1)


def initial_guess(rest_q, rest_pos, target_pos, lower, upper):
    """Rest posture with joint 1 (rotation about the base's vertical axis) turned so the tip's
    bearing from the base matches the target's."""
    turn = torch.atan2(target_pos[:, 1], target_pos[:, 0]) - torch.atan2(rest_pos[1], rest_pos[0])
    q = rest_q.clone()
    q[:, 0] = q[:, 0] + turn
    return torch.maximum(torch.minimum(q, upper), lower)


def world_to_goal_transforms(goal_to_world_base, goal_xy):
    """Per-episode 2D homogeneous world->goal transforms [n, 3, 3]: the base goal rotation,
    translated to each goal position. Used by the success check."""
    trans = goal_to_world_base.unsqueeze(0).repeat(goal_xy.shape[0], 1, 1)
    trans[:, 0:2, 2] = goal_xy
    return torch.linalg.inv(trans)


def _register():
    from mani_skill.envs.tasks.tabletop.push_t import PushTEnv
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.structs.pose import Pose
    from transforms3d.euler import euler2quat

    @register_env(ENV_ID, max_episode_steps=100)
    class PushTVariableGoalEnv(PushTEnv):
        def __init__(self, *args, goal_quadrants=(1, 2, 3, 4), start_quadrants=(1, 2, 3, 4),
                     goal_half_width=GOAL_HALF_WIDTH, random_tip_start=True, **kwargs):
            self.goal_quadrants = _check_quadrants(goal_quadrants, "goal_quadrants")
            self.start_quadrants = _check_quadrants(start_quadrants, "start_quadrants")
            self.goal_half_width = float(goal_half_width)
            self.random_tip_start = bool(random_tip_start)
            super().__init__(*args, **kwargs)

        def _load_scene(self, options: dict):
            super()._load_scene(options)
            n, dev = self.num_envs, self.device
            # Base goal-to-world transform (rotation fixed), copied per environment.
            self._goal_trans_base = torch.linalg.inv(self.world_to_goal_trans)
            self.world_to_goal_trans = self.world_to_goal_trans.unsqueeze(0).repeat(n, 1, 1).clone()
            self.goal_xy = self.goal_offset.to(dev).unsqueeze(0).repeat(n, 1).clone()
            self.goal_quadrant = torch.zeros(n, dtype=torch.long, device=dev)
            self.tip_start_xy = torch.zeros(n, 2, device=dev)
            self.tip_start_quadrant = torch.zeros(n, dtype=torch.long, device=dev)
            self.tee_start_quadrant = torch.zeros(n, dtype=torch.long, device=dev)
            self.tip_ik_error = torch.zeros(n, device=dev)
            self.tip_ik_fallback = torch.zeros(n, dtype=torch.bool, device=dev)
            self._ik = None

        def _setup_ik(self):
            """Kinematic chain from the robot base to the stick tip, and the rest tip pose (base frame)."""
            import pytorch_kinematics as pk

            with open(self.agent.urdf_path, "rb") as f:
                chain = pk.build_serial_chain_from_urdf(f.read(), end_link_name=self.agent.ee_link_name)
            chain = chain.to(device=self.device, dtype=torch.float32)
            lim = torch.as_tensor(chain.get_joint_limits(), dtype=torch.float32, device=self.device)
            rest = torch.as_tensor(self.agent.keyframes["rest"].qpos, dtype=torch.float32, device=self.device)
            joints = chain.get_joint_parameter_names()
            assert len(joints) == rest.numel(), (joints, rest.numel())

            def fk(q):
                m = chain.forward_kinematics(q).get_matrix()
                return m[:, :3, 3], m[:, :3, :3]

            rest_pos, rest_rot = fk(rest.unsqueeze(0))
            self._ik = dict(fk=fk, jacobian=lambda q: chain.jacobian(q), lower=lim[0], upper=lim[1],
                            rest=rest, rest_pos=rest_pos[0], rest_rot=rest_rot[0],
                            base=torch.tensor(ROBOT_BASE_XYZ, device=self.device))

        def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
            with torch.device(self.device):
                b = len(env_idx)
                self.table_scene.initialize(env_idx)  # robot rest pose + small joint noise, as PushT-v1
                center = self.goal_offset.to(self.device)

                # Goal: random position in an allowed quadrant, fixed rotation.
                offsets, quadrant = sample_goal_offsets(b, self.goal_quadrants, self.goal_half_width, device=self.device)
                goal_xy = center + offsets
                goal_xyz = torch.zeros((b, 3))
                goal_xyz[:, :2] = goal_xy
                goal_xyz[:, 2] = 1e-3
                self.goal_tee.set_pose(Pose.create_from_pq(p=goal_xyz, q=euler2quat(0, 0, self.goal_z_rot)))
                self.goal_xy[env_idx] = goal_xy
                self.goal_quadrant[env_idx] = quadrant
                self.world_to_goal_trans[env_idx] = world_to_goal_transforms(self._goal_trans_base, goal_xy)

                # Movable T: PushT-v1's original absolute spawn region, random rotation.
                tee_xyz = torch.zeros((b, 3))
                tee_xyz[:, 0] = center[0] + torch.rand(b) * self.tee_spawnbox_xlength + self.tee_spawnbox_xoffset
                tee_xyz[:, 1] = center[1] + torch.rand(b) * self.tee_spawnbox_ylength + self.tee_spawnbox_yoffset
                tee_xyz[:, 2] = 0.04 / 2 + 1e-3
                angle = torch.rand(b) * (2 * torch.pi)
                q = torch.zeros((b, 4))
                q[:, 0] = (angle / 2).cos()
                q[:, -1] = (angle / 2).sin()
                self.tee.set_pose(Pose.create_from_pq(p=tee_xyz, q=q))
                self.tee_start_quadrant[env_idx] = quadrant_of(tee_xyz[:, :2] - center)

                # Stick tip: random start away from the T, placed by IK (independent of the goal).
                if self.random_tip_start:
                    self._place_tip(env_idx, tee_xyz[:, :2] - center, center)
                else:
                    tip = self.ee_starting_pos2D[:2].to(self.device).expand(b, 2)
                    self.tip_start_xy[env_idx] = tip
                    self.tip_start_quadrant[env_idx] = quadrant_of(tip - center)
                    self.tip_ik_error[env_idx] = 0.0
                    self.tip_ik_fallback[env_idx] = False

                # End-effector marker, as in PushT-v1.
                xyz = torch.zeros((b, 3))
                xyz[:] = self.ee_starting_pos2D
                self.ee_goal_pos.set_pose(Pose.create_from_pq(p=xyz, q=euler2quat(0, np.pi / 2, 0)))

        def _place_tip(self, env_idx, tee_offset_xy, center):
            if self._ik is None:
                self._setup_ik()
            ik = self._ik
            b = len(env_idx)
            tip_off, tip_quad, _ = sample_tip_offsets(tee_offset_xy, self.start_quadrants)
            tip_world = center + tip_off
            target_pos = torch.zeros((b, 3))
            target_pos[:, :2] = tip_world - ik["base"][:2]
            target_pos[:, 2] = ik["rest_pos"][2]
            target_rot = ik["rest_rot"].expand(b, 3, 3)
            # Initial guess: the rest posture, with the base joint turned toward the target.
            rest = ik["rest"].expand(b, -1)
            q0 = initial_guess(rest, ik["rest_pos"], target_pos, ik["lower"], ik["upper"])
            q, err = dls_ik(ik["fk"], ik["jacobian"], q0, target_pos, target_rot, ik["lower"], ik["upper"])
            fallback = err > IK_TOLERANCE
            q = torch.where(fallback.unsqueeze(1), rest, q)  # PushT-v1's start if IK misses
            self.agent.reset(q)
            actual_world = ik["fk"](q)[0][:, :2] + ik["base"][:2]
            self.tip_start_xy[env_idx] = actual_world
            self.tip_start_quadrant[env_idx] = torch.where(fallback, quadrant_of(actual_world - center), tip_quad)
            self.tip_ik_error[env_idx] = torch.where(fallback, torch.zeros_like(err), err)
            self.tip_ik_fallback[env_idx] = fallback

    return PushTVariableGoalEnv


def _check_quadrants(quadrants, name):
    quadrants = tuple(sorted({int(q) for q in quadrants}))
    if not quadrants or not set(quadrants) <= {1, 2, 3, 4}:
        raise ValueError(f"{name} must be a non-empty subset of 1-4, got {quadrants}")
    return quadrants


try:
    PushTVariableGoalEnv = _register()
except ImportError:  # ManiSkill not installed (offline tests use the helpers only)
    PushTVariableGoalEnv = None

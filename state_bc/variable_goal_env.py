"""PushT-VariableGoal-v1: ManiSkill PushT-v1 with a random goal and random stick-tip start per episode.

Changes from PushT-v1 (ManiSkill v3.0.1):
- Goal: the goal T's position is sampled every episode within +-GOAL_HALF_WIDTH (x and y) of
  PushT-v1's fixed goal (the "centre"). Its rotation stays fixed. Quadrants around the centre, world
  frame: Q1 (+x, +y), Q2 (-x, +y), Q3 (-x, -y), Q4 (+x, -y). `goal_quadrants` limits which are
  sampled (all four for teachers; Q1-Q3 for transformer training).
- Success check: uses a per-environment goal transform. PushT-v1 computes one fixed transform at load
  time, which would score every episode against the original goal.
- Stick-tip start: sampled within +-TIP_HALF_WIDTH of the centre, in a quadrant allowed by
  `start_quadrants` (default all four), at least TIP_CLEARANCE from the movable T's centre. Joint
  angles come from a lookup table: inverse kinematics (damped least squares, on the GPU) is solved once
  per process for a TIP_GRID x TIP_GRID grid of cell centres covering the square, at PushT-v1's rest tip
  height and stick orientation. Each reset uses the cell containing the sampled point, so the tip starts
  within half a cell diagonal (about 3 mm) of it, and resets cost no IK. Cells the solver cannot reach
  within IK_TOLERANCE are avoided; if an episode has no usable cell, it keeps PushT-v1's start.
  `random_tip_start=False` restores PushT-v1's fixed tip start.
- Movable T: PushT-v1's original absolute spawn region and random rotation (unchanged).
- Starts are sampled independently of the goal, so a start never reveals the goal.
- Observations are unchanged: state mode still includes `extra/goal_pos`, which now varies.

Per-episode records on the env: goal_xy, goal_quadrant, tip_start_xy, tip_start_quadrant,
tee_start_quadrant, tip_ik_error, tip_ik_fallback. The lookup table's reachable fraction is in
env._ik["valid_fraction"] after the first reset.

Importing this module registers the environment.
"""

import numpy as np
import torch

ENV_ID = "PushT-VariableGoal-v1"
GOAL_HALF_WIDTH = 0.10   # metres; goals within +-10 cm of the centre in x and y
TIP_HALF_WIDTH = 0.15    # metres; stick-tip starts within +-15 cm of the centre in x and y
TIP_CLEARANCE = 0.15     # metres from the T's centre of mass; the T reaches about 0.14 m from it
TIP_CANDIDATES = 32      # rejection-sampling candidates per episode
TIP_GRID = 64            # lookup table resolution per axis (cells of about 4.7 mm)
IK_ITERATIONS = 200      # one-time solve for the lookup table
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


def tip_grid_offsets(half_width=TIP_HALF_WIDTH, n=TIP_GRID, device=None):
    """Cell centres [n*n, 2] of an n x n grid over [-half_width, half_width]^2; index = iy * n + ix.
    No centre lies on an axis, so a cell's quadrant equals the quadrant of any point inside it."""
    centres = (torch.arange(n, device=device, dtype=torch.float32) + 0.5) * (2 * half_width / n) - half_width
    gy, gx = torch.meshgrid(centres, centres, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)


def grid_index(offsets_xy, half_width=TIP_HALF_WIDTH, n=TIP_GRID):
    """Index of the grid cell containing each offset [m, 2]."""
    cell = 2 * half_width / n
    ix = ((offsets_xy[:, 0] + half_width) / cell).floor().long().clamp(0, n - 1)
    iy = ((offsets_xy[:, 1] + half_width) / cell).floor().long().clamp(0, n - 1)
    return iy * n + ix


def choose_tip_cells(tee_offset_xy, quadrants, valid, grid_offsets, half_width=TIP_HALF_WIDTH, n=TIP_GRID,
                     clearance=TIP_CLEARANCE, candidates=TIP_CANDIDATES, generator=None):
    """Pick a start cell per episode: sample points uniformly in an allowed quadrant, map each to its
    cell, keep the first cell that is reachable (valid) and at least `clearance` from the T. If none
    qualifies, keep the reachable candidate farthest from the T, else the farthest candidate.
    Returns cell indices and their distance to the T."""
    m, device = tee_offset_xy.shape[0], tee_offset_xy.device
    offs, _ = sample_goal_offsets(m * candidates, quadrants, half_width, generator, device)
    idx = grid_index(offs, half_width, n).view(m, candidates)
    dist = torch.linalg.norm(grid_offsets[idx] - tee_offset_xy.unsqueeze(1), dim=-1)
    reachable = valid[idx]
    ok = reachable & (dist >= clearance)
    fallback = torch.where(reachable.any(1), torch.where(reachable, dist, torch.full_like(dist, -1.0)).argmax(1),
                           dist.argmax(1))
    choice = torch.where(ok.any(1), ok.float().argmax(1), fallback)
    rows = torch.arange(m, device=device)
    return idx[rows, choice], dist[rows, choice]


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
            """Kinematic chain and the start lookup table, built once per process and device."""
            center = tuple(float(x) for x in self.goal_offset)
            key = (str(self.device), center, TIP_HALF_WIDTH, TIP_GRID)
            if key not in _IK_TABLES:
                _IK_TABLES[key] = _build_ik_table(self.agent, self.device, torch.tensor(center, device=self.device))
            self._ik = _IK_TABLES[key]

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
            cell, _ = choose_tip_cells(tee_offset_xy, self.start_quadrants, ik["valid"], ik["grid"])
            fallback = ~ik["valid"][cell]
            # PushT-v1's rest posture if no reachable cell was found for an episode.
            q = torch.where(fallback.unsqueeze(1), ik["rest"].expand(b, -1), ik["q"][cell])
            self.agent.reset(q)
            tip = torch.where(fallback.unsqueeze(1), ik["rest_tip_xy"].expand(b, 2), ik["tip_xy"][cell])
            self.tip_start_xy[env_idx] = tip
            self.tip_start_quadrant[env_idx] = quadrant_of(tip - center)
            self.tip_ik_error[env_idx] = torch.where(fallback, torch.zeros(b, device=tip.device), ik["error"][cell])
            self.tip_ik_fallback[env_idx] = fallback

    return PushTVariableGoalEnv


_IK_TABLES = {}


def _build_ik_table(agent, device, center):
    """Solve IK once for every grid cell centre. Returns joint angles, achieved tip positions (world xy),
    position errors and a reachability mask, plus PushT-v1's rest posture and tip."""
    import pytorch_kinematics as pk

    with open(agent.urdf_path, "rb") as f:
        chain = pk.build_serial_chain_from_urdf(f.read(), end_link_name=agent.ee_link_name)
    chain = chain.to(device=device, dtype=torch.float32)
    lim = torch.as_tensor(chain.get_joint_limits(), dtype=torch.float32, device=device)
    rest = torch.as_tensor(agent.keyframes["rest"].qpos, dtype=torch.float32, device=device)
    joints = chain.get_joint_parameter_names()
    assert len(joints) == rest.numel(), (joints, rest.numel())
    base = torch.tensor(ROBOT_BASE_XYZ, device=device)

    def fk(q):
        m = chain.forward_kinematics(q).get_matrix()
        return m[:, :3, 3], m[:, :3, :3]

    rest_pos, rest_rot = fk(rest.unsqueeze(0))
    grid = tip_grid_offsets(TIP_HALF_WIDTH, TIP_GRID, device)
    k = grid.shape[0]
    target_pos = torch.zeros((k, 3), device=device)
    target_pos[:, :2] = center + grid - base[:2]
    target_pos[:, 2] = rest_pos[0, 2]
    q0 = initial_guess(rest.expand(k, -1), rest_pos[0], target_pos, lim[0], lim[1])
    q, err = dls_ik(fk, lambda q: chain.jacobian(q), q0, target_pos, rest_rot.expand(k, 3, 3), lim[0], lim[1])
    tip_xy = fk(q)[0][:, :2] + base[:2]
    valid = err <= IK_TOLERANCE
    return dict(grid=grid, q=q, tip_xy=tip_xy, error=err, valid=valid, rest=rest,
                rest_tip_xy=rest_pos[0, :2] + base[:2], valid_fraction=valid.float().mean().item())


def _check_quadrants(quadrants, name):
    quadrants = tuple(sorted({int(q) for q in quadrants}))
    if not quadrants or not set(quadrants) <= {1, 2, 3, 4}:
        raise ValueError(f"{name} must be a non-empty subset of 1-4, got {quadrants}")
    return quadrants


try:
    PushTVariableGoalEnv = _register()
except ImportError:  # ManiSkill not installed (offline tests use the helpers only)
    PushTVariableGoalEnv = None

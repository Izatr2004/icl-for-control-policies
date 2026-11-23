"""Variable-goal Push-T: ManiSkill PushT-v1 with a different target position per episode.

Registers ``PushTVarGoal-v1``. Import this module before ``gym.make``.

What changes relative to PushT-v1 (ManiSkill 3.0.1, commit a4a4f92):
  * Goal T position is sampled per episode from four quadrants around ``QUAD_ORIGIN``.
    Goal orientation stays fixed at PushT-v1's 5*pi/3.
  * The success check (``pseudo_render_intersection``) uses a goal transform per
    environment, recomputed at every reset. In PushT-v1 it is one fixed matrix, so a
    moved goal would silently be scored against the old target.
  * The T block starts anywhere in ``TEE_BOX`` (independent of the goal quadrant), with
    its distance to the goal limited to ``START_GOAL_DIST``.
  * The stick tip starts at a random point in ``TIP_BOX``, drawn from a precomputed
    library of arm poses (``assets/tip_start_qpos.npz``, built by make_tip_starts.py),
    at least ``TIP_TEE_MIN`` from the block.

What does NOT change: observation (31 numbers, same order; ``extra/goal_pos`` now
varies), action space and controller, 100-step episodes, the dense reward (it already
reads the goal T's live pose), and the success rule (overlap >= 0.90).

Quadrants, with dx = x - QUAD_ORIGIN[0] (positive = away from the robot base) and y:
  Q1: dx > 0, y > 0    Q2: dx < 0, y > 0    Q3: dx < 0, y < 0    Q4: dx > 0, y < 0
A band of ``AXIS_GAP`` around each axis is excluded so neighbouring quadrants do not
share near-identical goals.

Usage:
  env = gym.make("PushTVarGoal-v1", obs_mode="state", control_mode="pd_joint_delta_pos",
                 goal_quadrants=(1, 2, 3, 4))
  env.reset(seed=0)                                    # random goals in those quadrants
  env.reset(options=dict(goal_xy=torch.tensor([[-0.2, 0.1]])))  # fixed goal(s), [b, 2]
  env.unwrapped.goal_quadrant                          # [num_envs] quadrant of current goal
"""
from pathlib import Path

import numpy as np
import torch

from mani_skill.envs.tasks.tabletop.push_t import PushTEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

# ---- Workspace (world frame, metres). Robot base is at x = -0.615. ----------------
# Measured stick-tip reach at table height: x in about [-0.40, 0.05], y in about
# [-0.45, 0.45] (see README). Regions below keep >= ~0.1 m margin inside that.
QUAD_ORIGIN = (-0.156, 0.0)          # PushT-v1's goal x; centred in y
GOAL_HALF = (0.10, 0.20)             # goals in origin +- these
AXIS_GAP = (0.02, 0.03)              # excluded band around each quadrant axis
TEE_BOX = ((-0.30, -0.02), (-0.28, 0.28))   # T centre-of-mass start region
TIP_BOX = ((-0.36, 0.00), (-0.30, 0.30))    # stick-tip start region
START_GOAL_DIST = (0.08, 0.30)       # allowed T-start-to-goal distance
TIP_TEE_MIN = 0.15                   # stick tip starts at least this far from the T
N_CANDIDATES = 64                    # rejection-sampling candidates per reset

_QUAD_SIGNS = {1: (1.0, 1.0), 2: (-1.0, 1.0), 3: (-1.0, -1.0), 4: (1.0, -1.0)}
_DEFAULT_TIP_FILE = Path(__file__).parent / "assets" / "tip_start_qpos.npz"


def quadrant_of(goal_xy: torch.Tensor) -> torch.Tensor:
    """Quadrant (1-4) of goal positions [b, 2]."""
    dx = goal_xy[:, 0] - QUAD_ORIGIN[0]
    dy = goal_xy[:, 1] - QUAD_ORIGIN[1]
    q = torch.full_like(dx, 0, dtype=torch.long)
    q[(dx > 0) & (dy > 0)] = 1
    q[(dx < 0) & (dy > 0)] = 2
    q[(dx < 0) & (dy < 0)] = 3
    q[(dx > 0) & (dy < 0)] = 4
    return q


def sample_goals(b: int, quadrants, device=None) -> torch.Tensor:
    """Uniform goal positions [b, 2] in the given quadrants (each quadrant equally likely)."""
    quads = torch.tensor(list(quadrants), device=device)
    pick = quads[torch.randint(len(quads), (b,), device=device)]
    signs = torch.tensor([_QUAD_SIGNS[int(k)] for k in range(1, 5)], device=device)[pick - 1]
    lo = torch.tensor(AXIS_GAP, device=device)
    hi = torch.tensor(GOAL_HALF, device=device)
    mag = lo + torch.rand(b, 2, device=device) * (hi - lo)
    return torch.tensor(QUAD_ORIGIN, device=device) + signs * mag


def _uniform_box(shape, box, device):
    lo = torch.tensor([box[0][0], box[1][0]], device=device)
    hi = torch.tensor([box[0][1], box[1][1]], device=device)
    return lo + torch.rand(*shape, 2, device=device) * (hi - lo)


def _first_valid(valid: torch.Tensor) -> torch.Tensor:
    """Index of the first True per row of [b, K]; 0 if none (vanishingly rare with K=64)."""
    return valid.float().argmax(dim=1)


@register_env("PushTVarGoal-v1", max_episode_steps=100)
class PushTVarGoalEnv(PushTEnv):
    def __init__(self, *args, goal_quadrants=(1, 2, 3, 4), tip_start_file=None, **kwargs):
        assert set(goal_quadrants) <= {1, 2, 3, 4} and len(goal_quadrants) > 0
        self.goal_quadrants = tuple(int(q) for q in goal_quadrants)
        lib = np.load(tip_start_file or _DEFAULT_TIP_FILE)
        self._tip_qpos_np = lib["qpos"]
        self._tip_xy_np = lib["tip_xy"]
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        # One goal transform per environment instead of PushT-v1's single matrix.
        self.world_to_goal_trans = self.world_to_goal_trans.unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.goal_quadrant = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.tip_qpos_lib = torch.as_tensor(self._tip_qpos_np, device=self.device)
        self.tip_xy_lib = torch.as_tensor(self._tip_xy_np, device=self.device)

    def _goal_world_to_goal(self, goal_xy: torch.Tensor) -> torch.Tensor:
        b = goal_xy.shape[0]
        c, s = np.cos(self.goal_z_rot), np.sin(self.goal_z_rot)
        g = torch.zeros(b, 3, 3, device=goal_xy.device)
        g[:, 0, 0], g[:, 0, 1], g[:, 1, 0], g[:, 1, 1] = c, -s, s, c
        g[:, 2, 2] = 1.0
        g[:, 0:2, 2] = goal_xy
        return torch.linalg.inv(g)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Parent sets up the table, default goal/T and robot rest pose; we override all three.
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)

            # ---- goal ----
            goal_xy = (options or {}).get("goal_xy")
            if goal_xy is None:
                goal_xy = sample_goals(b, self.goal_quadrants, self.device)
            else:
                goal_xy = torch.as_tensor(goal_xy, dtype=torch.float32, device=self.device).reshape(-1, 2)
                if goal_xy.shape[0] == 1 and b > 1:
                    goal_xy = goal_xy.expand(b, 2)
                elif goal_xy.shape[0] == self.num_envs and b < self.num_envs:
                    goal_xy = goal_xy[env_idx]
                assert goal_xy.shape[0] == b, "goal_xy must be [1, 2], [b, 2] or [num_envs, 2]"
            goal_p = torch.zeros(b, 3)
            goal_p[:, :2] = goal_xy
            goal_p[:, 2] = 1e-3
            goal_q = torch.tensor([np.cos(self.goal_z_rot / 2), 0.0, 0.0, np.sin(self.goal_z_rot / 2)],
                                  dtype=torch.float32).expand(b, 4)
            self.goal_tee.set_pose(Pose.create_from_pq(p=goal_p, q=goal_q))
            self.world_to_goal_trans[env_idx] = self._goal_world_to_goal(goal_xy)
            self.goal_quadrant[env_idx] = quadrant_of(goal_xy)

            # ---- T block start: independent of goal quadrant, distance-limited ----
            cand = _uniform_box((b, N_CANDIDATES), TEE_BOX, self.device)          # [b, K, 2]
            d = (cand - goal_xy[:, None, :]).norm(dim=-1)
            pick = _first_valid((d >= START_GOAL_DIST[0]) & (d <= START_GOAL_DIST[1]))
            tee_xy = cand[torch.arange(b), pick]
            tee_p = torch.zeros(b, 3)
            tee_p[:, :2] = tee_xy
            tee_p[:, 2] = 0.04 / 2 + 1e-3
            ang = torch.rand(b) * (2 * torch.pi)
            tee_q = torch.zeros(b, 4)
            tee_q[:, 0], tee_q[:, 3] = (ang / 2).cos(), (ang / 2).sin()
            self.tee.set_pose(Pose.create_from_pq(p=tee_p, q=tee_q))

            # ---- stick tip start: library pose away from the T ----
            n_lib = self.tip_xy_lib.shape[0]
            idx = torch.randint(n_lib, (b, N_CANDIDATES))
            dt = (self.tip_xy_lib[idx] - tee_xy[:, None, :]).norm(dim=-1)
            pick = _first_valid(dt >= TIP_TEE_MIN)
            qpos = self.tip_qpos_lib[idx[torch.arange(b), pick]]
            qpos = qpos + torch.randn_like(qpos) * self.robot_init_qpos_noise
            self.agent.reset(qpos)

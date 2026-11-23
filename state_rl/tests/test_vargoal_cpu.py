"""CPU-simulation checks for PushTVarGoal-v1 (no GPU needed; ~1-2 min).

Run from the repo root:  python -m unittest state_rl/tests/test_vargoal_cpu.py -v
Headless Linux without a GPU needs a software Vulkan driver (mesa lavapipe).
"""
import sys
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pusht_vargoal as pv  # noqa: E402
from mani_skill.utils.structs.pose import Pose  # noqa: E402


def make(**kw):
    return gym.make("PushTVarGoal-v1", obs_mode="state", control_mode="pd_joint_delta_pos",
                    num_envs=1, sim_backend="physx_cpu", **kw)


def place_tee(u, xy, ang):
    p = torch.tensor([[xy[0], xy[1], 0.021]])
    q = torch.tensor([[np.cos(ang / 2), 0, 0, np.sin(ang / 2)]], dtype=torch.float32)
    u.tee.set_pose(Pose.create_from_pq(p=p, q=q))


class SamplingTests(unittest.TestCase):
    def test_goals_land_in_requested_quadrants_with_gap(self):
        torch.manual_seed(0)
        for q in (1, 2, 3, 4):
            g = pv.sample_goals(20000, (q,))
            self.assertTrue((pv.quadrant_of(g) == q).all())
            dx, dy = (g[:, 0] - pv.QUAD_ORIGIN[0]).abs(), (g[:, 1] - pv.QUAD_ORIGIN[1]).abs()
            self.assertTrue((dx >= pv.AXIS_GAP[0]).all() and (dx <= pv.GOAL_HALF[0]).all())
            self.assertTrue((dy >= pv.AXIS_GAP[1]).all() and (dy <= pv.GOAL_HALF[1]).all())

    def test_quadrant_mix_is_balanced(self):
        torch.manual_seed(0)
        counts = torch.bincount(pv.quadrant_of(pv.sample_goals(40000, (1, 2, 3))), minlength=5)
        self.assertEqual(int(counts[4]), 0)
        self.assertTrue(((counts[1:4].float() / 40000 - 1 / 3).abs() < 0.02).all())


class EnvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = make()
        cls.u = cls.env.unwrapped

    def test_obs_and_spaces_unchanged(self):
        obs, _ = self.env.reset(seed=0)
        self.assertEqual(tuple(obs.shape), (1, 31))
        self.assertEqual(self.env.action_space.shape, (7,))
        from mani_skill.utils import gym_utils
        self.assertEqual(gym_utils.find_max_episode_steps_value(self.env), 100)

    def test_resets_respect_all_constraints(self):
        seen = set()
        for s in range(60):
            obs, _ = self.env.reset(seed=s)
            u = self.u
            goal = u.goal_tee.pose.p[0, :2]
            tee = u.tee.pose.p[0, :2]
            tip = u.agent.tcp.pose.p[0]
            q = int(u.goal_quadrant[0]); seen.add(q)
            self.assertEqual(q, int(pv.quadrant_of(goal[None])[0]))
            # observation carries the new goal at indices 21:24
            self.assertTrue(torch.allclose(obs[0, 21:23], goal, atol=1e-5))
            d = float((tee - goal).norm())
            self.assertGreaterEqual(d, pv.START_GOAL_DIST[0] - 1e-6)
            self.assertLessEqual(d, pv.START_GOAL_DIST[1] + 1e-6)
            self.assertTrue(pv.TEE_BOX[0][0] <= tee[0] <= pv.TEE_BOX[0][1])
            self.assertTrue(pv.TEE_BOX[1][0] <= tee[1] <= pv.TEE_BOX[1][1])
            # tip: near table, away from T (qpos noise can move it a little)
            self.assertLess(abs(float(tip[2]) - 0.0185), 0.03)
            self.assertGreater(float((tip[:2] - tee).norm()), pv.TIP_TEE_MIN - 0.04)
            self.assertFalse(bool(u.evaluate()["success"][0]))
        self.assertEqual(seen, {1, 2, 3, 4})

    def test_success_is_scored_against_the_moved_goal(self):
        for goal in ([-0.25, 0.18], [-0.06, -0.15], [-0.10, 0.05]):
            self.env.reset(seed=1, options=dict(goal_xy=torch.tensor([goal])))
            u = self.u
            place_tee(u, goal, u.goal_z_rot)
            # PushT-v1's pixel-grid overlap gives 0.9479 for a perfect placement at its own goal
            self.assertGreater(float(u.pseudo_render_intersection()[0]), 0.947)
            self.assertTrue(bool(u.evaluate()["success"][0]))
            # at PushT-v1's old fixed goal, the T must NOT count as success now
            place_tee(u, pv.QUAD_ORIGIN[0:1] + (-0.1,), u.goal_z_rot)
            self.assertFalse(bool(u.evaluate()["success"][0]))
            # a 4 cm offset from the new goal fails
            place_tee(u, [goal[0] + 0.04, goal[1]], u.goal_z_rot)
            self.assertFalse(bool(u.evaluate()["success"][0]))

    def test_matches_pusht_v1_at_its_original_goal(self):
        orig = gym.make("PushT-v1", obs_mode="state", num_envs=1, sim_backend="physx_cpu").unwrapped
        orig.reset(seed=0)
        self.env.reset(seed=0, options=dict(goal_xy=torch.tensor([[-0.156, -0.1]])))
        self.assertTrue(torch.allclose(self.u.world_to_goal_trans[0], orig.world_to_goal_trans, atol=1e-6))
        self.assertTrue(torch.allclose(self.u.goal_tee.pose.raw_pose, orig.goal_tee.pose.raw_pose, atol=1e-6))

    def test_reward_peaks_at_goal(self):
        goal = [-0.2, -0.12]
        obs, _ = self.env.reset(seed=2, options=dict(goal_xy=torch.tensor([goal])))
        u = self.u
        a = torch.zeros(1, 7)
        place_tee(u, [goal[0] + 0.15, goal[1]], u.goal_z_rot + 1.0)
        far = float(u.compute_dense_reward(obs, a, u.evaluate())[0])
        place_tee(u, goal, u.goal_z_rot)
        at = float(u.compute_dense_reward(obs, a, u.evaluate())[0])
        self.assertEqual(at, 3.0)
        self.assertLess(far, 1.5)

    def test_quadrant_restriction(self):
        env = make(goal_quadrants=(4,))
        for s in range(10):
            env.reset(seed=s)
            self.assertEqual(int(env.unwrapped.goal_quadrant[0]), 4)
        env.close()

    def test_episode_runs_100_steps(self):
        self.env.reset(seed=3)
        for t in range(100):
            _, _, term, trunc, _ = self.env.step(torch.zeros(1, 7))
        self.assertTrue(bool(trunc[0]))


if __name__ == "__main__":
    unittest.main()

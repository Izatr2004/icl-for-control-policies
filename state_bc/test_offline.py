"""Offline checks for pusht_state_bc.py (no ManiSkill or GPU needed).

Run: python -m unittest state_bc/test_offline.py -v   (from the repo root)
"""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pusht_state_bc as bc  # noqa: E402

D_S, D_A = 31, 7


def write_fake_replay(directory, num_episodes=20, seed=0):
    """Mimic ManiSkill's replayed state file: traj_i/obs [T+1, d_s], actions [T, d_a]."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(D_S, D_A)) / np.sqrt(D_S)
    h5_path = Path(directory) / "trajectory.state.pd_joint_delta_pos.physx_cuda.h5"
    episodes = []
    with h5py.File(h5_path, "w") as f:
        for i in range(num_episodes):
            T = int(rng.integers(30, 101))
            obs = rng.normal(size=(T + 1, D_S)).astype(np.float32)
            obs[:, 21:24] = [0.1, -0.2, 0.0]  # constant feature, like the fixed goal
            actions = (2.5 * np.tanh(obs[:-1] @ w)).astype(np.float32)  # some outside [-1, 1], like PPO
            g = f.create_group(f"traj_{i}")
            g["obs"], g["actions"] = obs, actions
            g["success"] = np.r_[np.zeros(T - 1, bool), True]
            episodes.append({"episode_id": i, "episode_seed": i, "elapsed_steps": T, "success": True})
    meta = {"env_info": {"env_id": "PushT-v1", "max_episode_steps": 100,
                         "env_kwargs": {"obs_mode": "state", "control_mode": "pd_joint_delta_pos"}},
            "episodes": episodes}
    h5_path.with_suffix(".json").write_text(json.dumps(meta))
    return h5_path


class LinearPolicy:
    def __init__(self, w):
        self.w = torch.as_tensor(w, dtype=torch.float32)

    def act(self, obs):
        return torch.tanh(obs @ self.w)


class FakeVecEnv:
    """Tiny stand-in for ManiSkillVectorEnv: state drifts by the action; success when obs[0] > 1."""

    def __init__(self, n):
        self.n = n
        self.single_observation_space = mock.Mock(shape=(D_S,))
        self.single_action_space = mock.Mock(shape=(D_A,))

    def reset(self, seed):
        g = torch.Generator().manual_seed(seed)
        self.state = torch.randn(self.n, D_S, generator=g)
        return self.state.clone(), {}

    def step(self, action):
        self.state[:, :D_A] += 0.1 * action
        success = self.state[:, 0] > 1.0
        return self.state.clone(), None, None, None, {"success": success}

    def close(self):
        pass


class FakeGoalVecEnv(FakeVecEnv):
    """Adds a base_env with per-episode goal quadrants, like PushT-VariableGoal-v1."""

    def reset(self, seed):
        out = super().reset(seed)
        self.base_env = mock.Mock(goal_quadrant=torch.arange(self.n) % 4 + 1,
                                  tip_start_quadrant=(torch.arange(self.n) + 1) % 4 + 1,
                                  tip_ik_fallback=torch.zeros(self.n, dtype=torch.bool),
                                  spec=["goal_quadrant", "tip_start_quadrant", "tip_ik_fallback"])
        return out


class FakeSharedRawEnv(FakeVecEnv):
    """Raw (unwrapped) env like PushTVarGoal-v1: no base_env; env.unwrapped has goal_quadrants (the sampling
    restriction), goal_quadrant (per episode) and goal_tee, but no goal_xy."""

    def __init__(self, n):
        super().__init__(n)
        self.unwrapped = types.SimpleNamespace(goal_quadrants=(1, 2, 3, 4))
        self.seeds, self.quadrants_at_reset = [], []

    def reset(self, seed):
        out = super().reset(seed)
        u = self.unwrapped
        self.seeds.append(seed)
        self.quadrants_at_reset.append(tuple(u.goal_quadrants))
        allowed = torch.tensor(u.goal_quadrants)
        u.goal_quadrant = allowed[torch.arange(self.n) % len(allowed)]
        p = torch.zeros(self.n, 3)
        p[:, :2] = torch.randn(self.n, 2, generator=torch.Generator().manual_seed(seed))
        u.goal_tee = types.SimpleNamespace(pose=types.SimpleNamespace(p=p))
        return out


def import_variable_goal_env():
    """variable_goal_env's helpers import without ManiSkill; the env class is then None."""
    import importlib

    sys.modules.pop("variable_goal_env", None)
    return importlib.import_module("variable_goal_env")


class VariableGoalTests(unittest.TestCase):
    def setUp(self):
        self.vg = import_variable_goal_env()

    def test_sampled_goals_stay_in_allowed_quadrants_and_box(self):
        torch.manual_seed(0)
        offsets, picked = self.vg.sample_goal_offsets(4000, (1, 2, 3))
        self.assertTrue(bool((offsets.abs() <= self.vg.GOAL_HALF_WIDTH).all()))
        np.testing.assert_array_equal(self.vg.quadrant_of(offsets).numpy(), picked.numpy())
        self.assertEqual(set(picked.tolist()), {1, 2, 3})
        counts = np.bincount(picked.numpy(), minlength=5)[1:4]
        self.assertTrue(np.all(np.abs(counts / 4000 - 1 / 3) < 0.03))  # balanced across quadrants

    def test_quadrant_convention(self):
        pts = torch.tensor([[0.05, 0.05], [-0.05, 0.05], [-0.05, -0.05], [0.05, -0.05]])
        self.assertEqual(self.vg.quadrant_of(pts).tolist(), [1, 2, 3, 4])

    def test_success_transform_maps_each_goal_to_origin(self):
        angle = 5 / 3 * np.pi
        base = torch.tensor([[np.cos(angle), -np.sin(angle), -0.156],
                             [np.sin(angle), np.cos(angle), -0.1], [0, 0, 1]], dtype=torch.float32)
        goals = torch.tensor([[-0.156, -0.1], [-0.08, -0.03], [-0.25, -0.19]])
        w2g = self.vg.world_to_goal_transforms(base, goals)
        homo = torch.cat([goals, torch.ones(3, 1)], 1).unsqueeze(-1)
        torch.testing.assert_close((w2g @ homo).squeeze(-1), torch.tensor([[0.0, 0.0, 1.0]] * 3),
                                   atol=1e-5, rtol=0)
        # the original goal reproduces PushT-v1's fixed transform
        torch.testing.assert_close(w2g[0], torch.linalg.inv(base), atol=1e-5, rtol=0)
        # rotation is unchanged: a point 1 cm along the goal's local x axis
        local = torch.tensor([0.01, 0.0, 1.0])
        world = (base @ local)[:2] - base[:2, 2] + goals[1]
        torch.testing.assert_close(w2g[1] @ torch.cat([world, torch.ones(1)]), local, atol=1e-5, rtol=0)

    def test_rejects_bad_quadrants(self):
        with self.assertRaises(ValueError):
            self.vg._check_quadrants((0, 5), "goal_quadrants")
        self.assertEqual(self.vg._check_quadrants([4, 1, 1], "x"), (1, 4))

    def test_grid_cells_contain_their_points_and_keep_quadrants(self):
        vg = self.vg
        grid = vg.tip_grid_offsets()
        self.assertEqual(grid.shape, (vg.TIP_GRID ** 2, 2))
        np.testing.assert_array_equal(vg.grid_index(grid).numpy(), np.arange(vg.TIP_GRID ** 2))
        torch.manual_seed(4)
        pts, quads = vg.sample_goal_offsets(5000, (1, 2, 3, 4), vg.TIP_HALF_WIDTH)
        cells = grid[vg.grid_index(pts)]
        half_cell = vg.TIP_HALF_WIDTH / vg.TIP_GRID
        self.assertLessEqual((cells - pts).abs().max().item(), half_cell + 1e-6)  # tip within half a cell
        np.testing.assert_array_equal(vg.quadrant_of(cells).numpy(), quads.numpy())

    def test_tip_cells_clear_the_T_respect_quadrants_and_avoid_unreachable(self):
        vg = self.vg
        torch.manual_seed(1)
        grid = vg.tip_grid_offsets()
        valid = torch.ones(len(grid), dtype=torch.bool)
        valid[grid[:, 0] > 0.10] = False  # pretend a strip is unreachable
        tee = torch.rand(2000, 2) * torch.tensor([0.2, 0.3]) + torch.tensor([-0.1, -0.1])  # PushT-v1 spawn box
        cell, dist = vg.choose_tip_cells(tee, (1, 2, 4), valid, grid)
        quads = vg.quadrant_of(grid[cell])
        self.assertNotIn(3, set(quads.tolist()))
        self.assertTrue(bool(valid[cell].all()))
        np.testing.assert_allclose(dist.numpy(), torch.linalg.norm(grid[cell] - tee, dim=1).numpy(), atol=1e-6)
        self.assertGreater((dist >= vg.TIP_CLEARANCE).float().mean().item(), 0.95)

    def test_starts_independent_of_goal(self):
        """Tip start sampling never looks at the goal, so start quadrant carries no goal information."""
        torch.manual_seed(2)
        goal_off, goal_q = self.vg.sample_goal_offsets(8000, (1, 2, 3, 4))
        tee = torch.zeros(8000, 2) + torch.tensor([0.0, 0.18])  # T out of the way
        grid = self.vg.tip_grid_offsets()
        cell, _ = self.vg.choose_tip_cells(tee, (1, 2, 3, 4), torch.ones(len(grid), dtype=torch.bool), grid)
        tip_q = self.vg.quadrant_of(grid[cell])
        table = np.zeros((4, 4))
        np.add.at(table, (goal_q.numpy() - 1, tip_q.numpy() - 1), 1)
        expected = np.outer(table.sum(1), table.sum(0)) / table.sum()
        self.assertLess(np.abs(table - expected).max() / expected.mean(), 0.15)

    def test_dls_ik_reaches_position_and_orientation(self):
        torch.manual_seed(3)
        A = torch.randn(3, 7) * 0.3

        def rz(a):
            c, s = a.cos(), a.sin()
            z, o = torch.zeros_like(a), torch.ones_like(a)
            return torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1), torch.stack([z, z, o], -1)], -2)

        def fk(q):
            return q @ A.T, rz(q[:, 6])

        def jac(q):
            J = torch.zeros(q.shape[0], 6, 7)
            J[:, :3] = A
            J[:, 5, 6] = 1.0
            return J

        q_true = torch.rand(16, 7) - 0.5
        target_pos, target_rot = fk(q_true)
        lower, upper = -torch.ones(7) * 2, torch.ones(7) * 2
        q, err = self.vg.dls_ik(fk, jac, torch.zeros(16, 7), target_pos, target_rot, lower, upper)
        self.assertLess(err.max().item(), 1e-3)
        torch.testing.assert_close(fk(q)[1], target_rot, atol=1e-3, rtol=0)
        self.assertTrue(bool(((q >= lower) & (q <= upper)).all()))

    def test_initial_guess_turns_base_toward_target(self):
        rest_q = torch.zeros(2, 7)
        rest_pos = torch.tensor([0.3, 0.3, 0.02])
        target = torch.tensor([[0.3, -0.3, 0.02], [0.0, 0.4, 0.02]])
        q = self.vg.initial_guess(rest_q, rest_pos, target, -torch.ones(7) * 3, torch.ones(7) * 3)
        torch.testing.assert_close(q[:, 0], torch.tensor([-np.pi / 2, np.pi / 4]), atol=1e-5, rtol=0)
        self.assertTrue(bool((q[:, 1:] == 0).all()))

    def test_env_class_is_none_without_maniskill(self):
        self.assertIsNone(self.vg.PushTVariableGoalEnv)


class OfflineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.h5 = write_fake_replay(self.dir)

    def tearDown(self):
        self.tmp.cleanup()

    def prepared(self, val_fraction="0.25"):
        out = self.dir / "data"
        bc.main(["prepare", "--h5", str(self.h5), "--out-dir", str(out), "--val-fraction", val_fraction])
        return out

    def test_load_drops_terminal_obs_and_clips_labels(self):
        episodes, _ = bc.load_replayed_demos(self.h5)
        self.assertEqual(len(episodes), 20)
        for ep in episodes:
            self.assertEqual(ep["obs"].shape[0], ep["actions"].shape[0])
            self.assertLessEqual(np.abs(ep["actions"]).max(), 1.0)
        self.assertGreater(max(np.abs(ep["raw_actions"]).max() for ep in episodes), 1.0)

    def test_split_is_by_episode_and_disjoint(self):
        tr, va = bc.split_episodes(999, 0.1, seed=0)
        self.assertEqual(len(tr) + len(va), 999)
        self.assertEqual(len(va), 100)
        self.assertFalse(set(tr) & set(va))
        np.testing.assert_array_equal(va, bc.split_episodes(999, 0.1, seed=0)[1])

    def test_prepare_writes_disjoint_train_and_val_files(self):
        out = self.prepared()
        tr, va = np.load(out / "demos_train.npz"), np.load(out / "demos_val.npz")
        self.assertFalse(set(tr["episode_ids"]) & set(va["episode_ids"]))
        self.assertEqual(len(np.unique(tr["episode_ids"])) + len(np.unique(va["episode_ids"])), 20)
        summary = json.loads((out / "demos_summary.json").read_text())
        self.assertGreater(summary["fraction_raw_actions_outside_bounds"], 0)

    def test_stats_center_constant_features_only(self):
        obs = np.random.default_rng(0).normal(size=(500, D_S)).astype(np.float32)
        obs[:, 21:24] = 0.5
        mean, std, scale = bc.compute_state_stats(obs)
        np.testing.assert_array_equal(scale[21:24], 1.0)
        np.testing.assert_allclose(mean, obs.mean(0), rtol=1e-5, atol=1e-6)

    def test_load_transitions_keeps_episode_ids_unique_across_files(self):
        a, b = self.dir / "a.npz", self.dir / "b.npz"
        bc.save_transitions(a, np.zeros((4, D_S)), np.zeros((4, D_A)), [0, 0, 1, 1])
        bc.save_transitions(b, np.ones((6, D_S)), np.ones((6, D_A)), [0, 0, 0, 5, 5, 5])
        obs, act, ids = bc.load_transitions([a, b])
        self.assertEqual(obs.shape, (10, D_S))
        self.assertEqual(len(np.unique(ids)), 4)

    def test_rejects_wrong_control_mode(self):
        meta_path = self.h5.with_suffix(".json")
        meta = json.loads(meta_path.read_text())
        meta["env_info"]["env_kwargs"]["control_mode"] = "pd_ee_delta_pos"
        meta_path.write_text(json.dumps(meta))
        with self.assertRaises(ValueError):
            bc.load_replayed_demos(self.h5)

    def test_train_on_multiple_files_reduces_val_mse_and_roundtrips(self):
        out = self.prepared()
        val = np.load(out / "demos_val.npz")
        extra = self.dir / "extra.npz"
        tr = np.load(out / "demos_train.npz")
        bc.save_transitions(extra, tr["obs"][:200], tr["actions"][:200], tr["episode_ids"][:200])
        untrained = bc.BCPolicy(D_S, D_A, np.zeros(D_S), np.ones(D_S))
        before = bc.action_mse(untrained, val["obs"], val["actions"], "cpu")
        run = self.dir / "run"
        bc.main(["train", "--train-data", str(out / "demos_train.npz"), str(extra),
                 "--val-data", str(out / "demos_val.npz"), "--out", str(run), "--iters", "2000",
                 "--log-every", "500", "--eval-every", "0", "--lr", "1e-3", "--device", "cpu"])
        info = json.loads((run / "data_info.json").read_text())
        self.assertEqual(info["train_transitions"], len(tr["obs"]) + 200)
        policy = bc.load_policy(run / "best.pt")
        after = bc.action_mse(policy, val["obs"], val["actions"], "cpu")
        self.assertLess(after, 0.7 * before)
        a = policy.act(torch.as_tensor(val["obs"][:5]))
        self.assertEqual(tuple(a.shape), (5, D_A))
        self.assertTrue(bool((a.abs() <= 1).all()))

    def test_rollout_labels_are_clean_and_episode_major(self):
        rng = np.random.default_rng(1)
        expert = LinearPolicy(rng.normal(size=(D_S, D_A)))
        driver = LinearPolicy(rng.normal(size=(D_S, D_A)))
        env = FakeVecEnv(n=8)
        res, obs, labels = bc.rollout(env, 12, driver, seed=3, labeler=expert, noise_std=0.5,
                                      generator=torch.Generator().manual_seed(0))
        self.assertEqual(obs.shape, (8, 12, D_S))
        self.assertEqual(labels.shape, (8, 12, D_A))
        # labels are the expert's actions on the visited states, unaffected by noise
        np.testing.assert_allclose(labels, expert.act(torch.as_tensor(obs)).numpy(), atol=1e-6)
        # episode-major: step t+1 of env i follows step t of env i
        self.assertFalse(np.allclose(obs[0, 1], obs[1, 0]))
        self.assertEqual(res["episodes"], 8)
        self.assertTrue(0 <= res["success_at_end"] <= res["success_once"] <= 1)

    def test_rollout_without_labeler_returns_metrics_only(self):
        env = FakeVecEnv(n=4)
        res = bc.rollout(env, 5, LinearPolicy(np.zeros((D_S, D_A))), seed=0)
        self.assertIsInstance(res, dict)

    def test_collect_writes_expert_labeled_transitions(self):
        rng = np.random.default_rng(2)
        expert = LinearPolicy(rng.normal(size=(D_S, D_A)))
        out = self.dir / "collected.npz"
        with mock.patch.object(bc, "make_vector_env", lambda n, **k: (FakeVecEnv(n), 10)), \
             mock.patch.object(bc, "load_expert", lambda *a, **k: expert):
            bc.main(["collect", "--expert", "x.pt", "--episodes", "10", "--num-envs", "4",
                     "--noise-std", "0", "0.3", "--seed-base", "100", "--out", str(out), "--device", "cpu"])
        d = np.load(out)
        self.assertEqual(d["obs"].shape, (10 * 10, D_S))
        self.assertEqual(len(np.unique(d["episode_ids"])), 10)
        np.testing.assert_allclose(d["actions"], expert.act(torch.as_tensor(d["obs"])).numpy(), atol=1e-6)
        meta = json.loads(out.with_suffix(".json").read_text())
        self.assertEqual([b["noise_std"] for b in meta["batches"]], [0.0, 0.3, 0.0])

    def test_rollout_reports_success_by_goal_quadrant(self):
        env = FakeGoalVecEnv(n=8)
        res = bc.rollout(env, 5, LinearPolicy(np.ones((D_S, D_A))), seed=1)
        self.assertEqual(sorted(res["by_goal_quadrant"]), ["Q1", "Q2", "Q3", "Q4"])
        self.assertEqual(sum(v["episodes"] for v in res["by_goal_quadrant"].values()), 8)
        self.assertNotIn("goal_quadrant_per_episode", res)  # only recorded when collecting

    def test_collect_records_goal_quadrant_and_noise_per_episode(self):
        expert = LinearPolicy(np.random.default_rng(3).normal(size=(D_S, D_A)))
        out = self.dir / "goal.npz"
        with mock.patch.object(bc, "make_vector_env", lambda n, **k: (FakeGoalVecEnv(n), 6)), \
             mock.patch.object(bc, "load_expert", lambda *a, **k: expert):
            bc.main(["collect", "--expert", "x.pt", "--episodes", "6", "--num-envs", "4",
                     "--noise-std", "0", "0.2", "--seed-base", "7", "--out", str(out), "--device", "cpu"])
        meta = json.loads(out.with_suffix(".json").read_text())
        self.assertEqual(meta["goal_quadrant_per_episode"], [1, 2, 3, 4, 1, 2])
        self.assertEqual(meta["tip_start_quadrant_per_episode"], [2, 3, 4, 1, 2, 3])
        self.assertEqual(meta["batches"][0]["tip_ik_fallback_rate"], 0.0)
        self.assertEqual(meta["noise_std_per_episode"], [0.0] * 4 + [0.2] * 2)
        self.assertEqual(meta["env_id"], bc.VARIABLE_GOAL_ENV_ID)

    def test_goal_quadrants_only_for_variable_goal_env(self):
        bc.configure_env(bc.VARIABLE_GOAL_ENV_ID, [1, 2, 3])
        self.assertEqual(bc.ENV_KWARGS, {"goal_quadrants": (1, 2, 3)})
        with self.assertRaises(ValueError):
            bc.configure_env("PushT-v1", [1, 2, 3])
        bc.configure_env(bc.VARIABLE_GOAL_ENV_ID, None, [1, 2, 3])
        self.assertEqual(bc.ENV_KWARGS, {"start_quadrants": (1, 2, 3)})
        with self.assertRaises(ValueError):
            bc.configure_env("PushT-v1", None, [4])
        bc.configure_env(bc.VARIABLE_GOAL_ENV_ID)
        self.assertEqual(bc.ENV_KWARGS, {})

    def test_shared_env_takes_goal_quadrants_but_not_start_quadrants(self):
        bc.configure_env(bc.SHARED_ENV_ID, [1, 2, 3])
        self.assertEqual(bc.ENV_KWARGS, {"goal_quadrants": (1, 2, 3)})
        with self.assertRaises(ValueError):
            bc.configure_env(bc.SHARED_ENV_ID, None, [1, 2, 3])
        bc.configure_env(bc.VARIABLE_GOAL_ENV_ID)

    def test_rollout_on_raw_env_records_goal_xy_from_goal_tee(self):
        env = FakeSharedRawEnv(6)
        res, obs, _ = bc.rollout(env, 4, LinearPolicy(np.ones((D_S, D_A))), seed=2,
                                 labeler=LinearPolicy(np.ones((D_S, D_A))))
        self.assertEqual(obs.shape, (6, 4, D_S))
        self.assertEqual(sorted(res["by_goal_quadrant"]), ["Q1", "Q2", "Q3", "Q4"])
        np.testing.assert_allclose(res["goal_xy_per_episode"], env.unwrapped.goal_tee.pose.p[:, :2].numpy())
        self.assertEqual(res["goal_quadrant_per_episode"], env.unwrapped.goal_quadrant.tolist())

    def test_eval_per_quadrant_uses_shared_seeds_restricts_goals_and_restores(self):
        env = FakeSharedRawEnv(5)
        res = bc.eval_per_quadrant(env, 3, LinearPolicy(np.ones((D_S, D_A))), base_seed=1001, bar=0.0)
        self.assertEqual(env.seeds, [2001, 3001, 4001, 5001])
        self.assertEqual(env.quadrants_at_reset, [(1,), (2,), (3,), (4,)])
        self.assertEqual(env.unwrapped.goal_quadrants, (1, 2, 3, 4))
        for q in range(1, 5):
            self.assertEqual(res[f"Q{q}"]["episodes"], 5)
            self.assertNotIn("by_goal_quadrant", res[f"Q{q}"])
        self.assertEqual(res["worst_success_once"], min(res[f"Q{q}"]["success_once"] for q in range(1, 5)))
        self.assertTrue(res["passes_bar"])
        self.assertFalse(bc.eval_per_quadrant(env, 3, LinearPolicy(np.ones((D_S, D_A))), 1001, bar=1.01)["passes_bar"])

    def test_released_ppo_loader_keeps_only_actor(self):
        ref = bc.ReleasedPPOActor(D_S, D_A)
        weights = {**ref.state_dict(), "critic.0.weight": torch.zeros(1), "actor_logstd": torch.zeros(1, D_A)}
        path = self.dir / "ppo.pt"
        torch.save(weights, path)
        actor = bc.load_released_ppo(path, D_S, D_A)
        x = torch.randn(3, D_S)
        torch.testing.assert_close(actor.act(x), ref.act(x))

    def test_flatten_with_names_matches_maniskill_order(self):
        tree = {"agent": {"qpos": torch.zeros(1, 7), "qvel": torch.zeros(1, 7)},
                "extra": {"tcp_pose": torch.zeros(1, 7), "goal_pos": torch.zeros(1, 3), "obj_pose": torch.zeros(1, 7)}}
        fields = bc.flatten_with_names(tree)
        self.assertEqual([n for n, _ in fields],
                         ["agent/qpos", "agent/qvel", "extra/tcp_pose", "extra/goal_pos", "extra/obj_pose"])
        self.assertEqual(sum(w for _, w in fields), 31)


if __name__ == "__main__":
    unittest.main()

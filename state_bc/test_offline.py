"""Offline checks for pusht_state_bc.py (no ManiSkill or GPU needed).

Run: python -m unittest state_bc/test_offline.py -v   (from the repo root)
"""

import json
import sys
import tempfile
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

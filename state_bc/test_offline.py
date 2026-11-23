"""Offline checks for pusht_state_bc.py (no ManiSkill or GPU needed).

Run: python -m unittest state_bc/test_offline.py -v   (from the repo root)
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

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
            obs[:, 28:31] = [0.1, -0.2, 0.0]  # constant feature, like the fixed goal
            actions = np.tanh(obs[:-1] @ w).astype(np.float32)
            g = f.create_group(f"traj_{i}")
            g["obs"], g["actions"] = obs, actions
            g["success"] = np.r_[np.zeros(T - 1, bool), True]
            episodes.append({"episode_id": i, "episode_seed": i, "elapsed_steps": T, "success": True})
    meta = {"env_info": {"env_id": "PushT-v1", "max_episode_steps": 100,
                         "env_kwargs": {"obs_mode": "state", "control_mode": "pd_joint_delta_pos"}},
            "episodes": episodes}
    h5_path.with_suffix(".json").write_text(json.dumps(meta))
    return h5_path


class OfflineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.h5 = write_fake_replay(self.dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_drops_terminal_obs(self):
        episodes, _ = bc.load_replayed_demos(self.h5)
        self.assertEqual(len(episodes), 20)
        for ep in episodes:
            self.assertEqual(ep["obs"].shape[0], ep["actions"].shape[0])

    def test_split_is_by_episode_and_disjoint(self):
        tr, va = bc.split_episodes(999, 0.1, seed=0)
        self.assertEqual(len(tr) + len(va), 999)
        self.assertEqual(len(va), 100)
        self.assertFalse(set(tr) & set(va))
        np.testing.assert_array_equal(va, bc.split_episodes(999, 0.1, seed=0)[1])

    def test_prepare_and_stats_use_train_split_only(self):
        out = self.dir / "data.npz"
        bc.main(["prepare", "--h5", str(self.h5), "--out", str(out), "--val-fraction", "0.25"])
        data = dict(np.load(out))
        self.assertFalse(set(data["train_episode_ids"]) & set(data["val_episode_ids"]))
        np.testing.assert_allclose(data["state_mean"], data["train_obs"].mean(0), rtol=1e-5, atol=1e-6)
        # constant goal features are centered only, never divided by ~0
        np.testing.assert_array_equal(data["state_scale"][28:31], 1.0)
        summary = json.loads(out.with_suffix(".summary.json").read_text())
        self.assertEqual(summary["features_only_centered"], [28, 29, 30])

    def test_rejects_wrong_control_mode(self):
        meta_path = self.h5.with_suffix(".json")
        meta = json.loads(meta_path.read_text())
        meta["env_info"]["env_kwargs"]["control_mode"] = "pd_ee_delta_pos"
        meta_path.write_text(json.dumps(meta))
        with self.assertRaises(ValueError):
            bc.load_replayed_demos(self.h5)

    def test_train_reduces_val_mse_and_checkpoint_roundtrips(self):
        data_path = self.dir / "data.npz"
        bc.main(["prepare", "--h5", str(self.h5), "--out", str(data_path)])
        data = dict(np.load(data_path))
        untrained = bc.BCPolicy(D_S, D_A, data["state_mean"], data["state_scale"])
        before = bc.action_mse(untrained, data["val_obs"], data["val_actions"], "cpu")
        run = self.dir / "run"
        bc.main(["train", "--dataset", str(data_path), "--out", str(run), "--iters", "2000",
                 "--log-every", "500", "--eval-every", "0", "--lr", "1e-3", "--device", "cpu"])
        policy = bc.load_policy(run / "best.pt")
        after = bc.action_mse(policy, data["val_obs"], data["val_actions"], "cpu")
        self.assertLess(after, 0.5 * before)
        # checkpoint carries its own normalization: raw observations in, bounded actions out
        a = policy.act(torch.as_tensor(data["val_obs"][:5]))
        self.assertEqual(tuple(a.shape), (5, D_A))
        self.assertTrue(bool((a.abs() <= 1).all()))
        np.testing.assert_allclose(policy.state_mean.numpy(), data["state_mean"])
        self.assertTrue((run / "train_log.csv").read_text().startswith("iter,train_mse"))

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

"""Offline tests for pusht_icl.py (no ManiSkill or GPU; needs torch, numpy, transformers).

Run: python -m unittest state_bc/test_icl_offline.py -v
"""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pusht_icl as icl  # noqa: E402
import pusht_state_bc as bc  # noqa: E402

D_FULL, D_A = 31, 7


def coded_split(n_teachers=2, G=6, E=5, T=12, success=None):
    """Synthetic demos whose states encode where they came from: field 0 = teacher, 1 = goal,
    2 = episode, 3 = time step (all before the goal fields, so they survive hide_goal)."""
    obs = np.zeros((n_teachers, G, E, T, D_FULL), np.float32)
    ti, gi, ei, t = np.meshgrid(*(np.arange(s) for s in (n_teachers, G, E, T)), indexing="ij")
    obs[..., 0], obs[..., 1], obs[..., 2], obs[..., 3] = ti, gi, ei, t
    obs[..., 21:24] = 99.0  # goal fields: must never reach the transformer
    actions = np.zeros((n_teachers, G, E, T, D_A), np.float32)
    actions[..., 0] = gi / G
    first = np.ones((n_teachers, G, E), np.int64) * 5 if success is None else success
    return obs, actions, first


def write_split(directory, split, obs, actions, first, teachers=icl.TEACHERS):
    G = obs.shape[1]
    goal_xy = np.stack([np.linspace(-0.2, -0.1, G), np.linspace(-0.1, 0.1, G)], 1).astype(np.float32)
    for i, t in enumerate(teachers):
        np.savez(Path(directory) / f"{split}_{t}.npz", obs=obs[i], actions=actions[i], first_success_step=first[i],
                 success_at_end=first[i] > 0, goal_xy=goal_xy, goal_quadrant=np.ones(G, np.int64))


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_hide_goal_drops_goal_fields_numpy_and_torch(self):
        x = np.arange(2 * 31, dtype=np.float32).reshape(2, 31)
        h = icl.hide_goal(x)
        self.assertEqual(h.shape, (2, 28))
        self.assertFalse(np.isin(x[:, 21:24], h).any())
        torch.testing.assert_close(icl.hide_goal(torch.as_tensor(x)), torch.as_tensor(h))

    def test_snippet_steps_keep_last_step_and_stride(self):
        self.assertEqual(icl.snippet_steps(100, 4).tolist()[-1], 99)
        self.assertEqual(len(icl.snippet_steps(100, 4)), 25)
        self.assertEqual(icl.snippet_steps(10, 3).tolist(), [0, 3, 6, 9])
        self.assertEqual(icl.snippet_steps(5, 1).tolist(), [0, 1, 2, 3, 4])

    def test_goal_sets_reproducible_and_leave_global_rng_alone(self):
        fn = lambda n, q: torch.rand(n, 2) + torch.tensor(q, dtype=torch.float32)[:1]  # noqa: E731
        torch.manual_seed(5)
        expected_next = torch.rand(1)
        torch.manual_seed(5)
        a = icl.sample_goal_set(fn, 4, (1, 2, 3), 11)
        self.assertEqual(torch.rand(1).item(), expected_next.item())
        torch.testing.assert_close(a, icl.sample_goal_set(fn, 4, (1, 2, 3), 11))
        self.assertFalse(torch.allclose(a, icl.sample_goal_set(fn, 4, (1, 2, 3), 12)))

    def test_prompts_come_from_one_teacher_and_goal_with_separate_query_episode(self):
        obs, actions, first = coded_split()
        write_split(self.dir, "train", obs, actions, first)
        data = icl.PromptData.load(self.dir, "train")
        self.assertEqual(data.obs.shape[-1], 28)
        self.assertFalse((data.obs == 99.0).any())
        k, stride = 3, 4
        s, a, qs, qa = data.sample(64, k, stride, torch.Generator().manual_seed(0))
        P = len(icl.snippet_steps(data.T, stride))
        self.assertEqual(tuple(s.shape), (64, k * P, 28))
        self.assertEqual(tuple(a.shape), (64, k * P, D_A))
        for b in range(64):
            self.assertTrue(bool((s[b, :, 0] == qs[b, 0]).all()))     # same teacher
            self.assertTrue(bool((s[b, :, 1] == qs[b, 1]).all()))     # same goal
            support_eps = set(s[b, :, 2].tolist())
            self.assertEqual(len(support_eps), k)
            self.assertNotIn(qs[b, 2].item(), support_eps)            # query from another episode
            self.assertEqual(s[b, P - 1, 3].item(), data.T - 1)       # each snippet ends at the last step
            torch.testing.assert_close(qa[b, 0], qs[b, 1] / data.G)   # label = teacher action at the query state

    def test_wrong_goal_support_keeps_the_same_queries(self):
        obs, actions, first = coded_split()
        write_split(self.dir, "val", obs, actions, first)
        data = icl.PromptData.load(self.dir, "val")
        right = data.sample(32, 2, 3, torch.Generator().manual_seed(1))
        wrong = data.sample(32, 2, 3, torch.Generator().manual_seed(1), wrong_goal=True)
        torch.testing.assert_close(right[2], wrong[2])
        torch.testing.assert_close(right[3], wrong[3])
        self.assertTrue(bool((wrong[0][:, 0, 1] != wrong[2][:, 1]).all()))   # support goal differs from query goal
        self.assertTrue(bool((wrong[0][:, 0, 0] == wrong[2][:, 0]).all()))   # same teacher

    def test_successful_episodes_are_used_for_support_first(self):
        first = np.full((2, 6, 5), -1, np.int64)
        first[:, :, 3] = 7  # only episode 3 reaches the goal
        obs, actions, _ = coded_split()
        write_split(self.dir, "train", obs, actions, first)
        data = icl.PromptData.load(self.dir, "train")
        s, _, qs, _ = data.sample(40, 1, 4, torch.Generator().manual_seed(2))
        self.assertTrue(bool((s[:, :, 2] == 3).all()))
        self.assertTrue(bool((qs[:, 2] != 3).all()))

    def test_k_must_leave_a_query_episode(self):
        obs, actions, first = coded_split(E=3)
        write_split(self.dir, "train", obs, actions, first)
        with self.assertRaises(ValueError):
            icl.PromptData.load(self.dir, "train").sample(4, 3, 2, torch.Generator().manual_seed(0))


def tiny_model(d_s=28, **kw):
    return icl.ICLTransformer(d_s, D_A, torch.zeros(d_s), torch.ones(d_s), n_layer=2, n_embd=32, n_head=4,
                              n_positions=kw.pop("n_positions", 128), **kw)


class ModelTests(unittest.TestCase):
    def test_shapes_with_and_without_context(self):
        m = tiny_model()
        q = torch.randn(3, 28)
        self.assertEqual(tuple(m(torch.zeros(3, 0, 28), torch.zeros(3, 0, D_A), q).shape), (3, D_A))
        self.assertEqual(tuple(m(torch.randn(3, 5, 28), torch.randn(3, 5, D_A), q).shape), (3, D_A))

    def test_context_changes_the_prediction(self):
        torch.manual_seed(0)
        m = tiny_model()
        q = torch.randn(2, 28)
        s = torch.randn(2, 4, 28)
        p1 = m(s, torch.zeros(2, 4, D_A), q)
        p2 = m(s, torch.ones(2, 4, D_A), q)
        self.assertFalse(torch.allclose(p1, p2))

    def test_rejects_prompts_longer_than_the_position_table(self):
        m = tiny_model(n_positions=9)
        with self.assertRaises(ValueError):
            m(torch.zeros(1, 5, 28), torch.zeros(1, 5, D_A), torch.zeros(1, 28))

    def test_save_load_roundtrip_and_actor_hides_goal_and_clamps(self):
        torch.manual_seed(1)
        m = icl.ICLTransformer(28, D_A, torch.randn(28), torch.rand(28) + 0.5, n_layer=2, n_embd=32, n_head=4)
        with tempfile.TemporaryDirectory() as d:
            icl.save_model(m, Path(d) / "m.pt")
            m2 = icl.load_model(Path(d) / "m.pt")
        s, a, q = torch.randn(2, 3, 28), torch.randn(2, 3, D_A), torch.randn(2, 28)
        torch.testing.assert_close(m.eval()(s, a, q), m2(s, a, q))
        full = torch.randn(2, 31)
        actor = icl.ICLActor(m2, s, a)
        out = actor.act(full)
        torch.testing.assert_close(out, m2(s, a, icl.hide_goal(full)).clamp(-1, 1))
        full2 = full.clone()
        full2[:, 21:24] += 5.0  # changing only the goal fields must not change the action
        torch.testing.assert_close(actor.act(full2), out)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        # The action is a per-goal constant that the (hidden-goal) query state does not reveal:
        # only the support can tell the model what to do.
        rng = np.random.default_rng(0)
        G, E, T = 16, 4, 6
        obs = rng.normal(size=(2, G, E, T, D_FULL)).astype(np.float32)
        target = rng.uniform(-0.8, 0.8, size=(G, D_A)).astype(np.float32)
        actions = np.broadcast_to(target[None, :, None, None], (2, G, E, T, D_A)).copy()
        first = np.ones((2, G, E), np.int64)
        write_split(self.dir, "train", obs, actions, first)
        write_split(self.dir, "val", obs, actions, first)

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, iters, out="run"):
        return ["train", "--data-dir", str(self.dir), "--out", str(self.dir / out), "--ks", "0", "1", "2",
                "--stride", "3", "--iters", str(iters), "--batch-size", "32", "--lr", "1e-3", "--warmup", "10",
                "--n-layer", "2", "--n-embd", "32", "--n-head", "4", "--log-every", "50", "--eval-every", "100",
                "--device", "cpu"]

    def test_training_learns_to_use_the_context_and_resumes(self):
        icl.main(self.args(300))
        run = self.dir / "run"
        for name in ("best.pt", "last.pt", "final.pt", "config.json", "train_log.jsonl"):
            self.assertTrue((run / name).exists(), name)
        val = json.loads((run / "train_log.jsonl").read_text().splitlines()[-1])["val"]
        # With a correct-goal support the model can copy the action; wrong-goal support or none cannot.
        self.assertLess(val["2"]["correct"], 0.5 * val["2"]["wrong"])
        self.assertLess(val["2"]["correct"], 0.5 * val["0"]["none"])
        # Resuming continues from the saved iteration instead of starting over.
        out = []
        with mock.patch("builtins.print", lambda *a, **k: out.append(" ".join(map(str, a)))):
            icl.main(self.args(400))
        self.assertTrue(any("resumed from" in line and "iteration 300" in line for line in out), out[:3])
        self.assertEqual(torch.load(run / "last.pt", weights_only=False)["iter"], 400)


class FakeGoalEnv:
    """Raw-env stand-in for PushTVarGoal-v1 with reset(options=dict(goal_xy=...)): the goal is copied
    into the state's goal fields, so tests can check how episodes are grouped by goal."""

    def __init__(self, n):
        self.n = n
        self.single_observation_space = mock.Mock(shape=(D_FULL,))
        self.single_action_space = mock.Mock(shape=(D_A,))
        self.unwrapped = types.SimpleNamespace()

    def reset(self, seed, options=None):
        goal = options["goal_xy"].float()
        self.state = torch.zeros(self.n, D_FULL)
        self.state[:, 21:23] = goal
        self.state[:, 0] = torch.arange(self.n) + 1000 * seed
        p = torch.zeros(self.n, 3)
        p[:, :2] = goal
        self.unwrapped.goal_quadrant = torch.ones(self.n, dtype=torch.long)
        self.unwrapped.goal_tee = types.SimpleNamespace(pose=types.SimpleNamespace(p=p))
        self.t = 0
        return self.state.clone(), {}

    def step(self, action):
        self.t += 1
        self.state[:, 1] = self.t
        return self.state.clone(), None, None, None, {"success": torch.full((self.n,), self.t >= 2)}

    def close(self):
        pass


class CollectTests(unittest.TestCase):
    def test_collect_groups_episodes_by_goal_for_every_teacher(self):
        fake_pv = types.SimpleNamespace(
            sample_goals=lambda n, q: torch.stack([torch.linspace(-0.25, -0.06, n), torch.linspace(-0.2, -0.04, n)], 1),
            quadrant_of=lambda g: torch.full((len(g),), 3, dtype=torch.long))
        policy = types.SimpleNamespace(act=lambda o: torch.tanh(o[:, :D_A]))
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.dict(sys.modules, {"pusht_vargoal": fake_pv}), \
                mock.patch.object(bc, "make_env", lambda n, **k: FakeGoalEnv(n)), \
                mock.patch.object(bc, "episode_length", lambda env: 4), \
                mock.patch.object(bc, "load_expert", lambda *a, **k: policy), \
                mock.patch.object(bc, "load_policy", lambda *a, **k: policy):
            icl.main(["collect", "--ppo-teacher", "p.pt", "--bc-teacher", "b.pt", "--splits", "val",
                      "--goals", "5", "--episodes", "3", "--num-envs", "4", "--out", d, "--device", "cpu"])
            for teacher in icl.TEACHERS:
                with np.load(Path(d) / f"val_{teacher}.npz") as f:
                    z = dict(f)
                self.assertEqual(z["obs"].shape, (5, 3, 4, D_FULL))
                self.assertEqual(z["actions"].shape, (5, 3, 4, D_A))
                # every episode of goal g was run with goal g
                np.testing.assert_allclose(z["obs"][:, :, 0, 21:23],
                                           np.broadcast_to(z["goal_xy"][:, None], (5, 3, 2)), atol=1e-6)
                # labels are the teacher's own actions on the visited states
                np.testing.assert_allclose(z["actions"], np.tanh(z["obs"][..., :D_A]), atol=1e-6)
                self.assertTrue((z["first_success_step"] == 2).all())
                meta = json.loads((Path(d) / f"val_{teacher}.json").read_text())
                self.assertEqual((meta["goals"], meta["episodes_per_goal"], meta["noise_std"]), (5, 3, 0.0))
            data = icl.PromptData.load(d, "val")
            self.assertEqual((data.n_teachers, data.G, data.E, data.T), (2, 5, 3, 4))


if __name__ == "__main__":
    unittest.main()

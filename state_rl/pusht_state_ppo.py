"""Goal-conditioned state PPO teacher for PushTVarGoal-v1 (Jimmy).

Training itself is ManiSkill's unchanged PPO baseline (examples/baselines/ppo/ppo.py at
commit a4a4f92), launched through run_ppo.py so the variable-goal env is registered.
This module holds everything around it:

  preflight  GPU checks of the variable-goal env (partial resets, per-env success check),
             plus observation_contract.json
  eval       per-quadrant closed-loop evaluation of PPO checkpoints, deterministic mean
             action clipped to [-1, 1]; selects a checkpoint on validation seeds and
             reports whether it clears the freeze bar in every quadrant

Metrics follow state_bc: success_once (reached the goal at some step) and success_at_end
(at the goal at step 100); episodes run the full 100 steps with no early reset.
Evaluation seeds: episodes for quadrant q use seed (base_seed + 1000 * q).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pusht_vargoal as pv  # noqa: E402  (registers PushTVarGoal-v1)

ENV_ID = "PushTVarGoal-v1"
CONTROL_MODE = "pd_joint_delta_pos"
OBS_FIELDS = [("agent/qpos", 7), ("agent/qvel", 7), ("extra/tcp_pose", 7),
              ("extra/goal_pos", 3), ("extra/obj_pose", 7)]  # same order as state_bc


def make_env(num_envs, sim_backend="physx_cuda", quadrants=(1, 2, 3, 4), **kw):
    import gymnasium as gym
    return gym.make(ENV_ID, num_envs=num_envs, obs_mode="state", control_mode=CONTROL_MODE,
                    sim_backend=sim_backend, goal_quadrants=quadrants, **kw)


# --------------------------------------------------------------------------------------
# Policy: deterministic mean of ManiSkill's state PPO actor
# --------------------------------------------------------------------------------------

class PPOActor(nn.Module):
    """actor_mean of examples/baselines/ppo/ppo.py (3 x 256 tanh)."""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, action_dim))

    @torch.no_grad()
    def act(self, raw_state):
        return self.actor_mean(raw_state).clamp(-1.0, 1.0)


def load_actor(path, state_dim=31, action_dim=7, device="cpu"):
    weights = torch.load(path, map_location=device, weights_only=False)
    actor = PPOActor(state_dim, action_dim)
    actor.load_state_dict({k: v for k, v in weights.items() if k.startswith("actor_mean.")})
    return actor.to(device).eval()


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------

@torch.no_grad()
def rollout(env, actor, seed, max_steps=100):
    """One batch of episodes (one per env) for max_steps with no resets."""
    obs, _ = env.reset(seed=seed)
    n = obs.shape[0]
    dev = obs.device
    success_once = torch.zeros(n, dtype=torch.bool, device=dev)
    first = torch.full((n,), -1, dtype=torch.long, device=dev)
    success = success_once
    for t in range(max_steps):
        obs, _, _, _, info = env.step(actor.act(obs.float()))
        success = info["success"].bool()
        first = torch.where((first < 0) & success, torch.full_like(first, t + 1), first)
        success_once |= success
    steps = first[first > 0].float()
    return {"episodes": int(n), "seed": int(seed),
            "success_once": success_once.float().mean().item(),
            "success_at_end": success.float().mean().item(),
            "mean_steps_to_first_success": steps.mean().item() if len(steps) else None}


def eval_per_quadrant(env, actor, base_seed, quadrants=(1, 2, 3, 4), max_steps=100):
    u = env.unwrapped
    out = {}
    for q in quadrants:
        u.goal_quadrants = (q,)
        r = rollout(env, actor, base_seed + 1000 * q, max_steps)
        assert bool((u.goal_quadrant == q).all()), "goals left the requested quadrant"
        out[f"Q{q}"] = r
    u.goal_quadrants = (1, 2, 3, 4)
    so = [out[f"Q{q}"]["success_once"] for q in quadrants]
    se = [out[f"Q{q}"]["success_at_end"] for q in quadrants]
    out["worst_success_once"] = min(so)
    out["mean_success_once"] = float(np.mean(so))
    out["mean_success_at_end"] = float(np.mean(se))
    return out


def checkpoint_iteration(p: Path):
    s = p.stem.split("_")[-1]
    return int(s) if s.isdigit() else 10**9  # final_ckpt sorts last


def cmd_eval(args):
    device = torch.device(args.device)
    ckpts = sorted(Path(args.run_dir).glob("*.pt"), key=checkpoint_iteration) if args.run_dir else []
    ckpts = ckpts[-args.last:] if args.last else ckpts
    ckpts += [Path(c) for c in args.ckpt]
    assert ckpts, "no checkpoints found"
    env = make_env(args.episodes, sim_backend=args.sim_backend)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in ckpts:
        actor = load_actor(c, device=device)
        r = eval_per_quadrant(env, actor, args.val_seed)
        rows.append({"checkpoint": str(c), **r})
        print(f"{c.name:>18}  worst {r['worst_success_once']:.0%}  mean {r['mean_success_once']:.0%}  "
              + "  ".join(f"Q{q} {r[f'Q{q}']['success_once']:.0%}/{r[f'Q{q}']['success_at_end']:.0%}"
                          for q in (1, 2, 3, 4)), flush=True)
    (out / "val_results.json").write_text(json.dumps(rows, indent=2))
    # Selection on validation only: best worst-quadrant success_once, then mean, then later checkpoint.
    best = max(rows, key=lambda r: (r["worst_success_once"], r["mean_success_once"],
                                    checkpoint_iteration(Path(r["checkpoint"]))))
    print("\nSelected on validation:", best["checkpoint"])
    result = {"selected": best["checkpoint"], "val": best, "freeze_bar": args.bar,
              "val_seed": args.val_seed, "episodes_per_quadrant": args.episodes}
    if args.test_seed is not None:
        test = eval_per_quadrant(env, load_actor(best["checkpoint"], device=device), args.test_seed)
        result.update(test=test, test_seed=args.test_seed,
                      passes_bar=all(test[f"Q{q}"]["success_once"] >= args.bar for q in (1, 2, 3, 4)))
        print("Test (used once):  " + "  ".join(
            f"Q{q} {test[f'Q{q}']['success_once']:.0%}/{test[f'Q{q}']['success_at_end']:.0%}" for q in (1, 2, 3, 4)))
        print(f"Freeze bar {args.bar:.0%} success_once in every quadrant:",
              "PASS" if result["passes_bar"] else "FAIL")
    (out / "selection.json").write_text(json.dumps(result, indent=2))
    env.close()


# --------------------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------------------

def _place_tee(u, xy, ang):
    from mani_skill.utils.structs.pose import Pose
    b = xy.shape[0]
    p = torch.zeros(b, 3, device=xy.device)
    p[:, :2], p[:, 2] = xy, 0.021
    q = torch.zeros(b, 4, device=xy.device)
    q[:, 0], q[:, 3] = np.cos(ang / 2), np.sin(ang / 2)
    u.tee.set_pose(Pose.create_from_pq(p=p, q=q))


def preflight(num_envs=8, sim_backend="physx_cuda", out=None):
    env = make_env(num_envs, sim_backend=sim_backend)
    u = env.unwrapped
    obs, _ = env.reset(seed=0)
    from mani_skill.utils import gym_utils
    checks = {}

    # 1. observation layout and spaces match state_bc's contract
    raw = u._get_obs_agent()
    extra = u._get_obs_extra(u.get_info())
    parts = [raw["qpos"], raw["qvel"], extra["tcp_pose"], extra["goal_pos"], extra["obj_pose"]]
    checks["obs_layout"] = bool(torch.allclose(torch.cat(parts, -1).float(), obs.float(), atol=1e-5))
    checks["dims"] = tuple(obs.shape[1:]) == (31,) and tuple(u.single_action_space.shape) == (7,)
    checks["max_episode_steps_100"] = gym_utils.find_max_episode_steps_value(env) == 100

    # 2. every env scores success against its own goal
    goals = u.goal_tee.pose.p[:, :2].clone()
    _place_tee(u, goals, u.goal_z_rot)
    checks["success_at_own_goal"] = bool(u.evaluate()["success"].all())
    shifted = goals.roll(1, 0)
    far = (shifted - goals).norm(dim=1) > 0.04
    _place_tee(u, shifted, u.goal_z_rot)
    checks["no_success_at_other_goal"] = bool((~u.evaluate()["success"][far]).all()) if far.any() else True

    # 3. partial reset changes only the reset envs (what PPO's auto-reset does)
    if num_envs >= 2:
        env.reset(seed=1)
        before_goal = u.goal_tee.pose.p[:, :2].clone()
        before_trans = u.world_to_goal_trans.clone()
        idx = torch.arange(0, num_envs, 2, device=u.device)
        keep = torch.arange(1, num_envs, 2, device=u.device)
        env.reset(options=dict(env_idx=idx))
        after_goal = u.goal_tee.pose.p[:, :2]
        checks["partial_reset_keeps_others"] = bool(
            torch.allclose(after_goal[keep], before_goal[keep]) and
            torch.allclose(u.world_to_goal_trans[keep], before_trans[keep]))
        checks["partial_reset_changes_reset_envs"] = bool((after_goal[idx] - before_goal[idx]).norm(dim=1).gt(1e-4).all())
        _place_tee(u, after_goal.clone(), u.goal_z_rot)
        checks["success_after_partial_reset"] = bool(u.evaluate()["success"].all())
        checks["quadrant_labels_after_partial_reset"] = bool((u.goal_quadrant == pv.quadrant_of(after_goal)).all())

    # 4. quadrant restriction
    u.goal_quadrants = (4,)
    env.reset(seed=2)
    checks["quadrant_restriction"] = bool((u.goal_quadrant == 4).all())
    u.goal_quadrants = (1, 2, 3, 4)

    contract = {
        "env_id": ENV_ID, "base_env": "PushT-v1", "obs_mode": "state", "control_mode": CONTROL_MODE,
        "state_dim": 31, "action_dim": 7, "max_episode_steps": 100,
        "control_freq": u.control_freq, "sim_freq": u.sim_freq, "sim_backend": sim_backend,
        "state_fields": [{"name": n, "start": s, "stop": s + k}
                         for (n, k), s in zip(OBS_FIELDS, np.cumsum([0] + [k for _, k in OBS_FIELDS[:-1]]).tolist())],
        "action_low": u.single_action_space.low.tolist(), "action_high": u.single_action_space.high.tolist(),
        "goal_fields_to_hide_from_transformer": ["extra/goal_pos (indices 21:24)"],
        "workspace": {"quad_origin": pv.QUAD_ORIGIN, "goal_half": pv.GOAL_HALF, "axis_gap": pv.AXIS_GAP,
                      "tee_box": pv.TEE_BOX, "tip_box": pv.TIP_BOX, "start_goal_dist": pv.START_GOAL_DIST,
                      "tip_tee_min": pv.TIP_TEE_MIN, "goal_z_rot": float(u.goal_z_rot)},
        "preflight": checks,
    }
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "observation_contract.json").write_text(json.dumps(contract, indent=2))
    env.close()
    for k, v in checks.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    if not all(checks.values()):
        raise SystemExit("Preflight failed")
    return contract


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("preflight")
    a.add_argument("--out", required=True)
    a.add_argument("--num-envs", type=int, default=8)
    a.add_argument("--sim-backend", default="physx_cuda")
    e = sub.add_parser("eval")
    e.add_argument("--run-dir", help="ppo.py run folder with ckpt_*.pt / final_ckpt.pt")
    e.add_argument("--ckpt", nargs="*", default=[])
    e.add_argument("--last", type=int, default=0, help="only the last N checkpoints of --run-dir")
    e.add_argument("--episodes", type=int, default=100, help="per quadrant")
    e.add_argument("--val-seed", type=int, default=1001)
    e.add_argument("--test-seed", type=int, default=None)
    e.add_argument("--bar", type=float, default=0.80)
    e.add_argument("--out", required=True)
    e.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    e.add_argument("--sim-backend", default="physx_cuda")
    args = p.parse_args(argv)
    if args.cmd == "preflight":
        preflight(args.num_envs, args.sim_backend, args.out)
    else:
        cmd_eval(args)


if __name__ == "__main__":
    main()

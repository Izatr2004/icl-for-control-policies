"""Full-state behavioral cloning (BC) teachers for ManiSkill PushT-v1.

Subcommands (run with the ManiSkill training interpreter):

    audit    Inspect the state observation and action spaces; write observation_contract.json.
    prepare  Convert a replayed state-mode demo file (.h5 + .json) into a compact .npz
             with a train/validation split by episode and normalization statistics.
    train    Train one MLP BC teacher, evaluating closed-loop success on validation seeds.
    eval     Evaluate a BC checkpoint, or the released PPO expert, on a seed range.

Pure data/model code is importable without ManiSkill, so it can be unit tested offline.
ManiSkill and gymnasium are imported lazily inside the simulator functions.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ENV_ID = "PushT-v1"
OBS_MODE = "state"
CONTROL_MODE = "pd_joint_delta_pos"
SIM_BACKEND = "physx_cuda"
# Features whose training std is below this are only centered, not scaled. Near-constant
# fields (for example the fixed goal position) would otherwise turn simulator noise
# into very large normalized values.
MIN_SCALED_STD = 1e-3


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------

def load_replayed_demos(h5_path: str | Path):
    """Return a list of episodes, each a dict with obs [T, d_s], actions [T, d_a], and metadata.

    The replayed file stores T+1 observations per episode; the last one has no action
    and is dropped, as in ManiSkill's own BC baseline.
    """
    import h5py

    h5_path = Path(h5_path)
    meta = json.loads(h5_path.with_suffix(".json").read_text())
    env_kwargs = meta["env_info"]["env_kwargs"]
    if env_kwargs.get("control_mode") not in (None, CONTROL_MODE):
        raise ValueError(f"Demo control mode {env_kwargs.get('control_mode')} != {CONTROL_MODE}")
    if env_kwargs.get("obs_mode") != OBS_MODE:
        raise ValueError(f"Demo obs mode {env_kwargs.get('obs_mode')} != {OBS_MODE}; replay with -o state")
    episodes = []
    with h5py.File(h5_path, "r") as f:
        for ep in meta["episodes"]:
            group = f[f"traj_{ep['episode_id']}"]
            obs = np.asarray(group["obs"], dtype=np.float32)
            actions = np.asarray(group["actions"], dtype=np.float32)
            if obs.ndim != 2 or actions.ndim != 2 or obs.shape[0] != actions.shape[0] + 1:
                raise ValueError(f"Unexpected shapes in traj_{ep['episode_id']}: obs {obs.shape}, actions {actions.shape}")
            success = np.asarray(group["success"]) if "success" in group else None
            episodes.append(dict(
                episode_id=int(ep["episode_id"]),
                episode_seed=ep.get("episode_seed"),
                obs=obs[:-1],
                actions=actions,
                success_at_end=bool(success[-1]) if success is not None else bool(ep.get("success", False)),
            ))
    return episodes, meta


def split_episodes(num_episodes: int, val_fraction: float, seed: int):
    """Split episode indices into train/validation sets, whole episodes only."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    order = rng.permutation(num_episodes)
    num_val = max(1, int(round(num_episodes * val_fraction)))
    return np.sort(order[num_val:]), np.sort(order[:num_val])


def compute_state_stats(states: np.ndarray):
    mean = states.mean(axis=0)
    std = states.std(axis=0)
    scale = np.where(std < MIN_SCALED_STD, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32), scale.astype(np.float32)


def build_dataset(episodes, val_fraction: float, split_seed: int):
    train_idx, val_idx = split_episodes(len(episodes), val_fraction, split_seed)

    def stack(indices):
        obs = np.concatenate([episodes[i]["obs"] for i in indices])
        act = np.concatenate([episodes[i]["actions"] for i in indices])
        ep = np.concatenate([np.full(len(episodes[i]["actions"]), episodes[i]["episode_id"]) for i in indices])
        return obs, act, ep

    train_obs, train_act, train_ep = stack(train_idx)
    val_obs, val_act, val_ep = stack(val_idx)
    mean, std, scale = compute_state_stats(train_obs)
    return dict(
        train_obs=train_obs, train_actions=train_act, train_episode_ids=train_ep,
        val_obs=val_obs, val_actions=val_act, val_episode_ids=val_ep,
        state_mean=mean, state_std=std, state_scale=scale,
        train_episodes=np.array([episodes[i]["episode_id"] for i in train_idx]),
        val_episodes=np.array([episodes[i]["episode_id"] for i in val_idx]),
        split_seed=np.array(split_seed), val_fraction=np.array(val_fraction),
    )


def dataset_summary(data) -> dict:
    acts = np.concatenate([data["train_actions"], data["val_actions"]])
    return {
        "state_dim": int(data["train_obs"].shape[1]),
        "action_dim": int(data["train_actions"].shape[1]),
        "train_episodes": int(len(data["train_episodes"])),
        "val_episodes": int(len(data["val_episodes"])),
        "train_transitions": int(len(data["train_obs"])),
        "val_transitions": int(len(data["val_obs"])),
        "action_min": acts.min(axis=0).round(4).tolist(),
        "action_max": acts.max(axis=0).round(4).tolist(),
        "fraction_actions_at_bound": float((np.abs(acts) >= 0.999).mean()),
        "features_only_centered": np.flatnonzero(data["state_std"] < MIN_SCALED_STD).tolist(),
    }


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------

class BCPolicy(nn.Module):
    """MLP teacher: normalized state -> action in [-1, 1].

    Normalization statistics are registered buffers, so a checkpoint carries its own
    preprocessing and the policy accepts raw environment observations.
    """

    def __init__(self, state_dim, action_dim, state_mean, state_scale, hidden=256, depth=2, tanh_output=True):
        super().__init__()
        layers, width = [], state_dim
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.ReLU()]
            width = hidden
        layers.append(nn.Linear(width, action_dim))
        self.net = nn.Sequential(*layers)
        self.tanh_output = tanh_output
        self.register_buffer("state_mean", torch.as_tensor(state_mean, dtype=torch.float32))
        self.register_buffer("state_scale", torch.as_tensor(state_scale, dtype=torch.float32))
        self.config = dict(state_dim=state_dim, action_dim=action_dim, hidden=hidden, depth=depth, tanh_output=tanh_output)

    def forward(self, raw_state):
        out = self.net((raw_state - self.state_mean) / self.state_scale)
        return torch.tanh(out) if self.tanh_output else out

    @torch.no_grad()
    def act(self, raw_state):
        return self(raw_state).clamp(-1.0, 1.0)


def save_policy(policy: BCPolicy, path, extra=None):
    torch.save({"config": policy.config, "state_dict": policy.state_dict(), "extra": extra or {}}, path)


def load_policy(path, device="cpu") -> BCPolicy:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    policy = BCPolicy(cfg["state_dim"], cfg["action_dim"], np.zeros(cfg["state_dim"]), np.ones(cfg["state_dim"]),
                      hidden=cfg["hidden"], depth=cfg["depth"], tanh_output=cfg["tanh_output"])
    policy.load_state_dict(ckpt["state_dict"])
    return policy.to(device).eval()


class ReleasedPPOActor(nn.Module):
    """Deterministic mean of ManiSkill's state PPO actor (examples/baselines/ppo/ppo.py)."""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, action_dim))

    @torch.no_grad()
    def act(self, raw_state):
        return self.actor_mean(raw_state).clamp(-1.0, 1.0)


def load_released_ppo(path, state_dim, action_dim, device="cpu"):
    weights = torch.load(path, map_location=device, weights_only=False)
    actor = ReleasedPPOActor(state_dim, action_dim)
    actor.load_state_dict({k: v for k, v in weights.items() if k.startswith("actor_mean.")})
    return actor.to(device).eval()


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------

@torch.no_grad()
def action_mse(policy, obs, actions, device, batch=65536):
    total, count = 0.0, 0
    for i in range(0, len(obs), batch):
        o = torch.as_tensor(obs[i:i + batch], device=device)
        a = torch.as_tensor(actions[i:i + batch], device=device)
        total += torch.nn.functional.mse_loss(policy(o), a, reduction="sum").item()
        count += a.numel()
    return total / max(count, 1)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    data = dict(np.load(args.dataset))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_obs = torch.as_tensor(data["train_obs"], device=device)
    train_act = torch.as_tensor(data["train_actions"], device=device)
    if args.tanh_output:
        # tanh cannot reach exactly +-1; keep targets just inside the bound.
        train_act = train_act.clamp(-0.999, 0.999)
    state_dim, action_dim = train_obs.shape[1], train_act.shape[1]
    policy = BCPolicy(state_dim, action_dim, data["state_mean"], data["state_scale"],
                      hidden=args.hidden, depth=args.depth, tanh_output=args.tanh_output).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters) if args.cosine else None
    gen = torch.Generator(device=device).manual_seed(args.seed)

    evaluator = None
    if args.eval_every > 0:
        evaluator = Evaluator(num_envs=args.eval_episodes, device=device)

    log_path = out / "train_log.csv"
    log = log_path.open("w")
    log.write("iter,train_mse,val_mse,lr,success_once,success_at_end,seconds\n")
    best = (-1.0, -1.0)
    start = time.time()
    running = 0.0
    for it in range(1, args.iters + 1):
        idx = torch.randint(0, len(train_obs), (args.batch_size,), device=device, generator=gen)
        loss = torch.nn.functional.mse_loss(policy(train_obs[idx]), train_act[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if sched:
            sched.step()
        running += loss.item() if it % 100 == 0 else 0.0

        last = it == args.iters
        if it % args.log_every == 0 or last:
            policy.eval()
            val_mse = action_mse(policy, data["val_obs"], data["val_actions"], device)
            row = dict(iter=it, train_mse=running / max(1, args.log_every // 100), val_mse=val_mse,
                       lr=opt.param_groups[0]["lr"], success_once="", success_at_end="")
            running = 0.0
            if evaluator and (it % args.eval_every == 0 or last):
                res = evaluator.run(policy, seed=args.val_seed)
                row.update(success_once=res["success_once"], success_at_end=res["success_at_end"])
                key = (res["success_at_end"], res["success_once"])
                save_policy(policy, out / f"ckpt_{it}.pt", {"iter": it, **res})
                if key > best:
                    best = key
                    save_policy(policy, out / "best.pt", {"iter": it, **res})
            policy.train()
            row["seconds"] = round(time.time() - start, 1)
            log.write(",".join(str(row[k]) for k in
                               ("iter", "train_mse", "val_mse", "lr", "success_once", "success_at_end", "seconds")) + "\n")
            log.flush()
            msg = f"iter {it:>7}  train_mse {row['train_mse']:.5f}  val_mse {val_mse:.5f}"
            if row["success_once"] != "":
                msg += f"  val success_once {row['success_once']:.0%}  success_at_end {row['success_at_end']:.0%}"
            print(msg, flush=True)
    log.close()
    save_policy(policy, out / "final.pt", {"iter": args.iters})
    if evaluator is None:
        save_policy(policy, out / "best.pt", {"iter": args.iters, "note": "no closed-loop eval; best = final"})
    print(f"Done in {(time.time() - start) / 60:.1f} min. Best validation (end, once): {best}")


# --------------------------------------------------------------------------------------
# Simulator: audit and evaluation
# --------------------------------------------------------------------------------------

def make_env(num_envs, obs_mode=OBS_MODE, **kwargs):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  (registers environments)

    return gym.make(ENV_ID, num_envs=num_envs, obs_mode=obs_mode, control_mode=CONTROL_MODE,
                    sim_backend=SIM_BACKEND, render_mode="rgb_array", **kwargs)


def flatten_with_names(tree, prefix=""):
    """Flatten a nested state dict in insertion order, as ManiSkill does; return names and widths."""
    fields = []
    for key, value in tree.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            fields += flatten_with_names(value, name + "/")
        else:
            width = 1 if value.ndim == 1 else int(value.shape[-1])
            fields.append((name, width))
    return fields


def audit(args):
    import mani_skill

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = make_env(1)
    try:
        obs, info = env.reset(seed=0)
        tree = env.unwrapped.get_obs(info, unflattened=True)
        fields = flatten_with_names(tree)
        width = sum(w for _, w in fields)
        assert obs.shape == (1, width), f"Flat obs {tuple(obs.shape)} does not match fields totalling {width}"
        offsets, start = {}, 0
        for name, w in fields:
            offsets[name] = [start, start + w]
            start += w
        names = [n for n, _ in fields]
        if not any("obj_pose" in n for n in names):
            raise AssertionError("State observation lacks obj_pose; this is not the full-state setting")
        space = env.unwrapped.single_action_space
        contract = {
            "env_id": ENV_ID, "obs_mode": OBS_MODE, "control_mode": CONTROL_MODE, "sim_backend": SIM_BACKEND,
            "mani_skill_version": getattr(mani_skill, "__version__", "unknown"),
            "state_dim": width, "state_fields_in_order": offsets,
            "action_dim": int(np.prod(space.shape)),
            "action_low": space.low.tolist(), "action_high": space.high.tolist(),
            "control_freq": getattr(env.unwrapped, "control_freq", None),
            "sim_freq": getattr(env.unwrapped, "sim_freq", None),
            "max_episode_steps": env.spec.max_episode_steps,
            "robot_uids": str(getattr(env.unwrapped, "robot_uids", None)),
            "success_rule": "evaluate(): T overlap with goal >= intersection_thresh",
            "intersection_thresh": getattr(env.unwrapped, "intersection_thresh", None),
        }
        if not (np.allclose(space.low, -1) and np.allclose(space.high, 1)):
            print("WARNING: action space is not [-1, 1]; the BC tanh head assumes normalized actions.")
        (out / "observation_contract.json").write_text(json.dumps(contract, indent=2))
        print(json.dumps(contract, indent=2))
    finally:
        env.close()


class Evaluator:
    """Closed-loop evaluation on a fixed batch of seeded initial states, no partial resets."""

    def __init__(self, num_envs, device, video_dir=None):
        from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

        env = make_env(num_envs, reconfiguration_freq=1)
        self.max_steps = env.spec.max_episode_steps
        if video_dir is not None:
            from mani_skill.utils.wrappers.record import RecordEpisode
            env = RecordEpisode(env, output_dir=str(video_dir), save_trajectory=False,
                                max_steps_per_video=self.max_steps, video_fps=30)
        # auto_reset=False: otherwise the final step's info is replaced by the next episode's.
        self.env = ManiSkillVectorEnv(env, num_envs, auto_reset=False, ignore_terminations=True, record_metrics=True)
        self.num_envs = num_envs
        self.device = device

    @torch.no_grad()
    def run(self, policy, seed):
        obs, _ = self.env.reset(seed=seed)
        success_once = torch.zeros(self.num_envs, dtype=torch.bool, device=obs.device)
        first_success = torch.full((self.num_envs,), -1, dtype=torch.long, device=obs.device)
        for t in range(self.max_steps):
            action = policy.act(obs.to(self.device).float())
            obs, _, _, _, info = self.env.step(action)
            success = info["success"].bool()
            first_success = torch.where((first_success < 0) & success, torch.full_like(first_success, t + 1), first_success)
            success_once |= success
        success_end = success
        steps = first_success[first_success > 0].float()
        return {
            "episodes": self.num_envs, "seed": seed,
            "success_once": success_once.float().mean().item(),
            "success_at_end": success_end.float().mean().item(),
            "mean_steps_to_first_success": steps.mean().item() if len(steps) else None,
        }

    def close(self):
        self.env.close()


def evaluate(args):
    device = torch.device(args.device)
    video_dir = Path(args.video_dir) if args.video_dir else None
    evaluator = Evaluator(args.episodes, device, video_dir=video_dir)
    try:
        if args.released_ppo:
            space = evaluator.env.single_observation_space
            act_space = evaluator.env.single_action_space
            policy = load_released_ppo(args.checkpoint, int(np.prod(space.shape)), int(np.prod(act_space.shape)), device)
        else:
            policy = load_policy(args.checkpoint, device)
        res = evaluator.run(policy, seed=args.seed)
    finally:
        evaluator.close()
    res["checkpoint"] = str(args.checkpoint)
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
    return res


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def prepare(args):
    episodes, meta = load_replayed_demos(args.h5)
    if args.max_episodes:
        episodes = episodes[:args.max_episodes]
    data = build_dataset(episodes, args.val_fraction, args.split_seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **data)
    summary = dataset_summary(data)
    summary.update(source_h5=str(args.h5), source_env_info=meta["env_info"],
                   source_commit=meta.get("commit_info"),
                   replayed_success_at_end=float(np.mean([e["success_at_end"] for e in episodes])))
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit")
    a.add_argument("--out", required=True)

    pr = sub.add_parser("prepare")
    pr.add_argument("--h5", required=True)
    pr.add_argument("--out", required=True)
    pr.add_argument("--val-fraction", type=float, default=0.1)
    pr.add_argument("--split-seed", type=int, default=0)
    pr.add_argument("--max-episodes", type=int, default=0, help="0 = use all episodes")

    t = sub.add_parser("train")
    t.add_argument("--dataset", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--iters", type=int, default=100_000)
    t.add_argument("--batch-size", type=int, default=1024)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--cosine", action=argparse.BooleanOptionalAction, default=True)
    t.add_argument("--hidden", type=int, default=256)
    t.add_argument("--depth", type=int, default=2)
    t.add_argument("--tanh-output", action=argparse.BooleanOptionalAction, default=True)
    t.add_argument("--log-every", type=int, default=1000)
    t.add_argument("--eval-every", type=int, default=10_000, help="0 disables closed-loop eval")
    t.add_argument("--eval-episodes", type=int, default=100)
    t.add_argument("--val-seed", type=int, default=1001)
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    e = sub.add_parser("eval")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--released-ppo", action="store_true", help="checkpoint is ManiSkill's released PPO .pt")
    e.add_argument("--episodes", type=int, default=100)
    e.add_argument("--seed", type=int, default=2001)
    e.add_argument("--out", default="")
    e.add_argument("--video-dir", default="")
    e.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = p.parse_args(argv)
    if args.cmd == "train" and args.log_every % 100:
        p.error("--log-every must be a multiple of 100")
    {"audit": audit, "prepare": prepare, "train": train, "eval": evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()

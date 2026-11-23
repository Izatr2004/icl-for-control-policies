"""Full-state behavioral cloning (BC) teachers for ManiSkill PushT-v1.

Subcommands (run with the ManiSkill training interpreter):

    audit    Inspect the state observation and action spaces; write observation_contract.json.
    prepare  Convert a replayed state-mode demo file (.h5 + .json) into train/validation
             transition files, split by whole episode.
    collect  Roll out a driver policy (the expert, or a BC checkpoint for DAgger) and label
             every visited state with the expert's deterministic action.
    train    Train one MLP BC teacher on one or more transition files, evaluating closed-loop
             success on validation seeds.
    eval     Evaluate a BC checkpoint, or the released PPO expert, on a seed range.

Transition files are .npz with obs [N, d_s], actions [N, d_a] and episode_ids [N],
plus a .json sidecar describing where they came from.

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
# Transition files
# --------------------------------------------------------------------------------------

def save_transitions(path, obs, actions, episode_ids, meta=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, obs=np.asarray(obs, np.float32), actions=np.asarray(actions, np.float32),
                        episode_ids=np.asarray(episode_ids, np.int64))
    meta = dict(meta or {}, transitions=int(len(obs)), episodes=int(len(np.unique(episode_ids))))
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return meta


def load_transitions(paths):
    """Concatenate transition files; episode ids are made unique across files."""
    obs, actions, episode_ids, offset = [], [], [], 0
    for p in paths:
        d = np.load(p)
        obs.append(d["obs"])
        actions.append(d["actions"])
        ids = d["episode_ids"].astype(np.int64)
        episode_ids.append(ids - ids.min() + offset if len(ids) else ids)
        offset = (episode_ids[-1].max() + 1) if len(ids) else offset
    return np.concatenate(obs), np.concatenate(actions), np.concatenate(episode_ids)


# --------------------------------------------------------------------------------------
# Released demos
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
                # The simulator clips commands to the action space, so clipped labels are
                # exactly what the expert executed.
                actions=np.clip(actions, -1.0, 1.0),
                raw_actions=actions,
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


def prepare(args):
    episodes, meta = load_replayed_demos(args.h5)
    if args.max_episodes:
        episodes = episodes[:args.max_episodes]
    train_idx, val_idx = split_episodes(len(episodes), args.val_fraction, args.split_seed)
    out = Path(args.out_dir)
    source = dict(source="released PPO demos, replayed to state obs", source_h5=str(args.h5),
                  source_env_info=meta["env_info"], source_commit=meta.get("commit_info"),
                  split_seed=args.split_seed, val_fraction=args.val_fraction)
    for name, idx in (("demos_train", train_idx), ("demos_val", val_idx)):
        eps = [episodes[i] for i in idx]
        save_transitions(out / f"{name}.npz",
                         np.concatenate([e["obs"] for e in eps]),
                         np.concatenate([e["actions"] for e in eps]),
                         np.concatenate([np.full(len(e["actions"]), e["episode_id"]) for e in eps]),
                         dict(source, split=name, episode_ids=[e["episode_id"] for e in eps]))
    raw = np.concatenate([e["raw_actions"] for e in episodes])
    summary = dict(
        state_dim=int(episodes[0]["obs"].shape[1]), action_dim=int(raw.shape[1]),
        episodes_in_file=len(meta["episodes"]), episodes_replayed=len(episodes),
        train_episodes=len(train_idx), val_episodes=len(val_idx),
        raw_action_min=raw.min(0).round(4).tolist(), raw_action_max=raw.max(0).round(4).tolist(),
        fraction_raw_actions_outside_bounds=float((np.abs(raw) > 1).mean()),
        replayed_success_at_end=float(np.mean([e["success_at_end"] for e in episodes])),
    )
    (out / "demos_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


# --------------------------------------------------------------------------------------
# Models
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
# Rollouts (shared by evaluation and data collection)
# --------------------------------------------------------------------------------------

@torch.no_grad()
def rollout(env, max_steps, driver, seed, labeler=None, noise_std=0.0, generator=None):
    """Run one batch of episodes, one per parallel env, for max_steps with no resets.

    driver picks the executed action; Gaussian noise (std on the [-1, 1] action scale) is
    added after labeling, so labels stay the labeler's clean actions. If labeler is given,
    returns obs [n, T, d_s] and labels [n, T, d_a] on the CPU, episode-major.
    """
    obs, _ = env.reset(seed=seed)
    n = obs.shape[0]
    device = obs.device
    success_once = torch.zeros(n, dtype=torch.bool, device=device)
    first_success = torch.full((n,), -1, dtype=torch.long, device=device)
    obs_buf, label_buf = [], []
    success = success_once
    for t in range(max_steps):
        o = obs.float()
        action = driver.act(o)
        if labeler is not None:
            label = action if labeler is driver else labeler.act(o)
            obs_buf.append(o.cpu())
            label_buf.append(label.cpu())
        if noise_std > 0:
            noise = torch.randn(action.shape, device=action.device, generator=generator)
            action = (action + noise_std * noise).clamp(-1.0, 1.0)
        obs, _, _, _, info = env.step(action)
        success = info["success"].bool()
        newly = (first_success < 0) & success
        first_success = torch.where(newly, torch.full_like(first_success, t + 1), first_success)
        success_once |= success
    steps = first_success[first_success > 0].float()
    result = {
        "episodes": int(n), "seed": int(seed), "noise_std": float(noise_std),
        "success_once": success_once.float().mean().item(),
        "success_at_end": success.float().mean().item(),
        "mean_steps_to_first_success": steps.mean().item() if len(steps) else None,
    }
    if labeler is None:
        return result
    return result, torch.stack(obs_buf, 1).numpy(), torch.stack(label_buf, 1).numpy()


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
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    obs_np, act_np, ep_np = load_transitions(args.train_data)
    val_obs, val_act, _ = load_transitions([args.val_data])
    mean, std, scale = compute_state_stats(obs_np)
    data_info = dict(train_files=[str(p) for p in args.train_data], val_file=str(args.val_data),
                     train_transitions=int(len(obs_np)), train_episodes=int(len(np.unique(ep_np))),
                     val_transitions=int(len(val_obs)),
                     features_only_centered=np.flatnonzero(std < MIN_SCALED_STD).tolist())
    (out / "data_info.json").write_text(json.dumps(data_info, indent=2))
    print(f"Training on {data_info['train_transitions']:,} transitions from "
          f"{data_info['train_episodes']:,} episodes in {len(args.train_data)} file(s)", flush=True)

    train_obs = torch.as_tensor(obs_np, device=device)
    train_act = torch.as_tensor(act_np, device=device).clamp(-1.0, 1.0)
    if args.tanh_output:
        # tanh cannot reach exactly +-1; keep targets just inside the bound.
        train_act = train_act.clamp(-0.999, 0.999)
    state_dim, action_dim = train_obs.shape[1], train_act.shape[1]
    policy = BCPolicy(state_dim, action_dim, mean, scale,
                      hidden=args.hidden, depth=args.depth, tanh_output=args.tanh_output).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters) if args.cosine else None
    gen = torch.Generator(device=device).manual_seed(args.seed)

    evaluator = Evaluator(num_envs=args.eval_episodes, device=device) if args.eval_every > 0 else None

    fields = ("iter", "train_mse", "val_mse", "lr", "success_once", "success_at_end", "seconds")
    log = (out / "train_log.csv").open("w")
    log.write(",".join(fields) + "\n")
    best = (-1.0, -1.0)
    start = time.time()
    running, samples = 0.0, 0
    for it in range(1, args.iters + 1):
        idx = torch.randint(0, len(train_obs), (args.batch_size,), device=device, generator=gen)
        loss = torch.nn.functional.mse_loss(policy(train_obs[idx]), train_act[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if sched:
            sched.step()
        if it % 100 == 0:
            running += loss.item()
            samples += 1

        last = it == args.iters
        if it % args.log_every == 0 or last:
            policy.eval()
            val_mse = action_mse(policy, val_obs, val_act.clip(-1, 1), device)
            row = dict(iter=it, train_mse=running / max(samples, 1), val_mse=val_mse,
                       lr=opt.param_groups[0]["lr"], success_once="", success_at_end="")
            running, samples = 0.0, 0
            if evaluator and (it % args.eval_every == 0 or last):
                res = evaluator.run(policy, seed=args.val_seed)
                row.update(success_once=res["success_once"], success_at_end=res["success_at_end"])
                save_policy(policy, out / f"ckpt_{it}.pt", {"iter": it, **res})
                key = (res["success_at_end"], res["success_once"])
                if key > best:
                    best = key
                    save_policy(policy, out / "best.pt", {"iter": it, **res})
                    (out / "best.json").write_text(json.dumps({"iter": it, **res}, indent=2))
            policy.train()
            row["seconds"] = round(time.time() - start, 1)
            log.write(",".join(str(row[k]) for k in fields) + "\n")
            log.flush()
            msg = f"iter {it:>7}  train_mse {row['train_mse']:.5f}  val_mse {val_mse:.5f}"
            if row["success_once"] != "":
                msg += f"  val success_once {row['success_once']:.0%}  success_at_end {row['success_at_end']:.0%}"
            print(msg, flush=True)
    log.close()
    save_policy(policy, out / "final.pt", {"iter": args.iters})
    if evaluator is None:
        save_policy(policy, out / "best.pt", {"iter": args.iters, "note": "no closed-loop eval; best = final"})
    else:
        evaluator.close()
    print(f"Done in {(time.time() - start) / 60:.1f} min. Best validation (end, once): {best}")


# --------------------------------------------------------------------------------------
# Simulator: audit, evaluation, collection
# --------------------------------------------------------------------------------------

def make_env(num_envs, obs_mode=OBS_MODE, **kwargs):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  (registers environments)

    return gym.make(ENV_ID, num_envs=num_envs, obs_mode=obs_mode, control_mode=CONTROL_MODE,
                    sim_backend=SIM_BACKEND, render_mode="rgb_array", **kwargs)


def episode_length(env):
    """Max episode steps. Vectorized GPU envs leave env.spec.max_episode_steps as None,
    so use ManiSkill's own lookup (as its PPO baseline does), with PushT-v1's registered 100 as fallback."""
    from mani_skill.utils import gym_utils

    steps = gym_utils.find_max_episode_steps_value(env)
    return int(steps) if steps is not None else 100


def make_vector_env(num_envs, video_dir=None, **kwargs):
    """Fixed-length episodes: no auto-reset (it would replace the final step's info) and
    terminations ignored, matching the evaluation convention of ManiSkill's PPO baseline."""
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    env = make_env(num_envs, **kwargs)
    max_steps = episode_length(env)
    if video_dir is not None:
        from mani_skill.utils.wrappers.record import RecordEpisode
        env = RecordEpisode(env, output_dir=str(video_dir), save_trajectory=False,
                            max_steps_per_video=max_steps, video_fps=30)
    return ManiSkillVectorEnv(env, num_envs, auto_reset=False, ignore_terminations=True, record_metrics=True), max_steps


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
        if not any("obj_pose" in n for n, _ in fields):
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
            "max_episode_steps": episode_length(env),
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
    """Closed-loop evaluation on a fixed batch of seeded initial states."""

    def __init__(self, num_envs, device, video_dir=None):
        self.env, self.max_steps = make_vector_env(num_envs, video_dir=video_dir, reconfiguration_freq=1)
        self.num_envs = num_envs
        self.device = device

    def run(self, policy, seed):
        return rollout(self.env, self.max_steps, policy, seed)

    def close(self):
        self.env.close()


def load_expert(path, env, device):
    state_dim = int(np.prod(env.single_observation_space.shape))
    action_dim = int(np.prod(env.single_action_space.shape))
    return load_released_ppo(path, state_dim, action_dim, device)


def evaluate(args):
    device = torch.device(args.device)
    video_dir = Path(args.video_dir) if args.video_dir else None
    evaluator = Evaluator(args.episodes, device, video_dir=video_dir)
    try:
        if args.released_ppo:
            policy = load_expert(args.checkpoint, evaluator.env, device)
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


def collect(args):
    """Roll out a driver and label every visited state with the expert.

    driver=expert (optionally with action noise) gives more, and slightly off-course,
    expert demonstrations. driver=bc gives one round of DAgger data: states the BC policy
    actually reaches, labeled with what the expert would do there.
    """
    device = torch.device(args.device)
    env, max_steps = make_vector_env(args.num_envs)
    try:
        expert = load_expert(args.expert, env, device)
        driver = expert if args.driver == "expert" else load_policy(args.driver_checkpoint, device)
        noise_levels = args.noise_std or [0.0]
        gen = torch.Generator(device=device).manual_seed(args.seed_base)
        batches = math.ceil(args.episodes / args.num_envs)
        all_obs, all_labels, per_batch = [], [], []
        for b in range(batches):
            sigma = noise_levels[b % len(noise_levels)]
            res, obs, labels = rollout(env, max_steps, driver, seed=args.seed_base + b,
                                       labeler=expert, noise_std=sigma, generator=gen)
            all_obs.append(obs)
            all_labels.append(labels)
            per_batch.append(res)
            print(f"batch {b + 1}/{batches}  noise {sigma:.2f}  driver success_once {res['success_once']:.0%}  "
                  f"success_at_end {res['success_at_end']:.0%}", flush=True)
    finally:
        env.close()
    obs = np.concatenate(all_obs)[:args.episodes]       # [episodes, T, d_s]
    labels = np.concatenate(all_labels)[:args.episodes]
    n, T = obs.shape[:2]
    meta = dict(
        driver=args.driver, driver_checkpoint=args.driver_checkpoint or None, expert=str(args.expert),
        labels="expert deterministic action, clipped to [-1, 1]", noise_std_cycle=noise_levels,
        seed_base=args.seed_base, num_envs=args.num_envs, steps_per_episode=int(T), batches=per_batch,
        driver_success_once=float(np.mean([r["success_once"] for r in per_batch])),
        driver_success_at_end=float(np.mean([r["success_at_end"] for r in per_batch])),
    )
    meta = save_transitions(args.out, obs.reshape(n * T, -1), labels.reshape(n * T, -1),
                            np.repeat(np.arange(n), T), meta)
    print(json.dumps({k: v for k, v in meta.items() if k != "batches"}, indent=2))


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    a = sub.add_parser("audit")
    a.add_argument("--out", required=True)

    pr = sub.add_parser("prepare")
    pr.add_argument("--h5", required=True)
    pr.add_argument("--out-dir", required=True)
    pr.add_argument("--val-fraction", type=float, default=0.1)
    pr.add_argument("--split-seed", type=int, default=0)
    pr.add_argument("--max-episodes", type=int, default=0, help="0 = use all episodes")

    c = sub.add_parser("collect")
    c.add_argument("--expert", required=True, help="released PPO checkpoint used for labels")
    c.add_argument("--driver", choices=("expert", "bc"), default="expert")
    c.add_argument("--driver-checkpoint", default="", help="BC checkpoint when --driver bc")
    c.add_argument("--episodes", type=int, required=True)
    c.add_argument("--num-envs", type=int, default=1024)
    c.add_argument("--noise-std", type=float, nargs="*", default=[0.0],
                   help="action noise std per batch, cycled; labels stay noise-free")
    c.add_argument("--seed-base", type=int, required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--device", default=default_device)

    t = sub.add_parser("train")
    t.add_argument("--train-data", nargs="+", required=True)
    t.add_argument("--val-data", required=True)
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
    t.add_argument("--device", default=default_device)

    e = sub.add_parser("eval")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--released-ppo", action="store_true", help="checkpoint is ManiSkill's released PPO .pt")
    e.add_argument("--episodes", type=int, default=100)
    e.add_argument("--seed", type=int, default=2001)
    e.add_argument("--out", default="")
    e.add_argument("--video-dir", default="")
    e.add_argument("--device", default=default_device)

    args = p.parse_args(argv)
    if args.cmd == "collect" and args.driver == "bc" and not args.driver_checkpoint:
        p.error("--driver bc needs --driver-checkpoint")
    {"audit": audit, "prepare": prepare, "collect": collect, "train": train, "eval": evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()

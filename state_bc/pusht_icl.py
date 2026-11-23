"""In-context learning (ICL) transformer for variable-goal Push-T (v1 of CODEX_CONTEXT.md).

A frozen GPT-2-style transformer sees demonstrations from one teacher for one hidden goal, then
controls the robot from a new start. It never sees the goal: the goal fields (state indices 21:24)
are removed from every state it receives. Teachers (goal-conditioned, they do see the goal):
Jimmy's PPO and the distilled BC teacher, both in PushTVarGoal-v1.

Subcommands (run with the ManiSkill training interpreter, pusht_vargoal.py next to this file):

    collect   For each split's goal set, roll out each teacher (noise-free) E times per goal
              from different starts and save {split}_{teacher}.npz: obs [G, E, T, 31],
              actions [G, E, T, 7], first_success_step [G, E], success_at_end [G, E],
              goal_xy [G, 2], goal_quadrant [G].
    train     Train the transformer on prompts sampled on the fly from the train split;
              validation action error (correct vs wrong-goal context) on the val split.
              Saves last.pt (for resuming) and best.pt.
    offline   Teacher-forced action error per context size, correct vs wrong-goal context.
    eval      Closed-loop success with frozen weights: per split, teacher, context size and
              condition (correct-goal, wrong-goal, no context), plus the teachers themselves.

Splits (goal positions, PushTVarGoal-v1 quadrants): train Q1-Q3, val Q1-Q3, test_in Q1-Q3
(unseen goals), test_q4 Q4 (held out from transformer training). Each split's goals are an
independent seeded draw (continuous positions, so no goal repeats across splits in practice).

Prompt layout (PushT_ICL_Transformer_Spec.md): [s_1, a_1, ..., s_N, a_N, s_q] -> a_q, one token
per state or action. Support = k whole demonstration episodes of the same teacher and goal,
each subsampled every `stride` steps ending at the last step (so the final T pose is included),
N = k * ceil(T / stride). The query state comes from a different episode of the same teacher and
goal; its label is the teacher's action there.

Pure data/model code imports without ManiSkill (transformers is needed for the model).
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

import pusht_state_bc as bc

GOAL_START, GOAL_STOP = 21, 24          # extra/goal_pos in the 31-number state
TEACHERS = ("ppo", "bc")
SPLITS = {"train": (1, 2, 3), "val": (1, 2, 3), "test_in": (1, 2, 3), "test_q4": (4,)}
GOAL_SEED = {"train": 11, "val": 12, "test_in": 13, "test_q4": 14}
DEMO_SEED_BASE = {"train": 1_000_000, "val": 2_000_000, "test_in": 3_000_000, "test_q4": 4_000_000}
EVAL_SEED_BASE = {"train": 7_100_000, "val": 7_200_000, "test_in": 7_300_000, "test_q4": 7_400_000}


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------

def hide_goal(x):
    """Drop the goal fields (21:24) from states [..., 31] -> [..., 28]."""
    if isinstance(x, np.ndarray):
        return np.concatenate([x[..., :GOAL_START], x[..., GOAL_STOP:]], axis=-1)
    return torch.cat([x[..., :GOAL_START], x[..., GOAL_STOP:]], dim=-1)


def snippet_steps(T, stride):
    """Time steps kept from each support episode: every `stride`-th, always including the last."""
    return torch.arange(T - 1, -1, -stride).flip(0)


def sample_goal_set(sample_fn, n, quadrants, seed):
    """Goal set for a split, reproducible from `seed` without touching the global RNG.
    sample_fn(n, quadrants) is pusht_vargoal.sample_goals (CPU)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return sample_fn(n, quadrants)


class PromptData:
    """Teacher demonstrations grouped by goal, as tensors (kept on the GPU during training).

    obs [n_teachers, G, E, T, d_s] (goal hidden), actions [n_teachers, G, E, T, d_a],
    success [n_teachers, G, E] (reached the goal at some step).
    """

    def __init__(self, obs, actions, success, goal_xy, goal_quadrant, teachers=TEACHERS):
        self.obs, self.actions, self.success = obs, actions, success
        self.goal_xy, self.goal_quadrant, self.teachers = goal_xy, goal_quadrant, tuple(teachers)
        self.n_teachers, self.G, self.E, self.T = obs.shape[:4]

    @classmethod
    def load(cls, data_dir, split, teachers=TEACHERS, device="cpu"):
        obs, actions, success, goals = [], [], [], None
        for t in teachers:
            with np.load(Path(data_dir) / f"{split}_{t}.npz") as d:
                if goals is None:
                    goals, quadrant = d["goal_xy"], d["goal_quadrant"]
                elif not np.allclose(goals, d["goal_xy"]):
                    raise ValueError(f"{split}: teachers were collected on different goal sets")
                obs.append(hide_goal(d["obs"]))
                actions.append(d["actions"])
                success.append(d["first_success_step"] > 0)
        as_t = lambda x, dt=torch.float32: torch.as_tensor(np.stack(x), dtype=dt, device=device)  # noqa: E731
        return cls(as_t(obs), as_t(actions), as_t(success, torch.bool),
                   torch.as_tensor(goals, device=device), torch.as_tensor(quadrant, device=device), teachers)

    def episode_order(self, ti, gi, generator):
        """[B, E] episode order per (teacher, goal): successful episodes first, random within."""
        scores = torch.rand(len(ti), self.E, generator=generator, device=self.obs.device)
        return (scores + self.success[ti, gi].float()).argsort(dim=1, descending=True)

    def support(self, ti, gi, episodes, stride):
        """States and actions of the given episodes [B, k], subsampled, flattened to [B, k*P, d]."""
        B, k = episodes.shape
        steps = snippet_steps(self.T, stride).to(self.obs.device)
        idx = (ti[:, None, None], gi[:, None, None], episodes[:, :, None], steps[None, None, :])
        n = k * len(steps)
        return (self.obs[idx].reshape(B, n, self.obs.shape[-1]),
                self.actions[idx].reshape(B, n, self.actions.shape[-1]))

    def sample(self, batch, k, stride, generator, wrong_goal=False):
        """One batch of prompts with k support episodes. With wrong_goal=True the support comes from a
        different goal of the same teacher; the query (episode, step, label) is identical to the
        wrong_goal=False draw from the same generator state."""
        if k >= self.E:
            raise ValueError(f"k={k} needs more than {self.E} episodes per goal")
        dev = self.obs.device
        ti = torch.randint(self.n_teachers, (batch,), generator=generator, device=dev)
        gi = torch.randint(self.G, (batch,), generator=generator, device=dev)
        order = self.episode_order(ti, gi, generator)
        shift = torch.randint(1, max(self.G, 2), (batch,), generator=generator, device=dev)
        gw = (gi + shift) % self.G
        order_w = self.episode_order(ti, gw, generator)
        qt = torch.randint(self.T, (batch,), generator=generator, device=dev)
        qe = order[:, k]
        query_s, query_a = self.obs[ti, gi, qe, qt], self.actions[ti, gi, qe, qt]
        sg, so = (gw, order_w) if wrong_goal else (gi, order)
        support_s, support_a = self.support(ti, sg, so[:, :k], stride)
        return support_s, support_a, query_s, query_a


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------

class ICLTransformer(nn.Module):
    """GPT-2 backbone (PushT_ICL_Transformer_Spec.md section 2) with numerical state/action
    projections and a linear action head read at the query token."""

    def __init__(self, d_s, d_a, state_mean, state_scale, n_layer=12, n_embd=256, n_head=8, n_positions=1024):
        super().__init__()
        from transformers import GPT2Config, GPT2Model

        self.hparams = dict(d_s=d_s, d_a=d_a, n_layer=n_layer, n_embd=n_embd, n_head=n_head, n_positions=n_positions)
        config = GPT2Config(
            vocab_size=1, n_positions=n_positions, n_embd=n_embd, n_layer=n_layer, n_head=n_head,
            n_inner=4 * n_embd, activation_function="gelu_new", resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
            layer_norm_epsilon=1e-5, initializer_range=0.02, scale_attn_weights=True,
            scale_attn_by_inverse_layer_idx=False, reorder_and_upcast_attn=False, add_cross_attention=False,
            use_cache=False, bos_token_id=None, eos_token_id=None)
        self.backbone = GPT2Model(config)
        self.state_encoder = nn.Linear(d_s, n_embd)
        self.action_encoder = nn.Linear(d_a, n_embd)
        self.action_head = nn.Linear(n_embd, d_a)
        self.register_buffer("state_mean", torch.as_tensor(state_mean, dtype=torch.float32))
        self.register_buffer("state_scale", torch.as_tensor(state_scale, dtype=torch.float32))

    def forward(self, support_s, support_a, query_s):
        """support_s [B, N, d_s], support_a [B, N, d_a] (N may be 0), query_s [B, d_s] -> [B, d_a]."""
        B, N = support_s.shape[:2]
        if 2 * N + 1 > self.hparams["n_positions"]:
            raise ValueError(f"prompt of {2 * N + 1} tokens exceeds {self.hparams['n_positions']}")
        q = self.state_encoder((query_s - self.state_mean) / self.state_scale)[:, None]
        if N:
            s = self.state_encoder((support_s - self.state_mean) / self.state_scale)
            a = self.action_encoder(support_a)
            q = torch.cat([torch.stack([s, a], dim=2).reshape(B, 2 * N, -1), q], dim=1)
        h = self.backbone(inputs_embeds=q).last_hidden_state[:, -1]
        return self.action_head(h)


def save_model(model, path, extra=None):
    torch.save({"hparams": model.hparams, "state_dict": model.state_dict(), **(extra or {})}, path)


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    hp = ckpt["hparams"]
    model = ICLTransformer(state_mean=torch.zeros(hp["d_s"]), state_scale=torch.ones(hp["d_s"]), **hp)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


class ICLActor:
    """Closed-loop controller: the frozen transformer with a fixed support per environment."""

    def __init__(self, model, support_s, support_a):
        self.model, self.support_s, self.support_a = model, support_s, support_a

    @torch.no_grad()
    def act(self, obs):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=obs.is_cuda):
            a = self.model(self.support_s, self.support_a, hide_goal(obs.float()))
        return a.float().clamp(-1.0, 1.0)


# --------------------------------------------------------------------------------------
# Training and offline evaluation
# --------------------------------------------------------------------------------------

@torch.no_grad()
def offline_eval(model, data, ks, stride, batches=4, batch=256, seed=0):
    """Teacher-forced query-action MSE per context size: correct-goal vs wrong-goal support (same queries)."""
    model.eval()
    dev = data.obs.device
    out = {}
    for k in ks:
        row = {}
        for cond in (("none",) if k == 0 else ("correct", "wrong")):
            g = torch.Generator(device=dev).manual_seed(seed + 1000 * k)
            total = 0.0
            for _ in range(batches):
                s, a, qs, qa = data.sample(batch, k, stride, g, wrong_goal=(cond == "wrong"))
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    pred = model(s, a, qs)
                total += torch.nn.functional.mse_loss(pred.float().clamp(-1, 1), qa).item()
            row[cond] = total / batches
        out[str(k)] = row
    model.train()
    return out


def lr_at(it, iters, lr, warmup):
    if it < warmup:
        return lr * (it + 1) / warmup
    progress = (it - warmup) / max(1, iters - warmup)
    return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ks = list(args.ks)
    train_data = PromptData.load(args.data_dir, "train", device=device)
    val_data = PromptData.load(args.data_dir, "val", device=device)
    pairs = len(snippet_steps(train_data.T, args.stride))
    print(f"train: {train_data.n_teachers} teachers x {train_data.G} goals x {train_data.E} episodes x {train_data.T} steps; "
          f"val: {val_data.G} goals. Context sizes {ks} episodes = up to {max(ks) * pairs} pairs "
          f"({2 * max(ks) * pairs + 1} tokens)", flush=True)

    flat = train_data.obs.reshape(-1, train_data.obs.shape[-1])
    pick = torch.randperm(len(flat), device=device)[:1_000_000]
    mean, _, scale = bc.compute_state_stats(flat[pick].cpu().numpy())
    model = ICLTransformer(flat.shape[-1], train_data.actions.shape[-1], mean, scale,
                           n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    start, best = 0, float("inf")
    last = out / "last.pt"
    if args.resume and last.exists():
        ckpt = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        opt.load_state_dict(ckpt["optimizer"])
        gen.set_state(ckpt["generator"].cpu())
        start, best = ckpt["iter"], ckpt["best"]
        print(f"resumed from {last} at iteration {start}", flush=True)
    config = {**vars(args), "pairs_per_episode": pairs, "train_goals": train_data.G, "episodes_per_goal": train_data.E}
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))

    log = (out / "train_log.jsonl").open("a")
    running, n_running, t0 = 0.0, 0, time.time()
    for it in range(start, args.iters):
        for group in opt.param_groups:
            group["lr"] = lr_at(it, args.iters, args.lr, args.warmup)
        k = ks[int(torch.randint(len(ks), (1,), generator=gen, device=device).item())]
        s, a, qs, qa = train_data.sample(args.batch_size, k, args.stride, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = model(s, a, qs)
        loss = torch.nn.functional.mse_loss(pred.float(), qa)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running += loss.item()
        n_running += 1
        done = it + 1
        if done % args.log_every == 0:
            rate = n_running / (time.time() - t0)
            print(f"iter {done} loss {running / n_running:.5f} lr {opt.param_groups[0]['lr']:.2e} "
                  f"{rate:.1f} it/s (ETA {(args.iters - done) / rate / 60:.0f} min)", flush=True)
            running, n_running, t0 = 0.0, 0, time.time()
        if done % args.eval_every == 0 or done == args.iters:
            val = offline_eval(model, val_data, ks, args.stride)
            score = float(np.mean([v.get("correct", v.get("none")) for v in val.values()]))
            print(f"iter {done} val MSE " + "  ".join(
                f"k={k}: " + "/".join(f"{c} {m:.4f}" for c, m in v.items()) for k, v in val.items()), flush=True)
            log.write(json.dumps({"iter": done, "val": val, "val_score": score}) + "\n")
            log.flush()
            if score < best:
                best = score
                save_model(model, out / "best.pt", {"iter": done, "val": val})
            torch.save({"hparams": model.hparams, "state_dict": model.state_dict(), "optimizer": opt.state_dict(),
                        "generator": gen.get_state(), "iter": done, "best": best}, last)
    save_model(model, out / "final.pt", {"iter": args.iters})
    log.close()
    print(f"Done. Best validation score {best:.5f}", flush=True)


def offline(args):
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    results = {}
    for split in args.splits:
        data = PromptData.load(args.data_dir, split, device=device)
        results[split] = offline_eval(model, data, args.ks, args.stride, batches=args.batches)
        print(split, json.dumps(results[split]), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))


# --------------------------------------------------------------------------------------
# Simulator: demonstration collection and closed-loop evaluation
# --------------------------------------------------------------------------------------

def load_teachers(args, env, device):
    return {"ppo": bc.load_expert(args.ppo_teacher, env, device), "bc": bc.load_policy(args.bc_teacher, device)}


def collect(args):
    """Noise-free demonstrations per split, goal and teacher; every teacher sees the same goals."""
    import pusht_vargoal as pv

    bc.configure_env(bc.SHARED_ENV_ID)
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = bc.make_env(args.num_envs)
    T = bc.episode_length(env)
    try:
        teachers = load_teachers(args, env, device)
        for split, n_goals in zip(args.splits, args.goals):
            goals = sample_goal_set(pv.sample_goals, n_goals, SPLITS[split], GOAL_SEED[split]).float()
            quadrant = pv.quadrant_of(goals)
            assert set(quadrant.tolist()) <= set(SPLITS[split]), split
            goal_of_episode = torch.arange(n_goals).repeat_interleave(args.episodes)
            n = len(goal_of_episode)
            batches = math.ceil(n / args.num_envs)
            for name, policy in teachers.items():
                path = out / f"{split}_{name}.npz"
                if path.exists() and not args.overwrite:
                    print("exists, skipping:", path, flush=True)
                    continue
                obs_all, act_all, first_all, end_all = [], [], [], []
                for b in range(batches):
                    ids = goal_of_episode[b * args.num_envs:(b + 1) * args.num_envs]
                    ids = torch.cat([ids, ids[:1].expand(args.num_envs - len(ids))])  # pad the last batch
                    goal_xy = goals[ids].to(device)
                    res, obs, labels = bc.rollout(env, T, policy, seed=DEMO_SEED_BASE[split] + b, labeler=policy,
                                                  reset_options={"goal_xy": goal_xy})
                    placed = torch.as_tensor(res["goal_xy_per_episode"])
                    if not torch.allclose(placed, goal_xy.cpu(), atol=1e-4):
                        raise AssertionError("environment did not place the requested goals")
                    obs_all.append(obs)
                    act_all.append(labels)
                    first_all.append(np.asarray(res["first_success_step_per_episode"]))
                    end_all.append(np.asarray(res["success_at_end_per_episode"]))
                    print(f"{split} {name} batch {b + 1}/{batches}: success once {res['success_once']:.0%} "
                          f"at end {res['success_at_end']:.0%}", flush=True)
                shape = (n_goals, args.episodes)
                obs = np.concatenate(obs_all)[:n].reshape(*shape, T, -1)
                act = np.concatenate(act_all)[:n].reshape(*shape, T, -1)
                first = np.concatenate(first_all)[:n].reshape(shape)
                end = np.concatenate(end_all)[:n].reshape(shape)
                np.savez(path, obs=obs.astype(np.float32), actions=act.astype(np.float32), first_success_step=first,
                         success_at_end=end, goal_xy=goals.numpy(), goal_quadrant=quadrant.numpy())
                meta = dict(split=split, teacher=name, goals=n_goals, episodes_per_goal=args.episodes, steps=T,
                            env_id=bc.SHARED_ENV_ID, quadrants=list(SPLITS[split]), goal_seed=GOAL_SEED[split],
                            demo_seed_base=DEMO_SEED_BASE[split], noise_std=0.0,
                            success_once=float((first > 0).mean()), success_at_end=float(end.mean()),
                            teacher_checkpoint=str(args.ppo_teacher if name == "ppo" else args.bc_teacher))
                path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
                print(json.dumps(meta), flush=True)
    finally:
        env.close()


def build_support(data, teacher, goal_ids, k, stride, seed=0):
    """Deterministic support for closed-loop evaluation: the first k episodes (successful first)
    of each listed goal for one teacher. Returns [len(goal_ids), k*P, d] states and actions."""
    dev = data.obs.device
    ti = torch.full((len(goal_ids),), data.teachers.index(teacher), device=dev)
    gi = torch.as_tensor(goal_ids, device=dev)
    order = data.episode_order(ti, gi, torch.Generator(device=dev).manual_seed(seed))
    return data.support(ti, gi, order[:, :k], stride)


def closed_loop(args):
    bc.configure_env(bc.SHARED_ENV_ID)
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    stride = args.stride
    results = {"checkpoint": str(args.checkpoint), "stride": stride, "starts_per_goal": args.starts, "splits": {}}
    G = args.goals
    # One simulator for all splits: ManiSkill's GPU physics is set up once per process.
    env = bc.make_env(G * args.starts)
    T = bc.episode_length(env)
    teachers = load_teachers(args, env, device)
    for split in args.splits:
        data = PromptData.load(args.data_dir, split, device=device)
        if data.G < G:
            raise ValueError(f"{split} has {data.G} goals; --goals {G} needs at least that many")
        goal_ids = torch.arange(G).repeat_interleave(args.starts)
        reset = {"goal_xy": data.goal_xy[goal_ids].float().to(device)}
        seed = EVAL_SEED_BASE[split]
        rows = {}
        for name, policy in teachers.items():
            rows[f"teacher_{name}"] = bc.rollout(env, T, policy, seed, reset_options=reset)
        wrong_ids = (goal_ids + G // 2) % G   # another goal of the same split, same teacher
        for teacher in args.context_teachers:
            for k in args.ks:
                for cond in (("none",) if k == 0 else ("correct", "wrong")):
                    ids = wrong_ids if cond == "wrong" else goal_ids
                    s, a = build_support(data, teacher, ids.tolist(), k, stride)
                    res = bc.rollout(env, T, ICLActor(model, s, a), seed, reset_options=reset)
                    rows[f"{teacher}_k{k}_{cond}"] = res
                    print(f"{split:<8} {teacher}-context k={k} {cond:<7} once {res['success_once']:.0%} "
                          f"end {res['success_at_end']:.0%}", flush=True)
        for name in teachers:
            r = rows[f"teacher_{name}"]
            print(f"{split:<8} {name} teacher (sees goal)  once {r['success_once']:.0%} end {r['success_at_end']:.0%}", flush=True)
        results["splits"][split] = {"goals": G, "episodes": G * args.starts, "seed": seed, "results": rows}
        if args.out:  # save after every split, so a disconnect keeps finished splits
            Path(args.out).write_text(json.dumps(results, indent=2))
    env.close()
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
    return results


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    teachers = argparse.ArgumentParser(add_help=False)
    teachers.add_argument("--ppo-teacher", required=True)
    teachers.add_argument("--bc-teacher", required=True)

    c = sub.add_parser("collect", parents=[teachers])
    c.add_argument("--splits", nargs="+", default=list(SPLITS))
    c.add_argument("--goals", type=int, nargs="+", default=[2048, 128, 128, 128], help="goals per split, same order")
    c.add_argument("--episodes", type=int, default=10, help="demonstrations per goal and teacher")
    c.add_argument("--num-envs", type=int, default=1024)
    c.add_argument("--out", required=True)
    c.add_argument("--overwrite", action="store_true")
    c.add_argument("--device", default=default_device)

    t = sub.add_parser("train")
    t.add_argument("--data-dir", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--ks", type=int, nargs="+", default=[0, 1, 2, 4, 8], help="support episodes per prompt, sampled per batch")
    t.add_argument("--stride", type=int, default=4, help="keep every stride-th step of a support episode")
    t.add_argument("--iters", type=int, default=100_000)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--weight-decay", type=float, default=0.0)
    t.add_argument("--warmup", type=int, default=1000)
    t.add_argument("--n-layer", type=int, default=12)
    t.add_argument("--n-embd", type=int, default=256)
    t.add_argument("--n-head", type=int, default=8)
    t.add_argument("--log-every", type=int, default=250)
    t.add_argument("--eval-every", type=int, default=2500)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    t.add_argument("--device", default=default_device)

    o = sub.add_parser("offline")
    o.add_argument("--checkpoint", required=True)
    o.add_argument("--data-dir", required=True)
    o.add_argument("--splits", nargs="+", default=["val", "test_in", "test_q4"])
    o.add_argument("--ks", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    o.add_argument("--stride", type=int, default=4)
    o.add_argument("--batches", type=int, default=8)
    o.add_argument("--out", default="")
    o.add_argument("--device", default=default_device)

    e = sub.add_parser("eval", parents=[teachers])
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--data-dir", required=True)
    e.add_argument("--splits", nargs="+", default=["test_in", "test_q4"])
    e.add_argument("--context-teachers", nargs="+", default=list(TEACHERS))
    e.add_argument("--ks", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    e.add_argument("--stride", type=int, default=4)
    e.add_argument("--goals", type=int, default=128, help="goals per split (at most the collected number)")
    e.add_argument("--starts", type=int, default=4, help="closed-loop episodes per goal, from new starts")
    e.add_argument("--out", default="")
    e.add_argument("--device", default=default_device)

    args = p.parse_args(argv)
    if args.cmd == "collect" and len(args.goals) != len(args.splits):
        p.error("--goals needs one number per split")
    {"collect": collect, "train": train, "offline": offline, "eval": closed_loop}[args.cmd](args)


if __name__ == "__main__":
    main()

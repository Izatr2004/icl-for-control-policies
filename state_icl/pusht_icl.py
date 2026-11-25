"""In-context goal inference on variable-goal Push-T: PPO-only pilot (Jimmy).

Follows CODEX_CONTEXT.md (variable-goal experiment) and PushT_ICL_Transformer_Spec.md (backbone and
token layout). The learner never sees the goal: prompts are demonstrations from one expert pushing to
one hidden goal, plus a query state from a separate episode with the same expert and goal; the label is
the expert's action at that state. This pilot uses only the PPO teacher (state_rl); the BC teacher's
prompts can be added later with the same code.

Subcommands
  collect  roll out the frozen PPO teacher on fixed goals; one .npz per split
  train    train the GPT-2-style learner on train-split prompts (Q1-Q3 goals); select on val (Q1-Q3)
  eval     frozen evaluation: offline action error and closed-loop success with correct-goal,
           wrong-goal and no context, plus the expert (PPO with the true goal) on the same starts

Splits (target positions; starts are anywhere, including Q4):
  train Q1-Q3 | val Q1-Q3 (separate goals) | test_id Q1-Q3 (separate goals) | test_ood Q4
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "state_rl"))
import pusht_vargoal as pv  # noqa: E402  (registers PushTVarGoal-v1)
from pusht_state_ppo import load_actor, make_env  # noqa: E402

OBS_DIM = 31
GOAL_COLS = (21, 22, 23)                                     # extra/goal_pos, removed for the learner
STATE_COLS = tuple(i for i in range(OBS_DIM) if i not in GOAL_COLS)
STATE_FIELDS = ["agent/qpos[7]", "agent/qvel[7]", "extra/tcp_pose[7]", "extra/obj_pose[7]"]
ACT_DIM = 7
T_EP = 100

SPLITS = {   # quadrants, goals per quadrant, goal/rollout seed
    "train":    dict(quadrants=(1, 2, 3), goals_per_quadrant=100, seed=11),
    "val":      dict(quadrants=(1, 2, 3), goals_per_quadrant=20, seed=12),
    "test_id":  dict(quadrants=(1, 2, 3), goals_per_quadrant=20, seed=13),
    "test_ood": dict(quadrants=(4,), goals_per_quadrant=60, seed=14),
}


# ======================================================================================
# Running jobs: (goal, episode) pairs in batches of the env's size
# ======================================================================================

@torch.no_grad()
def run_jobs(env, job_goals, policy, seed, record=False):
    """Run one 100-step episode per job. job_goals: [J, 2] (cpu). policy(obs, job_idx) -> actions.

    Jobs are processed in consecutive chunks of env.num_envs (the last chunk is padded and the padding
    discarded). Chunk c resets with seed + c, so the same jobs, env size and seed give the same starts
    whatever the policy (ManiSkill seeds torch on reset; the goals are given, not sampled).
    """
    u = env.unwrapped
    n = u.num_envs
    J = job_goals.shape[0]
    out = {k: [] for k in ("success_once", "success_at_end", "final_overlap", "max_overlap", "obs0")}
    if record:
        out["obs"], out["actions"] = [], []
    for c, start in enumerate(range(0, J, n)):
        idx = torch.arange(start, min(start + n, J))
        pad = torch.cat([idx, idx[-1:].expand(n - len(idx))])
        obs, _ = env.reset(seed=seed + c, options=dict(goal_xy=job_goals[pad].to(u.device)))
        assert torch.allclose(obs[:, 21:23].float().cpu(), job_goals[pad], atol=1e-5), "goal not applied"
        out["obs0"].append(obs[:len(idx)].float().cpu())
        so = torch.zeros(n, dtype=torch.bool, device=obs.device)
        mx = torch.zeros(n, device=obs.device)
        O, A = [], []
        for t in range(T_EP):
            a = policy(obs.float(), pad.to(obs.device)).clamp(-1.0, 1.0)
            if record:
                O.append(obs.float()); A.append(a)
            obs, _, _, _, info = env.step(a)
            s = info["success"].bool()
            so |= s
            ov = u.pseudo_render_intersection().float()
            mx = torch.maximum(mx, ov)
        keep = slice(0, len(idx))
        out["success_once"].append(so[keep].cpu()); out["success_at_end"].append(s[keep].cpu())
        out["final_overlap"].append(ov[keep].cpu()); out["max_overlap"].append(mx[keep].cpu())
        if record:
            out["obs"].append(torch.stack(O, 1)[keep].cpu()); out["actions"].append(torch.stack(A, 1)[keep].cpu())
    return {k: torch.cat(v) for k, v in out.items()}


# ======================================================================================
# collect
# ======================================================================================

def split_goals(quadrants, goals_per_quadrant, seed):
    torch.manual_seed(seed)
    return torch.cat([pv.sample_goals(goals_per_quadrant, (q,)) for q in quadrants]).cpu()


def cmd_collect(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = make_env(args.num_envs, sim_backend=args.sim_backend)
    actor = load_actor(args.teacher, device=env.unwrapped.device)
    meta = {"teacher": str(args.teacher), "episodes_per_goal": args.episodes, "splits": {}}
    for name in args.splits:
        cfg = dict(SPLITS[name])
        if args.goals_per_quadrant:
            cfg["goals_per_quadrant"] = args.goals_per_quadrant
        goals = split_goals(cfg["quadrants"], cfg["goals_per_quadrant"], cfg["seed"])
        G, E = len(goals), args.episodes
        t0 = time.time()
        r = run_jobs(env, goals.repeat_interleave(E, 0), lambda o, j: actor.act(o), seed=cfg["seed"] * 100_000,
                     record=True)
        quad = pv.quadrant_of(goals).numpy()
        np.savez_compressed(
            out / f"{name}.npz",
            obs=r["obs"].reshape(G, E, T_EP, OBS_DIM).numpy().astype(np.float32),
            actions=r["actions"].reshape(G, E, T_EP, ACT_DIM).numpy().astype(np.float32),
            goal_xy=goals.numpy().astype(np.float32), goal_quadrant=quad.astype(np.int64),
            success_once=r["success_once"].reshape(G, E).numpy(),
            success_at_end=r["success_at_end"].reshape(G, E).numpy(),
            final_overlap=r["final_overlap"].reshape(G, E).numpy().astype(np.float32))
        so = r["success_once"].float().reshape(G, E)
        meta["splits"][name] = {**{k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()},
                                "goals": G, "episodes": G * E, "success_once": so.mean().item(),
                                "goals_with_4plus_successes": int((so.sum(1) >= 4).sum()),
                                "seconds": round(time.time() - t0, 1)}
        print(f"{name:>8}: {G} goals x {E} episodes  teacher success-once {so.mean():.1%}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    (out / "collect_meta.json").write_text(json.dumps(meta, indent=2))


# ======================================================================================
# Learner (PushT_ICL_Transformer_Spec.md sections 2, 4-6)
# ======================================================================================

class ICLPolicy(nn.Module):
    def __init__(self, d_s, d_a, H=1, n_positions=1024):
        super().__init__()
        from transformers import GPT2Config, GPT2Model
        self.d_s, self.d_a, self.H = d_s, d_a, H
        self.config = GPT2Config(
            vocab_size=1, n_positions=n_positions, n_embd=256, n_layer=12, n_head=8, n_inner=1024,
            activation_function="gelu_new", resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
            layer_norm_epsilon=1e-5, initializer_range=0.02, scale_attn_weights=True,
            scale_attn_by_inverse_layer_idx=False, reorder_and_upcast_attn=False,
            add_cross_attention=False, use_cache=False, bos_token_id=None, eos_token_id=None)
        self.backbone = GPT2Model(self.config)
        self.backbone.wte.weight.requires_grad_(False)
        self.state_encoder = nn.Linear(d_s, 256)
        self.action_encoder = nn.Linear(H * d_a, 256)
        self.action_head = nn.Linear(256, H * d_a)
        for layer in (self.state_encoder, self.action_encoder, self.action_head):
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            nn.init.zeros_(layer.bias)

    def forward(self, support_states, support_actions, query_states):
        """support_states [B,N,d_s], support_actions [B,N,H*d_a] or [B,N,H,d_a], query_states [B,d_s]."""
        B, N = support_states.shape[:2]
        query = self.state_encoder(query_states).unsqueeze(1)
        if N:
            s = self.state_encoder(support_states)
            a = self.action_encoder(support_actions.reshape(B, N, -1))
            tokens = torch.cat((torch.stack((s, a), dim=2).flatten(1, 2), query), dim=1)
        else:
            tokens = query
        if tokens.shape[1] > self.config.n_positions:
            raise ValueError("Context exceeds the configured token budget")
        hidden = self.backbone(inputs_embeds=tokens).last_hidden_state
        return self.action_head(hidden[:, -1]).reshape(B, self.H, self.d_a)


def count_params(m):
    return sum(p.numel() for p in m.parameters()), sum(p.numel() for p in m.parameters() if p.requires_grad)


# ======================================================================================
# Episode banks and prompts
# ======================================================================================

class Bank:
    """One split's episodes on the device, states without goal fields, normalized with train stats."""

    def __init__(self, path, device, stats=None):
        d = np.load(path)
        states = d["obs"][..., list(STATE_COLS)]                       # [G,E,T,28]
        if stats is None:
            flat = states.reshape(-1, states.shape[-1]).astype(np.float64)
            stats = {"mean": flat.mean(0).tolist(), "std": np.maximum(flat.std(0), 1e-6).tolist()}
        self.stats = stats
        mean = np.asarray(stats["mean"], np.float32)
        std = np.asarray(stats["std"], np.float32)
        self.states = torch.as_tensor((states - mean) / std, device=device)
        self.actions = torch.as_tensor(d["actions"], device=device)    # already in [-1, 1]; not renormalized
        self.goal_xy = d["goal_xy"]
        self.goal_quadrant = d["goal_quadrant"]
        self.success = d["success_once"]
        self.G, self.E, self.T = states.shape[:3]
        self.good = [np.flatnonzero(self.success[g]) for g in range(self.G)]
        self.device = device

    def gather(self, g, sup, qe, qt, stride):
        """g [B], sup [B,k] episode ids, qe [B], qt [B] (numpy) -> support/query tensors."""
        B, k = sup.shape
        ts = np.arange(0, self.T, stride)
        L = len(ts)
        if k:
            gi = torch.as_tensor(np.repeat(g, k * L), device=self.device)
            ei = torch.as_tensor(np.repeat(sup.reshape(-1), L), device=self.device)
            ti = torch.as_tensor(np.tile(ts, B * k), device=self.device)
            ss = self.states[gi, ei, ti].reshape(B, k * L, -1)
            sa = self.actions[gi, ei, ti].reshape(B, k * L, -1)
        else:
            ss = self.states.new_zeros(B, 0, self.states.shape[-1])
            sa = self.actions.new_zeros(B, 0, self.actions.shape[-1])
        g_, e_, t_ = (torch.as_tensor(x, device=self.device) for x in (g, qe, qt))
        return ss, sa, self.states[g_, e_, t_], self.actions[g_, e_, t_]


def sample_train_prompts(bank, B, k, rng, stride):
    """Support: k successful episodes of a goal; query: any other episode of that goal, any step."""
    eligible = np.flatnonzero([len(x) >= k and bank.E > k for x in bank.good])
    g = rng.choice(eligible, B)
    sup = np.zeros((B, k), np.int64)
    qe = np.zeros(B, np.int64)
    for i in range(B):
        s = rng.choice(bank.good[g[i]], k, replace=False) if k else np.zeros(0, np.int64)
        sup[i] = s
        qe[i] = rng.choice(np.setdiff1d(np.arange(bank.E), s))
    qt = rng.integers(0, bank.T, B)
    return bank.gather(g, sup, qe, qt, stride)


def eval_support_order(bank, n_support_eps, seed):
    """Per goal, a fixed order of successful episodes among the first n_support_eps (support pool)."""
    rng = np.random.default_rng(seed)
    order = []
    for g in range(bank.G):
        pool = [e for e in bank.good[g] if e < n_support_eps]
        order.append(list(rng.permutation(pool)))
    return order


def wrong_goal_map(G):
    """Each goal's wrong-goal partner: a different goal of the same split (shift by G // 2)."""
    assert G >= 2, "wrong-goal context needs at least 2 goals in the split"
    return (np.arange(G) + max(1, G // 2)) % G


def context_for_goals(bank, goals, order, k, stride):
    """Support tensors [len(goals), k*L, .] built from the first k episodes of each goal's order."""
    sup = np.array([order[g][:k] for g in goals], np.int64).reshape(len(goals), k)
    dummy = np.zeros(len(goals), np.int64)
    ss, sa, _, _ = bank.gather(np.asarray(goals), sup, dummy, dummy, stride)
    return ss, sa


@torch.no_grad()
def offline_action_error(model, bank, ks, stride, n_support_eps, seed, batch=512):
    """Query-action MSE on episodes >= n_support_eps (all steps), context from episodes < n_support_eps."""
    order = eval_support_order(bank, n_support_eps, seed)
    wrong = wrong_goal_map(bank.G)
    res = {}
    for k in ks:
        ok = [g for g in range(bank.G) if len(order[g]) >= k and len(order[wrong[g]]) >= k]
        for cond in (("correct", "wrong") if k else ("none",)):
            se, n = 0.0, 0
            for g in ok:
                src = g if cond != "wrong" else wrong[g]
                ss, sa = context_for_goals(bank, [src], order, k, stride)
                qs = bank.states[g, n_support_eps:].reshape(-1, bank.states.shape[-1])
                qa = bank.actions[g, n_support_eps:].reshape(-1, ACT_DIM)
                for i in range(0, len(qs), batch):
                    q = qs[i:i + batch]
                    p = model(ss.expand(len(q), -1, -1), sa.expand(len(q), -1, -1), q)[:, 0]
                    se += (p - qa[i:i + batch]).pow(2).sum().item()
                    n += q.numel()
            res[f"{cond}_k{k}"] = {"mse": se / max(n, 1), "goals": len(ok)}
    return res


# ======================================================================================
# train
# ======================================================================================

def cmd_train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)
    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train = Bank(data / "train.npz", dev)
    val = Bank(data / "val.npz", dev, train.stats)
    L = len(range(0, T_EP, args.stride))
    assert 2 * max(args.ks) * L + 1 <= 1024, "context does not fit in 1024 tokens"
    model = ICLPolicy(len(STATE_COLS), ACT_DIM).to(dev)
    total, trainable = count_params(model)
    print(f"params total {total:,} trainable {trainable:,}", flush=True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    step, history, best = 0, [], None
    if args.resume:
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        step, history, best = ck["step"], ck["history"], ck["best"]
        rng = np.random.default_rng(args.seed + step)
        print(f"resumed at step {step} (model + optimizer)", flush=True)
    import transformers
    config = {k: v for k, v in vars(args).items() if k != "func"} | {
        "state_cols": list(STATE_COLS), "state_fields": STATE_FIELDS, "removed_goal_cols": list(GOAL_COLS),
        "action": "pd_joint_delta_pos, 7-d, already in [-1, 1]; not renormalized; clipped at execution",
        "pairs_per_demo": L, "params_total": total, "params_trainable": trainable,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "teachers": ["PPO (state_rl/teachers/ppo_vargoal_seed1_50M.pt)"], "selection_ks": args.select_ks}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    (out / "state_stats.json").write_text(json.dumps(train.stats))

    def save(path, with_opt):
        blob = {"model": model.state_dict(), "step": step, "config": config, "stats": train.stats,
                "history": history, "best": best}
        if with_opt:
            blob["optimizer"] = opt.state_dict()
        torch.save(blob, path)

    t0, last_log, losses = time.time(), time.time(), []
    model.train()
    while step < args.steps:
        k = int(rng.choice(args.ks))
        ss, sa, qs, qa = sample_train_prompts(train, args.batch, k, rng, args.stride)
        loss = nn.functional.mse_loss(model(ss, sa, qs)[:, 0], qa)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        step += 1
        losses.append(loss.item())
        if time.time() - last_log > 60 or step == args.steps:
            print(f"step {step}  loss {np.mean(losses[-200:]):.4f}  {step / (time.time() - t0):.1f} it/s",
                  flush=True)
            last_log = time.time()
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            v = offline_action_error(model, val, args.ks, args.stride, args.n_support_eps, seed=0)
            model.train()
            score = float(np.mean([v[f"correct_k{k}"]["mse"] for k in args.select_ks]))
            history.append({"step": step, "train_loss": float(np.mean(losses[-args.eval_every:])),
                            "val_select_mse": score, **{c: r["mse"] for c, r in v.items()}})
            line = "  ".join(f"{c} {r['mse']:.4f}" for c, r in v.items())
            print(f"[val] step {step}  select {score:.4f}  {line}", flush=True)
            if best is None or score < best["val_select_mse"]:
                best = {"step": step, "val_select_mse": score}
                save(out / "best.pt", with_opt=False)
            save(out / "last.pt", with_opt=True)
            (out / "history.json").write_text(json.dumps(history, indent=2))
        if args.max_minutes and time.time() - t0 > 60 * args.max_minutes:
            print(f"time limit reached at step {step}", flush=True)
            save(out / "last.pt", with_opt=True)
            break
    print("best:", best, flush=True)


# ======================================================================================
# eval
# ======================================================================================

def load_learner(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = ICLPolicy(len(STATE_COLS), ACT_DIM).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), ck


def summarize(r, quads):
    s = {m: r[m].float().mean().item() for m in ("success_once", "success_at_end", "final_overlap", "max_overlap")}
    s["episodes"] = int(len(r["success_once"]))
    for q in sorted(set(quads.tolist())):
        m = quads == q
        s[f"Q{q}_success_once"] = r["success_once"][m].float().mean().item()
    return s


def cmd_eval(args):
    dev = torch.device(args.device)
    model, ck = load_learner(args.learner, dev)
    cfg = ck["config"]
    stride, ks, nsup = cfg["stride"], args.ks, cfg["n_support_eps"]
    env = make_env(args.num_envs, sim_backend=args.sim_backend)
    teacher = load_actor(args.teacher, device=env.unwrapped.device)
    mean = torch.tensor(ck["stats"]["mean"], device=dev)
    std = torch.tensor(ck["stats"]["std"], device=dev)
    sc = torch.tensor(STATE_COLS, device=dev)
    results = {"learner": str(args.learner), "learner_step": ck["step"], "stride": stride,
               "n_support_eps": nsup, "starts_per_goal": args.starts, "splits": {}}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        bank = Bank(Path(args.data) / f"{split}.npz", dev, ck["stats"])
        order = eval_support_order(bank, nsup, seed=0)
        wrong = wrong_goal_map(bank.G)
        offline = offline_action_error(model, bank, ks, stride, nsup, seed=0)
        goals = [g for g in range(bank.G) if min(len(order[g]), len(order[wrong[g]])) >= max(ks)]
        job_goal_idx = np.repeat(goals, args.starts)
        job_goals = torch.as_tensor(bank.goal_xy[job_goal_idx])
        quads = torch.as_tensor(bank.goal_quadrant[job_goal_idx])
        seed = 10_000_000 + SPLITS[split]["seed"] * 100_000
        rows = {}

        def learner_policy(ss, sa):
            def act(obs, j):
                q = (obs.to(dev)[:, sc] - mean) / std
                jj = j.to(dev)
                return model(ss[jj], sa[jj], q)[:, 0].to(obs.device)
            return act

        ref = run_jobs(env, job_goals, lambda o, j: teacher.act(o), seed)
        rows["expert_true_goal"] = summarize(ref, quads)
        for k in ks:
            for cond in (("correct", "wrong") if k else ("none",)):
                src = job_goal_idx if cond != "wrong" else wrong[job_goal_idx]
                ss, sa = context_for_goals(bank, src, order, k, stride)
                r = run_jobs(env, job_goals, learner_policy(ss, sa), seed)
                assert torch.allclose(r["obs0"], ref["obs0"], atol=1e-4), "starts differ between conditions"
                rows[f"{cond}_k{k}"] = summarize(r, quads)
                rows[f"{cond}_k{k}"]["gap_to_expert_success_once"] = (
                    rows["expert_true_goal"]["success_once"] - rows[f"{cond}_k{k}"]["success_once"])
                print(f"{split:>8} {cond:>7} k={k}  success {rows[f'{cond}_k{k}']['success_once']:.1%} "
                      f"(at end {rows[f'{cond}_k{k}']['success_at_end']:.1%})  "
                      f"overlap {rows[f'{cond}_k{k}']['final_overlap']:.2f}  "
                      f"action MSE {offline[f'{cond}_k{k}']['mse']:.4f}", flush=True)
        print(f"{split:>8}  expert (true goal)  success {rows['expert_true_goal']['success_once']:.1%}", flush=True)
        results["splits"][split] = {"goals_evaluated": len(goals), "closed_loop": rows, "action_mse": offline}
        (out / "eval_results.json").write_text(json.dumps(results, indent=2))


# ======================================================================================

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect")
    c.add_argument("--teacher", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--splits", nargs="+", default=list(SPLITS))
    c.add_argument("--episodes", type=int, default=16, help="episodes per goal")
    c.add_argument("--goals-per-quadrant", type=int, default=0, help="override (tests)")
    c.add_argument("--num-envs", type=int, default=1024)
    c.add_argument("--sim-backend", default="physx_cuda")
    c.set_defaults(func=cmd_collect)

    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--steps", type=int, default=50_000)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--stride", type=int, default=4, help="keep every stride-th step of a demo")
    t.add_argument("--ks", type=int, nargs="+", default=[0, 1, 2, 4, 8], help="demos per prompt")
    t.add_argument("--select-ks", type=int, nargs="+", default=[1, 2, 4, 8])
    t.add_argument("--n-support-eps", type=int, default=10, help="eval: support pool = episodes < this")
    t.add_argument("--eval-every", type=int, default=2000)
    t.add_argument("--max-minutes", type=float, default=0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--resume", default=None)
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval")
    e.add_argument("--learner", required=True)
    e.add_argument("--teacher", required=True)
    e.add_argument("--data", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--splits", nargs="+", default=["val", "test_id", "test_ood"])
    e.add_argument("--ks", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    e.add_argument("--starts", type=int, default=8, help="rollout starts per goal")
    e.add_argument("--num-envs", type=int, default=512)
    e.add_argument("--sim-backend", default="physx_cuda")
    e.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    e.set_defaults(func=cmd_eval)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

"""Regenerate every figure and table for the ICL transformer v1 results from the committed result files.

Usage (from the repo root; needs only numpy and matplotlib, no GPU, no simulator):
    python state_bc/make_figures.py
    python state_bc/make_figures.py --run state_bc/results/transformer_v1_seed0 --out state_bc/results/transformer_v1_seed0/figures

Inputs (all in --run, copied from the Colab run's Drive folder):
    train_log.jsonl   validation action MSE every 2,500 steps (written by `pusht_icl.py train`)
    notebook.log      full notebook output; training loss every 500 steps is parsed from it
    action_error.json teacher-forced action MSE of best.pt on val / test_in / test_q4 (`pusht_icl.py offline`)
    closed_loop.json  closed-loop success of best.pt on test_in / test_q4 (`pusht_icl.py eval`)
    config.json       training arguments

Outputs (in --out): fig_training.{pdf,png}, fig_success_vs_k.{pdf,png}, fig_action_error_vs_k.{pdf,png},
tables.md (readable) and tables.tex (booktabs, for the paper).

Error bars on success rates are 95% Wilson score intervals for a binomial proportion with n = episodes per cell (512).
They treat episodes as independent. Episodes share goals (128 goals x 4 starts), and they come from one training seed,
so the intervals understate the full uncertainty; see the README for the seed caveat.
"""
import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
KS = [0, 1, 2, 4, 8]
SPLIT_NAMES = {"val": "Validation (Q1–Q3)", "test_in": "Test, Q1–Q3 goals", "test_q4": "Test, Q4 goals (held out)"}
TEACHER_NAMES = {"ppo": "PPO", "bc": "BC"}
BLUE, ORANGE, RED, GREY, INK = "#1f6feb", "#d4730b", "#cf222e", "#6e7781", "#1f2328"


def wilson(p, n, z=1.96):
    """95% Wilson score interval for a binomial proportion p observed over n trials."""
    if n == 0:
        return 0.0, 0.0
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def parse_train_loss(log_path, run_name):
    """Training loss lines ('iter N loss X lr Y ...') of the full run in notebook.log (the pilot is skipped)."""
    rows, active = [], False
    for line in Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("$ ") and " train " in line:
            active = run_name in line
            continue
        m = re.match(r"iter (\d+) loss ([0-9.eE+-]+) lr ([0-9.eE+-]+)", line)
        if active and m:
            rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    return rows


def style():
    plt.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GREY,
        "savefig.bbox": "tight", "savefig.dpi": 300, "font.family": "DejaVu Sans",
    })


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)


def fig_training(train_loss, val_log, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    it = [r[0] / 1000 for r in train_loss]
    a1.plot(it, [r[1] for r in train_loss], color=INK, lw=1.0)
    a1.set_yscale("log")
    a1.set_xlabel("training step (thousands)")
    a1.set_ylabel("training loss (action MSE)")
    a1.set_title("(a) Training loss, mean over the last 500 steps", loc="left")
    lr = a1.twinx()
    lr.plot(it, [r[2] * 1e4 for r in train_loss], color=GREY, lw=0.8, ls=":")
    lr.set_ylabel("learning rate (×1e-4)", color=GREY)
    lr.tick_params(axis="y", colors=GREY)
    lr.spines["right"].set_visible(True)
    lr.set_ylim(0, 1.05)

    vi = [r["iter"] / 1000 for r in val_log]
    a2.plot(vi, [r["val"]["4"]["correct"] for r in val_log], color=BLUE, lw=1.4, label="k = 4, correct-goal demos")
    a2.plot(vi, [r["val"]["1"]["correct"] for r in val_log], color=BLUE, lw=0.9, ls="--", label="k = 1, correct-goal demos")
    a2.plot(vi, [r["val"]["0"]["none"] for r in val_log], color=GREY, lw=1.4, label="no demos (k = 0)")
    a2.plot(vi, [r["val"]["4"]["wrong"] for r in val_log], color=RED, lw=1.4, label="k = 4, wrong-goal demos")
    a2.set_yscale("log")
    a2.set_xlabel("training step (thousands)")
    a2.set_ylabel("validation action MSE")
    a2.set_title("(b) Validation, 128 unseen Q1–Q3 goals", loc="left")
    a2.legend(frameon=False, loc="center right")
    fig.tight_layout(w_pad=3)
    save(fig, out, "fig_training")


def fig_success_vs_k(cl, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    x = list(range(len(KS)))
    for ax, split in zip(axes, ("test_in", "test_q4")):
        r = cl["splits"][split]["results"]
        n = cl["splits"][split]["episodes"]
        for t, col, dx in (("ppo", BLUE, -0.09), ("bc", ORANGE, 0.09)):
            for metric, ls, mk in (("success_once", "-", "o"), ("success_at_end", "--", "s")):
                ys, lo, hi = [], [], []
                for k in KS:
                    cell = r[f"{t}_k0_none"] if k == 0 else r[f"{t}_k{k}_correct"]
                    p = cell[metric]
                    a, b = wilson(p, n)
                    ys.append(100 * p), lo.append(100 * (p - a)), hi.append(100 * (b - p))
                label = f"{TEACHER_NAMES[t]} demos, {'once' if metric == 'success_once' else 'at end'}"
                ax.errorbar([xi + dx for xi in x], ys, yerr=[lo, hi], color=col, ls=ls, marker=mk, ms=3, lw=1.1,
                            capsize=1.8, elinewidth=0.8, label=label)
            wrong = [100 * r[f"{t}_k{k}_wrong"]["success_once"] for k in KS[1:]]
            ax.plot([xi + dx for xi in x[1:]], wrong, color=RED, marker="x", ms=3.5, lw=0.8,
                    ls="-" if t == "ppo" else ":", label=f"{TEACHER_NAMES[t]} demos, wrong goal (once)")
        for t, ls in (("ppo", "-"), ("bc", ":")):
            ax.axhline(100 * r[f"teacher_{t}"]["success_once"], color=GREY, lw=0.8, ls=ls)
            ax.axhline(100 * r[f"teacher_{t}"]["success_at_end"], color=GREY, lw=0.8, ls=ls, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["none"] + [str(k) for k in KS[1:]])
        ax.set_xlabel("demonstrations in the prompt (k)")
        ax.set_ylim(-3, 103)
        ax.set_title(f"({'a' if split == 'test_in' else 'b'}) {SPLIT_NAMES[split]}", loc="left")
    axes[0].set_ylabel("closed-loop success (%)")
    axes[0].text(4.35, 100 * cl["splits"]["test_in"]["results"]["teacher_ppo"]["success_once"] + 1.5, "teachers",
                 color=GREY, fontsize=6.5, ha="right")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(w_pad=1.5)
    save(fig, out, "fig_success_vs_k")


def fig_action_error(ae, out):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ks = KS[1:]
    for split, col in (("test_in", BLUE), ("test_q4", ORANGE)):
        s = ae[split]
        ax.plot(ks, [s[str(k)]["correct"] for k in ks], color=col, marker="o", ms=3, lw=1.3, label=f"{SPLIT_NAMES[split]}, correct")
        ax.plot(ks, [s[str(k)]["wrong"] for k in ks], color=col, marker="x", ms=3.5, lw=0.9, ls="--", label=f"{SPLIT_NAMES[split]}, wrong")
        ax.axhline(s["0"]["none"], color=col, lw=0.8, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_yscale("log")
    ax.set_xlabel("demonstrations in the prompt (k)")
    ax.set_ylabel("action MSE (teacher-forced)")
    ax.set_title("Action error vs k (dotted: no demos)", loc="left")
    ax.legend(frameon=False, fontsize=6, loc="center right")
    fig.tight_layout()
    save(fig, out, "fig_action_error_vs_k")


def pct_ci(p, n):
    a, b = wilson(p, n)
    return f"{100 * p:.1f} [{100 * a:.1f}, {100 * b:.1f}]"


def tables(cl, ae, cfg, train_loss, out):
    md, tex = [], []

    # Table 1: main closed-loop result at k = 4.
    md += ["## Table 1: closed-loop success at k = 4 (%, 95% Wilson interval, 512 episodes per cell)", "",
           "| Goals | Demos from | Metric | Correct-goal demos | Wrong-goal demos | No demos | Teacher (sees goal) |",
           "|---|---|---|---|---|---|---|"]
    tex += [r"% Table 1: closed-loop success at k = 4", r"\begin{tabular}{lllcccc}", r"\toprule",
            r"Goals & Demos & Metric & Correct goal & Wrong goal & No demos & Teacher \\", r"\midrule"]
    for split in ("test_in", "test_q4"):
        r, n = cl["splits"][split]["results"], cl["splits"][split]["episodes"]
        for t in ("ppo", "bc"):
            for metric, mname in (("success_once", "once"), ("success_at_end", "at end")):
                cells = [r[f"{t}_k4_correct"][metric], r[f"{t}_k4_wrong"][metric], r[f"{t}_k0_none"][metric],
                         r[f"teacher_{t}"][metric]]
                md.append(f"| {SPLIT_NAMES[split]} | {TEACHER_NAMES[t]} | {mname} | " + " | ".join(pct_ci(c, n) for c in cells) + " |")
                tex.append(f"{SPLIT_NAMES[split].replace('–', '--')} & {TEACHER_NAMES[t]} & {mname} & "
                           + " & ".join(f"{100 * c:.1f}" for c in cells) + r" \\")
        if split == "test_in":
            tex.append(r"\midrule")
    tex += [r"\bottomrule", r"\end{tabular}", ""]

    # Table 2: closed-loop success for every k (appendix).
    md += ["", "## Table 2: closed-loop success-once / success-at-end (%) for every k, correct-goal demos", "",
           "| Goals | Demos from | none | k = 1 | k = 2 | k = 4 | k = 8 | wrong goal, k = 1–8 (once) | Teacher |",
           "|---|---|---|---|---|---|---|---|---|"]
    tex += [r"% Table 2: success-once / success-at-end for every k", r"\begin{tabular}{llcccccc}", r"\toprule",
            r"Goals & Demos & none & $k{=}1$ & $k{=}2$ & $k{=}4$ & $k{=}8$ & Teacher \\", r"\midrule"]
    for split in ("test_in", "test_q4"):
        r = cl["splits"][split]["results"]
        for t in ("ppo", "bc"):
            f = lambda c: f"{100 * c['success_once']:.0f} / {100 * c['success_at_end']:.0f}"  # noqa: E731
            cells = [f(r[f"{t}_k0_none"])] + [f(r[f"{t}_k{k}_correct"]) for k in KS[1:]]
            wrong = [100 * r[f"{t}_k{k}_wrong"]["success_once"] for k in KS[1:]]
            md.append(f"| {SPLIT_NAMES[split]} | {TEACHER_NAMES[t]} | " + " | ".join(cells)
                      + f" | {min(wrong):.1f}–{max(wrong):.1f} | {f(r[f'teacher_{t}'])} |")
            tex.append(f"{SPLIT_NAMES[split].replace('–', '--')} & {TEACHER_NAMES[t]} & " + " & ".join(cells)
                       + f" & {f(r[f'teacher_{t}'])} " + r"\\")
    tex += [r"\bottomrule", r"\end{tabular}", ""]

    # Table 3: teacher-forced action error.
    md += ["", "## Table 3: teacher-forced action MSE of best.pt (correct-goal / wrong-goal demos)", "",
           "| Split | none | k = 1 | k = 2 | k = 4 | k = 8 |", "|---|---|---|---|---|---|"]
    tex += [r"% Table 3: teacher-forced action MSE", r"\begin{tabular}{lccccc}", r"\toprule",
            r"Split & none & $k{=}1$ & $k{=}2$ & $k{=}4$ & $k{=}8$ \\", r"\midrule"]
    for split in ("val", "test_in", "test_q4"):
        s = ae[split]
        cells = [f"{s['0']['none']:.4f}"] + [f"{s[str(k)]['correct']:.4f} / {s[str(k)]['wrong']:.3f}" for k in KS[1:]]
        md.append(f"| {SPLIT_NAMES[split]} | " + " | ".join(cells) + " |")
        tex.append(f"{SPLIT_NAMES[split].replace('–', '--')} & " + " & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", ""]

    # Table 4: hyperparameters (values from config.json; the 'why' column is the paper's justification).
    hp = [
        ("Layers / width / heads", f"{cfg['n_layer']} / {cfg['n_embd']} / {cfg['n_head']}", "GPT-2 configuration of Garg et al. (2022)"),
        ("Parameters", "9.75M", "follows from the above"),
        ("Optimizer", f"AdamW, lr {cfg['lr']:g}, weight decay {cfg['weight_decay']:g}", "lr and batch size as in Garg et al. (2022)"),
        ("Batch size", str(cfg["batch_size"]), "as in Garg et al. (2022)"),
        ("Schedule", f"linear warmup {cfg['warmup']} steps, cosine to 10%", "standard warmup + cosine decay; not tuned"),
        ("Training steps", f"{cfg['iters']:,}", "compute budget (~3.6 h on one L4); validation MSE still improving at the end"),
        ("Context sizes k (uniform)", ", ".join(map(str, cfg["ks"])), "covers no context to 8 demos (401 tokens)"),
        ("Snippet stride", f"every {cfg['stride']}th step, {cfg['pairs_per_episode']} pairs per demo", "fits k = 8 in the 1,024-token context"),
        ("Training goals / demos per goal", f"{cfg['train_goals']:,} / {cfg['episodes_per_goal']}", "Q1–Q3 goals; 2 teachers"),
        ("Precision", "bf16 autocast, gradient clip 1.0", "speed on an L4 GPU"),
        ("Model selection", "lowest mean validation MSE (correct, k = 0)", "selected step 100,000"),
    ]
    md += ["", "## Table 4: transformer hyperparameters", "", "| Setting | Value | Why |", "|---|---|---|"]
    md += [f"| {a} | {b} | {c} |" for a, b, c in hp]
    tex += [r"% Table 4: hyperparameters", r"\begin{tabular}{lll}", r"\toprule", r"Setting & Value & Reason \\", r"\midrule"]
    esc = lambda s: s.replace("%", "\\%")  # noqa: E731
    tex += [f"{a} & {esc(b)} & {esc(c)} " + r"\\" for a, b, c in hp]
    tex += [r"\bottomrule", r"\end{tabular}", ""]

    md += ["", f"Training loss: {train_loss[0][1]:.4f} at step {train_loss[0][0]:,}, {train_loss[-1][1]:.5f} at step "
           f"{train_loss[-1][0]:,} ({len(train_loss)} logged points).",
           "Intervals are 95% Wilson score intervals with n = 512 episodes, treating episodes as independent; they do not",
           "include variation across training seeds (one seed so far)."]
    (out / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "tables.tex").write_text("\n".join(tex), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, default=HERE / "results" / "transformer_v1_seed0")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.run / "figures"
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((args.run / "config.json").read_text())
    val_log = [json.loads(line) for line in (args.run / "train_log.jsonl").read_text().splitlines() if line.strip()]
    ae = json.loads((args.run / "action_error.json").read_text())
    cl = json.loads((args.run / "closed_loop.json").read_text())
    train_loss = parse_train_loss(args.run / "notebook.log", Path(cfg["out"]).name)
    assert train_loss and train_loss[-1][0] == cfg["iters"], "training loss not found for the full run in notebook.log"

    style()
    fig_training(train_loss, val_log, out)
    fig_success_vs_k(cl, out)
    fig_action_error(ae, out)
    tables(cl, ae, cfg, train_loss, out)
    print("wrote", ", ".join(sorted(p.name for p in out.iterdir())), "to", out)


if __name__ == "__main__":
    main()

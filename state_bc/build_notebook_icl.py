"""Build notebooks/Train_PushT_ICL_Transformer.ipynb from the state_bc modules.

Trains the v1 in-context learner (CODEX_CONTEXT.md, PushT_ICL_Transformer_Spec.md) on demonstrations from the two
frozen full-state teachers (Jimmy's PPO and the run 4 BC) in PushTVarGoal-v1. Jimmy's environment file and arm-pose
library are embedded from his branch (private repo); the teachers are read from the run 4 folder on Drive.
Rerun after changing pusht_icl.py, pusht_state_bc.py or the cells below:
    python state_bc/build_notebook_icl.py
"""

import base64
import hashlib
import subprocess
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = REPO / "notebooks" / "Train_PushT_ICL_Transformer.ipynb"
JIMMY_COMMIT = "c87d8ae3669c10894d765df2d0dd4b61c010e272"


def git_show(path: str) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), "show", f"{JIMMY_COMMIT}:{path}"],
                          check=True, capture_output=True).stdout


cells = []
md = lambda s: cells.append(new_markdown_cell(s.strip()))  # noqa: E731
code = lambda s: cells.append(new_code_cell(s.strip()))  # noqa: E731

md("""
# Push-T in-context learning transformer (v1)

**Question:** can a frozen transformer infer an unfamiliar Push-T goal from demonstrations and push the T there, without ever
seeing the goal's coordinates and without weight updates? (`CODEX_CONTEXT.md`.) See `state_bc/README.md` for the log.

**What it is trained on.** Prompts built from demonstrations by the two frozen full-state teachers in `PushTVarGoal-v1`:
Jimmy's goal-conditioned PPO and the BC teacher distilled from it (run 4), half the prompts each. One prompt = one teacher and
one goal: `k` demonstration episodes of that teacher reaching that goal from different starts (each subsampled every 4th step,
ending at the final step), then a state from **another** episode of the same teacher and goal. The target is the teacher's action at that
state. Every state the transformer sees has the goal fields (indices 21:24) removed; the demonstrations are its only clue.
Training goals are in Q1–Q3 only; Q4 goals are held out for the final test.

| Split | Goals | Use |
|---|---|---|
| `train` | 2,048 in Q1–Q3 | training prompts |
| `val` | 128 in Q1–Q3 (unseen) | checkpoint selection |
| `test_in` | 128 in Q1–Q3 (unseen) | final test, in distribution |
| `test_q4` | 128 in Q4 | final test, held-out quadrant |

10 noise-free demonstrations per goal and teacher. Context sizes `k` ∈ {0, 1, 2, 4, 8} episodes (up to 200 state-action pairs,
401 tokens). Model: GPT-2, 12 layers, width 256, 8 heads (9.75M parameters), query-action MSE, AdamW 1e-4, batch 64.

**Steps:** 1–4 setup → 5 demonstrations (cached on Drive, about 10 minutes the first time) → 6 **pilot** (3,000 steps, then checks that
correct-goal demonstrations beat wrong-goal ones) → 7 full training (resumes automatically if Colab disconnects: just rerun
the notebook) → 8 final evaluation.

Select **Runtime → Change runtime type → L4 GPU**. Upload only this notebook.
""")

code(f"""
from pathlib import Path
import os, sys, json, re, subprocess, datetime, shlex, shutil, base64, hashlib

MANISKILL_COMMIT = "a4a4f9272ad64b1564035874b605ceb687b63ed8"  # v3.0.1, same as the teacher notebooks
MS_ROOT = Path("/content/ManiSkill")
WORK_ROOT = Path("/content/pusht_state_bc")
WORK_ROOT.mkdir(parents=True, exist_ok=True)
DRIVE_BASE = Path("/content/drive/MyDrive/182_final_project/state_bc")
RUN_LABEL = "icl_transformer_v1"
JIMMY_COMMIT = "{JIMMY_COMMIT}"

# Frozen teachers (run 4 session on Drive)
TEACHER_DIR = DRIVE_BASE / "20261123T230947Z_run4_bc_from_jimmy" / "teachers"
PPO_TEACHER = TEACHER_DIR / "ppo_expert.pt"    # Jimmy's ppo_vargoal_seed1_50M.pt
BC_TEACHER = TEACHER_DIR / "bc_teacher.pt"     # distilled BC, run 4

# Demonstrations: cached on Drive and reused by later sessions
DATA_TAG = "icl_data_v1"
SPLITS = ["train", "val", "test_in", "test_q4"]
GOALS = [2048, 128, 128, 128]
EPISODES_PER_GOAL = 10

# Transformer
KS = [0, 1, 2, 4, 8]          # demonstration episodes per prompt
STRIDE = 4                    # keep every 4th step of a demonstration (25 pairs per episode)
BATCH_SIZE = 64
LR = 1e-4
PILOT_ITERS = 3_000
ITERS = 100_000
EVAL_EVERY = 2_500
SEED = 0
TRAIN_DIR = DRIVE_BASE / "icl_runs" / f"transformer_v1_seed{{SEED}}"   # fixed path, so reruns resume

# Final closed-loop evaluation
EVAL_GOALS = 128
EVAL_STARTS = 4               # new starts per goal (512 episodes per condition)

SESSION = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + RUN_LABEL
subprocess.run(["nvidia-smi"], check=True)
""")

md("## 1. Drive and logging\n\nThis session's logs and evaluations go to a new folder; demonstrations and the training run use fixed folders so they are reused.")
code("""
from google.colab import drive
drive.mount("/content/drive")
OUTPUT_ROOT = DRIVE_BASE / SESSION
OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
LOG = OUTPUT_ROOT / "notebook.log"
for p in (PPO_TEACHER, BC_TEACHER):
    assert p.exists(), f"Missing teacher {p}: check TEACHER_DIR"
print("Session folder:", OUTPUT_ROOT)


def run(command, show=None, **kwargs):
    \"\"\"Run a command and append all output to notebook.log. Prints every line, or only lines
    matching the regex `show`. On failure, raises with the last 40 lines.\"\"\"
    command = list(map(str, command))
    print("$", shlex.join(command), flush=True)
    pattern = re.compile(show) if show else None
    tail = []
    with LOG.open("a") as log:
        log.write("\\n$ " + shlex.join(command) + "\\n")
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1, **kwargs) as process:
            try:
                for line in process.stdout:
                    if pattern is None or pattern.search(line):
                        print(line, end="", flush=True)
                    log.write(line)
                    tail = (tail + [line.rstrip()])[-40:]
                code = process.wait()
            except BaseException:
                process.kill()
                raise
    if code:
        raise RuntimeError(f"Command failed (exit {code}). Log: {LOG}\\n" + "\\n".join(tail))
""")

md("## 2. Install ManiSkill v3.0.1 under Python 3.11, plus Hugging Face Transformers 4.46.3")
code("""
run(["apt-get", "update", "-qq"])
run(["apt-get", "install", "-y", "-qq", "libvulkan1", "libvulkan-dev", "vulkan-tools", "libglvnd-dev"])
if not (MS_ROOT / ".git").exists():
    run(["git", "clone", "https://github.com/mani-skill/ManiSkill.git", MS_ROOT])
run(["git", "-C", MS_ROOT, "checkout", "--detach", MANISKILL_COMMIT])
assert subprocess.check_output(["git", "-C", str(MS_ROOT), "rev-parse", "HEAD"], text=True).strip() == MANISKILL_COMMIT

run([sys.executable, "-m", "pip", "install", "-q", "uv==0.9.11"])
TRAIN_ENV = WORK_ROOT / "python311"
TRAIN_PY = TRAIN_ENV / "bin/python"
if not TRAIN_PY.exists():
    run([sys.executable, "-m", "uv", "venv", "--python", "3.11", "--seed", TRAIN_ENV])
constraints = WORK_ROOT / "constraints.txt"
constraints.write_text("numpy==1.26.4\\nopencv-python==4.11.0.86\\nsapien==3.0.3\\ntorch==2.6.0\\n")
run([TRAIN_PY, "-m", "pip", "install", "-q", "torch==2.6.0", "--index-url", "https://download.pytorch.org/whl/cu124"])
run([TRAIN_PY, "-m", "pip", "install", "-q", "--constraint", constraints, "-e", MS_ROOT,
     "numpy", "sapien", "h5py", "tyro", "imageio[ffmpeg]", "matplotlib", "transformers==4.46.3"])

for filename, library, vulkan in [
    ("/usr/share/vulkan/icd.d/nvidia_icd.json", "libGLX_nvidia.so.0", True),
    ("/usr/share/glvnd/egl_vendor.d/10_nvidia.json", "libEGL_nvidia.so.0", False),
]:
    path = Path(filename)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        icd = {"library_path": library}
        if vulkan:
            icd["api_version"] = "1.2.155"
        path.write_text(json.dumps({"file_format_version": "1.0.0", "ICD": icd}))
os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"
os.environ["MS_SKIP_ASSET_DOWNLOAD_PROMPT"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
run([TRAIN_PY, "-c", "import torch, sapien, mani_skill, transformers; print('torch', torch.__version__, 'mani_skill', "
     "mani_skill.__version__, 'transformers', transformers.__version__); assert torch.cuda.is_available(); "
     "print(torch.cuda.get_device_name(0))"])
(OUTPUT_ROOT / "pip-freeze.txt").write_text(subprocess.check_output([str(TRAIN_PY), "-m", "pip", "freeze"], text=True))
""")

md(f"""
## 3. Write the modules

`pusht_icl.py` (transformer, prompts, training, evaluation) and `pusht_state_bc.py` (environment helpers, teacher loaders)
are this repo's `state_bc/` files. `pusht_vargoal.py` and its arm-pose library are copied unchanged from Jimmy's branch at
`{JIMMY_COMMIT[:7]}`, checked by SHA-256.
""")
for name in ("pusht_icl.py", "pusht_state_bc.py"):
    cells.append(new_code_cell(f"%%writefile /content/pusht_state_bc/{name}\n"
                               + (HERE / name).read_text(encoding="utf-8").rstrip() + "\n"))
vargoal = git_show("state_rl/pusht_vargoal.py").decode("utf-8").replace("\r\n", "\n").rstrip()
cells.append(new_code_cell("%%writefile /content/pusht_state_bc/pusht_vargoal.py\n"
                           f"# Copied unchanged from jimmy-state-rl:state_rl/pusht_vargoal.py at {JIMMY_COMMIT[:7]}\n{vargoal}\n"))
npz = git_show("state_rl/assets/tip_start_qpos.npz")
b64 = base64.b64encode(npz).decode("ascii")
wrapped = "\n".join(b64[i:i + 4096] for i in range(0, len(b64), 4096))
code(f'''
# Jimmy's arm-pose library (base64), from jimmy-state-rl at {JIMMY_COMMIT[:7]}
TIP_LIBRARY = ("{hashlib.sha256(npz).hexdigest()}", """
{wrapped}
""")
data = base64.b64decode("".join(TIP_LIBRARY[1].split()))
assert hashlib.sha256(data).hexdigest() == TIP_LIBRARY[0], "tip_start_qpos.npz: SHA-256 mismatch"
(WORK_ROOT / "assets").mkdir(exist_ok=True)
(WORK_ROOT / "assets" / "tip_start_qpos.npz").write_bytes(data)
ICL = WORK_ROOT / "pusht_icl.py"
for name in ("pusht_icl.py", "pusht_state_bc.py", "pusht_vargoal.py"):
    shutil.copy2(WORK_ROOT / name, OUTPUT_ROOT / name)
TEACHERS = ["--ppo-teacher", PPO_TEACHER, "--bc-teacher", BC_TEACHER]
run([TRAIN_PY, ICL, "--help"], show=r"^\\s+(collect|train|offline|eval) ")
''')

md("""
## 4. Demonstrations

Noise-free rollouts of both teachers: for every goal of every split, 10 episodes per teacher from different starts, with the same
goals for both teachers. Saved as `{split}_{teacher}.npz` (states with the goal, so the data stays useful for other experiments;
the goal is removed when prompts are built). Cached on Drive under `icl_data_v1/`; a later session copies them instead of re-collecting.
""")
code("""
LOCAL_DATA = Path("/content") / DATA_TAG
DRIVE_DATA = DRIVE_BASE / DATA_TAG
expected = [f"{s}_{t}.npz" for s in SPLITS for t in ("ppo", "bc")]
if all((DRIVE_DATA / f).exists() for f in expected):
    print("Copying cached demonstrations from", DRIVE_DATA)
    shutil.copytree(DRIVE_DATA, LOCAL_DATA, dirs_exist_ok=True)
else:
    run([TRAIN_PY, ICL, "collect", *TEACHERS, "--splits", *SPLITS, "--goals", *GOALS,
         "--episodes", EPISODES_PER_GOAL, "--out", LOCAL_DATA], show=r"batch \\d+/\\d+|\\"success_once\\"|exists")
    shutil.copytree(LOCAL_DATA, DRIVE_DATA, dirs_exist_ok=True)
for f in expected:
    meta = json.loads((LOCAL_DATA / f).with_suffix(".json").read_text())
    print(f"{f:<18} {meta['goals']:>5} goals x {meta['episodes_per_goal']} episodes  "
          f"success once {meta['success_once']:.0%}  at end {meta['success_at_end']:.0%}")
""")

md("""
## 5. Pilot: 3,000 steps

Measures training speed (it/s and the ETA for the full run are printed) and checks that the model starts to use its context:
on validation goals, action error with **correct-goal** demonstrations should fall below **wrong-goal** and **no** demonstrations.
Then a small closed-loop check (64 validation goals, 2 starts each, PPO demonstrations, k = 0 and 4). Expect weak control this early;
the action-error gap is the signal. Stop here if correct and wrong are indistinguishable, and report back.
""")
code("""
PILOT_DIR = OUTPUT_ROOT / "pilot"
common = ["--data-dir", LOCAL_DATA, "--ks", *KS, "--stride", STRIDE, "--batch-size", BATCH_SIZE, "--lr", LR, "--seed", SEED]
run([TRAIN_PY, ICL, "train", *common, "--out", PILOT_DIR, "--iters", PILOT_ITERS, "--eval-every", 1000, "--log-every", 250],
    show=r"^(train|parameters|iter|Done)")
run([TRAIN_PY, ICL, "eval", *TEACHERS, "--checkpoint", PILOT_DIR / "final.pt", "--data-dir", LOCAL_DATA, "--splits", "val",
     "--context-teachers", "ppo", "--ks", 0, 4, "--stride", STRIDE, "--goals", 64, "--starts", 2,
     "--out", PILOT_DIR / "closed_loop_val.json"], show=r"^val ")
""")

md("""
## 6. Full training

100,000 steps (about 1–3 hours; the pilot's it/s gives the real figure). The run lives in a fixed Drive folder
(`icl_runs/transformer_v1_seed0/`) with `last.pt` saved every 2,500 steps: **if Colab disconnects, rerun the whole notebook** and it continues
where it stopped (demonstrations come from the Drive cache). `best.pt` has the lowest validation action error with correct-goal demonstrations.
""")
code("""
run([TRAIN_PY, ICL, "train", *common, "--out", TRAIN_DIR, "--iters", ITERS, "--eval-every", EVAL_EVERY, "--log-every", 500],
    show=r"^(train|parameters|resumed|iter \\d*(000|500) |Done)")
""")

md("## 7. Training curves\n\nValidation action error per context size: correct-goal (solid) vs wrong-goal (dashed) demonstrations, and no context.")
code("""
import matplotlib.pyplot as plt

rows = [json.loads(l) for l in (TRAIN_DIR / "train_log.jsonl").read_text().splitlines()]
it = [r["iter"] for r in rows]
fig, ax = plt.subplots(figsize=(8, 4))
for i, k in enumerate(map(str, KS)):
    colour = f"C{i}"
    if k == "0":
        ax.plot(it, [r["val"][k]["none"] for r in rows], color=colour, label="k=0 (no context)")
    else:
        ax.plot(it, [r["val"][k]["correct"] for r in rows], color=colour, label=f"k={k} correct")
        ax.plot(it, [r["val"][k]["wrong"] for r in rows], color=colour, ls="--", alpha=0.6)
ax.set(xlabel="iteration", ylabel="validation action MSE", yscale="log", title="Dashed: wrong-goal demonstrations")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUTPUT_ROOT / "training_curves.png", dpi=120)
plt.show()
""")

md("""
## 8. Final evaluation (frozen weights)

`best.pt` on goals it never trained on: `test_in` (Q1–Q3) and `test_q4` (held-out quadrant).
- **Action error** (teacher-forced) per context size, correct-goal vs wrong-goal demonstrations.
- **Closed loop:** 128 goals × 4 new starts per split, for PPO and BC demonstrations, k ∈ {0, 1, 2, 4, 8}, correct-goal vs wrong-goal
  (k = 0 has no context). The teachers themselves (which see the goal) run on the same starts as the reference.
""")
code("""
EVAL_DIR = OUTPUT_ROOT / "final_eval"
EVAL_DIR.mkdir(exist_ok=True)
BEST = TRAIN_DIR / "best.pt"
run([TRAIN_PY, ICL, "offline", "--checkpoint", BEST, "--data-dir", LOCAL_DATA, "--splits", "val", "test_in", "test_q4",
     "--ks", *KS, "--stride", STRIDE, "--out", EVAL_DIR / "action_error.json"])
run([TRAIN_PY, ICL, "eval", *TEACHERS, "--checkpoint", BEST, "--data-dir", LOCAL_DATA, "--splits", "test_in", "test_q4",
     "--ks", *KS, "--stride", STRIDE, "--goals", EVAL_GOALS, "--starts", EVAL_STARTS,
     "--out", EVAL_DIR / "closed_loop.json"], show=r"^test")
""")

code("""
res = json.loads((EVAL_DIR / "closed_loop.json").read_text())
for split, block in res["splits"].items():
    r = block["results"]
    print(f"\\n{split}: success-once / success-at-end, {block['episodes']} episodes per cell")
    print("  teachers (see the goal): " + "   ".join(
        f"{t} {r['teacher_' + t]['success_once']:.0%}/{r['teacher_' + t]['success_at_end']:.0%}" for t in ("ppo", "bc")))
    for t in ("ppo", "bc"):
        line = f"  {t} demos  k=0 none {r[f'{t}_k0_none']['success_once']:.0%}"
        for k in KS[1:]:
            c, w = r[f"{t}_k{k}_correct"], r[f"{t}_k{k}_wrong"]
            line += f" | k={k} correct {c['success_once']:.0%} wrong {w['success_once']:.0%}"
        print(line)
summary = {"session": SESSION, "train_dir": str(TRAIN_DIR), "checkpoint": str(BEST), "data": str(DRIVE_DATA),
           "action_error": json.loads((EVAL_DIR / "action_error.json").read_text()), "closed_loop": res}
(OUTPUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
print("\\nSaved", OUTPUT_ROOT / "summary.json")
""")

nb = new_notebook(cells=cells, metadata={
    "accelerator": "GPU",
    "colab": {"gpuType": "L4", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.validate(nb)
nbformat.write(nb, OUT)
print("Wrote", OUT, f"({OUT.stat().st_size / 1e6:.2f} MB)")

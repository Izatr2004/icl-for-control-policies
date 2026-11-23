"""Build notebooks/Train_PushT_VariableGoal_PPO_BC.ipynb (run 3b: warm-started PPO) from the state_bc modules.

The notebook embeds pusht_state_bc.py and variable_goal_env.py with %%writefile so it stays
self-contained in Colab. Edit those files or the cell text below, then rerun:
    python state_bc/build_notebook.py

Earlier notebooks are kept as run and no longer rebuilt:
run 1 = notebooks/Train_PushT_State_BC.ipynb, run 2 = notebooks/Train_PushT_State_BC_DAgger.ipynb.
"""

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
MODULES = ["pusht_state_bc.py", "variable_goal_env.py"]
OUT = HERE.parent / "notebooks" / "Train_PushT_VariableGoal_PPO_BC.ipynb"

cells = []
md = lambda s: cells.append(new_markdown_cell(s.strip()))  # noqa: E731
code = lambda s: cells.append(new_code_cell(s.strip()))  # noqa: E731

md("""
# Push-T variable goal, run 3b: warm-started goal-conditioned PPO expert → distilled BC teacher

**Run 3b.** Run 3 trained PPO from random weights and stalled at about 4–11% success after 50M transitions, so its 80% gate
stopped before BC. This run changes only the PPO step: it **starts from ManiSkill's released fixed-goal Push-T expert**
(same 31-number state, same joint controller), which already knows how to push the T, and uses **4,096 parallel
environments** as ManiSkill's own Push-T baseline does, with a lower learning rate for fine-tuning. See `state_bc/README.md`.

v1 of the project (`CODEX_CONTEXT.md`) needs teachers that can push the T to **any goal position**, with the goal
in their observation. The transformer will later see demos but never the goal, and will train on Q1–Q3 goals only.
This notebook makes the full-state teachers:

1. **Variable-goal environment** `PushT-VariableGoal-v1` (`variable_goal_env.py`): PushT-v1 with the goal T's
   position random each episode, within ±10 cm of PushT-v1's original goal, in quadrants Q1 (+x,+y), Q2 (−x,+y),
   Q3 (−x,−y), Q4 (+x,−y). Goal rotation fixed. The success check is fixed to use each episode's own goal.
   **The stick tip also starts at a random spot** (±15 cm of the same centre, away from the T), placed by inverse
   kinematics. The movable T spawns as in PushT-v1. Starts are independent of the goal. **Teachers train on all four quadrants.**
2. **Goal-conditioned PPO expert**, fine-tuned here from the released fixed-goal expert with ManiSkill's state PPO baseline
   (Jimmy's PPO can replace it later through `PPO_CHECKPOINT`).
3. **BC teacher distilled from that expert:** noisy expert rollouts plus one training run. **No DAgger** (run 2 showed it was not needed).
4. **Per-quadrant validation and test** of both teachers, then both are saved under `teachers/`.

| Setting | Value |
|---|---|
| Observation / controller | `obs_mode="state"` (31 numbers, now including a varying `goal_pos`), `pd_joint_delta_pos` (7) |
| PPO | ManiSkill `ppo.py`, **warm start from the released `ppo_pd_joint_delta_pos_ckpt.pt`**: 4,096 envs, 16 steps per rollout, 32 minibatches, 8 epochs, γ = 0.99, learning rate 1e-4, 50M transitions, seed 1 |
| BC teacher | MLP 2 × 256 ReLU, tanh output, MSE, Adam 3e-4 cosine, batch 1024, 200k iterations |
| Simulator | ManiSkill v3.0.1 (commit `a4a4f92`), GPU physics |
| Seeds | expert data 100,000+; BC validation data 300,000+; closed-loop validation 1001; test 5001 (once, at the end) |

Select **Runtime → Change runtime type → L4 GPU** and run the cells in order. Upload only this notebook.
Outputs go to `MyDrive/182_final_project/state_bc/<session>/`. PPO is the long step. Estimated total: about 1–2 hours, depending on PPO speed with 4,096 environments (not yet measured).
""")

code("""
from pathlib import Path
import os, sys, json, re, subprocess, datetime, shlex, shutil

MANISKILL_COMMIT = "a4a4f9272ad64b1564035874b605ceb687b63ed8"  # v3.0.1, same as the other notebooks
MS_ROOT = Path("/content/ManiSkill")
WORK_ROOT = Path("/content/pusht_state_bc")
WORK_ROOT.mkdir(parents=True, exist_ok=True)
USE_GOOGLE_DRIVE = True
DRIVE_SUBDIR = "182_final_project/state_bc"
RUN_LABEL = "run3b_warmstart"
ENV_ID = "PushT-VariableGoal-v1"

# PPO expert: fine-tuned from ManiSkill's released fixed-goal expert (run 3b), ManiSkill's Push-T env count
PPO_CHECKPOINT = ""          # set to an existing ppo.py checkpoint path to skip PPO training
PPO_WARM_START = True        # start from the released fixed-goal expert instead of random weights
PPO_LR = 1e-4                # run 3 used 3e-4 from scratch; lower to avoid erasing the pushing skill early
PPO_SEED = 1
PPO_TIMESTEPS = 50_000_000
PPO_NUM_ENVS = 4096          # run 3 used 1,024; ManiSkill's state Push-T baseline uses 4,096
PPO_NUM_STEPS = 16
PPO_MINIBATCHES = 32
PPO_EPOCHS = 8
PPO_EVAL_FREQ = 25           # iterations between ppo.py's own evaluations and checkpoints
PPO_CANDIDATES = 5           # evaluate this many of the latest checkpoints and keep the best
MIN_EXPERT_SUCCESS_ONCE = 0.80   # stop before BC if the expert is weaker than this on validation

# BC teacher
TEACHER_NAME = "bc_variable_goal_seed0"
SEED = 0
ITERS = 200_000
BATCH_SIZE = 1024
LR = 3e-4
EVAL_EVERY = 25_000
EVAL_EPISODES = 200          # 50 per quadrant on average
VAL_SEED = 1001
TEST_SEED = 5001

# Expert rollouts for BC
NUM_ENVS = 1024
EXPERT_EPISODES = 16_384                # 16 batches of 1,024
EXPERT_NOISE = [0.0, 0.1, 0.2, 0.3]     # action-noise std per batch, cycled; labels stay clean
EXPERT_SEED_BASE = 100_000
VAL_EPISODES = 1024                     # noise-free expert rollouts for validation action error
VAL_SEED_BASE = 300_000

SESSION = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + RUN_LABEL
subprocess.run(["nvidia-smi"], check=True)
print("Notebook Python:", sys.version)
""")

md("## 1. Save outputs to Drive\n\nEach session gets a new folder, so reruns never overwrite earlier results.")
code("""
if USE_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    OUTPUT_ROOT = Path("/content/drive/MyDrive") / DRIVE_SUBDIR / SESSION
else:
    OUTPUT_ROOT = WORK_ROOT / "outputs" / SESSION
OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
LOG = OUTPUT_ROOT / "notebook.log"
print("Output directory:", OUTPUT_ROOT)


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

md("""
## 2. Install ManiSkill v3.0.1 under Python 3.11

Same pinned install as runs 1–2, plus TensorBoard, which ManiSkill's PPO script imports.
""")
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
     "numpy", "sapien", "h5py", "tyro", "tensorboard", "imageio[ffmpeg]", "matplotlib"])

# Vulkan/EGL entries Colab is missing; needed for rendering evaluation videos.
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
run([TRAIN_PY, "-c", "import torch, sapien, mani_skill, numpy; print('torch', torch.__version__, 'CUDA', torch.version.cuda); "
     "print('mani_skill', mani_skill.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"])
(OUTPUT_ROOT / "pip-freeze.txt").write_text(subprocess.check_output([str(TRAIN_PY), "-m", "pip", "freeze"], text=True))
PPO_SCRIPT = MS_ROOT / "examples/baselines/ppo/ppo.py"
assert PPO_SCRIPT.is_file()
print("Training Python:", TRAIN_PY)
""")

md("""
## 3. Write the modules

`variable_goal_env.py` defines `PushT-VariableGoal-v1`. `pusht_state_bc.py` holds everything else
(`ppo`, `audit`, `check-env`, `collect`, `train`, `eval`). Repo copies live in `state_bc/`; this notebook is generated
from them by `state_bc/build_notebook.py`.
""")
for name in MODULES:
    source = (HERE / name).read_text(encoding="utf-8").rstrip()
    cells.append(new_code_cell(f"%%writefile /content/pusht_state_bc/{name}\n{source}\n"))
code("""
BC = WORK_ROOT / "pusht_state_bc.py"
for name in ("pusht_state_bc.py", "variable_goal_env.py"):
    shutil.copy2(WORK_ROOT / name, OUTPUT_ROOT / name)
run([TRAIN_PY, BC, "--help"])
""")

md("""
## 4. Check the environment

`audit` records the observation contract (now with a varying goal). `check-env` runs two checks **before** any time goes into PPO:
- **Success check:** places a stand-in T exactly on each episode's goal and confirms the success test counts it, then shifts it
  5 cm and confirms it does not. Catches a success check still pointing at the old fixed goal.
- **Tip starts:** compares where the simulator actually put the stick tip with where inverse kinematics aimed it (must be
  within 2 cm), requires at most 5% of episodes to fall back to PushT-v1's start, and reports tip height, distance to the T,
  and start quadrants.
""")
code("""
run([TRAIN_PY, BC, "audit", "--env-id", ENV_ID, "--out", OUTPUT_ROOT])
contract = json.loads((OUTPUT_ROOT / "observation_contract.json").read_text())
assert contract["control_mode"] == "pd_joint_delta_pos" and contract["state_dim"] == 31, contract
run([TRAIN_PY, BC, "check-env", "--env-id", ENV_ID, "--num-envs", 256, "--out", OUTPUT_ROOT / "env_check.json"])
""")

md("""
## 5. Train the goal-conditioned PPO expert (warm start)

Downloads ManiSkill's released fixed-goal Push-T expert and fine-tunes it on the variable-goal task with ManiSkill's
`ppo.py`, unchanged, run with the variable-goal environment registered (`--checkpoint` loads the released weights before
training). The first evaluations show how well the fixed-goal expert already does on moving goals. Only evaluation lines are shown;
everything is in `notebook.log`. Checkpoints land in `ppo/runs/<name>/`. Set `PPO_CHECKPOINT` in the config cell to skip this.
`eval_success_once_mean` is ppo.py's own 16-episode evaluation, a rough progress signal.
""")
code("""
PPO_DIR = OUTPUT_ROOT / "ppo"
PPO_DIR.mkdir(exist_ok=True)
PPO_RUN = f"ppo_variable_goal_warmstart_seed{PPO_SEED}" if PPO_WARM_START else f"ppo_variable_goal_seed{PPO_SEED}"
warm_start = []
if PPO_WARM_START and not PPO_CHECKPOINT:
    import urllib.request
    RELEASED = WORK_ROOT / "ppo_pd_joint_delta_pos_ckpt.pt"
    if not RELEASED.exists():
        urllib.request.urlretrieve("https://huggingface.co/datasets/haosulab/ManiSkill_Demonstrations/resolve/main/"
                                   "demos/PushT-v1/rl/ppo_pd_joint_delta_pos_ckpt.pt", RELEASED)
    shutil.copy2(RELEASED, PPO_DIR / "released_fixed_goal_expert.pt")
    # Baseline: the fixed-goal expert on moving goals, before any fine-tuning.
    run([TRAIN_PY, BC, "eval", "--env-id", ENV_ID, "--checkpoint", RELEASED, "--ppo", "--episodes", 200,
         "--seed", 1001, "--out", OUTPUT_ROOT / "released_expert_on_variable_goal.json"], show=r"success_(once|at_end)")
    warm_start = ["--checkpoint", RELEASED]
if not PPO_CHECKPOINT:
    run([TRAIN_PY, BC, "ppo", "--script", PPO_SCRIPT, *warm_start, "--learning-rate", PPO_LR,
         "--env-id", ENV_ID, "--control-mode", "pd_joint_delta_pos", "--exp-name", PPO_RUN, "--seed", PPO_SEED,
         "--num-envs", PPO_NUM_ENVS, "--num-steps", PPO_NUM_STEPS, "--num-minibatches", PPO_MINIBATCHES,
         "--update-epochs", PPO_EPOCHS, "--gamma", 0.99, "--total-timesteps", PPO_TIMESTEPS,
         "--num-eval-envs", 16, "--num-eval-steps", 100, "--eval-freq", PPO_EVAL_FREQ,
         "--save-model", "--no-track", "--no-capture-video"],
        show=r"eval_success|eval_return|SPS|Epoch: \\d*00,|rror|Traceback", cwd=PPO_DIR)
""")

md("""
## 6. Pick the PPO expert on validation seeds, per quadrant

Evaluates the final checkpoint and the latest periodic ones on 200 validation episodes, keeps the best
(success-once, then success-at-end), and stops if it is below `MIN_EXPERT_SUCCESS_ONCE`.
""")
code("""
def show_quadrants(name, res):
    row = "  ".join(f"{q} {v['success_once']:.0%}/{v['success_at_end']:.0%} (n={v['episodes']})"
                    for q, v in sorted(res.get("by_goal_quadrant", {}).items()))
    print(f"{name:<28} once {res['success_once']:.0%}  end {res['success_at_end']:.0%}   {row}")


if PPO_CHECKPOINT:
    candidates = [Path(PPO_CHECKPOINT)]
else:
    run_dir = PPO_DIR / "runs" / PPO_RUN
    periodic = sorted(run_dir.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    candidates = [run_dir / "final_ckpt.pt"] + periodic[-(PPO_CANDIDATES - 1):]
expert_evals = {}
for ckpt in candidates:
    out = OUTPUT_ROOT / f"ppo_val_{ckpt.stem}.json"
    run([TRAIN_PY, BC, "eval", "--env-id", ENV_ID, "--checkpoint", ckpt, "--ppo",
         "--episodes", EVAL_EPISODES, "--seed", VAL_SEED, "--out", out], show=r"^$")
    expert_evals[ckpt] = json.loads(out.read_text())
    show_quadrants(ckpt.name, expert_evals[ckpt])
EXPERT = max(expert_evals, key=lambda c: (expert_evals[c]["success_once"], expert_evals[c]["success_at_end"]))
TEACHERS = OUTPUT_ROOT / "teachers"
TEACHERS.mkdir(exist_ok=True)
shutil.copy2(EXPERT, TEACHERS / "ppo_expert.pt")
EXPERT_SOURCE = str(EXPERT)
print("Selected expert:", EXPERT)
if expert_evals[EXPERT]["success_once"] < MIN_EXPERT_SUCCESS_ONCE:
    raise RuntimeError(f"Expert success-once {expert_evals[EXPERT]['success_once']:.0%} is below "
                       f"{MIN_EXPERT_SUCCESS_ONCE:.0%}. Train PPO longer or check the environment before distilling BC.")
EXPERT = TEACHERS / "ppo_expert.pt"
""")

md("""
## 7. Collect expert data

16,384 expert episodes (about 1.6M transitions) with action noise cycling 0, 0.1, 0.2, 0.3 per batch, plus 1,024
noise-free episodes on separate seeds for validation action error. Goals cover all four quadrants. Labels are always
the expert's noise-free action.
""")
code("""
DATA_DIR = OUTPUT_ROOT / "data"
EXPERT_DATA, VAL_DATA = DATA_DIR / "expert_rollouts.npz", DATA_DIR / "expert_val.npz"
run([TRAIN_PY, BC, "collect", "--env-id", ENV_ID, "--expert", EXPERT, "--driver", "expert",
     "--episodes", EXPERT_EPISODES, "--num-envs", NUM_ENVS, "--noise-std", *EXPERT_NOISE,
     "--seed-base", EXPERT_SEED_BASE, "--out", EXPERT_DATA])
run([TRAIN_PY, BC, "collect", "--env-id", ENV_ID, "--expert", EXPERT, "--driver", "expert",
     "--episodes", VAL_EPISODES, "--num-envs", NUM_ENVS, "--noise-std", 0.0,
     "--seed-base", VAL_SEED_BASE, "--out", VAL_DATA])
""")

md("## 8. Smoke test\n\nShort BC run to catch problems before the main training.")
code("""
run([TRAIN_PY, BC, "train", "--env-id", ENV_ID, "--train-data", EXPERT_DATA, "--val-data", VAL_DATA,
     "--out", OUTPUT_ROOT / "smoke", "--seed", SEED, "--iters", 2000, "--log-every", 500,
     "--eval-every", 1000, "--eval-episodes", 32, "--val-seed", VAL_SEED])
""")

md("""
## 9. Train the BC teacher (no DAgger)

One training run on the expert rollouts. Logs validation action error every 1,000 iterations and closed-loop validation
success every `EVAL_EVERY`. `best.pt` has the highest validation success-at-end (ties broken by success-once).
""")
code("""
RUN_DIR = OUTPUT_ROOT / "runs" / TEACHER_NAME
config = dict(teacher=TEACHER_NAME, env_id=ENV_ID, seed=SEED, iters=ITERS, batch_size=BATCH_SIZE, lr=LR,
              eval_every=EVAL_EVERY, eval_episodes=EVAL_EPISODES, val_seed=VAL_SEED, test_seed=TEST_SEED,
              expert=str(EXPERT), expert_episodes=EXPERT_EPISODES, expert_noise=EXPERT_NOISE, dagger=False,
              maniskill_commit=MANISKILL_COMMIT, architecture="MLP 2x256 ReLU, tanh output")
RUN_DIR.mkdir(parents=True, exist_ok=False)
(RUN_DIR / "run_config.json").write_text(json.dumps(config, indent=2))
run([TRAIN_PY, BC, "train", "--env-id", ENV_ID, "--train-data", EXPERT_DATA, "--val-data", VAL_DATA, "--out", RUN_DIR,
     "--seed", SEED, "--iters", ITERS, "--batch-size", BATCH_SIZE, "--lr", LR, "--eval-every", EVAL_EVERY,
     "--eval-episodes", EVAL_EPISODES, "--val-seed", VAL_SEED], show=r"iter +\\d*[05]000 |Training on|Done")
shutil.copy2(RUN_DIR / "best.pt", TEACHERS / "bc_teacher.pt")
""")

md("## 10. Training curves")
code("""
import csv
import matplotlib.pyplot as plt

rows = list(csv.DictReader((RUN_DIR / "train_log.csv").open()))
it = [int(r["iter"]) for r in rows]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))
ax1.plot(it, [float(r["train_mse"]) for r in rows], label="train")
ax1.plot(it, [float(r["val_mse"]) for r in rows], label="validation")
ax1.set(xlabel="iteration", ylabel="action MSE", yscale="log", title="Action error")
ax1.legend()
ev = [r for r in rows if r["success_at_end"]]
ax2.plot([int(r["iter"]) for r in ev], [float(r["success_once"]) for r in ev], "o-", label="success once")
ax2.plot([int(r["iter"]) for r in ev], [float(r["success_at_end"]) for r in ev], "s-", label="success at end")
ax2.set(xlabel="iteration", ylabel="rate", ylim=(0, 1.02), title="Closed-loop validation")
ax2.legend()
fig.tight_layout()
fig.savefig(RUN_DIR / "training_curves.png", dpi=120)
plt.show()
""")

md("""
## 11. Final test on fresh seeds, per quadrant

Both teachers on the same 200 test episodes (seed 5001, never used for selection), broken down by goal quadrant.
Also saves a few BC videos.
""")
code("""
tests = {}
for name, ckpt, flags in (("ppo_expert", EXPERT, ["--ppo"]), ("bc_teacher", TEACHERS / "bc_teacher.pt", [])):
    out = TEACHERS / f"{name}_test_eval.json"
    run([TRAIN_PY, BC, "eval", "--env-id", ENV_ID, "--checkpoint", ckpt, *flags, "--episodes", EVAL_EPISODES,
         "--seed", TEST_SEED, "--out", out], show=r"^$")
    tests[name] = json.loads(out.read_text())
    show_quadrants(name, tests[name])
run([TRAIN_PY, BC, "eval", "--env-id", ENV_ID, "--checkpoint", TEACHERS / "bc_teacher.pt", "--episodes", 4,
     "--seed", TEST_SEED, "--video-dir", RUN_DIR / "videos"], show=r"^$")

summary = {"env_id": ENV_ID, "expert_checkpoint_source": EXPERT_SOURCE,
           "expert_validation": {str(k): v for k, v in expert_evals.items()},
           "test": tests, "teachers": sorted(p.name for p in TEACHERS.glob("*.pt"))}
(TEACHERS / "summary.json").write_text(json.dumps(summary, indent=2))
print("Teachers saved in", TEACHERS)
""")

md("## 12. Watch a rollout")
code("""
from IPython.display import Video, display
videos = sorted((RUN_DIR / "videos").glob("*.mp4"))
print(len(videos), "videos")
if videos:
    display(Video(str(videos[0]), embed=True, width=480))
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
print("Wrote", OUT)

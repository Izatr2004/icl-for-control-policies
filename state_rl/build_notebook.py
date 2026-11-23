"""Build notebooks/Train_PushT_State_PPO_VarGoal.ipynb (Kaggle) from the state_rl modules.

The notebook embeds pusht_vargoal.py, pusht_state_ppo.py, run_ppo.py and the tip-start
library, so it runs standalone. Edit those files or the cell text below, then rerun:
    python state_rl/build_notebook.py
"""
import base64
import hashlib
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=25_000_000, help="PPO total_timesteps")
ap.add_argument("--label", default="vargoal_pilot", help="run label (output folder suffix)")
ap.add_argument("--name", default="Train_PushT_State_PPO_VarGoal", help="notebook file name, no extension")
CLI = ap.parse_args()

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "notebooks" / f"{CLI.name}.ipynb"
MODULES = ["pusht_vargoal.py", "pusht_state_ppo.py", "run_ppo.py"]
LIB = HERE / "assets" / "tip_start_qpos.npz"
LIB_B64 = base64.b64encode(LIB.read_bytes()).decode()
LIB_SHA = hashlib.sha256(LIB.read_bytes()).hexdigest()

cells = []
SUBS = {"__MSTEPS__": str(CLI.steps // 1_000_000), "__STEPS__": f"{CLI.steps:_}", "__LABEL__": CLI.label}


def fill(text):
    for key, value in SUBS.items():
        text = text.replace(key, value)
    return text.strip()


md = lambda s: cells.append(new_markdown_cell(fill(s)))  # noqa: E731
code = lambda s: cells.append(new_code_cell(fill(s)))  # noqa: E731

md("""
# Push-T full-state PPO teacher, goal-conditioned (Jimmy)

Trains the goal-conditioned PPO teacher for the variable-goal experiment (`CODEX_CONTEXT.md`):
random block, stick-tip and goal positions across quadrants Q1–Q4, with the goal coordinates in the observation.
Nathan's goal-conditioned BC teacher is distilled from this one. See `state_rl/README.md` for the log.

| Setting | Value |
|---|---|
| Environment | `PushTVarGoal-v1` = ManiSkill `PushT-v1` with per-episode goals (`state_rl/pusht_vargoal.py`) |
| Observation / controller | `obs_mode="state"` (31 numbers, same layout as state_bc), `pd_joint_delta_pos` (7 numbers) |
| Episode | 100 steps at 20 Hz (unchanged) |
| Training | ManiSkill's **unchanged** `ppo.py` with its PushT state recipe: 1024 envs, 100-step rollouts, 8 epochs, 32 minibatches, gamma 0.99 |
| Budget | **__MSTEPS__M steps** (pilot was 25M: test 56–64% per quadrant, still rising) |
| Evaluation | deterministic mean action clipped to [-1, 1]; 100 episodes per quadrant; validation seed 1001, test seed 5001 (once) |
| Freeze bar | success-once ≥ 80% in **every** quadrant on test seeds |
| Simulator | ManiSkill v3.0.1 (commit `a4a4f92`), GPU physics |

**Kaggle setup:** Settings → Accelerator → **GPU T4** (one GPU is used), Settings → **Internet on**
(needs a phone-verified account). Upload only this notebook.
For the long run use **Save Version → Save & Run All (Commit)**: it runs in the background for up to 12 hours,
and everything in `/kaggle/working` is kept as the version's output. Interactive sessions lose files when they stop.
""")

code("""
from pathlib import Path
import os, sys, json, subprocess, datetime, shlex, shutil, base64, hashlib

MANISKILL_COMMIT = "a4a4f9272ad64b1564035874b605ceb687b63ed8"  # v3.0.1, same as state_bc and camera PPO
ON_KAGGLE = Path("/kaggle/working").exists()
BASE = Path("/kaggle/working") if ON_KAGGLE else Path("/content")
MS_ROOT = Path("/tmp/ManiSkill")          # outside /kaggle/working so it is not saved as output
WORK_ROOT = Path("/tmp/pusht_state_ppo")
WORK_ROOT.mkdir(parents=True, exist_ok=True)
RUN_LABEL = "__LABEL__"

# PPO (ManiSkill's PushT state recipe; only the env id, eval settings and budget differ)
SEED = 1
TOTAL_TIMESTEPS = __STEPS__
NUM_ENVS = 1024
NUM_STEPS = 100
UPDATE_EPOCHS = 8
NUM_MINIBATCHES = 32
GAMMA = 0.99
EVAL_FREQ = 10            # iterations between eval + checkpoint (1 iteration = 102,400 steps)
NUM_EVAL_ENVS = 64
RESUME_FROM = None        # path to a ppo.py checkpoint to continue from (weights only; fresh optimizer)

# Per-quadrant evaluation and checkpoint selection
EVAL_LAST = 10            # evaluate the last N checkpoints on validation seeds
EVAL_EPISODES = 100       # per quadrant
VAL_SEED = 1001
TEST_SEED = 5001
FREEZE_BAR = 0.80

SESSION = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + RUN_LABEL
OUTPUT_ROOT = BASE / "state_rl" / SESSION
OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
LOG = OUTPUT_ROOT / "notebook.log"
subprocess.run(["nvidia-smi"], check=True)
print("Platform:", "Kaggle" if ON_KAGGLE else "other", "| Notebook Python:", sys.version.split()[0])
print("Output directory:", OUTPUT_ROOT)


def run(command, **kwargs):
    \"\"\"Run a command, stream its output here, append it to notebook.log, and raise on failure.\"\"\"
    command = list(map(str, command))
    print("$", shlex.join(command), flush=True)
    tail = []
    with LOG.open("a") as log:
        log.write("\\n$ " + shlex.join(command) + "\\n")
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1, **kwargs) as process:
            try:
                for line in process.stdout:
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
## 1. Install ManiSkill v3.0.1 under Python 3.11

Same pinned install as the team's Colab notebooks (torch 2.6.0 + CUDA 12.4, SAPIEN 3.0.3).
""")
code("""
run(["apt-get", "update", "-qq"])
run(["apt-get", "install", "-y", "-qq", "libvulkan1", "libvulkan-dev", "vulkan-tools", "libglvnd-dev"])
if not (MS_ROOT / ".git").exists():
    run(["git", "clone", "https://github.com/mani-skill/ManiSkill.git", MS_ROOT])
run(["git", "-C", MS_ROOT, "checkout", "--detach", MANISKILL_COMMIT])
assert subprocess.check_output(["git", "-C", str(MS_ROOT), "rev-parse", "HEAD"], text=True).strip() == MANISKILL_COMMIT
PPO_PY = MS_ROOT / "examples/baselines/ppo/ppo.py"

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

# Vulkan/EGL entries the hosted images can be missing; SAPIEN needs them even without video.
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # Kaggle T4 x2: use one GPU
run([TRAIN_PY, "-c", "import torch, sapien, mani_skill, numpy; print('torch', torch.__version__, 'CUDA', torch.version.cuda); "
     "print('mani_skill', mani_skill.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"])
(OUTPUT_ROOT / "pip-freeze.txt").write_text(subprocess.check_output([str(TRAIN_PY), "-m", "pip", "freeze"], text=True))
""")

md("""
## 2. Write the state_rl code

Copies of `state_rl/` in the repo, generated by `state_rl/build_notebook.py`. Do not edit here.
""")
code("""
SRC = WORK_ROOT / "state_rl"
(SRC / "assets").mkdir(parents=True, exist_ok=True)
""")
for name in MODULES:
    text = (HERE / name).read_text(encoding="utf-8").rstrip()
    cells.append(new_code_cell(f"%%writefile /tmp/pusht_state_ppo/state_rl/{name}\n{text}\n"))
code(f"""
# Tip-start pose library (state_rl/assets/tip_start_qpos.npz, built by make_tip_starts.py)
LIB_B64 = "{LIB_B64}"
lib_bytes = base64.b64decode(LIB_B64)
assert hashlib.sha256(lib_bytes).hexdigest() == "{LIB_SHA}"
(SRC / "assets" / "tip_start_qpos.npz").write_bytes(lib_bytes)
CODE_COPY = OUTPUT_ROOT / "code"
shutil.copytree(SRC, CODE_COPY, dirs_exist_ok=True)
TOOL = SRC / "pusht_state_ppo.py"
RUN_PPO = SRC / "run_ppo.py"
run([TRAIN_PY, TOOL, "--help"])
""")

md("""
## 3. Preflight on the GPU simulator

Checks the observation layout, that every environment scores success against **its own** goal,
that partial resets (PPO's auto-reset) only change the reset environments, and the quadrant restriction.
Writes `observation_contract.json`. Stop if anything fails.
""")
code("""
run([TRAIN_PY, TOOL, "preflight", "--out", OUTPUT_ROOT, "--num-envs", 16])
contract = json.loads((OUTPUT_ROOT / "observation_contract.json").read_text())
assert all(contract["preflight"].values()) and contract["max_episode_steps"] == 100
""")

md("## 4. Smoke test: tiny PPO run + evaluation\n\nA few iterations to check the full pipeline before the long run.")
code("""
SMOKE = WORK_ROOT / "smoke"
SMOKE.mkdir(exist_ok=True)
run([TRAIN_PY, RUN_PPO, PPO_PY, "--env_id=PushTVarGoal-v1", "--seed=0", "--num_envs=64", "--num-steps=100",
     "--total_timesteps=12800", "--num_eval_envs=8", "--num_eval_steps=100", "--eval_freq=1",
     "--no-capture-video", "--exp-name=smoke"], cwd=SMOKE)
assert list((SMOKE / "runs/smoke").glob("*.pt")), "smoke run saved no checkpoint"
run([TRAIN_PY, TOOL, "eval", "--run-dir", SMOKE / "runs/smoke", "--last", 1, "--episodes", 8, "--out", SMOKE / "eval"])
""")

md("""
## 5. Train

ManiSkill's `ppo.py`, unchanged. Watch `eval_success_once_mean` (all quadrants mixed) and `SPS`.
Checkpoints and TensorBoard logs go to `runs/<run name>/` in the output folder.
""")
code("""
RUN_NAME = f"vargoal_state_seed{SEED}_{TOTAL_TIMESTEPS // 1_000_000}M"
args = [TRAIN_PY, RUN_PPO, PPO_PY, "--env_id=PushTVarGoal-v1", f"--seed={SEED}",
        f"--num_envs={NUM_ENVS}", f"--num-steps={NUM_STEPS}", f"--update_epochs={UPDATE_EPOCHS}",
        f"--num_minibatches={NUM_MINIBATCHES}", f"--gamma={GAMMA}", f"--total_timesteps={TOTAL_TIMESTEPS}",
        f"--num_eval_envs={NUM_EVAL_ENVS}", "--num_eval_steps=100", f"--eval_freq={EVAL_FREQ}",
        "--no-capture-video", f"--exp-name={RUN_NAME}"]
if RESUME_FROM:
    args.append(f"--checkpoint={RESUME_FROM}")
(OUTPUT_ROOT / "train_command.txt").write_text(shlex.join(map(str, args)))
run(args, cwd=OUTPUT_ROOT)
RUN_DIR = OUTPUT_ROOT / "runs" / RUN_NAME
""")

md("""
## 6. Per-quadrant evaluation, selection and freeze check

Evaluates the last checkpoints on validation seeds (100 episodes per quadrant), selects the one with the best
**worst-quadrant** success-once, then tests it once on fresh seeds against the 80% bar.
""")
code("""
EVAL_DIR = OUTPUT_ROOT / "eval"
run([TRAIN_PY, TOOL, "eval", "--run-dir", RUN_DIR, "--last", EVAL_LAST, "--episodes", EVAL_EPISODES,
     "--val-seed", VAL_SEED, "--test-seed", TEST_SEED, "--bar", FREEZE_BAR, "--out", EVAL_DIR])
sel = json.loads((EVAL_DIR / "selection.json").read_text())
shutil.copy2(sel["selected"], OUTPUT_ROOT / "teacher.pt")
print("\\nTeacher candidate:", OUTPUT_ROOT / "teacher.pt", "<- copy of", Path(sel["selected"]).name)
print("Passes freeze bar:", sel["passes_bar"])
""")

md("""
## 7. Save the results

On Kaggle, outputs persist only in a committed version (Save Version → Save & Run All).
In an interactive session, download the folder below before the session ends.
Log the results in `state_rl/README.md`.
""")
code("""
shutil.make_archive(str(OUTPUT_ROOT), "zip", OUTPUT_ROOT)
print("Folder:", OUTPUT_ROOT)
print("Zip:   ", str(OUTPUT_ROOT) + ".zip")
""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "kaggle": {"accelerator": "gpu", "isInternetEnabled": True},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, OUT)
print("Wrote", OUT)

"""Build notebooks/Train_PushT_State_BC.ipynb from pusht_state_bc.py.

The notebook embeds the module with %%writefile so it stays self-contained in Colab.
Edit pusht_state_bc.py or the cell text below, then rerun:  python state_bc/build_notebook.py
"""

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
MODULE = (HERE / "pusht_state_bc.py").read_text(encoding="utf-8")
OUT = HERE.parent / "notebooks" / "Train_PushT_State_BC.ipynb"

cells = []
md = lambda s: cells.append(new_markdown_cell(s.strip()))  # noqa: E731
code = lambda s: cells.append(new_code_cell(s.strip()))  # noqa: E731

md("""
# Push-T full-state BC teacher

Train a **behavioral cloning (BC) teacher** for ManiSkill `PushT-v1` from the released PPO demonstrations.

| Setting | Value |
|---|---|
| Observation | `obs_mode="state"`: joint positions and velocities, stick-tip pose, goal position, T-block pose (numbers only, no images) |
| Controller | `pd_joint_delta_pos`, shared by all four teacher types |
| Data | ManiSkill's released PPO demos for `pd_joint_delta_pos` (999 successful episodes), replayed into state observations |
| Teacher | MLP, two 256-unit ReLU hidden layers, tanh output, MSE loss (as in the proposal) |
| Simulator | ManiSkill v3.0.1 (commit `a4a4f92`), GPU physics, same install as the camera PPO notebook |

Select **Runtime → Change runtime type → GPU** (L4 or T4), then run the cells in order. Upload only this notebook.

Outputs go to **Google Drive** under `MyDrive/182_final_project/state_bc/<session>/`:
the observation contract, the processed dataset, training logs, checkpoints, evaluation JSON and videos.

Checkpoint selection uses **validation seeds** during training. The final number in step 10 uses a separate set of **test seeds** that were never used for selection.
""")

code("""
from pathlib import Path
import os, sys, json, subprocess, datetime, shlex, shutil, urllib.request

MANISKILL_COMMIT = "a4a4f9272ad64b1564035874b605ceb687b63ed8"  # v3.0.1, same as the camera PPO notebook
MS_ROOT = Path("/content/ManiSkill")
WORK_ROOT = Path("/content/pusht_state_bc")
WORK_ROOT.mkdir(parents=True, exist_ok=True)
USE_GOOGLE_DRIVE = True
DRIVE_SUBDIR = "182_final_project/state_bc"

# Teacher and training settings
TEACHER_NAME = "bc_seed0"     # one folder per teacher under runs/
SEED = 0                      # network init and minibatch sampling
SPLIT_SEED = 0                # which demo episodes go to validation
VAL_FRACTION = 0.1
ITERS = 100_000
BATCH_SIZE = 1024
LR = 3e-4
EVAL_EVERY = 10_000           # closed-loop validation every N iterations
EVAL_EPISODES = 100
VAL_SEED = 1001               # validation initial states (checkpoint selection)
TEST_SEED = 5001              # test initial states (final report only)

DEMO_BASE = "https://huggingface.co/datasets/haosulab/ManiSkill_Demonstrations/resolve/main/demos/PushT-v1/rl/"
DEMO_NAME = "trajectory.none.pd_joint_delta_pos.physx_cuda"
PPO_CKPT_NAME = "ppo_pd_joint_delta_pos_ckpt.pt"
SESSION = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
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
## 2. Install ManiSkill v3.0.1 under Python 3.11

Same pinned install as the camera PPO notebook: a separate Python 3.11 environment with CUDA PyTorch 2.6.0 and SAPIEN 3.0.3,
so all teachers run identical simulator code. Takes a few minutes the first time in a session.
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
     "numpy", "sapien", "h5py", "tyro", "imageio[ffmpeg]", "matplotlib"])

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
print("Training Python:", TRAIN_PY)
""")

md("""
## 3. Write the BC module

`pusht_state_bc.py` holds all data, training and evaluation code (`audit`, `prepare`, `train`, `eval`).
The repo copy lives in `state_bc/`; this notebook is built from it by `state_bc/build_notebook.py`.
""")
cells.append(new_code_cell("%%writefile /content/pusht_state_bc/pusht_state_bc.py\n" + MODULE.rstrip() + "\n"))
code("""
BC = WORK_ROOT / "pusht_state_bc.py"
shutil.copy2(BC, OUTPUT_ROOT / BC.name)
run([TRAIN_PY, BC, "--help"])
""")

md("""
## 4. Check the observation and action spaces

Writes `observation_contract.json`: every state field, in order, with its index range, plus action bounds,
control frequency and episode length. **Share this file with Jimmy**: his state PPO teachers must use the same contract.
""")
code("""
run([TRAIN_PY, BC, "audit", "--out", OUTPUT_ROOT])
contract = json.loads((OUTPUT_ROOT / "observation_contract.json").read_text())
assert contract["control_mode"] == "pd_joint_delta_pos" and contract["obs_mode"] == "state"
print("state_dim", contract["state_dim"], "action_dim", contract["action_dim"])
""")

md("""
## 5. Download and replay the released PPO demos

The raw file stores actions and simulator states, not observations. ManiSkill's replay tool restores each recorded
simulator state and records `state` observations (the official IL-baseline recipe, `--use-env-states`).
Then `prepare` splits **whole episodes** into train/validation and computes normalization from the training split only.
""")
code("""
DEMO_DIR = WORK_ROOT / "demos/PushT-v1/rl"
DEMO_DIR.mkdir(parents=True, exist_ok=True)
for name in (DEMO_NAME + ".h5", DEMO_NAME + ".json", PPO_CKPT_NAME):
    target = DEMO_DIR / name
    if not target.exists():
        print("Downloading", name)
        urllib.request.urlretrieve(DEMO_BASE + name, target)
    print(f"{name}: {target.stat().st_size / 1e6:.1f} MB")

replayed = DEMO_DIR / "trajectory.state.pd_joint_delta_pos.physx_cuda.h5"
if not replayed.exists():
    run([TRAIN_PY, "-m", "mani_skill.trajectory.replay_trajectory",
         "--traj-path", DEMO_DIR / (DEMO_NAME + ".h5"), "--use-env-states",
         "-c", "pd_joint_delta_pos", "-o", "state", "--save-traj", "--num-envs", 1024, "-b", "physx_cuda"])
assert replayed.exists(), sorted(p.name for p in DEMO_DIR.iterdir())

DATA_DIR = OUTPUT_ROOT / "data"
DATASET = DATA_DIR / "pusht_state_joint_ppo_demos.npz"
run([TRAIN_PY, BC, "prepare", "--h5", replayed, "--out", DATASET,
     "--val-fraction", VAL_FRACTION, "--split-seed", SPLIT_SEED])
summary = json.loads(DATASET.with_suffix(".summary.json").read_text())
assert summary["state_dim"] == contract["state_dim"] and summary["action_dim"] == contract["action_dim"]
""")

md("""
## 6. Sanity check: evaluate the released PPO expert

This is the teacher that produced the demos. Running it through the same evaluation code checks the
environment, seeds and success metric, and gives the target a BC teacher should approach.
If the checkpoint does not load (for example a shape mismatch), the cell reports it and continues.
""")
code("""
try:
    run([TRAIN_PY, BC, "eval", "--checkpoint", DEMO_DIR / PPO_CKPT_NAME, "--released-ppo",
         "--episodes", EVAL_EPISODES, "--seed", VAL_SEED, "--out", OUTPUT_ROOT / "released_ppo_val_eval.json"])
except RuntimeError as error:
    print("Released PPO evaluation failed; BC training does not depend on it.\\n", error)
""")

md("""
## 7. Smoke test

A short run (2,000 iterations, 16 evaluation episodes) to catch problems before the main job.
Its outputs go to a separate `smoke/` folder.
""")
code("""
run([TRAIN_PY, BC, "train", "--dataset", DATASET, "--out", OUTPUT_ROOT / "smoke",
     "--seed", SEED, "--iters", 2000, "--log-every", 500, "--eval-every", 1000, "--eval-episodes", 16,
     "--val-seed", VAL_SEED])
""")

md("""
## 8. Train the BC teacher

Logs validation action MSE every 1,000 iterations and closed-loop **validation** success every `EVAL_EVERY` iterations.
`best.pt` is the checkpoint with the highest validation success-at-end (ties broken by success-once).
Every evaluated checkpoint, `final.pt` and `train_log.csv` are kept. On an L4 this should take minutes, not hours.
""")
code("""
RUN_DIR = OUTPUT_ROOT / "runs" / TEACHER_NAME
config = dict(teacher=TEACHER_NAME, seed=SEED, split_seed=SPLIT_SEED, iters=ITERS, batch_size=BATCH_SIZE, lr=LR,
              eval_every=EVAL_EVERY, eval_episodes=EVAL_EPISODES, val_seed=VAL_SEED, test_seed=TEST_SEED,
              maniskill_commit=MANISKILL_COMMIT, dataset=str(DATASET), architecture="MLP 2x256 ReLU, tanh output")
RUN_DIR.mkdir(parents=True, exist_ok=False)
(RUN_DIR / "run_config.json").write_text(json.dumps(config, indent=2))
run([TRAIN_PY, BC, "train", "--dataset", DATASET, "--out", RUN_DIR, "--seed", SEED, "--iters", ITERS,
     "--batch-size", BATCH_SIZE, "--lr", LR, "--eval-every", EVAL_EVERY, "--eval-episodes", EVAL_EPISODES,
     "--val-seed", VAL_SEED])
""")

md("## 9. Training curves")
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
## 10. Final test on fresh seeds

Evaluates `best.pt` on `TEST_SEED` initial states, which were not used for checkpoint selection, and the released
PPO expert on the same states for the teacher-performance gap. Saves a few videos of the BC teacher.
""")
code("""
run([TRAIN_PY, BC, "eval", "--checkpoint", RUN_DIR / "best.pt", "--episodes", EVAL_EPISODES,
     "--seed", TEST_SEED, "--out", RUN_DIR / "test_eval.json"])
try:
    run([TRAIN_PY, BC, "eval", "--checkpoint", DEMO_DIR / PPO_CKPT_NAME, "--released-ppo", "--episodes", EVAL_EPISODES,
         "--seed", TEST_SEED, "--out", OUTPUT_ROOT / "released_ppo_test_eval.json"])
except RuntimeError as error:
    print("Released PPO test evaluation failed:\\n", error)
run([TRAIN_PY, BC, "eval", "--checkpoint", RUN_DIR / "best.pt", "--episodes", 4,
     "--seed", TEST_SEED, "--video-dir", RUN_DIR / "videos"])

result = {"teacher": TEACHER_NAME, "test": json.loads((RUN_DIR / "test_eval.json").read_text())}
ppo_test = OUTPUT_ROOT / "released_ppo_test_eval.json"
if ppo_test.exists():
    result["released_ppo_test"] = json.loads(ppo_test.read_text())
(RUN_DIR / "summary.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
print("Everything is saved in", OUTPUT_ROOT)
""")

md("""
## 11. Watch a rollout
""")
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

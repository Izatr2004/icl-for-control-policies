"""Build notebooks/Train_PushT_State_BC_DAgger.ipynb (run 2) from pusht_state_bc.py.

The notebook embeds the module with %%writefile so it stays self-contained in Colab.
Edit pusht_state_bc.py or the cell text below, then rerun:  python state_bc/build_notebook.py

Run 1's notebook (notebooks/Train_PushT_State_BC.ipynb) is kept as run; it is no longer rebuilt.
"""

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
MODULE = (HERE / "pusht_state_bc.py").read_text(encoding="utf-8")
OUT = HERE.parent / "notebooks" / "Train_PushT_State_BC_DAgger.ipynb"

cells = []
md = lambda s: cells.append(new_markdown_cell(s.strip()))  # noqa: E731
code = lambda s: cells.append(new_code_cell(s.strip()))  # noqa: E731

md("""
# Push-T full-state BC teacher, run 2: more expert data + DAgger

Run 1 trained BC on the 761 released demos that replayed and reached **19% success-once / 10% success-at-end**
on test seeds, versus **100% / 72%** for the released PPO expert. Its training error kept falling while validation
error stalled, a sign of too little data plus compounding errors. See `state_bc/README.md` for the full log.

This run changes only the **training data**. The teacher architecture, loss, optimizer and evaluation are unchanged.
1. **More expert data:** roll out the released PPO expert for 8,192 episodes, some with small action noise so the
   data includes slightly off-course states. Labels are always the expert's clean action.
2. **DAgger:** let the BC policy drive, label every state it reaches with the expert's action, add to the data,
   retrain. Four rounds.

| Setting | Value |
|---|---|
| Observation / controller | `obs_mode="state"` (31 numbers), `pd_joint_delta_pos` (7 numbers) |
| Teacher | MLP, 2 × 256 ReLU, tanh output, MSE, Adam 3e-4 with cosine decay, batch 1024, 100k iterations per training |
| Simulator | ManiSkill v3.0.1 (commit `a4a4f92`), GPU physics |
| Seeds | collection from 100,000+ (expert) and 200,000+ (DAgger); validation 1001; test 5001 (used once, at the end) |

Select **Runtime → Change runtime type → L4 GPU**, then run the cells in order. Upload only this notebook.
Outputs go to `MyDrive/182_final_project/state_bc/<session>/`. Expected time: roughly 35–50 minutes, mostly training.
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
RUN_LABEL = "run2_dagger"

# Teacher training (unchanged from run 1)
TEACHER_NAME = "bc_dagger_seed0"
SEED = 0
SPLIT_SEED = 0
VAL_FRACTION = 0.1
ITERS = 100_000
BATCH_SIZE = 1024
LR = 3e-4
EVAL_EVERY = 25_000           # closed-loop validation every N iterations (run 1 used 10,000)
EVAL_EPISODES = 100
VAL_SEED = 1001
TEST_SEED = 5001

# New in run 2: data collection
NUM_ENVS = 1024
EXPERT_EPISODES = 8192                  # 8 batches of 1024
EXPERT_NOISE = [0.0, 0.1, 0.2, 0.3]     # action-noise std per batch, cycled; labels stay clean
EXPERT_SEED_BASE = 100_000
DAGGER_ROUNDS = 4
DAGGER_EPISODES = 2048                  # per round
DAGGER_SEED_BASE = 200_000              # round r uses DAGGER_SEED_BASE + 1000 * r

DEMO_BASE = "https://huggingface.co/datasets/haosulab/ManiSkill_Demonstrations/resolve/main/demos/PushT-v1/rl/"
DEMO_NAME = "trajectory.none.pd_joint_delta_pos.physx_cuda"
PPO_CKPT_NAME = "ppo_pd_joint_delta_pos_ckpt.pt"
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

Same pinned install as run 1 and the camera PPO notebook. Skips work already done in this Colab session.
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

`pusht_state_bc.py` holds all code: `audit`, `prepare`, `collect`, `train`, `eval`.
The repo copy is `state_bc/pusht_state_bc.py`; this notebook is generated from it by `state_bc/build_notebook.py`.
""")
cells.append(new_code_cell("%%writefile /content/pusht_state_bc/pusht_state_bc.py\n" + MODULE.rstrip() + "\n"))
code("""
BC = WORK_ROOT / "pusht_state_bc.py"
shutil.copy2(BC, OUTPUT_ROOT / BC.name)
run([TRAIN_PY, BC, "--help"])
""")

md("## 4. Check the observation and action spaces\n\nWrites `observation_contract.json` (unchanged from run 1).")
code("""
run([TRAIN_PY, BC, "audit", "--out", OUTPUT_ROOT])
contract = json.loads((OUTPUT_ROOT / "observation_contract.json").read_text())
assert contract["control_mode"] == "pd_joint_delta_pos" and contract["obs_mode"] == "state"
assert contract["max_episode_steps"] == 100, contract["max_episode_steps"]
""")

md("""
## 5. Released demos and expert checkpoint

Same as run 1: download, replay into state observations, split whole episodes 90/10.
The released demos stay in the training data, and their validation split is used for validation action error.
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
EXPERT = DEMO_DIR / PPO_CKPT_NAME

replayed = DEMO_DIR / "trajectory.state.pd_joint_delta_pos.physx_cuda.h5"
if not replayed.exists():
    run([TRAIN_PY, "-m", "mani_skill.trajectory.replay_trajectory",
         "--traj-path", DEMO_DIR / (DEMO_NAME + ".h5"), "--use-env-states",
         "-c", "pd_joint_delta_pos", "-o", "state", "--save-traj", "--num-envs", 1024, "-b", "physx_cuda"])

DATA_DIR = OUTPUT_ROOT / "data"
run([TRAIN_PY, BC, "prepare", "--h5", replayed, "--out-dir", DATA_DIR,
     "--val-fraction", VAL_FRACTION, "--split-seed", SPLIT_SEED])
DEMOS_TRAIN, DEMOS_VAL = DATA_DIR / "demos_train.npz", DATA_DIR / "demos_val.npz"
""")

md("""
## 6. Released PPO expert on validation seeds

Reference point. Run 1 measured **98% success-once / 65% success-at-end** here.
""")
code("""
run([TRAIN_PY, BC, "eval", "--checkpoint", EXPERT, "--released-ppo", "--episodes", EVAL_EPISODES,
     "--seed", VAL_SEED, "--out", OUTPUT_ROOT / "released_ppo_val_eval.json"])
""")

md("""
## 7. Collect more expert data (new)

The expert drives 8,192 episodes of 100 steps (about 819k transitions). Batches cycle through action-noise levels
0, 0.1, 0.2 and 0.3, so some episodes wander off the expert's usual path and record how it corrects.
Labels are always the expert's noise-free action. The printed success rates show how much noise the expert tolerates.
""")
code("""
EXPERT_DATA = DATA_DIR / "expert_rollouts.npz"
run([TRAIN_PY, BC, "collect", "--expert", EXPERT, "--driver", "expert", "--episodes", EXPERT_EPISODES,
     "--num-envs", NUM_ENVS, "--noise-std", *EXPERT_NOISE, "--seed-base", EXPERT_SEED_BASE, "--out", EXPERT_DATA])
""")

md("## 8. Smoke test\n\nShort run on the new data to catch problems before the long jobs.")
code("""
run([TRAIN_PY, BC, "train", "--train-data", DEMOS_TRAIN, EXPERT_DATA, "--val-data", DEMOS_VAL,
     "--out", OUTPUT_ROOT / "smoke", "--seed", SEED, "--iters", 2000, "--log-every", 500,
     "--eval-every", 1000, "--eval-episodes", 16, "--val-seed", VAL_SEED])
""")

md("""
## 9. Round 0: train on released demos + expert rollouts

This alone measures how much **more expert data** helps, before any DAgger.
""")
code("""
RUNS = OUTPUT_ROOT / "runs" / TEACHER_NAME
config = dict(teacher=TEACHER_NAME, seed=SEED, split_seed=SPLIT_SEED, iters=ITERS, batch_size=BATCH_SIZE, lr=LR,
              eval_every=EVAL_EVERY, eval_episodes=EVAL_EPISODES, val_seed=VAL_SEED, test_seed=TEST_SEED,
              expert_episodes=EXPERT_EPISODES, expert_noise=EXPERT_NOISE, dagger_rounds=DAGGER_ROUNDS,
              dagger_episodes=DAGGER_EPISODES, maniskill_commit=MANISKILL_COMMIT,
              architecture="MLP 2x256 ReLU, tanh output", training="from scratch each round on all data so far")
RUNS.mkdir(parents=True, exist_ok=False)
(RUNS / "run_config.json").write_text(json.dumps(config, indent=2))


def train_round(r, train_files):
    out = RUNS / f"round{r}"
    run([TRAIN_PY, BC, "train", "--train-data", *train_files, "--val-data", DEMOS_VAL, "--out", out,
         "--seed", SEED, "--iters", ITERS, "--batch-size", BATCH_SIZE, "--lr", LR,
         "--eval-every", EVAL_EVERY, "--eval-episodes", EVAL_EPISODES, "--val-seed", VAL_SEED])
    return out


progress = []
train_files = [DEMOS_TRAIN, EXPERT_DATA]
round_dir = train_round(0, train_files)
best = json.loads((round_dir / "best.json").read_text())
progress.append(dict(round=0, train_files=len(train_files), val_success_once=best["success_once"],
                     val_success_at_end=best["success_at_end"], best_iter=best["iter"], on_policy=None))
(RUNS / "dagger_progress.json").write_text(json.dumps(progress, indent=2))
print(progress[-1])
""")

md("""
## 10. DAgger rounds (new)

Each round: the latest BC teacher drives 2,048 episodes, the expert labels every state it visits,
those transitions join the training data, and a fresh BC teacher trains on everything so far.
`on_policy` is the driving BC policy's own success while collecting, a second view of progress.
""")
code("""
for r in range(1, DAGGER_ROUNDS + 1):
    previous = RUNS / f"round{r - 1}" / "best.pt"
    dagger_data = DATA_DIR / f"dagger_round{r}.npz"
    run([TRAIN_PY, BC, "collect", "--expert", EXPERT, "--driver", "bc", "--driver-checkpoint", previous,
         "--episodes", DAGGER_EPISODES, "--num-envs", NUM_ENVS, "--seed-base", DAGGER_SEED_BASE + 1000 * r,
         "--out", dagger_data])
    collected = json.loads(dagger_data.with_suffix(".json").read_text())
    train_files.append(dagger_data)
    round_dir = train_round(r, train_files)
    best = json.loads((round_dir / "best.json").read_text())
    progress.append(dict(round=r, train_files=len(train_files), val_success_once=best["success_once"],
                         val_success_at_end=best["success_at_end"], best_iter=best["iter"],
                         on_policy=dict(success_once=collected["driver_success_once"],
                                        success_at_end=collected["driver_success_at_end"])))
    (RUNS / "dagger_progress.json").write_text(json.dumps(progress, indent=2))
    print(progress[-1])
""")

md("## 11. Progress across rounds")
code("""
import csv
import matplotlib.pyplot as plt

print(f"{'round':>5} {'val once':>9} {'val end':>8} {'driver once (collect)':>22}")
for p in progress:
    driver = f"{p['on_policy']['success_once']:.0%}" if p["on_policy"] else "-"
    print(f"{p['round']:>5} {p['val_success_once']:>9.0%} {p['val_success_at_end']:>8.0%} {driver:>22}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))
rounds = [p["round"] for p in progress]
ax1.plot(rounds, [p["val_success_once"] for p in progress], "o-", label="success once")
ax1.plot(rounds, [p["val_success_at_end"] for p in progress], "s-", label="success at end")
ax1.axhline(0.98, ls="--", c="gray", lw=1, label="PPO expert once (run 1)")
ax1.set(xlabel="DAgger round", ylabel="validation rate", ylim=(0, 1.02), title="Best checkpoint per round")
ax1.legend()
for r in rounds:
    rows = list(csv.DictReader((RUNS / f"round{r}" / "train_log.csv").open()))
    ax2.plot([int(x["iter"]) for x in rows], [float(x["val_mse"]) for x in rows], label=f"round {r}")
ax2.set(xlabel="iteration", ylabel="validation action MSE", yscale="log", title="Action error on held-out demos")
ax2.legend()
fig.tight_layout()
fig.savefig(RUNS / "dagger_progress.png", dpi=120)
plt.show()
""")

md("""
## 12. Final test on fresh seeds

Picks the round with the best **validation** success (success-at-end, then success-once, later round on ties),
then evaluates it once on the test seeds next to the PPO expert. Saves a few videos.
""")
code("""
chosen = max(progress, key=lambda p: (p["val_success_at_end"], p["val_success_once"], p["round"]))
FINAL = RUNS / f"round{chosen['round']}" / "best.pt"
shutil.copy2(FINAL, RUNS / "teacher.pt")
print("Selected round", chosen["round"], FINAL)

run([TRAIN_PY, BC, "eval", "--checkpoint", FINAL, "--episodes", EVAL_EPISODES,
     "--seed", TEST_SEED, "--out", RUNS / "test_eval.json"])
run([TRAIN_PY, BC, "eval", "--checkpoint", EXPERT, "--released-ppo", "--episodes", EVAL_EPISODES,
     "--seed", TEST_SEED, "--out", OUTPUT_ROOT / "released_ppo_test_eval.json"])
run([TRAIN_PY, BC, "eval", "--checkpoint", FINAL, "--episodes", 4, "--seed", TEST_SEED,
     "--video-dir", RUNS / "videos"])

result = {"teacher": TEACHER_NAME, "selected_round": chosen["round"], "progress": progress,
          "test": json.loads((RUNS / "test_eval.json").read_text()),
          "released_ppo_test": json.loads((OUTPUT_ROOT / "released_ppo_test_eval.json").read_text()),
          "run1_test_for_comparison": {"success_once": 0.19, "success_at_end": 0.10}}
(RUNS / "summary.json").write_text(json.dumps(result, indent=2))
print(json.dumps({k: result[k] for k in ("selected_round", "test", "released_ppo_test", "run1_test_for_comparison")}, indent=2))
print("Everything is saved in", OUTPUT_ROOT)
""")

md("## 13. Watch a rollout")
code("""
from IPython.display import Video, display
videos = sorted((RUNS / "videos").glob("*.mp4"))
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

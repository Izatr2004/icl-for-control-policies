"""Build notebooks/Train_PushT_VarGoal_BC_from_Jimmy.ipynb (run 4) from the state_bc modules.

Run 4 distills the full-state BC teacher from Jimmy's frozen goal-conditioned PPO teacher, in Jimmy's
shared environment PushTVarGoal-v1. The repository is private, so Colab cannot clone it: Jimmy's files
are read here with `git show <JIMMY_COMMIT>:<path>` and embedded in the notebook (code with %%writefile,
binary files as base64 with a SHA-256 check). Rerun after changing pusht_state_bc.py or the cells below:
    python state_bc/build_notebook_run4.py

Run 3b's notebook (build_notebook.py) is kept as run and no longer rebuilt.
"""

import base64
import hashlib
import json
import subprocess
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = REPO / "notebooks" / "Train_PushT_VarGoal_BC_from_Jimmy.ipynb"

JIMMY_COMMIT = "c87d8ae3669c10894d765df2d0dd4b61c010e272"   # origin/jimmy-state-rl, frozen teacher
JIMMY_CODE = ["state_rl/pusht_vargoal.py", "state_rl/pusht_state_ppo.py"]
JIMMY_BINARY = {  # path in Jimmy's branch -> path under WORK_ROOT in Colab
    "state_rl/assets/tip_start_qpos.npz": "assets/tip_start_qpos.npz",
    "state_rl/teachers/ppo_vargoal_seed1_50M.pt": "jimmy_ppo_vargoal_seed1_50M.pt",
}
JIMMY_SELECTION = "state_rl/teachers/ppo_vargoal_seed1_50M_selection.json"


def git_show(path: str) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), "show", f"{JIMMY_COMMIT}:{path}"],
                          check=True, capture_output=True).stdout


cells = []
md = lambda s: cells.append(new_markdown_cell(s.strip()))  # noqa: E731
code = lambda s: cells.append(new_code_cell(s.strip()))  # noqa: E731

selection = json.loads(git_show(JIMMY_SELECTION))
recorded = {split: {q: [round(selection[split][q]["success_once"], 3), round(selection[split][q]["success_at_end"], 3)]
                    for q in ("Q1", "Q2", "Q3", "Q4")} for split in ("val", "test")}

md(f"""
# Push-T variable goal, run 4: BC teacher distilled from Jimmy's PPO teacher

The group uses **one** variable-goal environment for all transformer data: Jimmy's `PushTVarGoal-v1`
(`state_rl/pusht_vargoal.py`, branch `jimmy-state-rl`, commit `{JIMMY_COMMIT[:7]}`). The plan's teacher pair is
**PPO (Jimmy) → distilled BC (this notebook)**. Run 3b's own environment and teachers are kept as a record only. See `state_bc/README.md`.

1. **Environment:** goal T position random per episode around (−0.156, 0): ±10 cm in x, ±20 cm in y, with a 2 cm / 3 cm band
   around each axis excluded. Quadrants by the sign of the offset: Q1 (+x,+y), Q2 (−x,+y), Q3 (−x,−y), **Q4 (+x,−y)**, the transformer's
   held-out quadrant. The movable T starts anywhere in a fixed box, 8–30 cm from the goal; the stick tip starts from a library of 600 arm poses.
2. **Expert:** Jimmy's frozen `ppo_vargoal_seed1_50M.pt`, embedded below, checked by SHA-256. It is re-evaluated here first, with his
   per-quadrant protocol, as a reproduction check and a gate.
3. **BC teacher distilled from it:** noisy expert rollouts plus one training run, exactly as run 3b. **No PPO training, no DAgger.**
4. **Per-quadrant validation and test** of both teachers with Jimmy's protocol (100 episodes per quadrant; quadrant q uses seed base + 1000·q;
   validation base 1001, test base 5001), so BC and PPO are scored on the same episodes. Freeze bar: success-once ≥ 80% in every quadrant.

| Setting | Value |
|---|---|
| Observation / controller | `obs_mode="state"` (31 numbers, goal at 21:24), `pd_joint_delta_pos` (7) |
| BC teacher | MLP 2 × 256 ReLU, tanh output, MSE, Adam 3e-4 cosine, batch 1024, 200k iterations |
| Expert data | 16,384 episodes, action noise 0 / 0.1 / 0.2 / 0.3 cycled per batch, labels always the expert's clean action |
| Simulator | ManiSkill v3.0.1 (commit `a4a4f92`), GPU physics |
| Seeds | expert data 100,000+; BC validation data 300,000+; closed-loop validation during training 1001 (all quadrants mixed); per-quadrant validation 2001–5001; test 6001–9001 (once, at the end) |

Select **Runtime → Change runtime type → L4 GPU** and run the cells in order. Upload only this notebook.
Outputs go to `MyDrive/182_final_project/state_bc/<session>/`. Estimated total: about 30–40 minutes (install about 8).
""")

code(f"""
from pathlib import Path
import os, sys, json, re, subprocess, datetime, shlex, shutil, base64, hashlib

MANISKILL_COMMIT = "a4a4f9272ad64b1564035874b605ceb687b63ed8"  # v3.0.1, same as the other notebooks
MS_ROOT = Path("/content/ManiSkill")
WORK_ROOT = Path("/content/pusht_state_bc")
WORK_ROOT.mkdir(parents=True, exist_ok=True)
USE_GOOGLE_DRIVE = True
DRIVE_SUBDIR = "182_final_project/state_bc"
RUN_LABEL = "run4_bc_from_jimmy"
ENV_ID = "PushTVarGoal-v1"
JIMMY_COMMIT = "{JIMMY_COMMIT}"

# Per-quadrant protocol shared with state_rl/pusht_state_ppo.py
QUAD_EPISODES = 100          # per quadrant
QUAD_VAL_SEED = 1001         # quadrant q uses 1001 + 1000 * q
QUAD_TEST_SEED = 5001        # quadrant q uses 5001 + 1000 * q; used once, at the end
FREEZE_BAR = 0.80            # success-once in every quadrant

# BC teacher
TEACHER_NAME = "bc_from_jimmy_ppo_seed0"
SEED = 0
ITERS = 200_000
BATCH_SIZE = 1024
LR = 3e-4
EVAL_EVERY = 25_000
EVAL_EPISODES = 200          # closed-loop validation during training, all quadrants mixed
VAL_SEED = 1001

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

Same pinned install as runs 1–3b and Jimmy's notebooks.
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

md(f"""
## 3. Write the modules and Jimmy's files

`pusht_state_bc.py` is this repo's BC module. `pusht_vargoal.py` (the shared environment) and `pusht_state_ppo.py`
(Jimmy's preflight checks) are copied unchanged from Jimmy's branch at commit `{JIMMY_COMMIT[:7]}`. The arm-pose library and the frozen
PPO teacher follow as base64; their SHA-256 must match the committed files.
""")
cells.append(new_code_cell("%%writefile /content/pusht_state_bc/pusht_state_bc.py\n"
                           + (HERE / "pusht_state_bc.py").read_text(encoding="utf-8").rstrip() + "\n"))
for path in JIMMY_CODE:
    text = git_show(path).decode("utf-8").replace("\r\n", "\n").rstrip()
    cells.append(new_code_cell(f"%%writefile /content/pusht_state_bc/{Path(path).name}\n"
                               f"# Copied unchanged from jimmy-state-rl:{path} at {JIMMY_COMMIT[:7]}\n{text}\n"))

blobs = []
for src, dst in JIMMY_BINARY.items():
    data = git_show(src)
    b64 = base64.b64encode(data).decode("ascii")
    wrapped = "\n".join(b64[i:i + 4096] for i in range(0, len(b64), 4096))
    blobs.append(f'    "{dst}": ("{hashlib.sha256(data).hexdigest()}", """\n{wrapped}\n"""),')
code("# Jimmy's binary files (base64), from jimmy-state-rl at " + JIMMY_COMMIT[:7] + "\nJIMMY_FILES = {\n"
     + "\n".join(blobs) + "\n}\n"
     + """for name, (sha, text) in JIMMY_FILES.items():
    data = base64.b64decode("".join(text.split()))
    assert hashlib.sha256(data).hexdigest() == sha, f"{name}: SHA-256 mismatch"
    (WORK_ROOT / name).parent.mkdir(parents=True, exist_ok=True)
    (WORK_ROOT / name).write_bytes(data)
    print(f"{name}: {len(data):,} bytes, sha256 {sha[:12]}... OK")
del JIMMY_FILES""")

code(f"""
BC = WORK_ROOT / "pusht_state_bc.py"
for name in ("pusht_state_bc.py", "pusht_vargoal.py", "pusht_state_ppo.py"):
    shutil.copy2(WORK_ROOT / name, OUTPUT_ROOT / name)
TEACHERS = OUTPUT_ROOT / "teachers"
TEACHERS.mkdir(exist_ok=True)
EXPERT = TEACHERS / "ppo_expert.pt"
shutil.copy2(WORK_ROOT / "jimmy_ppo_vargoal_seed1_50M.pt", EXPERT)
# Jimmy's recorded results for this teacher (Kaggle T4), success-once / success-at-end per quadrant
JIMMY_RECORDED = {json.dumps(recorded)}
run([TRAIN_PY, BC, "--help"], show=r"eval-quadrants")
""")

md("""
## 4. Check the environment

- `audit` records the observation contract.
- Jimmy's `preflight`: observation layout, success at each environment's own goal and not at another's, partial resets, quadrant restriction.
- Our `check-env`: a stand-in T on every goal must succeed, and shifted 5 cm must fail; reports goal ranges and quadrant counts.
""")
code("""
run([TRAIN_PY, BC, "audit", "--env-id", ENV_ID, "--out", OUTPUT_ROOT])
contract = json.loads((OUTPUT_ROOT / "observation_contract.json").read_text())
assert contract["control_mode"] == "pd_joint_delta_pos" and contract["state_dim"] == 31, contract
run([TRAIN_PY, WORK_ROOT / "pusht_state_ppo.py", "preflight", "--out", OUTPUT_ROOT / "jimmy_preflight", "--num-envs", 8])
run([TRAIN_PY, BC, "check-env", "--env-id", ENV_ID, "--num-envs", 256, "--out", OUTPUT_ROOT / "env_check.json"])
""")

md("""
## 5. Re-evaluate Jimmy's PPO teacher (reproduction check and gate)

Same protocol as Jimmy's selection: 100 episodes per quadrant, validation base seed 1001 and test base seed 5001. Numbers can differ a
little from his (different GPU: L4 here, T4 on Kaggle). Stops before collecting data if the teacher misses the 80% bar in any quadrant.
""")
code("""
def show(name, r):
    print(f"{name:<14} worst {r['worst_success_once']:.0%}  mean once {r['mean_success_once']:.0%}  "
          f"mean end {r['mean_success_at_end']:.0%}   "
          + "  ".join(f"Q{q} {r[f'Q{q}']['success_once']:.0%}/{r[f'Q{q}']['success_at_end']:.0%}" for q in (1, 2, 3, 4)))


expert_evals = {}
for split, seed in (("val", QUAD_VAL_SEED), ("test", QUAD_TEST_SEED)):
    out = TEACHERS / f"ppo_expert_{split}_quadrants.json"
    run([TRAIN_PY, BC, "eval-quadrants", "--env-id", ENV_ID, "--checkpoint", EXPERT, "--ppo", "--episodes", QUAD_EPISODES,
         "--seed", seed, "--bar", FREEZE_BAR, "--out", out], show=r"^$")
    expert_evals[split] = json.loads(out.read_text())
    show(f"PPO {split}", expert_evals[split])
    print(" " * 14, "Jimmy recorded:", "  ".join(f"{q} {o:.0%}/{e:.0%}" for q, (o, e) in JIMMY_RECORDED[split].items()))
if not expert_evals["test"]["passes_bar"]:
    raise RuntimeError("Jimmy's teacher misses the 80% bar here; compare with his numbers before distilling BC.")
""")

md("""
## 6. Collect expert data

16,384 expert episodes (about 1.6M transitions) with action noise cycling 0, 0.1, 0.2, 0.3 per batch, plus 1,024
noise-free episodes on separate seeds for validation action error. Goals cover all four quadrants. Labels are always
the expert's noise-free action. Each episode's goal position and quadrant are saved in the `.json` sidecar.
""")
code("""
DATA_DIR = OUTPUT_ROOT / "data"
EXPERT_DATA, VAL_DATA = DATA_DIR / "expert_rollouts.npz", DATA_DIR / "expert_val.npz"
run([TRAIN_PY, BC, "collect", "--env-id", ENV_ID, "--expert", EXPERT, "--driver", "expert",
     "--episodes", EXPERT_EPISODES, "--num-envs", NUM_ENVS, "--noise-std", *EXPERT_NOISE,
     "--seed-base", EXPERT_SEED_BASE, "--out", EXPERT_DATA], show=r"^batch|transitions|driver_success")
run([TRAIN_PY, BC, "collect", "--env-id", ENV_ID, "--expert", EXPERT, "--driver", "expert",
     "--episodes", VAL_EPISODES, "--num-envs", NUM_ENVS, "--noise-std", 0.0,
     "--seed-base", VAL_SEED_BASE, "--out", VAL_DATA], show=r"^batch|transitions|driver_success")
""")

md("## 7. Smoke test\n\nShort BC run to catch problems before the main training.")
code("""
run([TRAIN_PY, BC, "train", "--env-id", ENV_ID, "--train-data", EXPERT_DATA, "--val-data", VAL_DATA,
     "--out", OUTPUT_ROOT / "smoke", "--seed", SEED, "--iters", 2000, "--log-every", 500,
     "--eval-every", 1000, "--eval-episodes", 32, "--val-seed", VAL_SEED])
""")

md("""
## 8. Train the BC teacher (no DAgger)

One training run on the expert rollouts. Logs validation action error every 1,000 iterations and closed-loop validation
success every `EVAL_EVERY` (200 episodes, all quadrants mixed). `best.pt` has the highest validation success-at-end
(ties broken by success-once).
""")
code("""
RUN_DIR = OUTPUT_ROOT / "runs" / TEACHER_NAME
config = dict(teacher=TEACHER_NAME, env_id=ENV_ID, seed=SEED, iters=ITERS, batch_size=BATCH_SIZE, lr=LR,
              eval_every=EVAL_EVERY, eval_episodes=EVAL_EPISODES, val_seed=VAL_SEED,
              quad_val_seed=QUAD_VAL_SEED, quad_test_seed=QUAD_TEST_SEED, quad_episodes=QUAD_EPISODES,
              expert="Jimmy's ppo_vargoal_seed1_50M.pt", jimmy_commit=JIMMY_COMMIT,
              expert_episodes=EXPERT_EPISODES, expert_noise=EXPERT_NOISE, dagger=False,
              maniskill_commit=MANISKILL_COMMIT, architecture="MLP 2x256 ReLU, tanh output")
RUN_DIR.mkdir(parents=True, exist_ok=False)
(RUN_DIR / "run_config.json").write_text(json.dumps(config, indent=2))
run([TRAIN_PY, BC, "train", "--env-id", ENV_ID, "--train-data", EXPERT_DATA, "--val-data", VAL_DATA, "--out", RUN_DIR,
     "--seed", SEED, "--iters", ITERS, "--batch-size", BATCH_SIZE, "--lr", LR, "--eval-every", EVAL_EVERY,
     "--eval-episodes", EVAL_EPISODES, "--val-seed", VAL_SEED], show=r"iter +\\d*[05]000 |Training on|Done")
shutil.copy2(RUN_DIR / "best.pt", TEACHERS / "bc_teacher.pt")
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
## 10. Per-quadrant validation and final test

BC teacher with the same protocol as the PPO teacher in step 5: validation base seed 1001, then test base seed 5001 (used once).
Also saves a few BC videos.
""")
code("""
bc_evals = {}
for split, seed in (("val", QUAD_VAL_SEED), ("test", QUAD_TEST_SEED)):
    out = TEACHERS / f"bc_teacher_{split}_quadrants.json"
    run([TRAIN_PY, BC, "eval-quadrants", "--env-id", ENV_ID, "--checkpoint", TEACHERS / "bc_teacher.pt",
         "--episodes", QUAD_EPISODES, "--seed", seed, "--bar", FREEZE_BAR, "--out", out], show=r"^$")
    bc_evals[split] = json.loads(out.read_text())
print("Test (100 episodes per quadrant), success-once / success-at-end:")
show("PPO (Jimmy)", expert_evals["test"])
show("BC (distilled)", bc_evals["test"])
print("BC passes the 80% bar in every quadrant:", "PASS" if bc_evals["test"]["passes_bar"] else "FAIL")
run([TRAIN_PY, BC, "eval", "--env-id", ENV_ID, "--checkpoint", TEACHERS / "bc_teacher.pt", "--episodes", 4,
     "--seed", 9999, "--video-dir", RUN_DIR / "videos"], show=r"^$")

summary = {"env_id": ENV_ID, "jimmy_commit": JIMMY_COMMIT, "expert": "Jimmy's ppo_vargoal_seed1_50M.pt",
           "jimmy_recorded": JIMMY_RECORDED, "ppo_expert": expert_evals, "bc_teacher": bc_evals,
           "teachers": sorted(p.name for p in TEACHERS.glob("*.pt"))}
(TEACHERS / "summary.json").write_text(json.dumps(summary, indent=2))
print("Teachers saved in", TEACHERS)
""")

md("## 11. Watch a rollout")
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
print("Wrote", OUT, f"({OUT.stat().st_size / 1e6:.1f} MB)")

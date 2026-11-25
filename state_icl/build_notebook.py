"""Build notebooks/Train_PushT_ICL_PPO_Pilot.ipynb (Kaggle) for the PPO-only ICL pilot.

Embeds state_rl/{pusht_vargoal,pusht_state_ppo}.py, the tip-start library, state_icl/pusht_icl.py and the
frozen PPO teacher, so the notebook runs standalone. Rerun after editing:  python state_icl/build_notebook.py
"""
import base64
import hashlib
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--bc", default=None, help="Nathan's bc_teacher.pt to embed (adds BC demos and prompts)")
CLI = ap.parse_args()
OUT = ROOT / "notebooks" / "Train_PushT_ICL_TwoSeed.ipynb"
FILES = {"state_rl/pusht_vargoal.py": ROOT / "state_rl/pusht_vargoal.py",
         "state_rl/pusht_state_ppo.py": ROOT / "state_rl/pusht_state_ppo.py",
         "state_icl/pusht_icl.py": ROOT / "state_icl/pusht_icl.py",
         "state_icl/bc_teacher.py": ROOT / "state_icl/bc_teacher.py"}
TEACHER = ROOT / "state_rl/teachers/ppo_vargoal_seed1_50M.pt"


def actor_only_bytes(path):
    """Kaggle imports are capped at 1 MB: embed only the actor weights (all load_actor reads)."""
    import io
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=True)
    buf = io.BytesIO()
    torch.save({k: v for k, v in sd.items() if k.startswith("actor_mean.")}, buf)
    return buf.getvalue()


BLOBS = {"state_rl/assets/tip_start_qpos.npz": (ROOT / "state_rl/assets/tip_start_qpos.npz").read_bytes(),
         "state_rl/teachers/ppo_vargoal_seed1_50M_actor.pt": actor_only_bytes(TEACHER)}
TEACHER_SHA = hashlib.sha256(TEACHER.read_bytes()).hexdigest()
if CLI.bc:
    BLOBS["state_icl/teachers/bc_teacher.pt"] = Path(CLI.bc).read_bytes()
TEACHER_NAMES = ["ppo"] + (["bc"] if CLI.bc else [])

cells = []
md = lambda s: cells.append(new_markdown_cell(s.strip().replace("__TSHA__", TEACHER_SHA[:16])))  # noqa: E731
code = lambda s: cells.append(new_code_cell(s.strip().replace("{TEACHER_NAMES_LITERAL}", repr(TEACHER_NAMES))))  # noqa: E731

md("""
# Push-T in-context goal inference: two learner seeds in parallel (Jimmy)

A GPT-2-style learner (12 layers, width 256, 8 heads; `PushT_ICL_Transformer_Spec.md`) sees demonstrations of a frozen
teacher pushing to a **hidden** goal and outputs the teacher's action at a new state. Goal coordinates are removed from every
input (28 of the 31 numbers). Trained on Q1–Q3 goals, selected on separate Q1–Q3 goals, tested frozen on unseen Q1–Q3 goals
and on Q4 goals (`CODEX_CONTEXT.md`). Teachers in this notebook: **__TEACHERS__**.

| Setting | Value |
|---|---|
| Data | Nathan's conventions: goals train 2,048 / val 128 / test_in 128 (Q1–Q3) / test_q4 128 (Q4), 10 noise-free episodes per goal and teacher, same goal/demo/eval seeds, same file format |
| Data cache | if a previous run's output is attached as input, its demos are reused; otherwise collected here (few minutes) |
| Prompt | k ∈ {0, 1, 2, 4, 8} successful demos (whole episode, every 4th step, ending at step 100) + one query step from another episode |
| Training | **two seeds at once, one per T4**; fp16 autocast, Adam, batch 64, grad clip 1.0; settings in the first cell |
| Selection | lowest val action MSE with correct demos (k = 1–8, averaged over teachers) |
| Evaluation | per teacher: correct-goal / wrong-goal / no demos and the teacher itself (true goal); 4 starts per goal, identical across conditions |

The PPO teacher is embedded as its actor (`state_rl/teachers/ppo_vargoal_seed1_50M.pt`, sha256 __TSHA__): Kaggle caps imports at 1 MB.

**Kaggle:** Settings → Accelerator → **GPU T4 x2**, Internet on. Save Version → **Save & Run All (Commit)**.
""".replace("__TEACHERS__", " + ".join(TEACHER_NAMES)))

code("""
from pathlib import Path
import os, sys, json, subprocess, datetime, shlex, shutil, base64, hashlib

MANISKILL_COMMIT = "a4a4f9272ad64b1564035874b605ceb687b63ed8"
BASE = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/content")
MS_ROOT = Path("/tmp/ManiSkill")
WORK = Path("/tmp/pusht_icl"); WORK.mkdir(parents=True, exist_ok=True)
SESSION = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_icl_two_seed"
OUTPUT_ROOT = BASE / "state_icl" / SESSION
OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
LOG = OUTPUT_ROOT / "notebook.log"

TEACHERS = {TEACHER_NAMES_LITERAL}
SEEDS = [0, 1]             # one per GPU
TRAIN_STEPS = 50_000
WARMUP, LR_MIN_FRAC = 0, 1.0   # constant lr; Nathan's schedule is WARMUP=1000, LR_MIN_FRAC=0.1
MAX_MINUTES = 600          # stop training early if it is slower than expected (Kaggle limit: 12 h)
EVAL_EVERY = 2000
subprocess.run(["nvidia-smi"], check=True)
print("Output:", OUTPUT_ROOT)


def run(command, **kwargs):
    command = list(map(str, command))
    print("$", shlex.join(command), flush=True)
    tail = []
    with LOG.open("a") as log:
        log.write("\\n$ " + shlex.join(command) + "\\n")
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                              **kwargs) as p:
            for line in p.stdout:
                print(line, end="", flush=True); log.write(line); tail = (tail + [line.rstrip()])[-40:]
            code = p.wait()
    if code:
        raise RuntimeError(f"Command failed (exit {code})\\n" + "\\n".join(tail))


def run_parallel(jobs):
    # jobs: {label: (command, gpu)}. Runs them at once, one GPU each; prints their output prefixed by label.
    import time as _t
    procs = {}
    for label, (command, gpu) in jobs.items():
        command = list(map(str, command))
        print(f"[{label}] $", shlex.join(command), flush=True)
        log = open(OUTPUT_ROOT / f"{label}.log", "w")
        procs[label] = (subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True,
                                         env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}), log, [0])
    while procs:
        _t.sleep(30)
        for label, (p, log, pos) in list(procs.items()):
            log.flush()
            lines = (OUTPUT_ROOT / f"{label}.log").read_text().splitlines()
            for line in lines[pos[0]:]:
                if "warn" not in line.lower():   # skip library warnings
                    print(f"[{label}] {line}", flush=True)
            pos[0] = len(lines)
            if p.poll() is not None:
                log.close(); del procs[label]
                if p.returncode:
                    raise RuntimeError(f"{label} failed (exit {p.returncode}); see {label}.log")
""")

md("## 1. Install ManiSkill v3.0.1 (Python 3.11) and Transformers")
code("""
run(["apt-get", "update", "-qq"])
run(["apt-get", "install", "-y", "-qq", "libvulkan1", "libvulkan-dev", "vulkan-tools", "libglvnd-dev"])
if not (MS_ROOT / ".git").exists():
    run(["git", "clone", "https://github.com/mani-skill/ManiSkill.git", MS_ROOT])
run(["git", "-C", MS_ROOT, "checkout", "--detach", MANISKILL_COMMIT])
run([sys.executable, "-m", "pip", "install", "-q", "uv==0.9.11"])
ENV = WORK / "python311"; PY = ENV / "bin/python"
if not PY.exists():
    run([sys.executable, "-m", "uv", "venv", "--python", "3.11", "--seed", ENV])
constraints = WORK / "constraints.txt"
constraints.write_text("numpy==1.26.4\\nopencv-python==4.11.0.86\\nsapien==3.0.3\\ntorch==2.6.0\\n")
run([PY, "-m", "pip", "install", "-q", "torch==2.6.0", "--index-url", "https://download.pytorch.org/whl/cu124"])
run([PY, "-m", "pip", "install", "-q", "--constraint", constraints, "-e", MS_ROOT,
     "numpy", "sapien", "h5py", "tyro", "transformers==4.46.3"])
for filename, library, vulkan in [("/usr/share/vulkan/icd.d/nvidia_icd.json", "libGLX_nvidia.so.0", True),
                                  ("/usr/share/glvnd/egl_vendor.d/10_nvidia.json", "libEGL_nvidia.so.0", False)]:
    path = Path(filename)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        icd = {"library_path": library}
        if vulkan:
            icd["api_version"] = "1.2.155"
        path.write_text(json.dumps({"file_format_version": "1.0.0", "ICD": icd}))
os.environ.update(VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json", MS_SKIP_ASSET_DOWNLOAD_PROMPT="1",
                  PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="0")
run([PY, "-c", "import torch, transformers, mani_skill; print('torch', torch.__version__, 'transformers', "
     "transformers.__version__, 'mani_skill', mani_skill.__version__); assert torch.cuda.is_available()"])
(OUTPUT_ROOT / "pip-freeze.txt").write_text(subprocess.check_output([str(PY), "-m", "pip", "freeze"], text=True))
""")

md("## 2. Write the code and the frozen PPO teacher\n\nGenerated by `state_icl/build_notebook.py`. Do not edit here.")
code("""
SRC = WORK / "repo"
for d in ("state_rl/assets", "state_rl/teachers", "state_icl/teachers"):
    (SRC / d).mkdir(parents=True, exist_ok=True)
""")
for rel, path in FILES.items():
    cells.append(new_code_cell(f"%%writefile /tmp/pusht_icl/repo/{rel}\n{path.read_text(encoding='utf-8').rstrip()}\n"))
blob_lines = []
for rel, b in BLOBS.items():
    blob_lines.append(f'("{rel}", "{hashlib.sha256(b).hexdigest()}", "{base64.b64encode(b).decode()}"),')
code("BLOBS = [\n" + "\n".join(blob_lines) + "\n]\n" + """
for rel, sha, b64 in BLOBS:
    data = base64.b64decode(b64)
    assert hashlib.sha256(data).hexdigest() == sha, rel
    (SRC / rel).write_bytes(data)
shutil.copytree(SRC, OUTPUT_ROOT / "code", dirs_exist_ok=True)
ICL = SRC / "state_icl/pusht_icl.py"
SPECS = {"ppo": f"ppo=ppo:{SRC / 'state_rl/teachers/ppo_vargoal_seed1_50M_actor.pt'}",
         "bc": f"bc=bc:{SRC / 'state_icl/teachers/bc_teacher.pt'}"}
TEACHER_SPECS = [SPECS[t] for t in TEACHERS]
run([PY, ICL, "--help"])
""")

md("## 3. Smoke test on the GPU (tiny collect, 20 training steps, tiny eval)")
code("""
SMOKE = WORK / "smoke"
shutil.rmtree(SMOKE, ignore_errors=True)
for spec in TEACHER_SPECS:
    run([PY, ICL, "collect", "--teacher", spec, "--out", SMOKE / "data", "--episodes", 4, "--goals", 6, "--num-envs", 24])
run([PY, ICL, "train", "--data", SMOKE / "data", "--teachers", *TEACHERS, "--out", SMOKE / "run", "--steps", 20,
     "--eval-every", 10, "--batch", 16, "--ks", 0, 1, 2, "--select-ks", 1, 2, "--n-support-eps", 3])
run([PY, ICL, "eval", "--learner", SMOKE / "run/best.pt", "--teachers", *TEACHER_SPECS, "--data", SMOKE / "data",
     "--out", SMOKE / "eval", "--ks", 0, 1, "--starts", 2, "--num-envs", 12, "--splits", "test_q4"])
""")

md("""## 4. Teacher demonstrations: reuse the attached cache, or collect

To reuse demos from an earlier run: Add Input → Notebook Output Files → this notebook (or the collection run).""")
code("""
cached = sorted(Path("/kaggle/input").glob("**/train_ppo.npz")) if Path("/kaggle/input").exists() else []
DATA = OUTPUT_ROOT / "data"
DATA.mkdir(exist_ok=True)
if cached:
    for f in cached[0].parent.glob("*_*.*"):                # npz + json of every split and teacher
        (DATA / f.name).symlink_to(f)
    print("Using cached demos from", cached[0].parent)
for spec in TEACHER_SPECS:                                  # collects only what is missing
    run([PY, ICL, "collect", "--teacher", spec, "--out", DATA, "--episodes", 10, "--num-envs", 1024])
for f in sorted(DATA.glob("*.json")):
    m = json.loads(f.read_text())
    print(f"{f.stem:>14}: {m['goals']} goals x {m['episodes_per_goal']}  success-once {m['success_once']:.1%}  "
          f"at end {m['success_at_end']:.1%}")
""")

md("## 5. Train two seeds at once (one per T4)")
code("""
RUNS = {seed: OUTPUT_ROOT / f"run_seed{seed}" for seed in SEEDS}
run_parallel({f"train_seed{seed}": ([PY, ICL, "train", "--data", DATA, "--teachers", *TEACHERS, "--out", RUNS[seed],
                                     "--steps", TRAIN_STEPS, "--eval-every", EVAL_EVERY, "--warmup", WARMUP,
                                     "--lr-min-frac", LR_MIN_FRAC, "--max-minutes", MAX_MINUTES, "--seed", seed], gpu)
              for gpu, seed in enumerate(SEEDS)})
""")

md("## 6. Frozen evaluation of both seeds (one per T4)")
code("""
run_parallel({f"eval_seed{seed}": ([PY, ICL, "eval", "--learner", RUNS[seed] / "best.pt", "--teachers", *TEACHER_SPECS,
                                    "--data", DATA, "--out", RUNS[seed] / "eval"], gpu)
              for gpu, seed in enumerate(SEEDS)})
for seed in SEEDS:
    res = json.loads((RUNS[seed] / "eval" / "eval_results.json").read_text())
    print(f"\\n=== seed {seed} (step {res['learner_step']}) ===")
    for teacher, splits in res["teachers"].items():
        for split, r in splits.items():
            print(f"[{teacher}] {split}  ({r['goals_evaluated']} goals x {res['starts_per_goal']} starts)")
            for cond, x in r["closed_loop"].items():
                mse = r["action_mse"].get(cond, {}).get("mse")
                print(f"  {cond:>17}  success {x['success_once']:6.1%}  at end {x['success_at_end']:6.1%}  "
                      f"overlap {x['final_overlap']:.2f}" + (f"  action MSE {mse:.4f}" if mse is not None else ""))
""")

md("## 7. Save\n\nOutputs persist in the committed version: demos (`data/`, reusable as a cache), both runs, evaluations.")
code("""
for f in DATA.iterdir():                                    # store cached demos as real files in this output
    if f.is_symlink():
        target = f.resolve(); f.unlink(); shutil.copy2(target, f)
print("Folder:", OUTPUT_ROOT)
""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}, "kaggle": {"accelerator": "gpu", "isInternetEnabled": True}})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, OUT)
print("Wrote", OUT, f"({OUT.stat().st_size / 1e6:.1f} MB)")

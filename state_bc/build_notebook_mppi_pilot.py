"""Build notebooks/Pilot_PushT_MPPI_Expert.ipynb: a Colab pilot of the sampling-based MPC (MPPI) Push-T expert.

Measures, on a Colab GPU, how fast the batched MPPI expert runs and how often it succeeds at several search budgets, then
extrapolates the cost of generating BC demonstrations and renders videos. Rerun after changing
scripted_pusher/mppi/mppi_gpu.py or the cells below:
    python state_bc/build_notebook_mppi_pilot.py
"""

import base64
import hashlib
import subprocess
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = REPO / "notebooks" / "Pilot_PushT_MPPI_Expert.ipynb"
JIMMY_COMMIT = "c87d8ae3669c10894d765df2d0dd4b61c010e272"
MPPI = HERE / "scripted_pusher" / "mppi" / "mppi_gpu.py"


def git_show(path: str) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), "show", f"{JIMMY_COMMIT}:{path}"],
                          check=True, capture_output=True).stdout


cells = []
md = lambda s: cells.append(new_markdown_cell(s.strip()))  # noqa: E731
code = lambda s: cells.append(new_code_cell(s.strip()))  # noqa: E731

md("""
# Pilot: sampling-based MPC (MPPI) expert for Push-T

**Goal:** measure, before committing compute, how fast a sampling-based planner runs on a Colab GPU and how often it succeeds,
so we can decide whether to use it as the non-RL demonstration source for the BC teacher (v2).

**What the expert does.** At every control step it copies the current simulator state into N parallel copies, tries N smooth
candidate stick-tip motions over the next H steps in the simulator itself, scores each by how close the T ends up to the goal
pose (plus small terms for staying near the T and effort; this is not PPO's reward), averages the candidates weighted by score,
executes the first step, and repeats. No learned policy and no PPO involvement.

**This pilot runs** three search budgets (N × H = 128 × 8, 256 × 10, 512 × 12), 32 episodes each (8 per goal quadrant, 100
steps, the team's validation seeds), and reports success-once, success-at-end, measured seconds per episode, and the extrapolated
cost of 4,096 and 16,384 demonstrations. It then renders top-down videos of the best setting.

Select **Runtime → Change runtime type → L4 GPU**. Upload only this notebook. Expected time: about 20–40 minutes.
""")

code(f"""
from pathlib import Path
import os, sys, json, re, subprocess, datetime, shlex, shutil, base64, hashlib

MANISKILL_COMMIT = "a4a4f9272ad64b1564035874b605ceb687b63ed8"  # v3.0.1, same as the teacher notebooks
MS_ROOT = Path("/content/ManiSkill")
WORK_ROOT = Path("/content/pusht_state_bc")
WORK_ROOT.mkdir(parents=True, exist_ok=True)
DRIVE_BASE = Path("/content/drive/MyDrive/182_final_project/state_bc")
RUN_LABEL = "mppi_pilot"
JIMMY_COMMIT = "{JIMMY_COMMIT}"

EPISODES_PER_BATCH = 8           # episodes run in lockstep; each gets N simulator copies
CONFIGS = [(128, 8), (256, 10), (512, 12)]   # (samples N, horizon H)
QUADRANTS = [1, 2, 3, 4]

SESSION = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + RUN_LABEL
subprocess.run(["nvidia-smi"], check=True)
""")

md("## 1. Drive and logging")
code("""
from google.colab import drive
drive.mount("/content/drive")
OUTPUT_ROOT = DRIVE_BASE / SESSION
OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
LOG = OUTPUT_ROOT / "notebook.log"
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

md("## 2. Install ManiSkill v3.0.1 under Python 3.11 (same as the other notebooks)")
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
run([TRAIN_PY, "-c", "import torch, sapien, mani_skill, pytorch_kinematics; print('torch', torch.__version__, 'mani_skill', "
     "mani_skill.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"])
""")

md(f"""
## 3. Write the modules

`mppi_gpu.py` is this repo's `state_bc/scripted_pusher/mppi/mppi_gpu.py`. `pusht_vargoal.py` and its arm-pose library are
copied unchanged from Jimmy's branch at `{JIMMY_COMMIT[:7]}`, checked by SHA-256.
""")
cells.append(new_code_cell("%%writefile /content/pusht_state_bc/mppi_gpu.py\n" + MPPI.read_text(encoding="utf-8").rstrip() + "\n"))
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
MPPI_PY = WORK_ROOT / "mppi_gpu.py"
for name in ("mppi_gpu.py", "pusht_vargoal.py"):
    shutil.copy2(WORK_ROOT / name, OUTPUT_ROOT / name)
''')

md("""
## 4. Self-check on the GPU

Episodes run as groups of parallel simulator copies. Check that every copy matches its episode's real state after a reset and
after a few steps (both differences should be 0.0), and that copies share their episode's goal. Stop here if not.
""")
code("""
run([TRAIN_PY, MPPI_PY, "--self-check", "--episodes-per-batch", 2, "--samples", 8, "--horizon", 4,
     "--out", OUTPUT_ROOT / "self_check"], cwd=WORK_ROOT)
""")

md("""
## 5. Pilot: three search budgets

32 episodes per budget (8 per quadrant). Prints success and measured seconds per control step for each batch of 8 episodes.
""")
code("""
results = {}
for samples, horizon in CONFIGS:
    out = OUTPUT_ROOT / f"N{samples}_H{horizon}"
    run([TRAIN_PY, MPPI_PY, "--episodes-per-batch", EPISODES_PER_BATCH, "--samples", samples, "--horizon", horizon,
         "--quadrants", *QUADRANTS, "--out", out], show=r"^Q\\d", cwd=WORK_ROOT)
    results[(samples, horizon)] = json.loads((out / "summary.json").read_text())["summary"]
""")

md("## 6. Summary and cost extrapolation")
code("""
print(f"{'N x H':>9} | {'once':>5} {'end':>5} | {'s/episode':>9} | {'4,096 demos':>11} | {'16,384 demos':>12} | per quadrant (once/end)")
for (samples, horizon), s in results.items():
    per_ep = s["seconds_per_episode"]
    quads = "  ".join(f"Q{q[1]} {v['success_once']:.0%}/{v['success_at_end']:.0%}" for q, v in s["results"].items())
    print(f"{samples:>4} x {horizon:<3} | {s['mean_success_once']:>5.0%} {s['mean_success_at_end']:>5.0%} | {per_ep:>9.1f} | "
          f"{per_ep * 4096 / 3600:>9.1f} h | {per_ep * 16384 / 3600:>10.1f} h | {quads}")
print("\\nGPU:", next(iter(results.values()))["gpu"], "| episodes run in lockstep:", EPISODES_PER_BATCH,
      "| cost scales roughly with N x H and improves with more episodes per batch if GPU memory allows.")
(OUTPUT_ROOT / "pilot_summary.json").write_text(json.dumps({f"N{k[0]}_H{k[1]}": v for k, v in results.items()}, indent=1))
""")

md("## 7. Videos of the best setting (top-down, drawn from the simulator state each step)")
code("""
import numpy as np, imageio.v2 as imageio, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from IPython.display import Video, display

T_POLY = np.array([(-0.1, -0.0625), (0.1, -0.0625), (0.1, -0.0125), (0.025, -0.0125),
                   (0.025, 0.1375), (-0.025, 0.1375), (-0.025, -0.0125), (-0.1, -0.0125)])
GOAL_THETA = 5 / 3 * np.pi

def poly(xy, th):
    c, s = np.cos(th), np.sin(th)
    return T_POLY @ np.array([[c, -s], [s, c]]).T + xy

best = max(results, key=lambda k: (results[k]["mean_success_once"], -results[k]["seconds_per_episode"]))
frames_all = np.load(OUTPUT_ROOT / f"N{best[0]}_H{best[1]}" / "frames.npy", allow_pickle=True).item()
video_dir = OUTPUT_ROOT / "videos"
video_dir.mkdir(exist_ok=True)
for q, frames in frames_all.items():
    for e in range(len(frames[0]["t_th"])):
        path = video_dir / f"N{best[0]}_H{best[1]}_Q{q}_ep{e}.mp4"
        writer = imageio.get_writer(path, fps=20, codec="libx264", quality=7, macro_block_size=1)
        trail = []
        for t, f in enumerate(frames):
            fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=100)
            ax.set_facecolor("#f6f6f2")
            ax.add_patch(Polygon(poly(f["goal"][e], GOAL_THETA), closed=True, fc="#cfcfcf", ec="#7a7a7a", ls="--"))
            ok = bool(f["success"][e])
            ax.add_patch(Polygon(poly(f["t_xy"][e], f["t_th"][e]), closed=True, fc="#2e9e4f" if ok else "#c8453a",
                                 ec="#333", alpha=0.9))
            trail.append(f["tip"][e][:2])
            tr = np.array(trail[-40:])
            ax.plot(tr[:, 0], tr[:, 1], color="#1f6feb", lw=1, alpha=0.6)
            ax.plot(*f["tip"][e][:2], "o", color="#1f6feb")
            ax.set_xlim(-0.45, 0.15), ax.set_ylim(-0.38, 0.38), ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"MPPI N{best[0]} H{best[1]}  Q{q} ep{e}  step {t + 1}/100{'  SUCCESS' if ok else ''}", fontsize=8)
            fig.tight_layout(); fig.canvas.draw()
            writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3]); plt.close(fig)
        writer.close()
        print("saved", path)
for path in sorted(video_dir.glob("*.mp4"))[:4]:
    display(Video(str(path), embed=True, width=360))
""")

md("""
## 8. What to send back

`pilot_summary.json` and the `videos/` folder in this session's Drive folder (or a screenshot of the section 6 table).
Decision rule: if a budget reaches about 80% success-once in every quadrant at an affordable cost, use it to generate the
BC demonstrations (with action-noise injection, as in run 4). If not, the next lever is a larger budget or more episodes per
batch; the table shows how cost scales.
""")

nb = new_notebook(cells=cells, metadata={
    "accelerator": "GPU",
    "colab": {"gpuType": "L4", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
})
nbformat.validate(nb)
nbformat.write(nb, OUT)
print("wrote", OUT, f"({OUT.stat().st_size / 1e6:.2f} MB, {len(cells)} cells)")

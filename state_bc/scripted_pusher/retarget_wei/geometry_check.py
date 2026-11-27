"""Overlay the Wei pusher contact points (in the slider's own frame) on our T outline, to check shape and angle convention."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from scripted_pusher import T_POLY

d = np.load(sys.argv[1])
st, sl, ends = d["state"], d["slider"], d["ends"]
v_sl = np.r_[np.linalg.norm(np.diff(sl[:, :2], axis=0), axis=1), 0] + np.r_[np.abs(np.diff(sl[:, 2])), 0] * 0.05
moving = v_sl > 0.0015                         # the slider is being pushed this step
rel = st[:, :2] - sl[:, :2]
pts = []
for th, r in zip(sl[moving, 2], rel[moving]):
    c, s = np.cos(-th), np.sin(-th)
    pts.append([c * r[0] - s * r[1], s * r[0] + c * r[1]])
pts = np.array(pts)
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
for ax, (name, M) in zip(axes, [("as is", np.eye(2)), ("rot +90", np.array([[0, -1], [1, 0]])),
                                ("rot 180", -np.eye(2)), ("rot -90", np.array([[0, 1], [-1, 0]]))]):
    ax.add_patch(Polygon(T_POLY, closed=True, fc="#f3c9c4", ec="#c8453a"))
    p = pts @ M.T
    ax.scatter(p[::7, 0], p[::7, 1], s=1, c="#1f6feb", alpha=0.3)
    ax.set_xlim(-0.2, 0.2), ax.set_ylim(-0.15, 0.2), ax.set_aspect("equal"), ax.set_title(f"Wei contacts, {name}")
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=70)
# how well does each rotation fit: fraction of contacts within 0.5-1.5 cm outside our outline
from scripted_pusher import dist_to_poly, point_in_poly
for name, M in [("as is", np.eye(2)), ("rot +90", np.array([[0, -1], [1, 0]])), ("rot 180", -np.eye(2)),
                ("rot -90", np.array([[0, 1], [-1, 0]]))]:
    p = pts[::20] @ M.T
    dd = np.array([0 if point_in_poly(x, T_POLY) else dist_to_poly(x, T_POLY) for x in p])
    inside = np.mean([point_in_poly(x, T_POLY) for x in p])
    print(f"{name:8s} inside our T {inside:.0%}, median distance to outline {np.median(dd)*100:.1f} cm")

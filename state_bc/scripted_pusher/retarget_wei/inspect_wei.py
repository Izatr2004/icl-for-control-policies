"""Inspect the Wei et al. Push-T demos: shapes, ranges, speeds, goal, and how the slider pose relates to its geometry."""
import sys

import numpy as np
import zarr

root = zarr.open(sys.argv[1], mode="r")
ends = root["meta/episode_ends"][:]
act, st, sl, tg = (root[f"data/{k}"][:] for k in ("action", "state", "slider_state", "target"))
print("frames", len(act), "episodes", len(ends))
for name, a in (("action", act), ("pusher state", st), ("slider", sl), ("target", tg)):
    print(f"{name:12s} shape {a.shape}  min {a.min(0).round(3)}  max {a.max(0).round(3)}")
starts = np.concatenate([[0], ends[:-1]])
lens = ends - starts
print("episode length (10 Hz steps): min", lens.min(), "median", int(np.median(lens)), "max", lens.max())
# speeds
v_push = np.linalg.norm(np.diff(st[:, :2], axis=0), axis=1) / 0.1
v_sl = np.linalg.norm(np.diff(sl[:, :2], axis=0), axis=1) / 0.1
mask = np.ones(len(v_push), bool)
mask[ends[:-1] - 1] = False
print("pusher speed m/s: median %.3f  p90 %.3f  max %.3f" % (np.median(v_push[mask]), np.percentile(v_push[mask], 90), v_push[mask].max()))
print("slider speed m/s (when moving >1mm/step): median %.3f" % np.median(v_sl[mask][v_sl[mask] > 0.01]))
# start / end per episode
d_start = np.linalg.norm(sl[starts, :2] - tg[starts, :2], axis=1)
th_start = np.abs((sl[starts, 2] - tg[starts, 2] + np.pi) % (2 * np.pi) - np.pi)
d_end = np.linalg.norm(sl[ends - 1, :2] - tg[ends - 1, :2], axis=1)
th_end = np.abs((sl[ends - 1, 2] - tg[ends - 1, 2] + np.pi) % (2 * np.pi) - np.pi)
print("start dist to goal m: median %.3f max %.3f; start angle err rad: median %.2f max %.2f" % (
    np.median(d_start), d_start.max(), np.median(th_start), th_start.max()))
print("end dist to goal m: median %.4f p90 %.4f; end angle err: median %.3f p90 %.3f" % (
    np.median(d_end), np.percentile(d_end, 90), np.median(th_end), np.percentile(th_end, 90)))
# action vs state lag: is action the next pusher position?
nxt = st[1:, :2] - act[:-1]
print("|action_t - pusher_{t+1}| median %.4f m" % np.median(np.linalg.norm(nxt[mask], axis=1)))
# path length of the pusher per episode
pl = np.array([np.sum(np.linalg.norm(np.diff(st[a:b, :2], axis=0), axis=1)) for a, b in zip(starts, ends)])
print("pusher path length per episode m: median %.2f max %.2f" % (np.median(pl), pl.max()))
# time spent "active" (pusher moving > 5 mm/s)
act_frac = np.mean(v_push[mask] > 0.005)
print("fraction of steps with pusher moving: %.2f" % act_frac)
# how close does the pusher get to the slider centre? (bounds T size)
dmin = np.linalg.norm(st[:, :2] - sl[:, :2], axis=1)
print("pusher-to-slider-pose distance: min %.3f p1 %.3f p5 %.3f" % (dmin.min(), np.percentile(dmin, 1), np.percentile(dmin, 5)))
np.savez(sys.argv[2], ends=ends, action=act, state=st, slider=sl, target=tg)

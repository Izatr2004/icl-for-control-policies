"""For each executed push: compare the model's predicted T pose change with what actually happened."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gymnasium as gym
import numpy as np
import torch

import pusht_vargoal  # noqa: F401
from planner_pusher import N_LOC, R_LOC, PlannerParams, PlannerPusher, predict
from scripted_pusher import wrap

params = PlannerParams(**json.loads(sys.argv[1])) if len(sys.argv) > 1 else PlannerParams()
env = gym.make("PushTVarGoal-v1", obs_mode="state_dict", control_mode="pd_joint_delta_pos", sim_backend="physx_cpu", num_envs=1)
pol = PlannerPusher(params, seed=0, randomize=False)
rows, steps_in = [], {"approach": 0, "push": 0, "retreat": 0, "hold": 0, "plan": 0}
for ep in range(10):
    obs, _ = env.reset(seed=3000 + ep)
    pol.reset(obs)
    prev_phase, start = None, None
    for t in range(100):
        a = pol.act(obs)
        steps_in[pol.phase] = steps_in.get(pol.phase, 0) + 1
        s = pol._parse(obs)
        if pol.phase == "push" and prev_phase != "push":
            start = (np.array([*s["t_xy"], s["t_th"]]), pol.ci, pol.length)
        if prev_phase == "push" and pol.phase != "push" and start is not None:
            pose0, ci, L = start
            new, *_ = predict(pose0, np.array([L]), params)
            pred = new[ci, 0] - pose0
            act = np.array([*s["t_xy"], s["t_th"]]) - pose0
            rows.append((L, np.linalg.norm(pred[:2]), np.linalg.norm(act[:2]), wrap(pred[2]), wrap(act[2]),
                         np.linalg.norm(pred[:2] - act[:2]), abs(wrap(pred[2] - act[2]))))
        prev_phase = pol.phase
        obs, *_ = env.step(torch.as_tensor(a, dtype=torch.float32)[None])
r = np.array(rows)
print("pushes:", len(r))
print("push length L        median %.3f" % np.median(r[:, 0]))
print("predicted |dxy|      median %.3f   actual |dxy| median %.3f" % (np.median(r[:, 1]), np.median(r[:, 2])))
print("predicted |dth|      median %.3f   actual |dth| median %.3f" % (np.median(np.abs(r[:, 3])), np.median(np.abs(r[:, 4]))))
print("position error       median %.3f   p90 %.3f" % (np.median(r[:, 5]), np.percentile(r[:, 5], 90)))
print("angle error          median %.3f   p90 %.3f" % (np.median(r[:, 6]), np.percentile(r[:, 6], 90)))
print("angle sign agreement %.0f%%" % (100 * np.mean(np.sign(r[:, 3]) == np.sign(r[:, 4]))))
print("actual/predicted translation ratio median %.2f" % np.median(r[:, 2] / np.maximum(r[:, 1], 1e-4)))
print("steps per phase over 10 episodes:", steps_in)

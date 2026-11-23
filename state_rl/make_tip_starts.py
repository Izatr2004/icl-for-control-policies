"""Build the library of robot start poses used by PushTVarGoal-v1.

Drives the stick tip from ManiSkill's rest pose to random (x, y) points at table
height with the end-effector position controller, and keeps the final joint
positions when the tip lands within 1 cm. Each reset then draws one of these
poses, which randomizes the tip start without solving inverse kinematics on
the GPU. Joint positions are controller-independent, so the library also works
with pd_joint_delta_pos.

Run once (CPU simulation is enough):  python state_rl/make_tip_starts.py
"""
import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # noqa: F401
from mani_skill.utils.structs.pose import Pose

from pusht_vargoal import TIP_BOX  # single source of truth for the region

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=600)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default=str(Path(__file__).parent / "assets" / "tip_start_qpos.npz"))
args = p.parse_args()

env = gym.make("PushT-v1", obs_mode="state", control_mode="pd_ee_delta_pos",
               num_envs=1, sim_backend="physx_cpu", robot_init_qpos_noise=0.0)
u = env.unwrapped
rng = np.random.default_rng(args.seed)
qpos, tips, errs = [], [], []
for i in range(args.n):
    env.reset(seed=args.seed)
    u.tee.set_pose(Pose.create_from_pq(p=torch.tensor([[5.0, 5.0, 0.02]])))  # block out of the way
    z0 = u.agent.tcp.pose.p[0, 2].item()
    tgt = torch.tensor([rng.uniform(*TIP_BOX[0]), rng.uniform(*TIP_BOX[1]), z0])
    for _ in range(60):
        err = tgt - u.agent.tcp.pose.p[0]
        env.step(torch.clamp(err / 0.1, -1, 1).unsqueeze(0))
    for _ in range(10):  # settle with zero command
        env.step(torch.zeros(1, 3))
    tip = u.agent.tcp.pose.p[0]
    e = float((tip[:2] - tgt[:2]).norm())
    if e < 0.01 and abs(float(tip[2]) - z0) < 0.01:
        qpos.append(u.agent.robot.get_qpos()[0].numpy())
        tips.append(tip[:2].numpy())
    errs.append(e)

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
np.savez(args.out, qpos=np.array(qpos, np.float32), tip_xy=np.array(tips, np.float32))
meta = dict(n_requested=args.n, n_kept=len(qpos), seed=args.seed, tip_box=TIP_BOX,
            max_err_kept_m=0.01, mani_skill="3.0.1")
Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=2))
print(meta)

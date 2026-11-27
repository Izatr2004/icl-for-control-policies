"""Quick sweep: success within a long horizon (and within 100 steps) for several parameter settings."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gymnasium as gym
import numpy as np
import torch

import pusht_vargoal  # noqa: F401
from scripted_pusher import PusherParams, ScriptedPusher

configs = json.loads(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
horizon = int(sys.argv[3]) if len(sys.argv) > 3 else 300
quads = [1, 2, 3, 4]
for name, cfg in configs.items():
    randomize = cfg.pop("randomize", True)
    params = PusherParams(**cfg)
    firsts, ends = [], []
    for q in quads:
        env = gym.make("PushTVarGoal-v1", obs_mode="state_dict", control_mode="pd_joint_delta_pos",
                       sim_backend="physx_cpu", num_envs=1, goal_quadrants=(q,), max_episode_steps=horizon)
        pusher = ScriptedPusher(params, seed=q, randomize=randomize)
        for i in range(n):
            obs, _ = env.reset(seed=1001 + 1000 * q + i)
            pusher.reset(obs)
            first, succ = None, []
            for t in range(horizon):
                obs, r, term, trunc, info = env.step(torch.as_tensor(pusher.act(obs), dtype=torch.float32)[None])
                succ.append(bool(info["success"][0]))
                if first is None and succ[-1]:
                    first = t + 1
            firsts.append(first)
            ends.append(succ[-1])
        env.close()
    ok = [f for f in firsts if f]
    print(f"{name:12s} success within {horizon}: {len(ok)}/{len(firsts)}  within 100: {sum(f <= 100 for f in ok)}"
          f"  at end of {horizon}: {sum(ends)}  median first {np.median(ok) if ok else float('nan'):.0f}", flush=True)

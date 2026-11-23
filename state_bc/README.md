# Full-state BC teachers (Nathan)

Behavioral cloning teachers for ManiSkill `PushT-v1` with `obs_mode="state"` and `pd_joint_delta_pos`,
trained on the released ManiSkill PPO demonstrations.

- **Colab:** upload [`notebooks/Train_PushT_State_BC.ipynb`](../notebooks/Train_PushT_State_BC.ipynb) and run it on a GPU runtime.
  Outputs go to `MyDrive/182_final_project/state_bc/<session>/`.
- **Code:** [`pusht_state_bc.py`](pusht_state_bc.py) has `audit`, `prepare`, `train` and `eval` subcommands.
- **Notebook source:** edit `pusht_state_bc.py` or `build_notebook.py`, then run `python state_bc/build_notebook.py`.
  The notebook embeds the module, so never edit the notebook's module cell by hand.
- **Offline tests** (no ManiSkill or GPU needed): `python -m unittest state_bc/test_offline.py -v`.

## Setup

| | |
|---|---|
| ManiSkill | v3.0.1, commit `a4a4f92`, same install as the camera PPO notebook |
| State (31 numbers) | `agent/qpos` (7), `agent/qvel` (7), `extra/tcp_pose` (7), `extra/goal_pos` (3), `extra/obj_pose` (7). The notebook's `observation_contract.json` is the authority. |
| Action (7 numbers) | joint position deltas, normalized to [-1, 1] |
| Data | 999 successful released PPO episodes, replayed with `--use-env-states` into state observations; 90/10 train/validation split by episode |
| Teacher | MLP, 2×256 ReLU, tanh output, MSE; normalization stored inside the checkpoint |
| Selection | closed-loop validation success on seeds from 1001; final test on seeds from 5001 |

Features with a training std below 1e-3 (such as the fixed goal position) are centered but not scaled,
so simulator noise does not get amplified.

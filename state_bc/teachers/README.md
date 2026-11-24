# Full-state BC teacher weights (Nathan)

Frozen weights from the runs logged in [`../README.md`](../README.md). They were copied from Nathan's Google Drive
(`MyDrive/182_final_project/state_bc/<session>/`) on 2026-11-23, and each loads on CPU with the code in `pusht_state_bc.py`.

| File | What it is | Environment | Test result (success-once / at end) | Drive session | SHA-256 |
|---|---|---|---|---|---|
| **`bc_vargoal_run4.pt`** | **The frozen full-state BC teacher for v1.** Distilled from Jimmy's PPO teacher (run 4) | `PushTVarGoal-v1` (Jimmy's shared env) | Q1 92/90, Q2 94/93, Q3 89/84, Q4 91/88 (100 episodes per quadrant, seeds 6001–9001) | `20261123T230947Z_run4_bc_from_jimmy/teachers/bc_teacher.pt` | `3b18812bed110bf55f7c5a05ac8ed0a8c476a9d94ad1faf8e323df8b57d1b58d` |
| `bc_fixedgoal_run2.pt` | Fixed-goal BC teacher (run 2, DAgger round 2) | `PushT-v1` (fixed goal) | 98 / 78 (100 episodes, seed 5001) | `20261123T031621Z_run2_dagger/runs/bc_dagger_seed0/teacher.pt` (copy of `round2/best.pt`) | `0e014d117583d0bdb3cdca6d9de5f4eebd7381a9d823196bc13fff2e872eeb45` |
| `bc_run3b_own_env.pt` | BC distilled from run 3b's PPO. A record only; not used for transformer data | `PushT-VariableGoal-v1` (our own env) | 99 / 33.5 (200 episodes, seed 5001) | `20261123T213152Z_run3b_warmstart/teachers/bc_teacher.pt` | `0d0d6b5d27b26c45b25993e43bd899755452f0c091e8678e5383b83f22fd9569` |
| `ppo_run3b_own_env.pt` | Nathan's PPO, warm-started from ManiSkill's released expert (run 3b). A record only | `PushT-VariableGoal-v1` (our own env) | 98.5 / 29 (200 episodes, seed 5001) | `20261123T213152Z_run3b_warmstart/teachers/ppo_expert.pt` | `fbd5c8c061b1189668abf7ba1ceedbb9d623f0b302bb72819a39068b06634f82` |

**Not duplicated here:** v1's PPO teacher is Jimmy's `state_rl/teachers/ppo_vargoal_seed1_50M.pt` (branch `jimmy-state-rl`, commit c87d8ae,
SHA-256 `430807142173472f7721a2c11768bb793b00e92c7178abb1d6ae811265ffe917`). Run 4's Drive copy of it (`teachers/ppo_expert.pt`) is byte-identical (checked 2026-11-23).

**Loading:**

```python
import pusht_state_bc as bc
bc_teacher = bc.load_policy("state_bc/teachers/bc_vargoal_run4.pt", device="cpu")      # BCPolicy; .act(state[31]) -> action[7]
ppo = bc.load_released_ppo("state_bc/teachers/ppo_run3b_own_env.pt", 31, 7)             # ManiSkill ppo.py actor mean
```

All of them take the full 31-number state, goal included (`obs_mode="state"`), and output 7 `pd_joint_delta_pos` actions in [-1, 1].
BC checkpoints store their own normalization statistics and training config (`config`, `extra`).

**Transformer weights** (`pusht_icl.py`, about 39 MB each) are not here yet. They will be added once the v1 run finishes and is evaluated.

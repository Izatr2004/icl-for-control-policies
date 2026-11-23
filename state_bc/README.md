# Full-state BC teachers (Nathan)

Behavioral cloning (BC) teachers for ManiSkill `PushT-v1` with full simulator state (`obs_mode="state"`) and
joint control (`pd_joint_delta_pos`). This file is the running log for this part of the project: every run,
its results, what we concluded, and what we changed next. **Add to it; don't rewrite history.**

Contents: [Role](#role-in-the-project) · [How to run](#how-to-run) · [Setup](#setup-shared-with-the-team) ·
[Run 1](#run-1-bc-on-released-demos) · [Run 2](#run-2-more-expert-data--dagger) · [Next steps](#next-steps-proposed-not-started) ·
[Run 3](#run-3-variable-goal-ppo-expert--distilled-bc) ·
[Decisions](#decisions-log) · [Ideas and open questions](#ideas-and-open-questions) · [Changelog](#changelog)

## Role in the project

The project asks whether a frozen GPT-2-style transformer can infer an unseen control policy from a few
demonstrations (Garg et al. 2022, with policies in place of functions). The team builds the policy population
as a 2 × 2:

| | BC | PPO |
|---|---|---|
| **Full state** | **Nathan (this folder)** | Jimmy |
| **Camera** (RGB + robot joints, no block pose) | Izat | Neil |

A BC teacher is one "function" in Garg's sense. The transformer is trained across **many** teachers, so this part
must eventually produce several **competent and behaviorally different** BC teachers, plus rollouts from each
for the transformer's context and queries. State-based BC and PPO teachers share one observation/action contract,
so they can be mixed in one state-based learner.

## How to run

- **Transformer (v1):** [`notebooks/Train_PushT_ICL_Transformer.ipynb`](../notebooks/Train_PushT_ICL_Transformer.ipynb), built by
  `build_notebook_icl.py`; code in `pusht_icl.py`; tests `python -m unittest state_bc/test_icl_offline.py` (needs `transformers`).
- **Teacher notebook (run 4, BC from Jimmy's PPO in the shared `PushTVarGoal-v1`):**
  [`notebooks/Train_PushT_VarGoal_BC_from_Jimmy.ipynb`](../notebooks/Train_PushT_VarGoal_BC_from_Jimmy.ipynb).
  Upload to Colab, choose an L4 GPU, run top to bottom. Outputs go to `MyDrive/182_final_project/state_bc/<session>/`.
- **Earlier notebooks (kept as run, not rebuilt):** run 1 [`Train_PushT_State_BC.ipynb`](../notebooks/Train_PushT_State_BC.ipynb),
  run 2 [`Train_PushT_State_BC_DAgger.ipynb`](../notebooks/Train_PushT_State_BC_DAgger.ipynb) (both fixed-goal `PushT-v1`), and
  runs 3/3b [`Train_PushT_VariableGoal_PPO_BC.ipynb`](../notebooks/Train_PushT_VariableGoal_PPO_BC.ipynb) (our own `PushT-VariableGoal-v1`).
- **Code:** [`pusht_state_bc.py`](pusht_state_bc.py), with subcommands `ppo`, `audit`, `check-env`, `prepare`, `collect`, `train`, `eval`,
  `eval-quadrants`; [`variable_goal_env.py`](variable_goal_env.py) defines our `PushT-VariableGoal-v1`. `--env-id PushTVarGoal-v1` uses
  Jimmy's shared environment (`pusht_vargoal.py` must sit next to `pusht_state_bc.py`); `--env-id PushT-v1` the original.
- **Notebook source:** notebooks embed the modules. Edit them or the builder, then run `python state_bc/build_notebook_run4.py`
  (run 4; needs Jimmy's commit c87d8ae in the local clone, `git fetch` first). Never edit the notebook's module cells by hand.
- **Offline tests** (no ManiSkill or GPU): `python -m unittest state_bc/test_offline.py -v`.
  Needs `torch`, `numpy`, `h5py`, `nbformat`.

## Setup shared with the team

Measured by the `audit` step in run 1 (`observation_contract.json` in each session folder is the authority).

| Item | Value |
|---|---|
| ManiSkill | v3.0.1, commit `a4a4f92`, same pinned install as Neil's camera PPO notebook (Python 3.11, torch 2.6.0 + CUDA 12.4, SAPIEN 3.0.3) |
| Task | `PushT-v1`, robot `panda_stick`, GPU physics (`physx_cuda`) |
| Episode | 100 control steps at 20 Hz (sim at 100 Hz) |
| Success | T overlap with goal ≥ 0.90 (`evaluate()`); there is no end-zone condition in this version |
| State (31) | `agent/qpos` [0:7], `agent/qvel` [7:14], `extra/tcp_pose` [14:21], `extra/goal_pos` [21:24], `extra/obj_pose` [24:31] |
| Action (7) | joint position deltas, normalized to [-1, 1]; the simulator clips commands to this range |
| Metrics | `success_once`: the T reached the goal at some step. `success_at_end`: still at the goal at step 100. Episodes run the full 100 steps with no early reset. |

## Run 1: BC on released demos

**Date:** 2026-11-22 (Pacific). **Drive session:** `state_bc/20261123T023221Z`. **Notebook:** `Train_PushT_State_BC.ipynb`. Colab L4.

### What it did
1. Downloaded ManiSkill's released PPO demos for `pd_joint_delta_pos` (999 successful episodes) and the PPO checkpoint that produced them.
2. Replayed the demos with `--use-env-states` to get state observations.
3. Split **whole episodes** 90/10 into train/validation (split seed 0).
4. Trained an MLP teacher, 31 → 256 → 256 → 7, ReLU hidden layers, tanh output, MSE loss.
   Adam at learning rate 3e-4 with cosine decay, batch size 1024, 100k iterations, training seed 0.
   State normalization uses training-split mean and std, stored inside the checkpoint.
5. Evaluated closed-loop success on 100 validation episodes (seed 1001) every 10k iterations, keeping the best checkpoint.
6. Tested the selected checkpoint once on 100 fresh episodes (seed 5001), next to the released PPO expert.

### Data facts
- Only **761 of 999** demos survived the replay (685 train / 76 validation, 66,233 / 7,047 transitions).
  The replay tool drops episodes that are not successful at their last step when replayed; we did not investigate further.
- Recorded demo actions range up to about ±3.1, and **11.5%** are at or beyond ±1. PPO outputs unbounded means and the
  simulator clips them, so labels clipped to [-1, 1] equal what the expert actually executed.
- Features only centered (training std < 1e-3): indices 21, 22, 23 (fixed goal position) and 26 (block height).

### Results

| Policy | Seeds | Success once | Success at end | Mean steps to first success |
|---|---|---|---|---|
| Released PPO expert | validation (1001) | 98% | 65% | 31.5 |
| **BC teacher, run 1** | validation (1001), best checkpoint | 26% | 12% | |
| Released PPO expert | test (5001) | **100%** | **72%** | 30.5 |
| **BC teacher, run 1** | test (5001) | **19%** | **10%** | 43.7 |

Training curve (validation, 100 episodes):

| Iteration | 10k | 20k | 30k | 40k | 50k | 60k | 70k | 80k | 90k | 100k |
|---|---|---|---|---|---|---|---|---|---|---|
| Success once | 13% | 15% | 17% | 19% | 21% | 20% | 22% | 23% | 22% | 26% |
| Success at end | 5% | 4% | 10% | 6% | 7% | 8% | 7% | 9% | 12% | 12% |

Action MSE (normalized action scale): training fell from 0.034 to **0.0076**, while validation fell from 0.060 to
**0.043** and stayed flat from about 30k iterations. Training took **3.1 minutes** including evaluations.

### Diagnosis
- **Overfitting:** validation error is about 6× training error and stopped improving early. More iterations or a
  larger network would not fix this; more data would.
- **Compounding errors:** BC only sees states the expert visited. A small mistake leads to an unfamiliar state, which leads
  to a bigger mistake. Push-T is contact-heavy, so small errors change where the block goes.
- **Not enough data:** about 66k transitions from 685 episodes.
- **Not the cause:** the pipeline. The released expert reaches 98–100% through the same evaluation code, so the
  environment, seeds and metrics work.

### Problems found while running
- The first attempt (session `20261123T021918Z`) crashed in the smoke test: `env.spec.max_episode_steps` is `None` for
  GPU-vectorized environments. Fixed by using ManiSkill's `gym_utils.find_max_episode_steps_value`, as its PPO baseline does.
  The same bug had silently made the released PPO check fail and written `max_episode_steps: null` into that session's contract.

### Conclusion
The pipeline works end to end, but at 10% success-at-end this is not a competent teacher. Since the expert checkpoint
loads and runs, we are not limited to the 999 recorded demos.

## Run 2: more expert data + DAgger

**Date:** 2026-11-22 (Pacific). **Drive session:** `state_bc/20261123T031621Z_run2_dagger`.
**Notebook:** `Train_PushT_State_BC_DAgger.ipynb`. Colab L4. **Status:** completed; results [below](#run-2-results).

### What changes, and why

| Change | Run 1 | Run 2 | Why |
|---|---|---|---|
| Expert rollouts | none | 8,192 episodes by the released expert (about 819k transitions) | Fix overfitting with about 12× more data |
| Action noise during expert rollouts | none | std 0, 0.1, 0.2, 0.3 on the [-1, 1] scale, cycled per batch of 1,024; labels stay noise-free | Record how the expert recovers from slightly off-course states |
| DAgger | none | 4 rounds: the latest BC teacher drives 2,048 episodes, the expert labels every visited state, retrain | Fix compounding errors by training on the states BC actually reaches |
| Training per round | one run | from scratch on all data so far (released demos + expert rollouts + DAgger rounds) | Standard DAgger aggregation |
| Validation evaluation frequency | every 10k iterations | every 25k iterations | Five trainings instead of one; checkpoints are coarser but the run stays under about an hour |
| Released demo labels | raw | clipped to [-1, 1] at load time | Same values the training already used, now explicit in the data files |
| Data files | one `.npz` with both splits | separate transition files (`demos_train`, `demos_val`, `expert_rollouts`, `dagger_round{r}`), each with a `.json` sidecar | Needed to aggregate DAgger data and to record where each transition came from |
| Collection seeds | | expert 100,000+, DAgger 200,000+r·1000 | Kept away from validation (1001) and test (5001) seeds |

**Unchanged:** teacher architecture, loss, optimizer, learning rate, batch size, 100k iterations, normalization rule,
validation and test seeds, metrics, checkpoint selection by validation success.

### What to look at when it runs
- **Round 0 vs run 1:** the effect of more expert data alone.
- **Rounds 1–4:** whether DAgger keeps improving validation success and the driving policy's own success during collection.
- **Expert success under noise** (printed during collection): if the expert fails often at noise 0.3, that level adds mostly bad states.
- **Validation action MSE:** should drop well below run 1's 0.043 floor if overfitting was the problem.
- **Target:** success-once close to the expert's 98–100%.

### Run 2 results

**Headline (test seeds 5001, 100 episodes, selected checkpoint = round 2):**

| Policy | Success once | Success at end | Mean steps to first success |
|---|---|---|---|
| BC teacher, run 1 | 19% | 10% | 43.7 |
| **BC teacher, run 2** | **98%** | **78%** | 31.4 |
| Released PPO expert | 100% | 72% | 30.5 |

The BC teacher now matches the expert. Its 78% vs 72% success-at-end is within noise for 100 episodes
(standard error about 4–5 points), so read it as "same as the expert", not "better".

**Expert rollouts (8,192 episodes, 819,200 transitions).** Expert success by action-noise level, averaged over two batches each:

| Noise std | 0.0 | 0.1 | 0.2 | 0.3 |
|---|---|---|---|---|
| Success once | 98% | 97% | 93% | 65–66% |
| Success at end | 69–70% | 68–69% | 46–47% | 15–17% |

Noise up to 0.1 barely hurts the expert. At 0.3 it often fails, so those episodes are mostly off-course states.

**Per round (validation seeds 1001, best of 4 evaluations per round):**

| Round | Training data | Transitions | Val success once | Val success at end | Driving BC success once / at end (during collection) | Final val action MSE |
|---|---|---|---|---|---|---|
| 0 | demos + expert rollouts | 885,433 | 98% | 71% | | 0.026 |
| 1 | + DAgger round 1 | 1,090,233 | 97% | 73% | 96% / 70% (round 0 teacher) | 0.030 |
| 2 | + DAgger round 2 | 1,295,033 | **100%** | **77%** | 96% / 69% (round 1 teacher) | 0.030 |
| 3 | + DAgger round 3 | 1,499,833 | 99% | 69% | 97% / 73% (round 2 teacher) | 0.032 |
| 4 | + DAgger round 4 | 1,704,633 | 98% | 68% | 97% / 68% (round 3 teacher) | 0.033 |

Each training took 2.8 minutes. The demo replay took about 2 minutes, and the first audit in a fresh runtime downloads the PhysX GPU library.

### Run 2 analysis
- **More expert data did almost all the work.** Round 0, before any DAgger, already reached 98% / 71% on validation,
  up from run 1's 26% / 12%. Run 1's problem was mainly too little data.
- **DAgger added little or nothing measurable.** Rounds 1–4 sit between 97–100% success-once and 68–77% success-at-end,
  all within the noise of 100-episode evaluations. Round 2 was selected because it had the highest validation number,
  but picking the best of 20 noisy evaluations inflates that number. The fresh test seeds (98% / 78%) are the honest estimate.
- **Why DAgger had little to fix:** the round 0 teacher already succeeded 96% of the time while driving, so it rarely
  reached states the expert data did not cover. The noisy expert rollouts probably supplied most of the recovery data DAgger is meant to add.
- **Validation action MSE rose slightly with DAgger** (0.026 → 0.033). Validation states come from expert demos, while
  later rounds add more BC-visited states, so the same 2 × 256 network spreads over a wider mix. Closed-loop success did not drop, so this is not a problem.
- **The remaining gap is success-at-end,** which the expert shares: both reach the goal almost always but leave it in roughly 25–30% of episodes.

### Run 2 conclusion
The first full-state BC teacher is competent: `runs/bc_dagger_seed0/teacher.pt` (copy of `round2/best.pt`) in the session folder.
For future BC teachers distilled from an expert, **expert rollouts with action noise plus round 0 training is probably enough.**
DAgger can be dropped or cut to one round to save about 15 minutes per teacher. That is an inference from one run with one expert,
so check it again when distilling a different expert.

## Next steps (proposed, not started)
1. **ICL rollout collection:** add a `collect` mode where the teacher both drives and labels (recording its own actions),
   and agree on the rollout format with Jimmy and whoever trains the transformer.
2. **Teacher diversity:** start producing several behaviorally different BC teachers (see [Making many different teachers](#making-many-different-teachers)).
   With the run 2 recipe, each teacher costs about 10 minutes of collection and training.
3. **Diversity checks:** action differences on shared states and a teacher-identification classifier, before scaling up.

(Superseded on 2026-11-23 by the variable-goal v1 design; see the v1 plan and run 3.)

## Run 3: variable-goal PPO expert + distilled BC

**Status:** built 2026-11-23, not yet run. **Notebook:** `Train_PushT_VariableGoal_PPO_BC.ipynb`.

### Purpose
v1 needs teachers that push the T to **any** goal, with the goal in their observation (see the v1 plan). This run builds
the variable-goal environment, trains a goal-conditioned PPO expert, distills a BC teacher from it, and validates both per quadrant.

### Environment: `PushT-VariableGoal-v1` ([`variable_goal_env.py`](variable_goal_env.py))
| | PushT-v1 | PushT-VariableGoal-v1 |
|---|---|---|
| Goal T position | fixed at (−0.156, −0.100) | uniform within **±10 cm** of that point in x and y, new every episode |
| Goal quadrants | | Q1 (+x,+y), Q2 (−x,+y), Q3 (−x,−y), Q4 (+x,−y) around that point, world frame; quadrant drawn uniformly from the allowed set (`goal_quadrants`, default all four), then a uniform point inside it |
| Goal rotation | 5π/3 | unchanged |
| Success check | one fixed goal transform computed at load | **per-environment transform, recomputed at every reset** |
| Movable T start | random position in a fixed box, random rotation | unchanged (same absolute box) |
| Stick tip start | fixed near (−0.32, 0.28), ±0.02 rad joint noise | **random within ±15 cm of the centre** (quadrant from `start_quadrants`, default all four), at least 15 cm from the T's centre, placed by inverse kinematics at PushT-v1's rest tip height and stick orientation. Independent of the goal. Added 2026-11-23; see below. |
| Observation | `goal_pos` constant | `goal_pos` varies (same 31 fields) |
| Reward | pose-based dense reward | unchanged; its translation term already reads the live goal pose, and its rotation term uses the fixed goal rotation |

The goal square (x −0.256 to −0.056, y −0.200 to 0.000) lies inside the movable T's original spawn box, so goals stay in a region the
original task already covers. **This quadrant definition is the default agreed for now (2026-11-23); the group can change it** via
`GOAL_HALF_WIDTH` and the centre in `variable_goal_env.py`.

**Built-in check (`check-env`):** places a stand-in T exactly on each episode's goal and runs the environment's overlap computation;
all must count as success, and none may after a 5 cm shift. It also compares the simulator's actual stick-tip position after reset
with the inverse-kinematics target (must agree within 2 cm, which catches a wrong robot-base offset or kinematic chain) and allows
at most 5% of episodes to fall back to PushT-v1's start. Runs before PPO so a broken environment cannot waste a training run.

### Pipeline and settings
| Step | Settings |
|---|---|
| PPO expert | ManiSkill `ppo.py`, unchanged, run with the new env registered: 1,024 envs, 16 steps, 32 minibatches, 8 epochs, γ 0.99, 50M transitions, seed 1. Same as Neil's PPO settings but with state observations. |
| Expert selection | final checkpoint plus the 4 latest periodic ones, evaluated on 200 validation episodes (seed 1001); best success-once, then success-at-end. Stops if below 80% success-once. |
| Expert data | 16,384 episodes (about 1.6M transitions), action noise cycling 0 / 0.1 / 0.2 / 0.3 per batch, labels noise-free; goals in all four quadrants |
| BC validation data | 1,024 noise-free expert episodes on separate seeds (300,000+) for validation action error |
| BC teacher | MLP 2 × 256 ReLU, tanh output, MSE, Adam 3e-4 cosine, batch 1024, **200k iterations**, **no DAgger** |
| Evaluation | 200 episodes per evaluation (about 50 per quadrant); success reported overall and per goal quadrant |
| Outputs | `teachers/ppo_expert.pt`, `teachers/bc_teacher.pt`, per-quadrant test results in `teachers/summary.json` |

### Changes from run 2, and why
| Change | Why |
|---|---|
| Variable-goal environment | v1 design: goal-conditioned teachers |
| Train our own PPO expert | The released expert only knows the fixed goal. Jimmy's goal-conditioned PPO can replace it later via `PPO_CHECKPOINT`. |
| No released demos; validation data from expert rollouts | Released demos all use the fixed goal |
| No DAgger | Run 2: round 0 already matched the expert |
| Expert episodes 8,192 → 16,384 | More goal variety to cover |
| BC iterations 100k → 200k | Goal-conditioned mapping is harder; about 6 minutes on an L4 if run 2's speed holds |
| Evaluation episodes 100 → 200 | About 50 per quadrant, so per-quadrant numbers are less noisy |
| Selection of the expert by success-once | The expert's job is to reach goals; consistent with the success-once proposal for competence |

### Robot start position (discussed and built 2026-11-23)
The first version of run 3 kept PushT-v1's robot start: a fixed rest posture with about ±0.02 rad joint noise, so the stick tip always
started near (−0.32, 0.28). `CODEX_CONTEXT.md` says starting positions (stick tip and T) should cover Q1–Q4, so this was changed
before running:
- **Sampling:** quadrant uniformly from `start_quadrants`, then a uniform point within ±15 cm of the centre in that quadrant.
  32 candidates per episode; the first at least 15 cm from the T's centre of mass is kept (the T reaches about 14 cm from it),
  else the farthest. Sampled independently of the goal.
- **Placement:** batched damped-least-squares inverse kinematics on the GPU (60 iterations) using `pytorch_kinematics`, which ManiSkill
  already uses for its end-effector controllers. Target: the sampled point at PushT-v1's rest tip height, with the rest stick orientation.
  Initial guess: the rest posture with the base joint turned toward the target. Joint limits enforced.
- **Fallback:** if inverse kinematics misses by more than 1 cm, that episode uses PushT-v1's rest posture, and is flagged.
- **Speed fix: lookup table (2026-11-23).** The first launch of run 3 trained PPO at only about **6,700 SPS** (about 2 hours for 50M
  transitions). Likely cause: during PPO training, some of the 1,024 environments reset on nearly every step, and each reset ran
  60 IK iterations. Not confirmed by a controlled comparison. That run was stopped. Now IK is solved **once per process** for a 64 × 64 grid
  of cell centres covering the ±15 cm square (200 iterations, one GPU batch). Each reset samples a point as before, uses the joint angles of the
  cell containing it (tip within about 3 mm of the sampled point; the actual tip position is recorded), and skips cells the solver could not reach.
  `check-env` reports the reachable fraction of the grid.
- **Records per episode:** `tip_start_xy`, `tip_start_quadrant`, `tee_start_quadrant`, `tip_ik_error`, `tip_ik_fallback`. Collected data
  saves these with the goal records, so start hold-outs and the 2 × 2 test grid can be built from it later.
- `random_tip_start=False` restores PushT-v1's tip start.
- **Unverified until it runs:** reachability of the whole ±15 cm square and the fallback rate; `check-env` reports both.
  Varied starts also make the task harder, so PPO may need more than 50M transitions.

**This does not stop the transformer from "cheating" with its weights.** The start state is visible in the query observation, so the
weights can use it directly; extra visible variety does not force reading the context. What forces context use is that the correct
action depends on something **hidden** (the goal) that only the demos reveal. Start randomization still helps: it matches the context,
makes teachers robust to varied starts, and enables start hold-outs.

**Remaining ways the transformer could avoid the context, and the guards:**
- **Goal leakage in inputs:** drop `goal_pos` (indices 21:24) and anything derived from it from transformer inputs.
- **Query states that hint at the goal:** a mid-episode query state from a teacher rollout (block already moving toward the goal) partly reveals the goal.
  Closed-loop evaluation from fresh starts avoids this; for action-error metrics, report early-step queries separately.
- **Starts correlated with goals:** keep start sampling independent of the goal (it is in `PushT-VariableGoal-v1`).
- **A good average policy:** the no-context and wrong-goal-context controls measure how far the transformer gets without the right demos.
  The ICL effect is the gap between correct context and those controls.

### What to look at when it runs
- `check-env` must pass.
- PPO's own `eval_success_once_mean` should climb; if it stays low, 50M transitions may not be enough for variable goals.
- Per-quadrant success of the expert and BC teacher: any weak quadrant matters, especially Q4, since the transformer will be tested there.
- BC vs expert gap, as in run 2.

### Run 3 result: PPO expert did not learn the task (2026-11-23)
**Drive session:** `state_bc/20261123T194232Z_run3_variable_goal`. PPO trained the full 50M transitions at roughly 8,000–10,000 SPS
(about 1.5 hours). The notebook's 80% gate then stopped the run before BC, as designed.

Final checkpoints on 200 validation episodes (seed 1001), success-once / success-at-end:

| Checkpoint | Overall | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|---|
| ckpt_2976 | 6% / 1% | 7% / 2% | 12% / 0% | 4% / 0% | 4% / 2% |
| ckpt_3001 | 6% / 3% | 7% / 4% | 10% / 5% | 5% / 2% | 4% / 2% |
| **ckpt_3026** (selected) | **11% / 9%** | 16% / 12% | 10% / 7% | 9% / 7% | 9% / 7% |
| ckpt_3051 | 4% / 1% | 2% / 0% | 5% / 2% | 5% / 2% | 4% / 2% |
| final_ckpt | 4% / 2% | 7% / 2% | 2% / 0% | 5% / 4% | 4% / 2% |

ppo.py's own 16-episode evaluations first showed a success (0.0625) partway through training.

**Reading:** the expert is far from usable (needs ≥80%). Success is low in every quadrant alike, so no single quadrant is the problem.
The last checkpoints hover at 4–11% rather than climbing, though 200-episode evaluations are noisy at this level.
Most likely the task became much harder than PushT-v1 (moving goals **and** random tip starts at once), and 50M transitions from random
weights is not enough; PushT-v1's released expert itself comes from ManiSkill's 4,096-environment `ppo_fast.py` setup.
A bug in the new environment is less likely: `check-env` passed (success check follows each goal; tip placement matches the simulator).
Not yet ruled out: an episode length of 100 steps being too short for far starts, and the reward shaping not guiding the stick to distant T positions.

### Run 3b plan: warm-started PPO (built 2026-11-23; ran the same day, results below)
Same notebook (`Train_PushT_VariableGoal_PPO_BC.ipynb`, rebuilt), session label `run3b_warmstart`. Only the PPO step changes:

| Setting | Run 3 | Run 3b | Why |
|---|---|---|---|
| PPO initial weights | random | **ManiSkill's released fixed-goal expert** (`ppo_pd_joint_delta_pos_ckpt.pt`), loaded with ppo.py's `--checkpoint` | It already pushes the T well from random T starts, with the same 31-number state (`goal_pos` included) and controller; it only has to learn that the goal moves and the tip starts elsewhere |
| Parallel environments | 1,024 | **4,096** | ManiSkill's own state Push-T baseline setting; more data per update. Changes the batch to 65,536 transitions (2,048 per minibatch) |
| Learning rate | 3e-4 | **1e-4** | Fine-tuning; a full-size step early could erase the pushing skill. A judgment call, not tuned |
| Transitions | 50M | 50M (unchanged) | |
| New measurement | | The released expert's success on the variable-goal task **before** fine-tuning (200 validation episodes), saved as `released_expert_on_variable_goal.json` | Shows how much of the task it already solves |

Everything else (environment, `check-env`, the 80% gate, expert data, BC, evaluation) is unchanged. Considered but not included: a diagnostic run with
fixed tip starts to see which change made run 3 hard; worth adding if 3b also stalls.

**Expected time (estimate, not measured):** install about 8 minutes; checks and the baseline evaluation a few minutes; PPO 35–90 minutes depending on
speed with 4,096 environments (run 3 managed 8,000–10,000 SPS with 1,024); expert selection, data collection, BC training and testing about 20–25 minutes.
Roughly **1¼ to 2 hours** in total.

### Run 3b result: both variable-goal teachers work (2026-11-23)
**Drive session:** `state_bc/20261123T213152Z_run3b_warmstart`. The whole notebook took about **65 minutes**, from session start at 21:32 UTC
to the summary at 22:37 UTC, below the estimate. Results were read straight from the Drive files (`summary.json`,
`released_expert_on_variable_goal.json`, `env_check.json`, `notebook.log`).

**`check-env`** passed:
- tip placement: no fallbacks, all 64 × 64 grid cells reachable, tip vs IK gap at most 0.01 mm
- closest tip start to the T: 15.07 cm
- stand-in T: overlap at least 0.948 on the goal, at most 0.339 when shifted 5 cm (threshold 0.9)
- goal quadrant and tip-start quadrant agree 23% of the time, close to the 25% expected if they are independent

**Released fixed-goal expert on moving goals, before fine-tuning:** 1% / 0.5% success-once / at-end (200 validation episodes; Q1 0%, Q2 0%, Q3 2%, Q4 2%).
It knows how to push, but not to a moving goal from a moving tip start, so the warm start began from essentially zero success.

**PPO speed and progress:**
- **Speed:** 4,096 environments ran at about **17,500 SPS** (16,500–18,500), about twice run 3's rate, so 50M transitions took about 48 minutes.
- **Progress:** ppo.py's 16-episode evaluations, success-once:

| Transitions | 0 | 1.6M–19.7M | 21.3M | 22.9M | 24.6M | 27.9M | 31.1M | 32.8M | 34.4M–49.2M |
|---|---|---|---|---|---|---|---|---|---|
| eval_success_once | 0 | 0–6% | 19% | 31% | 50% | 81% | 88% | 100% | 88–100% |

**Reading:** nothing happened for the first 20M transitions, then success-once rose from 0 to about 100% within about 12M. The risk named in the pre-run
assessment (near zero at 20–25M) was only just avoided. **Run 3b changed two things at once** (warm start and 4,096 environments), so this result
does not show which one mattered. The baseline of 1% suggests the larger batch or the transferred pushing skill helped only after a long search,
not that the expert already solved the task.

**Expert selection** (last five checkpoints, 200 validation episodes, seed 1001, success-once / success-at-end):

| Checkpoint | Overall | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|---|
| ckpt_676 | 98.5% / 28.5% | 98% / 33% | 98% / 22% | 100% / 35% | 98% / 20% |
| ckpt_701 | 98.5% / 28.5% | 95% / 18% | 100% / 27% | 100% / 37% | 100% / 33% |
| ckpt_726 | 99% / 39% | 96% / 40% | 100% / 44% | 100% / 28% | 100% / 47% |
| ckpt_751 | 97% / 24% | 96% / 21% | 98% / 24% | 98% / 37% | 96% / 11% |
| **final_ckpt** (selected) | **99.5% / 28.5%** | 100% / 32% | 98% / 24% | 100% / 32% | 100% / 24% |

**Expert data:** 16,384 episodes (1.64M transitions), with action noise 0 / 0.1 / 0.2 / 0.3 giving driver success-once of about 98% / 97.5% / 87.5% / 52%
(83.7% overall). Labels are always the expert's noise-free action.

**BC teacher:** best validation checkpoint in 5.7 minutes of training, at 97.5% / 34% on validation. Its validation MSE was still falling slowly at 145k iterations.

**Test** (200 episodes, seed 5001, never used for selection):

| Teacher | Overall | Q1 | Q2 | Q3 | Q4 | Steps to first success |
|---|---|---|---|---|---|---|
| PPO expert (`teachers/ppo_expert.pt`) | **98.5% / 29%** | 100% / 36% | 96% / 22% | 100% / 33% | 98% / 22% | 40.8 |
| BC teacher (`teachers/bc_teacher.pt`) | **99% / 33.5%** | 96% / 44% | 100% / 43% | 100% / 25% | 100% / 20% | 42.3 |

(Q1–Q4 have 45–55 test episodes each.)

**Conclusions:**
- **Both goal-conditioned full-state teachers are usable.** Success-once is at least 96% in every goal quadrant, including Q4, the transformer's OOD test quadrant.
  BC matches its expert, as in run 2, again without DAgger.
- **Success-at-end is low (about 30%, vs the fixed-goal expert's 72%).** The policies reach the goal in about 40 of 100 steps, then often push the T off it.
  This is likely because ManiSkill's PPO counts an episode as done at first success during training, so the policy is never rewarded for staying on the goal.
  This matters for transformer data. A demo's final T pose is a less reliable hint of the hidden goal than first-success, so **truncating demos at first success** (or
  recording the first-success step) looks worthwhile. Success-once should be the competence measure, as proposed earlier.
- **PPO and BC teachers behave almost alike,** so this pair adds little teacher variety by itself; goal variety does the work.

### Jimmy's shared variable-goal environment (found 2026-11-23, after run 3b)
Jimmy pushed `origin/jimmy-state-rl` (commit c87d8ae) with his own environment, `PushTVarGoal-v1` (`state_rl/pusht_vargoal.py`), and a frozen
goal-conditioned PPO teacher (`state_rl/teachers/ppo_vargoal_seed1_50M.pt`). His README says everyone using variable goals should use his file,
so success is measured the same way. **It is not the same task as our `PushT-VariableGoal-v1`:**

| | Ours (`PushT-VariableGoal-v1`) | Jimmy's (`PushTVarGoal-v1`) |
|---|---|---|
| Quadrant origin | (−0.156, −0.1), PushT-v1's goal | (−0.156, 0.0), PushT-v1's goal x, centred in y |
| Goal region | ±10 cm square, no gap | ±10 cm in x, ±20 cm in y, 2 cm / 3 cm band around each axis excluded |
| T start | PushT-v1's original fixed spawn box (absolute, so independent of the goal), random rotation | Anywhere in a larger fixed box, 8–30 cm from the goal, independent of goal quadrant |
| Tip start | ±15 cm square, IK lookup table, 15 cm from the T | Fixed box, library of 600 precomputed arm poses, 15 cm from the T |
| PPO teacher, test success-once / at-end | 98.5% / 29% (run 3b) | 90–97% / 86–97% per quadrant |

**Implications:**
- The transformer data must come from **one** environment, or the Q4 test means different things for different teachers. Jimmy's was written as the
  shared one (his decisions log: "Jimmy writes the shared variable-goal env"), and the plan calls for BC **distilled from the PPO teacher**.
- Jimmy's teacher holds the goal much better (at-end about 90% vs our 29%) with the same `ppo.py` recipe, which removes the success-at-end concern
  raised above.
- **Proposed next step (run 4):** distill the BC teacher from Jimmy's frozen PPO teacher in `PushTVarGoal-v1`. This reuses run 3b's collect/train/eval
  pipeline; only the environment and expert change, and no PPO training is needed (about 30–40 minutes). Run 3b's teachers stay as a record
  (and as evidence that the warm start works), but are not used for transformer data.
- **Why Jimmy's and not ours (discussed 2026-11-23, to take to the group):** mostly role and task design, not code quality.
  - For Jimmy's: PPO is his role and the plan distills BC from it; his file was announced as the shared one; his goal region is wider
    (±20 cm in y), which makes the Q4 test less mild (a weakness noted for ours above); the axis gap separates quadrants cleanly;
    his T starts vary more; his teacher holds the goal.
  - For ours: higher success-once (98.5% vs about 93%); a more thorough `check-env`; exact tip placement over the whole start square;
    and a `start_quadrants` option that makes the Q4-start hold-out possible, which Jimmy's lacks.
  - Options: adopt Jimmy's (recommended); adopt ours (Jimmy re-trains or uses run 3b's expert); or adopt Jimmy's and port `start_quadrants` into it.
- **Which teacher is better? (discussed 2026-11-23)** The numbers are from different tasks, so they cannot be compared directly; Jimmy's task is
  harder (wider goals, T starts up to 30 cm from the goal). For transformer data, holding the goal matters (cleaner demos), which favours Jimmy's.
  Proposed test to settle it: **cross-evaluate** both PPO experts in both environments (same 31-number state and checkpoint format), 200 episodes
  each, per quadrant, success-once and at-end.
- Agree the transformer rollout format with Jimmy first. He proposes `.npz` with `obs [N, 31]`, `actions [N, 7]`, `episode_ids [N]`, plus
  `goal_xy [N, 2]` and `goal_quadrant [N]`.

### Run 4 plan: BC distilled from Jimmy's PPO teacher (built 2026-11-23; ran the same day, results below)
Notebook `notebooks/Train_PushT_VarGoal_BC_from_Jimmy.ipynb`, built by `state_bc/build_notebook_run4.py`, session label `run4_bc_from_jimmy`.

| Item | Run 4 |
|---|---|
| Environment | Jimmy's `PushTVarGoal-v1`, `state_rl/pusht_vargoal.py` copied unchanged from `jimmy-state-rl` at commit c87d8ae |
| Expert | Jimmy's frozen `ppo_vargoal_seed1_50M.pt`, same commit |
| PPO training | none |
| Expert data, BC model, training | unchanged from run 3b: 16,384 episodes with noise 0 / 0.1 / 0.2 / 0.3, MLP 2 × 256, 200k iterations, no DAgger |
| Evaluation | Jimmy's per-quadrant protocol for **both** teachers: 100 episodes per quadrant, quadrant q uses seed base + 1000·q, validation base 1001, test base 5001 (once), freeze bar success-once ≥ 80% in every quadrant |

**How Jimmy's files get into Colab:** the GitHub repo is private, so Colab cannot clone it. The builder reads Jimmy's files with
`git show c87d8ae:<path>` and embeds them in the notebook: the two Python files as `%%writefile` cells, and the arm-pose library (22 KB) and the
teacher (1.1 MB) as base64, each checked against its SHA-256. The notebook is about 1.6 MB as a result.

**Checks in the notebook before any data is collected:**
- `audit`
- Jimmy's own `preflight`: observation layout, per-environment success, partial resets, quadrant restriction
- our `check-env`
- a **re-evaluation of Jimmy's teacher** on his validation and test seeds, printed next to his recorded numbers. The GPU differs (L4 vs his T4),
  so small differences are expected. The notebook stops if the teacher misses the bar in any quadrant.

**Code changes** (`pusht_state_bc.py`):
- `--env-id PushTVarGoal-v1` imports `pusht_vargoal`, and `--goal-quadrants` works with it.
- `rollout` also works on an unwrapped environment, and records each episode's goal position from the goal T when the environment does not
  expose `goal_xy`, so collected data has `goal_xy` and `goal_quadrant` per episode, as in Jimmy's proposed transformer format.
- New `eval-quadrants` subcommand that reproduces Jimmy's protocol: raw environment, no reconfiguration, goals restricted one quadrant at a time.

Offline tests: 28, all passing. The new ones cover the shared environment's options, goal records on an unwrapped environment, and the
per-quadrant protocol (seeds, restriction, restore, freeze bar). The builder check confirms every notebook cell parses and the embedded files
decode to the committed hashes. Not yet run on a GPU.

**Expected:** BC close to Jimmy's teacher (about 90%+ success-once per quadrant) and, unlike run 3b, high success-at-end. Expected time: about 30–40 minutes.

### Run 4 result: BC teacher passes the bar in every quadrant (2026-11-23)
**Drive session:** `state_bc/20261123T230947Z_run4_bc_from_jimmy`. The whole notebook took about **14 minutes** (23:10 to 23:23 UTC, L4), well under the estimate.
Results were read from the Drive files (`teachers/summary.json`, `env_check.json`, `notebook.log`).

**Checks:**
- `audit`: 31-number state with the goal at 21:24, `pd_joint_delta_pos`, 100 steps.
- Jimmy's `preflight`: all 10 checks pass.
- Our `check-env` passes: stand-in T on goal overlap ≥ 0.948, shifted 5 cm ≤ 0.339. Goals span x −0.256 to −0.057 and y −0.199 to 0.198; quadrants 67 / 56 / 72 / 61 of 256.

**Jimmy's teacher reproduced** (100 episodes per quadrant, success-once / success-at-end; Jimmy's Kaggle T4 numbers in brackets):

| Split | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| Validation (base 1001) | 91 / 86 [93 / 83] | 96 / 92 [93 / 90] | 92 / 88 [91 / 86] | 95 / 92 [96 / 94] |
| Test (base 5001) | 94 / 88 [94 / 86] | 95 / 95 [97 / 97] | 93 / 91 [92 / 87] | 93 / 92 [90 / 89] |

Every number is within 3 points of his. The episodes are seeded identically, so the small gaps come from GPU physics (L4 vs T4), not a protocol difference.

**Expert data:** 16,384 episodes, 1.64M transitions. Driver success-once by noise level 0 / 0.1 / 0.2 / 0.3: about 94% / 96% / 92% / 76%. Success-at-end by noise level: 90% / 71% / 31% / 14%.
Action noise mostly knocks the T off the goal *after* reaching it, which matters for transformer demos (see below).

**BC training:** 200k iterations in 5.7 minutes. Validation MSE fell from 0.0168 to 0.0031 with no sign of overfitting (train MSE 0.0039).
Closed-loop validation (200 mixed episodes) stayed at 89–94% success-once and 85–89% success-at-end from 25k iterations on. The best checkpoint
scored 94.5% / 89.5%.

**Test** (100 episodes per quadrant, base seed 5001, used once; success-once / success-at-end):

| Teacher | Q1 | Q2 | Q3 | Q4 | Worst once | Mean once / end |
|---|---|---|---|---|---|---|
| PPO (Jimmy, `teachers/ppo_expert.pt`) | 94 / 88 | 95 / 95 | 93 / 91 | 93 / 92 | 93% | 93.8% / 91.5% |
| **BC (distilled, `teachers/bc_teacher.pt`)** | 92 / 90 | 94 / 93 | 89 / 84 | 91 / 88 | **89%** | **91.5% / 88.8%** |

On validation, BC scored 88 / 81, 93 / 93, 91 / 88 and 90 / 88 (worst 88%). Both teachers **pass the 80% freeze bar in every quadrant**.

**Conclusions:**
- **The full-state BC teacher is done.** BC is about 2 points below its expert on average, within the noise of 100 episodes per quadrant (about ±3 points).
  This is the third time distillation without DAgger has matched the expert closely.
- **Success-at-end is solved:** about 89%, against 33% in run 3b, because Jimmy's teacher holds the goal. Demos mostly end with the T on the goal.
- **Q4 is no weaker than the other quadrants** for either teacher, so a transformer failure on Q4 goals will not be explained by weak teachers there.
- **For transformer data, use noise-free teacher rollouts.** The noisy rollouts here are for BC robustness only. With noise 0.2–0.3, most episodes end
  off the goal (success-at-end 14–31%), which would blur the hidden-goal signal in the context.
- Both teachers act almost identically (BC imitates PPO), as expected. They add little variety to the transformer's training mix on their own.

**Should DAgger or more tuning improve it? (discussed 2026-11-23) No; leave the teacher as is.**
- The gap to the expert (about 2 points) is within the noise of 100 episodes per quadrant, and BC cannot exceed its expert, so the most there is to gain is about 2 points.
- In run 2, DAgger added nothing measurable over noisy expert rollouts. The noise injection already covers the off-course states DAgger would add.
- The teacher's job is to clear the bar in every quadrant and give clean demos. It does both. Time is better spent on transformer data collection.
- If the gap ever matters, first evaluate with more episodes (for example 500 per quadrant) to see whether it is real, before changing training.

### Next: the transformer (discussed 2026-11-23)
Both v1 teachers are frozen and in the same environment: Jimmy's PPO and the run 4 BC. `CODEX_CONTEXT.md`'s first experiment is numerical only
("a balanced mixture of PPO and BC prompts"); camera teachers are a later extension. So the transformer can start now, and the full-state pair is
what it needs. No branch has transformer code yet.

**Settle with the group first:**
1. Who builds and trains the transformer (still open).
2. The prompt data format, with Jimmy. Proposed: `.npz` with `obs`, `actions`, `episode_ids`, `goal_xy` and `goal_quadrant`, plus a `teacher` field.
3. The data design:
   - number of training goals (Q1–Q3), validation goals (Q1–Q3) and Q4 test goals
   - episodes per goal
   - support snippet length and how many snippets (`N` ≤ 511 pairs; one 100-step episode is 100 pairs)
   - whether to truncate demos at first success

**Proposed stages:**
1. **Prompt data generator** (teacher side, natural next step for this folder). For each sampled goal (fixed through
   `reset(options=dict(goal_xy=...))` in `PushTVarGoal-v1`) and each teacher, roll out several noise-free episodes from different starts and save
   them grouped by goal. Support and query come from different episodes; labels are the teacher's actions. Drop `extra/goal_pos` (21:24) only
   when building prompts, so the teachers still see the goal. Separate goal sets for train (Q1–Q3), validation (Q1–Q3) and test (Q4).
2. **Transformer training** to the spec (GPT-2, 12 layers, width 256, 8 heads, one token per state or action, query-action MSE), with prompts
   sampled on the fly from stage 1's data.
**Time estimate (2026-11-23, rough, not measured; Nathan will train it):**
- The model is small: about 9.5M parameters.
- Cost per step depends mostly on prompt length: `N` = 64 pairs is 129 tokens, `N` = 200 is 401 tokens.
- On an L4 with batch 64, expect roughly 0.03–0.1 s per step. That makes 100k steps about 1–3 hours. Garg's 500k steps would be about 5–14 hours,
  too long for one Colab session, so checkpoint to Drive and support resuming.
- Prompt data: about 10 minutes, scaled from run 4's collection speed.
- Closed-loop evaluation: about 15–30 minutes for all conditions.
- Plan: a 30-minute pilot first, checking that correct-goal context beats wrong-goal context; then about 100k steps. First full pass: about 2–4 hours
  of GPU time. Measure steps per second in the pilot before committing to a longer run.

3. **Frozen closed-loop evaluation:** unseen Q1–Q3 goals and Q4 goals; correct-goal, wrong-goal and no context; context-size sweep. Plus the
   baseline from `CODEX_CONTEXT.md`: goal-conditioned BC trained on Q1–Q3 data only, given the goal estimated from a demo's final T pose.

### Transformer v1: built (2026-11-23, not yet run)
Nathan trains it. Notebook [`notebooks/Train_PushT_ICL_Transformer.ipynb`](../notebooks/Train_PushT_ICL_Transformer.ipynb), built by
`state_bc/build_notebook_icl.py`; code in [`pusht_icl.py`](pusht_icl.py), with subcommands `collect`, `train`, `offline` and `eval`.

**What it is trained on:**
- **Teachers:** the two frozen full-state teachers in `PushTVarGoal-v1`, Jimmy's PPO and the run 4 BC, read from the run 4 Drive folder. Half
  the prompts come from each.
- **Demonstrations:** noise-free; 10 episodes per goal and teacher, from different starts, with the same goals for both teachers.

| Split | Goals | Quadrants | Use |
|---|---|---|---|
| `train` | 2,048 | Q1–Q3 | training prompts |
| `val` | 128 | Q1–Q3, unseen | checkpoint selection |
| `test_in` | 128 | Q1–Q3, unseen | final test |
| `test_q4` | 128 | Q4 | final test, held-out quadrant |

- **One prompt:** one teacher and one goal. It holds `k` ∈ {0, 1, 2, 4, 8} whole demonstration episodes (the same `k` within a batch, chosen at
  random per batch). Each episode is subsampled every 4th step and always includes the final step, so the final T pose is in the context: 25 pairs
  per episode, at most 200 pairs (401 tokens). Then one **query state from a different episode** of the same teacher and goal. The target is the
  teacher's action there. Successful demonstrations are used first.
- **The goal is hidden:** fields 21:24 are removed from every state the transformer sees (28 numbers instead of 31). Teachers still see the
  goal when generating data. Robot state and T pose stay.
- **Model, to `PushT_ICL_Transformer_Spec.md`:** Hugging Face `GPT2Model`, 12 layers, width 256, 8 heads, `gelu_new`, no dropout, learned
  positions, 1,024 positions; one linear projection per state or action token and a linear action head; 9.75M parameters, random init.
  States are normalized with training statistics.
- **Training:** query-action MSE, AdamW 1e-4 (1,000-step warmup, cosine to 10%), batch 64, gradient clip 1.0, bf16 autocast,
  100k steps.
- **Validation:** every 2,500 steps, validation action MSE per `k`, with correct-goal vs wrong-goal demonstrations on identical queries.
  `best.pt` has the lowest mean correct-goal (or no-context) error.
- **Resuming:** `last.pt` holds model, optimizer and sampler state, in a fixed Drive folder (`icl_runs/transformer_v1_seed0/`), so rerunning the
  notebook continues an interrupted run.

**Evaluation (frozen weights, unseen goals):**
- **Action error:** teacher-forced, per `k`, correct vs wrong goal, on `val`, `test_in` and `test_q4`.
- **Closed loop:** 128 goals × 4 new starts per split, demonstrations from PPO or BC, `k` ∈ {0, 1, 2, 4, 8}, correct-goal vs wrong-goal (the
  demonstrations of another goal from the same split and teacher), no context at `k = 0`. All conditions share the same starts (same seeds and
  goals). The two teachers, which see the goal, run on those starts too as the reference.
- Not included yet: the goal-estimate baseline (BC trained on Q1–Q3 data, given the goal from a demonstration's final T pose).

**Pilot first:** 3,000 steps, then the action-error gap between correct-goal and wrong-goal demonstrations on validation goals, plus a small
closed-loop check. Training prints it/s and an ETA, so the pilot measures the real speed.

**Changes elsewhere:** `pusht_state_bc.rollout` accepts `reset_options` (to fix goals) and, when labeling, returns each episode's first-success
step and success-at-end.

**Offline tests:** `test_icl_offline.py`, 13 tests. They cover:
- the goal is hidden
- snippets end at the final step
- prompts come from one teacher and one goal, with the query from a different episode
- wrong-goal prompts keep identical queries
- successful demonstrations come first
- model shapes, save/load, and that the actor ignores the goal fields
- a tiny model trained on CPU learns to use the context (correct-goal error below half of wrong-goal and no-context error)
- resuming
- `collect` groups episodes by goal

All 41 tests in both files pass. Not yet run on a GPU.

**Design choices to revisit after the pilot:**
- Whole subsampled episodes as snippets, rather than truncating at first success.
- One query per prompt, as in the spec, rather than predicting at every state token (more data per step).
- 2,048 training goals.
- The stride of 4.

### Pre-run assessment (2026-11-23, before running)
The overall design is sound: the hidden goal forces context use, and Q4 goals test generalization. Risks, most likely first:
1. **PPO may not reach 80% in 50M transitions.** The task is now harder (moving goals and varied tip starts), and the fixed-goal
   camera PPO needed about 10M transitions before any success. If `eval_success_once_mean` is still near zero around 20–25M,
   expect to need about 100M or more, or a staged start (fixed tip first). The notebook's 80% gate stops before BC if so.
2. **First contact with real kinematics.** The inverse-kinematics code assumes the chain's root sits at the robot base (−0.615, 0, 0)
   and uses `pytorch_kinematics` calls not exercised offline. A mistake shows up immediately in `check-env` (crash, large tip gap,
   or high fallback rate), so it costs minutes, not a PPO run.
3. **100-step episodes** may be tight when the tip starts far from the T and the goal is far from the T. Watch steps-to-success.
4. **Corners of the ±15 cm tip square may be unreachable** with a downward stick; that shows up as fallbacks in `check-env`.

Research-level points to keep in mind:
- **Goal inference may be easy.** With a fixed goal rotation, the final T pose in a successful demo nearly is the goal. The transformer
  can succeed by reading it, so the explicit "estimate goal from final T pose" baseline may match it. That still counts as ICL.
  But the expert's success-at-end is only about 70%, so roughly 30% of demos end with the T pushed back off the goal; truncating demos at
  first success, or filtering, would make the context cleaner.
- **Q4 is closer to recombination than extrapolation.** Q4 goals have +x offsets (seen in Q1) and −y offsets (seen in Q3), and all goals
  lie within a 20 cm square, smaller than the T itself. This is a mild OOD test; describe it that way, or widen the goal square later.
- **PPO and distilled BC behave almost alike,** so the "balanced mixture of PPO and BC prompts" adds little teacher variety. Goal variety does the work.

### Idea: also vary the goal T's orientation (discussed 2026-11-23, deferred)
The movable T's *start* rotation is already random; only the *goal* rotation is fixed. Varying it would make the hidden goal
richer (x, y and angle), give more task variety, and allow an orientation hold-out. Reasons to wait:
- `CODEX_CONTEXT.md` fixes target orientation for the first numerical experiment.
- PPO reaching the goal reliably is already the main risk; rotating the T into arbitrary angles is much harder and may not fit in 100 steps.
- Code changes needed: the dense reward's rotation term uses the fixed `goal_z_rot` (must become per-episode), and the goal pose and
  success transform must carry each episode's rotation.
- It would not make goal inference harder in principle: the final T pose in a successful demo still reveals the whole goal.

Suggested path: run v1 with fixed rotation; if PPO handles moving goals comfortably, add a rotation range (for example ±45° first,
widening later) as a v2 axis, possibly with its own hold-out range.

**As an OOD hold-out axis** (follow-up, 2026-11-23): a good test, plausibly stronger than the Q4 position hold-out, because the
pushing strategy changes with the target angle (which way and how far to turn the T), not just where to push.
- **Gap vs beyond:** on a full circle, a held-out arc (say 270°–360°) sits between seen angles on both sides, so it is a gap in coverage.
  For genuine extrapolation, train on a limited range (for example ±30°) and test beyond it (for example 30°–60°).
- **Teachers must see the goal angle:** state observations include only the goal's *position* (`extra/goal_pos`), not its rotation.
  Goal-conditioned teachers would need the goal angle added to their observation (for example as sin/cos). The transformer would still hide it.
- **Teachers train on the full range;** only the transformer's training goals exclude the held-out angles.
- **Keep axes separate in testing,** as with the start hold-out: in-distribution, position hold-out only, angle hold-out only, and both.

## Decisions log

| Date | Decision | Reason |
|---|---|---|
| 2026-11-22 | ManiSkill `PushT-v1` instead of Gym-PushT | Group decision; articulated arm, GPU simulation, released experts |
| 2026-11-22 | `obs_mode="state"` is "full state": numbers only, no images | Full state vs camera is the modality comparison; Izat and Neil do camera |
| 2026-11-22 | `pd_joint_delta_pos` for all four people | Group decision; one action format keeps the state vs camera comparison clean. Neil's branch file still says `pd_ee_delta_pos` and is out of date. |
| 2026-11-22 | ManiSkill v3.0.1 (commit `a4a4f92`) with Neil's install recipe | Identical simulator code across teachers |
| 2026-11-22 | BC trains on the released PPO demos first | Available immediately; tests the pipeline before teacher diversity work |
| 2026-11-22 | Outputs in Nathan's Google Drive under `182_final_project/state_bc/` | Each person saves to their own Drive; results get compiled later |
| 2026-11-22 | Teacher: MLP 2 × 256 ReLU, tanh output, MSE | Matches the proposal's BC teacher |
| 2026-11-22 | Hyperparameters: Adam 3e-4, batch 1024, 100k iterations | Same as ManiSkill's BC baseline; not tuned |
| 2026-11-22 | Features with std < 1e-3 are centered, not scaled | The spec's 1e-6 floor would amplify simulator noise on near-constant fields like the goal position |
| 2026-11-22 | Checkpoints chosen on validation seeds; test seeds used once at the end | Never select on test data |
| 2026-11-22 | Run 2 adds expert rollouts and DAgger, other settings unchanged | Run 1 diagnosis: overfitting and compounding errors |
| 2026-11-22 | No Claude co-author lines in commits | Nathan's preference |
| 2026-11-22 | Accept run 2's round 2 checkpoint as the first full-state BC teacher | 98% / 78% on test seeds, matching the expert's 100% / 72% |
| 2026-11-22 | Treat DAgger as optional for future expert-distilled BC teachers | Run 2 round 0 (no DAgger) already matched the expert; later rounds were within noise |
| 2026-11-22 | **v1 of the project focuses on out-of-distribution tests**; Garg-style parameterized policy families are deferred | Group decision. Izat is updating the repo's project context to match; BC work waits for that update. |
| 2026-11-23 | **v1 design adopted: variable-goal Push-T** (`CODEX_CONTEXT.md` on main, commit 9915e87) | Teachers are goal-conditioned (see goal) and trained on goals in Q1–Q4. The transformer never sees the goal and trains on Q1–Q3 goals only; the OOD test is Q4 goals. Hidden goal makes context necessary. |
| 2026-11-23 | Train a goal-conditioned PPO expert inside the BC notebook | Not blocked on Jimmy; his PPO can replace it later |
| 2026-11-23 | Default quadrants: goals within ±10 cm of PushT-v1's goal, quadrants by sign of the offset | No group definition yet; easy to change |
| 2026-11-23 | No DAgger for the variable-goal BC teacher | Run 2 showed it added nothing measurable |
| 2026-11-23 | Randomize the stick-tip start (±15 cm, by quadrant, independent of the goal) | `CODEX_CONTEXT.md` puts starts in Q1–Q4; also enables start hold-outs. Not an anti-cheating measure: the hidden goal is. |
| 2026-11-23 | Run 3b: warm-start PPO from the released fixed-goal expert, 4,096 envs, learning rate 1e-4 | Run 3's from-scratch PPO stalled at 4–11%; Jimmy's PPO status unknown, so train now rather than wait |
| 2026-11-23 | **Accept run 3b's teachers as the variable-goal full-state teachers** (`teachers/ppo_expert.pt`, `teachers/bc_teacher.pt` in session `20261123T213152Z_run3b_warmstart`) | Test success-once 98.5% (PPO) and 99% (BC), at least 96% in every quadrant including Q4 |
| 2026-11-23 | Use success-once as the competence measure; consider truncating transformer demos at first success | Run 3b teachers reach the goal reliably but end on it only about 30% of the time |
| 2026-11-23 | **Freeze run 4's BC teacher as the full-state BC teacher** (`teachers/bc_teacher.pt` in session `20261123T230947Z_run4_bc_from_jimmy`) | Test success-once 89–94% in every quadrant (bar 80%), about 2 points below Jimmy's PPO; success-at-end 84–93% |
| 2026-11-23 | **Use Jimmy's `PushTVarGoal-v1` and his frozen PPO teacher; run 4 distills the BC teacher from it** | One shared environment for all transformer data; PPO is Jimmy's role and BC is distilled from it; his teacher holds the goal (at-end about 90%) and his goal region is wider. Run 3b's teachers are kept as a record, not used for transformer data. Cross-evaluation skipped for now. |

## v1 plan: variable-goal Push-T (from `CODEX_CONTEXT.md`, 2026-11-23)

**Question:** can a frozen transformer infer an unfamiliar Push-T goal from demonstrations and control the robot to it,
without goal coordinates or weight updates? Target position varies; target orientation stays fixed for now.

**What this means for the BC teacher:**
- **Goal-conditioned BC:** same idea as run 2, but the expert is a goal-conditioned PPO teacher (Jimmy) trained on random
  starts and goals in Q1–Q4. The context calls the pair "PPO and distilled BC". The released fixed-goal expert no longer applies.
- The BC input keeps `extra/goal_pos` (indices 21:24), which now varies. The current normalization only centers it, because it was constant in runs 1–2; with varying goals it will be scaled normally.
- **Validate per quadrant:** report success separately for goals in Q1, Q2, Q3 and Q4, then freeze the teacher before any transformer data is collected.
- **Goal-estimate baseline (likely BC work):** a goal-conditioned BC trained only on Q1–Q3 source data, given a goal estimated
  from the final block pose of a successful demonstration.
- **Transformer data:** per prompt, one expert and one goal, several episodes from different starts; support and query from different episodes;
  teacher actions as labels. Transformer inputs drop the goal fields; robot state and block pose stay.

**Environment work needed first (shared by everyone).** In ManiSkill v3.0.1, `PushT-v1` hard-codes the goal:
- `goal_offset` and `goal_z_rot` are class constants. `_initialize_episode` always places the goal T there.
- `world_to_goal_trans`, which the **success check** uses, is computed once in `_load_scene` as a single matrix.
  A variable goal needs a per-environment transform, recomputed at every reset. Otherwise success is measured against the old goal.
- The dense reward's translation term already reads the goal T's live pose, but its rotation term uses `goal_z_rot` (fine while orientation is fixed).
- Start positions: the T spawns in a fixed box relative to the goal, and the stick tip starts at a fixed point.
  The plan randomizes both across Q1–Q4.
- Episodes are 100 steps; farther start-to-goal distances may need a longer limit.

**Open questions:** where the quadrant origin is and what the reachable sampling region is; who writes the shared variable-goal environment;
when Jimmy's goal-conditioned PPO will be ready; whether the goal-estimate baseline is part of the BC work.

**Idea under discussion (2026-11-23): also hold out Q4 *starts* during transformer training.** `CODEX_CONTEXT.md` currently
holds out Q4 *goals* only and lists start hold-outs as a later experiment. Assessment:
- Not a bad idea, but **don't apply both hold-outs to one test set.** If the transformer fails on "Q4 start + Q4 goal", you cannot
  tell which unfamiliarity caused it. Use separate test cells: Q1–3 start + Q1–3 goal (in distribution), Q1–3 start + **Q4 goal**,
  **Q4 start** + Q1–3 goal, and Q4 start + Q4 goal.
- **They test different things.** A goal hold-out tests in-context goal inference (the context is the only source of the goal).
  A start hold-out tests whether the learned "act toward the inferred goal" behavior extends to unfamiliar states, which is mostly
  generalization of the weights, unless the model learns to copy how support demos from nearby states behaved.
- **A start hold-out is leaky:** trajectories from Q1–Q3 starts still pass through Q4 states. It holds out initial configurations,
  not Q4 states, so describe it that way.
- **Define "start in Q4":** by the T-block's starting position, the stick tip's, or both.

## Ideas and open questions

### ICL context and DAgger (answered 2026-11-22)
**Do DAgger states or expert labels go into the transformer's context?** No. DAgger only changes how *this teacher is
trained*. Once trained, the BC teacher is a fixed function from state to action, and that function is what the
transformer must infer. ICL data comes from rolling out the **finished BC teacher itself** and recording **its own actions**.
Putting the PPO expert's labels in the context would show the transformer a different policy, the expert,
not the teacher being queried. The `collect` command currently always labels with the expert; ICL rollout collection
will need a mode where the teacher both drives and labels.

### Making many different teachers
The goal is teachers that all succeed but choose consistently different actions in the same state. Neil found that
short fine-tunes of one parent PPO policy stayed almost identical (contact points about 4 mm apart, routes about 1 mm apart).
- **BC distilled from each of Jimmy's state PPO teachers:** cheap, inherits his diversity, gives matched BC/PPO pairs. Depends on Jimmy.
  The same `collect` + DAgger code works with any of his checkpoints as the expert.
- **BC on strategy clusters of demos:** group demos by how they push (first contact side, approach direction, push order).
  Do not group by start position, which makes teachers that cover different regions instead of acting differently in the same state.
- **Procedural family:** one base policy plus a parameter z, for example a posture preference in the arm's redundant
  joint directions that changes joint actions without changing the stick tip's motion. Unlimited, controllable teachers,
  closest to Garg's random functions, but synthetic, so an extra experiment rather than a replacement.
- **For Jimmy:** independent from-scratch PPO seeds (state PPO is cheap) and strong reward differences.
- **Caveat:** DAgger toward one expert makes BC teachers more alike. Fine for proving the pipeline, not for diversity.

### Checks before scaling up the teacher count
- Competence: success-once on fixed seeds above an agreed bar.
- Action differences between teachers on a shared set of states.
- Identifiability: a small classifier that guesses the teacher from a short snippet. If it cannot, the transformer cannot use context either.

### Number of teachers vs task diversity (discussed 2026-11-22)
Garg's models each trained on one function class, but drew a **new random function for every prompt**
(roughly 30 million per model). Raventós et al. (2023) found that with a *finite* pool of tasks, a transformer only learns a general
in-context algorithm past a diversity threshold (thousands of tasks in their linear-regression setup). Below it, it memorizes the pool.
With tens of trained teachers, ours will likely learn to recognize training-like teachers rather than infer arbitrary new ones.
Options: frame the claim around a small family and report performance vs number of training teachers; add a procedural family for Garg-scale diversity; or both.
(The paper's exact numbers are from memory; check before citing.)

### Proposed pivot: out-of-distribution generalization (discussed 2026-11-22, not decided)
**Group idea:** train with starting states in only 3 of 4 quadrants, then test whether the transformer can act correctly in the
4th quadrant when given demonstrations from there in context.

**Assessment:**
- It is a real and interesting question, close to Garg's input-distribution-shift tests: can context substitute for training coverage?
- **Key risk: with a single teacher, the transformer has no reason to read the context.** If the same policy generates every prompt,
  the query action is fully determined by the query state, so the weights can learn the policy directly and ignore the demos.
  ICL appears in Garg's setup because the function changes from prompt to prompt and only the context reveals which one.
  So something must vary across prompts that only the context reveals.
- **Ways to keep context necessary:**
  1. Keep multiple teachers, and add the quadrant hold-out as an extra test axis.
  2. **Hidden goal:** randomize the goal pose and remove it from the observation, so demos are the only way to learn where to push.
     Gives continuous, unlimited task diversity like Garg's random w. The OOD version holds out goals in one quadrant.
     Cost: PushT-v1's goal is fixed, so this needs goal-randomized experts (new PPO training, practical in state mode).
- **Design if adopted:** train on teachers T_train with starts in Q1–Q3. Test four cells: {training teacher, held-out teacher} × {Q1–Q3, Q4}.
  Context conditions: correct teacher from Q4, correct teacher from Q1–Q3, none, wrong teacher.
  Baseline: a plain BC policy trained only on Q1–Q3 (pure extrapolation without context).
- **Variant: move the goal but keep it visible** (asked 2026-11-22). Then every teacher is one goal-conditioned policy
  π(state, goal), and the goal in the query observation already says where to push. The weights can learn π directly,
  so context is again redundant in training. This is the "goal conditioning alone is not demonstration-conditioned
  policy inference" case from `CODEX_CONTEXT.md`. A held-out goal quadrant then tests ordinary goal-conditioned extrapolation, not ICL.
  It becomes an ICL question only if something else still varies per prompt (multiple teachers), with the goal shift as the OOD axis.
  Middle grounds: show the goal only coarsely or with noise, so demos add information.
- **Practical notes:** quadrants would split the T-block spawn region (0.2 × 0.3 m), which needs custom resets.
  Teachers trained on the full region already act competently in Q4, so they can supply Q4 demos.

### Garg-style policy families (discussed 2026-11-22; OOD tests set aside for now)
Goal: many policies per "function class", fair evaluation when policies agree on some states, and evidence the transformer is not memorizing.

**1. Make policies parameterized, like Garg's random w.** Train one policy conditioned on a hidden parameter z, π(s, z).
Draw a fresh z for every prompt, so one training run yields unlimited teachers. z is never in the observation.
Candidate classes, each with infinite members:
- **Reward-parameterized family:** one PPO policy trained across sampled reward weights, e.g. the proposal's
  r = r_task − λx·vx² − λy·vy², with z = (λx, λy) as an extra input. Strong ranges are needed; Neil's short fine-tunes barely differed.
- **Skill / diversity family:** SMERL- or DIAYN-style latent skills, where a discriminator rewards behaviors that are distinguishable
  from z, with diversity only rewarded when the task is solved. Targets "competent and distinguishable" directly.
- **Null-space posture family:** π_z(s) = base(s) + (I − J⁺J)·z in joint space. Different joint actions, same stick-tip motion to first order.
  Different actions by construction; risk that the base policy is unfamiliar with the shifted arm postures.
- BC fits as **distillation of each conditioned expert into a z-conditioned BC network** (the run 2 recipe), or classes are defined by behavior family and the training method becomes secondary.

**2. Same answers are fine; evaluate like Garg.** Garg scores prediction error on the query, not recovery of w. With few
examples many w fit the data, and the best predictor averages over them. For policies:
- Baselines on the same queries: **no-context / mean policy** (best predictor without context), **oracle** (knows z; the floor),
  and a **fit-z baseline** that searches for the z that best explains the context and predicts π(s_q, ẑ), the analog of least squares.
  ICL is shown when the transformer beats the mean policy and approaches fit-z as k grows.
- Normalize error per query state by how much actions vary across z there, and report it on **informative states**
  (high variance across z) separately, since near the goal all policies may act alike.
- Check identifiability first: if the fit-z baseline cannot recover z from k demos, no learner can.

**3. Prove it is not memorizing.**
- Fresh continuous z per training prompt: effectively no repeats, as in Garg.
- Test on new z from the same distribution.
- **Diversity sweep** (Raventós-style): train with finite pools of 16, 256, 4,096 and unlimited z, and test on new z.
  Memorization shows as good training-pool error but poor new-z error at small pools.

**Data:** with one z per parallel env, 1,024 teachers roll out at once on the GPU, so about 100k episodes take minutes.

### Open questions for the group
- **Success metric:** the expert reaches the goal in 98–100% of episodes but stays there in only 65–72%, because ManiSkill's
  PPO ends episodes at first success during training. Proposal: report success-once as the competence measure, for all teacher types.
- **Who builds and trains the transformer?** Everyone owns a teacher type.
- **Rollout format** for the transformer (fields, support vs query rollouts, failed episodes), shared with Jimmy.
- **Teacher counts and splits:** how many teachers per type, and the train/validation/test split by teacher.
- **Why 238 demos fail to replay:** possibly a version difference from the commit that generated them. Not investigated.

## Changelog
- **2026-11-22:** Run 1 notebook and module created (`audit`, `prepare`, `train`, `eval`), offline tests added.
- **2026-11-22:** Fixed `max_episode_steps` being `None` on GPU environments (run 1 crash).
- **2026-11-22:** Run 1 completed: 19% / 10% test success. Diagnosis recorded above.
- **2026-11-22:** Run 2: added `collect` (expert rollouts with action noise, and DAgger labeling), multi-file training,
  per-source transition files, and the run 2 notebook. Offline tests extended to 12 (rollout and collection logic use a fake environment).
- **2026-11-22:** Run 2 completed: 98% / 78% test success (expert 100% / 72%). Results, analysis and proposed next steps recorded above.
- **2026-11-23:** Adopted the variable-goal v1 design from `CODEX_CONTEXT.md`; recorded what it means for the BC teacher and the environment changes needed.
- **2026-11-23:** Run 3 built: `variable_goal_env.py` (`PushT-VariableGoal-v1`), `ppo` and `check-env` subcommands, `--env-id` and
  `--goal-quadrants` options, per-quadrant success in every evaluation, goal quadrant and noise level recorded per collected episode,
  and the run 3 notebook. Offline tests extended to 19 (goal sampling, quadrant convention, per-goal success transform, per-quadrant reporting).
- **2026-11-23:** Random stick-tip starts added to `PushT-VariableGoal-v1` (inverse-kinematics placement, T clearance, fallback, per-episode
  start records), `--start-quadrants`, tip checks in `check-env`, and start records saved with collected data. Offline tests extended to 24
  (tip sampling and clearance, start/goal independence, IK solver on a test arm, initial guess). Notebook rebuilt.
- **2026-11-23:** First run 3 launch: PPO at about 6,700 SPS. Stopped. Replaced per-reset IK with a one-time 64 × 64 lookup table
  (`tip_grid_offsets`, `grid_index`, `choose_tip_cells`, `_build_ik_table`); `check-env` reports grid reachability. Offline tests: 25. Notebook rebuilt.
- **2026-11-23:** Second run 3 launch (with the lookup table): about **8,000 SPS** five minutes in, about 20% faster than 6,700. The per-reset IK was
  only part of the cost; most of it is the environment step itself (physics plus PushT's per-step pseudo-render success check). At this speed,
  50M transitions take about 1 hour 45 minutes. Possible future speed-ups: more parallel environments (ManiSkill's own state Push-T baseline uses 4,096
  with `ppo_fast.py`), which also changes PPO's batch size, so it is a training change, not just a speed change.
- **2026-11-23:** Run 3 PPO stalled at 4–11% success after 50M transitions (80% gate stopped before BC). Notebook updated for run 3b: warm start
  from the released expert, 4,096 envs, learning rate 1e-4, and a baseline evaluation of the released expert on moving goals.
- **2026-11-23:** Run 3b completed in about 65 minutes. The released expert scored 1% on moving goals before fine-tuning; PPO ran at about 17,500 SPS,
  stayed near zero for 20M transitions, then reached about 100%. Test success-once / at-end: PPO expert 98.5% / 29%, BC teacher 99% / 33.5%, every quadrant at least 96%.
  Results recorded above.
- **2026-11-23:** Found Jimmy's shared `PushTVarGoal-v1` and frozen PPO teacher; decided to use them. Run 4 built: `build_notebook_run4.py` and
  `Train_PushT_VarGoal_BC_from_Jimmy.ipynb` (Jimmy's files embedded at c87d8ae), `--env-id PushTVarGoal-v1` support, goal records on unwrapped
  environments, the `eval-quadrants` subcommand. Offline tests: 28.
- **2026-11-23:** Run 4 completed in about 14 minutes. Jimmy's teacher reproduced within 3 points of his numbers. BC test success-once / at-end per
  quadrant: Q1 92 / 90, Q2 94 / 93, Q3 89 / 84, Q4 91 / 88. Both teachers pass the 80% bar in every quadrant. Results recorded above.
- **2026-11-23:** Transformer v1 built: `pusht_icl.py` (`collect`, `train`, `offline`, `eval`), `build_notebook_icl.py`,
  `Train_PushT_ICL_Transformer.ipynb`, `test_icl_offline.py` (13 tests). `rollout` gained `reset_options` and per-episode success records.
  Offline tests: 41 in total.

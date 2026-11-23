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

- **Current notebook (run 3, variable goal):** [`notebooks/Train_PushT_VariableGoal_PPO_BC.ipynb`](../notebooks/Train_PushT_VariableGoal_PPO_BC.ipynb).
  Upload to Colab, choose an L4 GPU, run top to bottom. Outputs go to `MyDrive/182_final_project/state_bc/<session>/`.
- **Earlier notebooks (kept as run, not rebuilt):** run 1 [`Train_PushT_State_BC.ipynb`](../notebooks/Train_PushT_State_BC.ipynb),
  run 2 [`Train_PushT_State_BC_DAgger.ipynb`](../notebooks/Train_PushT_State_BC_DAgger.ipynb). Both use the fixed-goal `PushT-v1`.
- **Code:** [`pusht_state_bc.py`](pusht_state_bc.py), with subcommands `ppo`, `audit`, `check-env`, `prepare`, `collect`, `train`, `eval`;
  [`variable_goal_env.py`](variable_goal_env.py) defines `PushT-VariableGoal-v1`. The default environment is now the variable-goal one;
  pass `--env-id PushT-v1` for the original.
- **Notebook source:** the notebook embeds both modules. Edit them or `build_notebook.py`, then run
  `python state_bc/build_notebook.py`. Never edit the notebook's module cells by hand.
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

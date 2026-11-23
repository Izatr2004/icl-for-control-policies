# Full-state BC teachers (Nathan)

Behavioral cloning (BC) teachers for ManiSkill `PushT-v1` with full simulator state (`obs_mode="state"`) and
joint control (`pd_joint_delta_pos`). This file is the running log for this part of the project: every run,
its results, what we concluded, and what we changed next. **Add to it; don't rewrite history.**

Contents: [Role](#role-in-the-project) · [How to run](#how-to-run) · [Setup](#setup-shared-with-the-team) ·
[Run 1](#run-1-bc-on-released-demos) · [Run 2](#run-2-more-expert-data--dagger) ·
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

- **Current notebook (run 2):** [`notebooks/Train_PushT_State_BC_DAgger.ipynb`](../notebooks/Train_PushT_State_BC_DAgger.ipynb).
  Upload to Colab, choose an L4 GPU, run top to bottom. Outputs go to `MyDrive/182_final_project/state_bc/<session>/`.
- **Run 1 notebook (kept as run):** [`notebooks/Train_PushT_State_BC.ipynb`](../notebooks/Train_PushT_State_BC.ipynb).
- **Code:** [`pusht_state_bc.py`](pusht_state_bc.py), with subcommands `audit`, `prepare`, `collect`, `train`, `eval`.
- **Notebook source:** the notebook embeds the module. Edit `pusht_state_bc.py` or `build_notebook.py`, then run
  `python state_bc/build_notebook.py`. Never edit the notebook's module cell by hand.
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

**Status:** built, not yet run. **Notebook:** `Train_PushT_State_BC_DAgger.ipynb`.

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

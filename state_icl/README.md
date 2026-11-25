# state_icl: in-context goal inference, PPO-only pilot (Jimmy)

**Question** (`CODEX_CONTEXT.md`): can a frozen transformer infer an unfamiliar Push-T goal from demonstrations and
control the robot, without goal coordinates or test-time weight updates?

This folder trains and evaluates the learner from `PushT_ICL_Transformer_Spec.md` on prompts from **one** teacher:
the frozen PPO teacher in `state_rl/teachers/ppo_vargoal_seed1_50M.pt`. The main experiment trains on a balanced
mix of PPO and BC prompts; that needs Nathan's frozen BC teacher and uses the same code (add its demos as more
prompt sources). This pilot doubles as the PPO-only ablation.

## Files

| File | What |
|---|---|
| `pusht_icl.py` | `collect` (teacher demos per split), `train`, `eval`; the learner `ICLPolicy` |
| `build_notebook.py` | builds `notebooks/Train_PushT_ICL_PPO_Pilot.ipynb` (Kaggle; embeds code and teacher) |
| `tests/test_icl_cpu.py` | spec section-9 checks; data tests need `ICL_TEST_DATA` (a tiny `collect` output) |

Tiny CPU run (mesa lavapipe): `collect --num-envs 1 --sim-backend physx_cpu --goals-per-quadrant 1 --episodes 3`,
then `train --steps 40 --batch 8 --ks 0 1 2 --select-ks 1 2 --n-support-eps 2`, then `eval --num-envs 1
--sim-backend physx_cpu --ks 0 1 --starts 1`.

## Setup

**Learner (fixed by the spec):** Hugging Face `GPT2Model`, 12 layers, width 256, 8 heads, FFN 1024, `gelu_new`,
learned positions, 1024 tokens, no dropout, random init; linear state/action encoders and action head
(normal 0.02 init). One token per state and per action: `[s1, a1, ..., sN, aN, s_query]`; prediction read at the
query position; loss = query-action MSE; no tanh, clipped to [-1, 1] only when executed. 9,751,303 parameters
(256 in the frozen, unused word embedding). transformers 4.46.3.

**Inputs:** the 31-number state obs with `extra/goal_pos` (columns 21–23) removed → 28 numbers: `agent/qpos`,
`agent/qvel`, `extra/tcp_pose`, `extra/obj_pose`. Nothing else in the obs depends on the goal (the goal angle is
fixed). States normalized with mean/std of the train split (std floor 1e-6), frozen for val/test. Actions are
`pd_joint_delta_pos` commands, already in [-1, 1]; not renormalized.

**Splits** (target positions; starts anywhere, including Q4), 16 teacher episodes per goal:

| Split | Goals | Quadrants | Seed |
|---|---|---|---|
| train | 300 | 100 each in Q1, Q2, Q3 | 11 |
| val | 60 | 20 each in Q1–Q3 | 12 |
| test_id | 60 | 20 each in Q1–Q3 | 13 |
| test_ood | 60 | Q4 | 14 |

**Prompts (training):** pick a train goal; k demos = k different **successful** episodes of that goal, each the
whole 100-step episode at every 4th step (25 pairs); query = one step of a **different** episode of the same goal
(any step, successful or not); label = the teacher's action there. k is drawn per batch from {0, 1, 2, 4, 8}
(max 401 tokens). Same k within a batch (no padding).

**Training:** Adam 1e-4, batch 64, grad-norm clip 1.0, 50k steps (8 h cap), seed 0. Val every 2000 steps.
**Selection:** lowest val action MSE with correct demos averaged over k = 1, 2, 4, 8 (val goals only; never test).
`last.pt` keeps the optimizer state for resuming (lesson from the PPO continuation).

**Evaluation (frozen):** per goal, the support pool is the successful episodes among episodes 0–9 in a fixed random
order; context with k demos = the first k (so contexts are nested). Episodes 10–15 are the offline query set.
- *Action error:* MSE against the teacher's actions on all steps of the query episodes.
- *Closed loop:* 8 new starts per goal; one predicted action per step, 100 steps. Conditions: correct-goal demos,
  wrong-goal demos (from goal `(i + G/2) mod G` of the same split, same teacher, same k), no demos (k = 0), and the
  **expert** (PPO with the true goal). The same goals, seeds and env batch give identical starts in every condition;
  the code asserts this.
- Reported: success-once, success-at-end, final and max overlap, per-quadrant success, gap to the expert.
- Goals are evaluated only if they and their wrong-goal partner have ≥ 8 successful pool episodes (the teacher
  succeeds ~93%, so most goals qualify); the expert is measured on the same goals.

**Not in this pilot:** BC prompts (main experiment), the goal-estimate baseline (needs the goal-conditioned BC
policy), extra learner seeds.

## Decisions log

| Date | Decision | Reason |
|---|---|---|
| 2026-11-23 | PPO-only pilot first; same code for the PPO+BC model | BC teacher not frozen yet; Jimmy: "test w a ppo only pilot" |
| 2026-11-23 | Pilot sizes above (300/60/60/60 goals, 16 episodes, stride 4, k up to 8, Adam 1e-4, batch 64) | Proposed by Claude, approved by Jimmy; spec leaves them to a pilot |
| 2026-11-23 | Support demos only from successful episodes | Demos must convey the goal; the doc's goal-estimate baseline also uses successful demos |
| 2026-11-23 | Selection on val correct-context MSE (k = 1–8) | Pinned before running; spec requires a validation-only rule |
| 2026-11-23 | Wrong goal = shifted goal in the same split | Same expert, same k, different target, as the doc requires |

## Runs

*(none yet)*

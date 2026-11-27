# Scripted Push-T pusher (prototype, 2026-11-26)

Goal: a demo-like scripted expert for `PushTVarGoal-v1`, in the spirit of OCBench's scripted policies, to train a BC teacher
that is independent of the PPO teacher (v2: PPO and BC as separate teacher types).

**Status: prototype, not usable as a teacher yet.** Validation protocol (100 steps, seeds 1001 + 1000·q + i, 25 episodes
per quadrant), success-once / success-at-end:

| Q1 | Q2 | Q3 | Q4 | Mean |
|---|---|---|---|---|
| 8% / 8% | 0% / 0% | 8% / 4% | 4% / 4% | **5% / 4%** |

The freeze bar is 80% success-once in every quadrant; the PPO teacher gets 93–95%.

## How it works (`scripted_pusher.py`)
- **Plan:** score candidate pushes (contact points along the T outline × push lengths) with a quasi-static ellipsoidal
  limit-surface model. Pick randomly among the top 3.
- **Approach:** move the stick tip to a stand-off point behind the contact. If the straight path would hit the T, lift over it.
- **Push:** keep the target a few cm ahead of the tip (the joint PD controller only applies force in proportion to its
  tracking error). While in contact, steer among nearby contacts (sliding along the edge). Stop on stall or when no
  nearby push helps.
- **Hold:** when within 7 mm and 0.05 rad of the goal, back away and keep still.
- **Actions:** native `pd_joint_delta_pos`. The desired tip motion is converted with damped least squares on the arm Jacobian
  (`pytorch_kinematics` on ManiSkill's `panda_stick.urdf`; checked to match the simulator's tcp exactly), so recorded actions
  can be used directly as BC labels.
- **Demo-like randomness:** smooth (Ornstein–Uhlenbeck) jitter on the tip target, faded out near the goal; per-episode speed
  and stand-off; top-3 sampling of pushes.

## What was learned while building it
- **Push force:** commanding small position steps stalls against the T (friction 3). A 3 cm lead target fixed it.
- **The tip riding onto the T** (it ended up on top, at height 0.04) caused many no-progress pushes. Fixed by requiring a full
  descent before pushing, and lifting off if the tip is ever above the T.
- **Inside corners** (where the stem meets the bar) wedge the tip. Those contacts are excluded.
- **Model calibration** (`calibrate.py`): the model's rotation direction is right 98% of the time, but the original limit-surface
  length (0.055 m) over-predicted rotation about 2×. The fitted value is 0.08 m.
- **Diagnosis of the remaining failure:**
  - With 300–400 steps instead of 100, success only rises to about 10–25% (`sweep.py`), so it is not just speed.
  - The one-push-ahead greedy planner reaches the goal position but often stalls on the angle: any push that fixes the angle
    also moves the T, and the greedy cost sees no single push that improves both.
  - Repositioning also takes 40–60 of the 100 steps.

## Next steps, most promising first
1. **Plan more than one push ahead:** a short search over 2–3 push sequences, or decompose into rotate-then-translate
   primitives. This targets the angle stall.
2. **Use the simulator as the model:** on the GPU, clone the state and evaluate candidate pushes in parallel environments.
   This is an MPC "expert", still independent of PPO, though less demo-like.
3. **Plan routes around the T on the table** instead of lifting, to cut approach time.

## Running it
Needs ManiSkill 3.0.1 (commit `a4a4f92`), `pytorch_kinematics`, `imageio[ffmpeg]`, `matplotlib`, and `pusht_vargoal.py` plus
`assets/tip_start_qpos.npz` from `jimmy-state-rl` (`state_rl/`) next to these files. It runs on CPU (`physx_cpu`), at about
2 s per episode.

```
python run_pusher.py --episodes 25 --quadrants 1 2 3 4 --videos 2 --video-successes --out results
```

**Videos** (`videos/`) are top-down replays drawn from the simulator state at every step:
- the T turns green while it is inside the 90% overlap
- the grey dashed T is the goal
- the blue line is the tip's path; a hollow dot means the tip is lifted

`highlights.mp4` strings together 4 successes, a near miss and 2 typical failures. Rendering 3D ManiSkill videos needs
Vulkan, which did not work on this machine; they can be recorded on Colab.

# icl for control policies

This project studies whether a frozen transformer can infer a held-out
control policy from a few demonstrations.

The initial experiment uses ManiSkill `PushT-v1` with the Panda arm and stick
end effector. Start with numerical state observations and a single controller
mode shared across teachers. Validate a released PPO expert and its
demonstrations before building teacher-level splits and testing whether
correct context improves prediction and rollout performance compared with
empty and wrong-teacher context. Follow with RGB and robot proprioception.

The saved course proposal still describes Gym-PushT.

See [CODEX_CONTEXT.md](CODEX_CONTEXT.md) for the research handoff and a
starter prompt for new Codex sessions.

## Camera-based PPO teacher

Current choices are recorded in [Camera_PPO_Decisions.md](Camera_PPO_Decisions.md).
Use [Train_PushT_RGB_Joint_PPO_50M_MB32.ipynb](notebooks/Train_PushT_RGB_Joint_PPO_50M_MB32.ipynb)
in Colab with an L4 runtime. It trains the original RGB PPO baseline from random
weights for a 50M-transition budget using `pd_joint_delta_pos` and the original
RGB/proprioception inputs, including `tcp_pose` but excluding numerical object
and goal poses. A tqdm bar retains success-once, success-at-end, return, and SPS
statistics together. All evaluation rows are also saved as CSV. The new run uses 1024 environments,
16 rollout steps, and 32 minibatches (512 samples each), with a launch check
to prevent accidentally repeating the completed run's 8-minibatch configuration.
The previous joint-control notebook is preserved for reference.

The self-contained notebook installs a pinned ManiSkill baseline, verifies inputs
and rendering, and runs a small PPO smoke test before the main training cell.
Checkpoints and run metadata are saved to Google Drive by default.
Training uses a separate Python 3.11 environment, including CUDA PyTorch, so it
does not depend on the notebook kernel's Python version. Installation output is
streamed and saved in `setup.log`, with error details included on failure.

The earlier experimental notebook is preserved as
[Train_PushT_RGB_PPO.ipynb](notebooks/Train_PushT_RGB_PPO.ipynb). Its custom
observation adapter is available as
[scripts/train_rgb_ppo.py](scripts/train_rgb_ppo.py). It restricts both actor
and critic to one camera image plus joint positions and velocities and uses
end-effector position control. Its checkpoints are incompatible with the new
joint-control setup. Neither notebook's long training has been executed locally;
use the preflight and smoke checks in Colab.

Offline notebook/monitor checks: `python3 -B -m unittest discover -s tests -v`.

## Teacher diversity probe

[Probe_PushT_Teacher_Diversity.ipynb](notebooks/Probe_PushT_Teacher_Diversity.ipynb)
is the next self-contained Colab L4 experiment. The selected first teacher,
`ckpt_2776.pt`, scored 97% success-once and 96% end success in the user's
100-episode validation. The notebook loads that teacher independently into
an original-reward control branch and an action-penalty branch, initially
fine-tuning each for 1M requested transitions. It calibrates the penalty from
the parent's observed scales and compares original-task competence, command
magnitude, and actions on shared observations. Logs, per-episode results,
checkpoints, and videos are saved to a new Drive output directory. Its parent
checkpoint path is preset to the user's selected run; update it if moved.
The GPU probe has not yet been run locally.

The magnitude probes subsequently retained high success but produced modest
command-size changes. The next selected experiment is
[Probe_PushT_Contact_Route.ipynb](notebooks/Probe_PushT_Contact_Route.ipynb),
a self-contained Colab notebook for opposing crossbar contact-patch preferences
and left/right approach-route preferences. It runs four independent 1M-step
fine-tunes from the original teacher and reuses the completed control when its
settings match. Force-based contact diagnostics, approach trajectories, plots,
original-task evaluations, and shared-observation actions check whether the
intended spatial differences actually emerge. The new spatial GPU probe remains
unexecuted here; preflight and smoke checks are included.

The next PPO-only probe is
[Probe_PushT_Directional_Velocity.ipynb](notebooks/Probe_PushT_Directional_Velocity.ipynb).
It tests the original directional movement idea using measured physical TCP
velocity, with x-heavy and y-heavy quadratic penalties. Each branch starts from
the original parent for 1M requested transitions; a matching completed control
is reused. Coefficients are calibrated using sampled parent rollouts. Matched
evaluations report original-task competence, axis motion before first success
and over the whole episode, paired differences, common-observation actions,
plots, and videos. This new GPU experiment has not been executed locally.

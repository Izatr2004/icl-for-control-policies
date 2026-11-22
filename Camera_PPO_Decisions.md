# Camera-Based PPO Teacher: Decisions and Possible Extensions

## Current training setup (November 21, 2026)

- **Notebook:** `notebooks/Train_PushT_RGB_Joint_PPO_50M_MB32.ipynb`.
- **Method:** Original ManiSkill RGB PPO script, without the custom observation adapter.
- **Controller:** `pd_joint_delta_pos`, matching the RGB baseline's default.
- **Actor and critic inputs:** Baseline RGB plus robot proprioception, including joint positions, joint velocities, and stick-tip pose (`tcp_pose`). Numerical block and target poses remain excluded. Audit the actual installed fields before training.
- **Initialization and budget:** Fresh random weights; 50,000,000 requested transitions in one job. With 1024 environments and 16 steps per rollout, complete batches yield 49,987,584 transitions.
- **L4 configuration:** 1024 environments, 16 rollout steps, 32 minibatches (512 samples each), 8 optimization passes, discount 0.99, and 16 evaluation environments. Keep seed 1, learning rate 0.0003, no learning-rate annealing, and the other original settings. Run name: `rgb_joint_seed1_50m_mb32`.
- **Monitoring:** tqdm displays the latest success-once, success-at-end, return, and SPS together. Save full logs, evaluation CSV, checkpoints, source/configuration, and resolved package versions to Drive.
- **Unchanged choices:** ManiSkill `PushT-v1`, original task reward, one current RGB frame, and the baseline camera configuration.

This supersedes the earlier controller and joint-only input choices below. Preserve the earlier checkpoints as exploratory runs, not compatible starting weights for this setup. The user-supplied published success curve stayed near zero until roughly 10M transitions; the earlier zero success at about 5M alone does not establish that the approach cannot learn.

## Completed 50M run and next comparison

The completed run actually used 1024 environments and 8 minibatches (2048 samples each). It learned before collapsing near 49M transitions. In a matched 100-episode evaluation, `ckpt_2851.pt` achieved 72% success-once and 32% success-at-end; the final checkpoint achieved 4% and 0%. Approximate KL rose sharply near collapse, consistent with update instability, but this does not establish the cause.

The new notebook starts fresh and changes only the minibatch count to 32 and the run name. Keep both success metrics on tqdm and in CSV. More minibatches also mean more optimizer updates per rollout; this comparison does not assume improved stability. Preserve earlier checkpoints for comparison.

## Successful 32-minibatch teacher and diversity probe

The user completed the 32-minibatch run and supplied its evaluation table. It reached sustained high success late in training, without the previous run's comparable collapse. In a matched 100-episode checkpoint comparison (seed 1001), `ckpt_2776.pt` achieved **97% success-once and 96% success-at-end**, versus the final checkpoint's 100% and 76%. The user selected `ckpt_2776.pt` as the first teacher. These are validation results; a fresh-seed test remains outstanding. One run comparison does not establish the minibatch change's reliability across seeds.

Selected source checkpoint in Colab:
`/content/drive/MyDrive/182_final_project/camera_ppo_joint/20261121T212436151405Z/runs/rgb_joint_seed1_50m_mb32/ckpt_2776.pt`.

The next authorized experiment is `notebooks/Probe_PushT_Teacher_Diversity.ipynb`, a self-contained Colab notebook. It re-evaluates the parent, calibrates a clipped normalized joint-command penalty, and fine-tunes independent control and preference branches for 1M requested transitions each (999,424 executed). Both load all parent weights with fresh Adam, seed 2, learning rate 3e-5 and target KL 0.05; the rollout configuration remains 1024 environments, 16 steps, 32 minibatches and 8 epochs. The initial expected penalty is 5% of the parent's mean absolute reward on the calibration distribution. This is an exploratory budget and coefficient, not a validated optimum.

Evaluation uses original task rewards and 100-step episodes, matched seeded initial states, per-episode success/return/command measures, shared-observation deterministic action comparisons, and diagnostic videos. The command cost is not physical energy. All descendants share a parent. Neither useful diversity nor affordable fine-tuning has yet been demonstrated. Inspect this probe before expanding toward a larger teacher population. Offline checks pass; the new GPU evaluation and fine-tuning still require execution in Colab.

## Spatial preferences selected after magnitude probes

The user executed the magnitude probe. On 100 matched validation episodes (seed 2001), parent/control/penalty had end success 95%/98%/98%, and command MSE 0.29625/0.29266/0.28933. A 3x penalty (coefficient 0.281359), independently initialized from the parent for 1M requested transitions, retained 98% end success with command MSE 0.28229 (3.54% below the control). The initial penalty-versus-control command-cost confidence interval included zero; no interval was provided for the 3x comparison. These modest changes have not established a useful preference family. Smoothness was discussed but not implemented or tested.

The user explicitly selected **contact point and route** as the next direction. `notebooks/Probe_PushT_Contact_Route.ipynb` implements four independent 1M-step probes: contact at either crossbar end cap in the object's local frame, and left/right sine-shaped approach routes with 6 cm maximum lateral offset. Both reward costs are bounded by .15 per step and disabled at task success. These coefficients and budgets are exploratory. All variants retain RGB/robot inputs and the original task success rule; object pose is reward/diagnostic information only.

The contact reward uses near-surface distance to the selected patch. Separate force queries detect tool–T contact and measure tool position at detected contact; this is a contact-location proxy, not the true contact manifold. Tool collision links are selected from the installed URDF and printed. Route shaping stops at first detected contact, maintains per-environment reset state below the vector auto-reset wrapper, and does not expose its phase/anchors to the reactive policy. Contact queries sample the final simulation substep and may miss brief contacts. The notebook includes actual GPU preflight checks for penalty bounds and partial-reset isolation before the training smoke tests.

The existing control checkpoint is preset to `/content/drive/MyDrive/182_final_project/teacher_diversity_probe/20261122T013725865607Z/runs/probe_control/final_ckpt.pt`. It is reused only if provenance/settings match; otherwise one additional control is trained. Outputs include original-task success, paired initial states, patch/route metrics, trajectory traces, common-observation actions, videos, and plots. This new spatial notebook has not been executed on a GPU here. Inspect its preflight and results before counting any spatial variant as an established new teacher.

## Directional TCP-velocity probe

The user completed the spatial probe: contact-left/right retained 97%/98% end success, and route-left/right 96%/99%, but opposing contact first-position means differed by only about 4.3 mm and approach lateral means by about 0.8 mm. The supplied plots showed substantially overlapping paths. These are competent descendants, not demonstrated distinct route/contact strategies. The clipped route cost and proximity-gated contact cost are possible reward-design limitations; their causal contribution has not been established.

The user is responsible for the **PPO portion** of the project and requested the original directional velocity idea next. `notebooks/Probe_PushT_Directional_Velocity.ipynb` is a self-contained Colab L4 notebook. It measures world-frame TCP displacement over each control interval, before vector auto-reset, and subtracts `lambda_x*vx^2 + lambda_y*vy^2`. This is a physical interval-average velocity cost, not joint-command size, instantaneous substep velocity, or energy. The exact quadratic is uncapped and remains active after success; this choice differs from the older prose's bounded-penalty wording and avoids a clipping plateau.

The two branch ratios are 10:1 and 1:10. Coefficients are calibrated separately to 15% of mean absolute task reward using 20 stochastic parent rollouts (seed 4001), then frozen. Both branches independently load the selected parent and use fresh Adam, seed 2, learning rate 3e-5, KL .05, and 1M requested transitions. The prior original-reward control is reused only when settings/provenance match; otherwise an additional control is trained. These ratios, strength, and budget are exploratory.

Evaluation uses 100 matched deterministic validation episodes (seed 3001), original task rewards, whole-episode and before-first-success axis velocity statistics, travel distances, task success/timing, paired batch-bootstrap intervals, common-observation actions, traces, plots, and videos. The original observation/controller contract is preserved. GPU preflight validates physical units, reward arithmetic, and resets; offline checks pass, but no execution or results for this new notebook are claimed locally. SMERL-inspired discriminator training was discussed, not implemented or selected as the immediate experiment.

## Earlier pilot decisions (superseded where noted above)

- **Environment:** ManiSkill `PushT-v1`.
- **Method:** Train using the existing RGB PPO baseline.
- **Compute:** Start on Colab with an L4 GPU.
- **Initial scope:** Train one baseline teacher. Decide the final teacher count and splits after pilot results.
- **Reward:** Start with the unchanged baseline reward.
- **Controller:** Use `pd_ee_delta_pos`. The policy controls stick-tip displacement and learns where to make contact.
- **Actor and critic inputs:** Both receive RGB plus robot joint positions and velocities. Neither receives privileged numerical object state.
- **Observation history:** One current image, without frame stacking.
- **Camera:** Use the baseline camera view and resolution, after checking a rendered frame.

## Options to explore

- **Different reward functions:** Test movement-speed penalties, including different penalties along the table's two axes, to encourage distinct policies.
- **Contact preferences:** Consider rewards encouraging contact near different regions of the block if simpler variations produce insufficient diversity.
- **Multiple training seeds:** Check whether independent training runs already produce competent, distinguishable policies.
- **Motion history:** Add recent image frames if a single image leaves important block motion ambiguous.
- **Camera settings:** Adjust viewpoint or resolution if visibility or visual detail limits learning.
- **Training scale:** Tune parallel environment count and minibatch sizes based on L4 memory use and throughput.

## Still to decide

- Final teacher population and training/validation/test splits.
- Training budget and checkpoint-saving frequency.
- Validation procedure, checkpoint selection, and final evaluation.
- Demonstration count, storage format, and whether to include failed episodes.

Before full training, run a short check of rendering, observation fields, controller behavior, GPU memory use, and throughput.

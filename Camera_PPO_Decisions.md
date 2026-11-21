# Camera-Based PPO Teacher: Decisions and Possible Extensions

## Initial decisions

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

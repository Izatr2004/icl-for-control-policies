# Push-T policy ICL: Codex handoff

To start a Codex session from this repository, paste:

> Read `CODEX_CONTEXT.md` for project context, then inspect the repository and Git status before making changes. Use ManiSkill `PushT-v1` for new implementation work. Treat the experimental design below as project background and follow my current request for the scope of this session.

## 1. Read this first

This is a research project about **in-context learning of control policies**, inspired by Garg et al., *What Can Transformers Learn In-Context? A Case Study of Simple Function Classes* (2022).

The central question is:

> Can a transformer trained across a family of control policies infer an unseen policy from a few demonstrations, without changing its weights at test time?

We use Push-T as the case study. Start with numerical observations, then investigate camera images plus robot proprioception. This is a controlled policy-inference experiment, not primarily a large pretrained VLA or language-model benchmark.

The CS 182 team is Izat Rasyad, Nathan Jimenez, Jimmy Liu, and Neil Chen. Available hardware is **four RTX 3090 GPUs, 24 GB each, plus a machine with many CPU cores**. Access to that hardware from a new Codex session must be checked; do not assume it is the local machine.

### Status and authority

The latest saved proposal, `CS182_Vision_to_Action_ICL_Proposal.tex`, was read in full for this handoff. It is version 6, last modified November 12, 2026, Pacific time. Its title is *In-Context Learning of Push-T Action Policies*.

| Item | Current status |
|---|---|
| Research goal: Garg-style ICL for control policies | Established project direction |
| Push-T only, no dot/blob alignment task | Established scope |
| Small randomly initialized GPT-2-style learner | Primary proposed learner |
| Both BC-trained and PPO-trained teacher populations | In the saved proposal |
| Numerical inputs first, vision plus robot state later | In the saved proposal |
| Gym-PushT with Pymunk | Environment currently written in the saved proposal |
| ManiSkill `PushT-v1`, with a simulated articulated arm | Selected for implementation. The saved proposal still describes Gym-PushT. |
| Released ManiSkill PPO teachers and demonstrations | Verified available; not downloaded or evaluated in this conversation |
| ACT and Diffusion Policy | Available imitation baselines discussed as possible tools/comparisons; not selected as the primary ICL learner |
| Flow-matching head and action-chunk comparisons | Proposed experiments, subject to core pilot feasibility |
| Llama-style backbone and flow-based teachers | Later extensions |
| Implementation, trained models, experimental results | None established by this handoff |

Use the latest direct user instruction if it changes these choices. Do not silently report the saved proposal as already using ManiSkill. Do not claim a live Overleaf project was edited: the established artifacts are the `.tex` and PDF files.

## 2. What the Garg analogy means

In Garg's setup, training across many sampled functions teaches a transformer how to use examples to predict outputs of a new function. Here, the target functions are policies.

For a deterministic teacher indexed by `j`:

```text
teacher: f_j(state) -> action
context D_j: observation/action examples from teacher j
learner: (D_j, query observation) -> predicted teacher-j action
```

During training, sample a teacher, construct a prompt from its demonstrations, and optimize prediction of its query actions. The learner's weights acquire a prior over the policy population and the ability to use context. At test time, freeze the learner and provide demonstrations from a held-out teacher.

Important distinctions:

- Training supplies general knowledge about the policy family. Context supplies evidence about the particular teacher. These are complementary, not mutually exclusive explanations.
- There is no separate test-time weight-update or fine-tuning phase in the primary ICL evaluation.
- A single teacher with many starting states is not the same experiment as learning across a family of held-out policies.
- A new teacher can be sampled from the same training distribution. That alone does not establish out-of-distribution state generalization. Evaluate state-distribution shifts separately if desired.
- Goal conditioning alone is not demonstration-conditioned policy inference.
- Action chunking alone is not ICL. Predicting a sequence does not establish adaptation to demonstrations.
- BC and PPO teachers are two training-method populations, not automatically distinct mathematical function classes.

This is separate from an older project comparing LLM/VLM/VLA ICL. Do not import that project's OC-Bench evaluation, language inputs, or large VLA backbones into this experiment unless requested.

## 3. Environment choice and migration implications

### Environment in the saved proposal: Gym-PushT

- Repository: https://github.com/huggingface/gym-pusht
- Gymnasium API, **Pymunk 2D physics**, not MuJoCo.
- Circular pusher interacting with a T-shaped block; no articulated robot arm.
- `state` observation: pusher `(x,y)` and T-block `(x,y,theta)`, five values.
- `pixels_agent_pos`: RGB image, by default 96 by 96, plus pusher `(x,y)`.
- Actions are 2D pusher target positions. The proposal normalizes them to `[-1,1]^2` for learning.
- The default five-value observation omits velocities. Do not describe it as complete simulator state. Adding velocities needs a wrapper and consistent data.
- Here, the phrase “robot state” means pusher position, not robot joint proprioception.

### Selected environment: ManiSkill `PushT-v1`

- Repository: https://github.com/mani-skill/ManiSkill
- Built on SAPIEN/PhysX, with GPU-parallel simulation.
- Uses a Panda arm with a stick end-effector (`panda_stick`). It models articulated robot dynamics and contacts.
- Task is based on the real-world Push-T setup associated with Diffusion Policy. It is not the original 2D Gym-PushT environment.
- Supports numerical observation modes and rendered camera observations with robot proprioception.
- General observation structure includes `agent` data such as joint positions and velocities, task-specific `extra`, and camera `sensor_data`.
- Released datasets include several controller modes: `pd_ee_delta_pos`, `pd_ee_delta_pose`, and `pd_joint_delta_pos`.

Do not carry Gym-PushT assumptions into ManiSkill. In particular:

1. Inspect actual observation and action spaces. Action dimensions and semantics depend on the controller; do not hard-code a 2D action.
2. Replace proposal notation `A in R^(H x 2)` with `A in R^(H x d_action)` if migrating.
3. Check the installed task's reward, success test, termination, and horizon. Published task cards describe 90% T overlap and an end-effector end-zone step, with an easy variant, but code/version details should be verified before reproducing results.
4. Preserve consistent controller mode, normalization, control frequency, and simulator settings across teachers and learners.
5. Do not treat different controller checkpoints as interchangeable members of one policy family.
6. For vision plus proprioception, explicitly select allowed fields. Do not accidentally expose privileged object state through auxiliary observation fields.
7. Gym-PushT human demonstrations are not directly interchangeable with ManiSkill arm trajectories.


## 4. ManiSkill data and expert availability

Official Push-T release:

https://huggingface.co/datasets/haosulab/ManiSkill_Demonstrations/tree/main/demos/PushT-v1/rl

Verified contents include:

- `ppo_pd_ee_delta_pos_ckpt.pt`
- `ppo_pd_ee_delta_pose_ckpt.pt`
- `ppo_pd_joint_delta_pos_ckpt.pt`
- Matching HDF5 trajectory files, JSON metadata, and sample videos.

The official generation script trains **state-based PPO** and rolls it out to produce successful demonstrations. It is here:

https://github.com/mani-skill/ManiSkill/blob/main/scripts/data_generation/rl.sh

The released checkpoints above are state-based experts. A pretrained camera-based Push-T PPO checkpoint was **not found in that release**. This is not a claim that none exists anywhere. ManiSkill does provide RGB PPO training code and Push-T configurations:

https://github.com/mani-skill/ManiSkill/blob/main/examples/baselines/ppo/examples.sh

After installing a compatible ManiSkill version, the documented download command is:

```bash
python -m mani_skill.utils.download_demo PushT-v1
```

The raw demonstration format stores actions and simulator states with JSON metadata. Rendered observations may be omitted for compression. Replay can produce state or RGB datasets. Match simulator/backend/controller settings and follow the official replay guidance; do not assume arbitrary replay is exact.

- Dataset format: https://maniskill.readthedocs.io/en/latest/user_guide/datasets/demos.html
- IL setup/replay guidance: https://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/setup.html
- Replay script: https://github.com/mani-skill/ManiSkill/blob/main/scripts/data_generation/replay_for_il_baselines.sh

**Rendering images for state-expert trajectories does not produce a camera-trained expert.** It produces image/action labels whose actions were chosen using privileged state.

A suggested modality experiment is to retain the same state-based teachers and collect aligned records `(sim_state, robot_proprioception, RGB, teacher_action)`. Train one learner on numerical observations and another on RGB plus proprioception. This holds teacher behavior fixed. Training separate visual teachers is another experiment because it also changes teacher competence and behavior.

## 5. Teacher policies and policy-family design

### Custom teacher architectures in the saved proposal

| Teacher | Proposed architecture and training |
|---|---|
| BC | MLP with two 256-unit ReLU hidden layers, tanh output, supervised action MSE |
| PPO | Gaussian actor with two 256-unit ReLU hidden layers, action mean and learned standard deviations; separate value MLP with the same hidden architecture |

For the initial deterministic PPO teacher, freeze the actor and validate its clipped mean policy. The value network is used for RL training, not action labeling. These are **proposed custom architectures**, not verified descriptions of the released ManiSkill checkpoints. Instantiate released weights using their matching implementation and configuration.

For action horizon `H > 1`, the proposal uses consecutive actions from teacher rollouts as chunk targets. Start with `H=1` to establish the policy-inference mechanism. For larger horizons, recognize that subsequent actions depend on dynamics and subsequent states. With partial observations or stochastic teachers, a chunk may not be a deterministic function of a single observation.

### Proposed diversity mechanisms

- BC: trajectory-level resampling and initialization seeds, with different demonstrated strategies if needed. The saved proposal refers to compatible human Push-T demonstrations. Using ManiSkill PPO-generated demonstrations for BC is a subsequently discussed option.
- PPO: seeds and bounded pusher-speed penalties, while holding goal, dynamics, and action conventions fixed.
- Original reward proposal: `r_j = r_align - lambda_x,j * v_p,x^2 - lambda_y,j * v_p,y^2`.

If using ManiSkill, the velocity definition and reward integration need to be adapted and validated for the arm's end-effector. The old reward formula has not already been implemented there.

**The central unresolved design challenge is obtaining successful, distinguishable policies.** Seeds or resampling alone may yield nearly identical teachers. On shared states, check both task competence and consistent action differences. Conversely, simply creating poor teachers is not useful diversity. Keep these pilot diagnostics separate from final held-out evaluation.

Hold out entire teachers, and where applicable their reward settings, not merely episodes from the same teacher. BC teachers trained on the same underlying data can overlap heavily; record data lineage. Exact teacher counts and train/validation/test proportions are not yet fixed.

## 6. Existing imitation-learning code and distinctions

ManiSkill has the following code:

| Method | What was verified | Caveat |
|---|---|---|
| Plain BC | `examples/baselines/bc/bc.py`, a state-based 2-by-256 ReLU MLP trained with MSE | Docs still label standard BC WIP. Example scripts cover PushCube/PickCube; adapt and validate for Push-T. Its source uses a linear output, unlike the proposal's tanh output. |
| ACT | State and RGB Push-T commands in `examples/baselines/act/baselines.sh` | Training examples, not evidence that this session trained a policy |
| Diffusion Policy | Official implementation and README under `examples/baselines/diffusion_policy` | Verify the chosen configuration, data, and controller before running |

Links:

- https://github.com/mani-skill/ManiSkill/blob/main/examples/baselines/bc/README.md
- https://github.com/mani-skill/ManiSkill/blob/main/examples/baselines/bc/bc.py
- https://github.com/mani-skill/ManiSkill/blob/main/examples/baselines/act/baselines.sh
- https://github.com/mani-skill/ManiSkill/blob/main/examples/baselines/diffusion_policy/README.md
- https://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/baselines.html

ACT predicts continuous action chunks with a transformer trained as a conditional VAE. The original method fixes its latent variable to zero for deterministic evaluation and can ensemble overlapping chunks. Diffusion Policy samples continuous action chunks by iteratively denoising random action vectors, using a CNN or transformer denoiser. Denoising iterations are computation steps, not environment steps.

Both methods use action sequences; an ACT-versus-diffusion comparison does not isolate chunking. Neither original method automatically implements demonstration-conditioned ICL. Flow matching is also distinct from diffusion; the saved proposal explicitly proposes flow matching, and the recent diffusion discussion has not silently replaced it.

Keep separate the **teacher architecture**, the **ICL context-processing backbone**, and the **learner's action-generation head**. Changing all three together would make causal interpretation difficult.

## 7. ICL learner and training

Primary learner from the proposal:

- Randomly initialized GPT-2-style causal transformer.
- 12 layers, hidden width 256, 8 attention heads.
- Learned projections for numerical observations and actions.
- Later, a small CNN for images plus a robot-state projection.
- Optional small Llama-style backbone comparison with matched parameter/training budgets, encoders, and action heads.

“GPT-2-style” means the architecture family, not a pretrained language checkpoint or a text tokenizer. Actual parameter count depends on implementation; report it from instantiated code. Do not hard-code a language vocabulary that the numerical experiment does not need.

Conceptual prompt:

```text
[(O_1, A_1), ..., (O_k, A_k), O_query] -> predicted A_query
```

Support comes from one teacher and query observations from separate rollouts of that teacher. Ground-truth query actions are prediction targets and must not leak into their input positions. The saved evaluation measures `k` equal-duration **support snippets**, so a snippet can contain several observation/action pairs. Tokenization, snippet duration, episode delimiters, observation history, and maximum token budget remain implementation choices to specify explicitly.

The proposal trains separate learners on BC and PPO teacher populations with matched budgets. Compare MSE and flow-matching heads with teacher data fixed within each group. Initial horizons are `H=1` and `H=8`; execute **one action before replanning** to isolate prediction horizon from execution horizon. Whole-chunk open-loop execution would be an additional comparison, not the saved default.

Optional flow teacher: three-layer, 256-unit SiLU MLP mapping numerical observation, noisy action chunk, and flow time to a velocity field, trained on the same data as a chunked BC teacher. This is a proposed extension, not implemented. Account for sampling variability when using stochastic teachers or heads.

## 8. Evaluation and evidence for ICL

Proposed primary comparison:

- Frozen learner, held-out teachers and separate query rollouts.
- Context counts `k = 0, 1, 2, 4, 8` equal-duration support snippets.
- Correct target-teacher context versus no context and another teacher's context.
- Three learner seeds, plus variation over held-out teachers and evaluation starts.
- Action prediction error, alignment success, overlap, and gap to teacher performance.
- Match query states, rollout starts, support duration, and observation history across conditions.

Use both fixed-query prediction and closed-loop evaluation. Once policies execute different actions, their visited states diverge; distinguish that from matched-state action error. More successful rollout behavior does not necessarily mean closer imitation of a particular teacher, and low action error alone does not guarantee successful control.

A useful positive result is improved held-out-teacher prediction and task performance with correct context, beyond no/wrong-context controls. Context length alone is not sufficient evidence if a pooled policy can ignore it. Do not report results or claim this mechanism works until tested.

For vision, report remaining information differences from numerical input. If the teacher uses information absent from the learner's observation/history, some prediction error may reflect partial observability. For sampled action distributions, a single demonstrated action is not necessarily the only valid action; interpret pointwise errors alongside success and sampling variability.

## 9. Recommended next steps for an implementation session

These are proposed engineering steps, not completed work:

1. Read repository instructions and inspect existing code before creating a new implementation. Check GPU/CPU access and dependencies. Keep any user work intact.
2. Pin the environment/package versions. Create/reset/step the task, inspect observation/action fields, and verify rendering and success reporting.
3. Download a released state-based PPO expert and matching data. Choose one controller mode. Validate checkpoint compatibility and run a small teacher evaluation before generating a large dataset.
4. Fit a simple numerical BC baseline and evaluate rollouts. This checks the data and action pipeline; it does not yet establish ICL.
5. Pilot a small policy family and assess distinguishability on shared states. Decide how to create more variation if contexts do not identify teachers.
6. Build teacher-level splits and episodic support/query sampling. Implement the smallest numerical `H=1` ICL learner with no-context and wrong-context controls.
7. Measure throughput, memory, convergence, teacher success, and context sensitivity. Estimate the full compute budget from those pilots.
8. Expand to more teachers and learner seeds, then `H=8` and modality comparisons. Add flow matching, full ACT/diffusion comparisons, or Llama only after the core result is interpretable.

Do not launch an exhaustive experiment grid merely because four GPUs are available. The proposal intentionally gives no unmeasured “hundreds of GPU-hours” estimate. Independent jobs across GPUs are a reasonable initial use; distributed training is not required by the plan.

## 10. Proposal constraints and references

If asked to update the proposal later:

- Preserve the four-student team and one team submission.
- Abstract at most six sentences.
- Introduction/background must summarize at least eight **papers**, at least one sentence per paper. A software repository does not count as a paper.
- Include key questions, hypothesis, methods/experiments, and expected compute.
- LaTeX/PDF, 12-point font, one-inch margins. Check the course topics document for the authoritative page limit; the existing artifact was arranged as two main pages plus references.
- Do not use the excluded A. Zhao et al. ViT ICL case study, arXiv:2505.20872.
- Brown/GPT-3 was replaced by Akyurek et al. because it is more directly relevant to this controlled function-learning experiment.
- User prefers plain student-style writing and **no em dashes**.

Eight papers in the saved proposal:

1. Garg et al. (2022), *What Can Transformers Learn In-Context? A Case Study of Simple Function Classes*. https://arxiv.org/abs/2208.01066
2. Akyurek et al. (2023), *What Learning Algorithm Is In-Context Learning? Investigations with Linear Models*. https://arxiv.org/abs/2211.15661
3. Fu et al. (2025), *In-Context Imitation Learning via Next-Token Prediction* (ICRT). https://icrt.dev/
4. Chi et al. (2023), *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*. https://diffusion-policy.cs.columbia.edu/
5. Zhao et al. (2023), *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT). https://roboticsproceedings.org/rss19/p016.html
6. Schulman et al. (2017), *Proximal Policy Optimization Algorithms*. https://arxiv.org/abs/1707.06347
7. Lipman et al. (2023), *Flow Matching for Generative Modeling*. https://arxiv.org/abs/2210.02747
8. Touvron et al. (2023), *LLaMA: Open and Efficient Foundation Language Models*. https://arxiv.org/abs/2302.13971

The ninth reference is Gym-PushT software. If adopting ManiSkill in the written proposal, update the environment description and references rather than leaving contradictory details.

Additional official ManiSkill references:

- Push-T task implementation: https://maniskill.readthedocs.io/en/latest/_modules/mani_skill/envs/tasks/tabletop/push_t.html
- Observation modes: https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html
- GPU simulation: https://maniskill.readthedocs.io/en/latest/user_guide/concepts/gpu_simulation.html
- RL baselines: https://maniskill.readthedocs.io/en/latest/user_guide/reinforcement_learning/baselines.html
- ManiSkill3 RSS 2025 paper: https://roboticsproceedings.org/rss21/p021.html

The links document what was checked in this discussion. Inspect the installed versions and repository commit when implementing; their exact interfaces and behavior may change.

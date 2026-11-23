# Push-T goal inference with in-context learning

> Read `CODEX_CONTEXT.md` and inspect the repository and Git status before making changes. Follow the variable-goal experiment below. Use the transformer spec for the compatible backbone and tokenization details. Follow my current request for the scope of this session.

## Research question

Can a frozen transformer infer an unfamiliar Push-T goal from demonstrations and successfully control the robot, without explicit goal coordinates or test-time weight updates?

Modify ManiSkill `PushT-v1` to vary the target position. Keep target orientation fixed for the initial numerical-observation experiment. Starting positions refer to the robot's pushing tip and the movable T-block within a reachable, valid workspace. The robot base stays fixed.

| Stage | Starting positions | Target positions | Explicit target coordinates |
|---|---|---|---|
| PPO teacher training | Q1–Q4 | Q1–Q4 | Yes |
| BC teacher training | Q1–Q4 | Q1–Q4 | Yes |
| Transformer training | Q1–Q4 | Q1–Q3 only | Hidden |
| Transformer validation | Q1–Q4 | Separate goals and episodes in Q1–Q3 | Hidden |
| In-distribution test | Q1–Q4 | Separate unseen goals in Q1–Q3 | Hidden |
| Out-of-distribution test | Q1–Q4 | Q4 | Hidden. Demonstrations convey the goal. |

Q4 starts and trajectories that pass through Q4 are allowed during transformer training. The holdout applies to target positions so the experiment isolates generalization to unfamiliar goals. Define the quadrant origin and reachable sampling region in the implementation before collecting data.

## Goal-conditioned teachers

Train PPO with randomized starts and targets across Q1–Q4, using numerical robot state, movable block pose, and explicit target coordinates. Collect demonstrations across that workspace and train a numerical BC teacher with the same goal-conditioned input. If BC learns from PPO demonstrations, describe the comparison as PPO and distilled BC. Validate both teachers across all quadrants and freeze them before collecting transformer supervision.

## Transformer prompts and training

Each prompt uses one expert and one target. Collect multiple episodes with that same expert and target from different starting configurations. Use distinct episodes for support demonstrations and query observations. Teacher actions on query observations serve as labels.

Keep robot state and the movable block pose in both support and query inputs. Exclude target coordinates, goal-relative features, and simulator-state fields that reveal the target. Demonstration movements and final block poses can convey the intended destination.

Train one randomly initialized GPT-2-style learner on a balanced mixture of PPO and BC prompts. Use 12 causal layers, width 256, eight attention heads, numerical state and action projections, and query-action MSE. The initial prediction horizon is one action. Use only Q1–Q3 targets for training and select checkpoints and hyperparameters on separate Q1–Q3 validation goals.

Use [the transformer spec](PushT_ICL_Transformer_Spec.md) for backbone and tokenization details. Its earlier split focuses on held-out policies. The primary split for this experiment holds out goal locations.

## Frozen evaluation

Evaluate on unseen Q1–Q3 goals and separately on Q4 goals. For each target, provide expert demonstrations for that goal and roll out from separate initial configurations. Predict and execute one action at a time with frozen weights. Q4 demonstrations are available as evaluation context.

Compare correct-goal demonstrations, equally sized wrong-goal demonstrations from the same expert, and no context. Match query states, rollout starts, and support duration across conditions. Vary context size and report action error, closed-loop success, alignment overlap, and the gap to the expert.

Include a baseline that estimates the goal from the final block pose in a successful demonstration and passes it to a goal-conditioned BC policy trained on the same Q1–Q3 source data.

## Status and extensions

The existing fixed-target teacher and transformer pilots are preliminary work. The variable-goal setup above is the new experiment design. The hypothesis is that correct demonstrations improve control of unfamiliar targets, including Q4.

Later experiments can hold out starting configurations, compare action chunking, or add vision and proprioception. In the visual setting, hide the target outline in the transformer's rendered observations.

## Table 1: closed-loop success at k = 4 (%, 95% Wilson interval, 512 episodes per cell)

| Goals | Demos from | Metric | Correct-goal demos | Wrong-goal demos | No demos | Teacher (sees goal) |
|---|---|---|---|---|---|---|
| Test, Q1–Q3 goals | PPO | once | 91.2 [88.4, 93.4] | 1.4 [0.7, 2.8] | 1.4 [0.7, 2.8] | 94.1 [91.8, 95.9] |
| Test, Q1–Q3 goals | PPO | at end | 87.5 [84.4, 90.1] | 1.0 [0.4, 2.3] | 0.4 [0.1, 1.4] | 89.1 [86.1, 91.5] |
| Test, Q1–Q3 goals | BC | once | 91.6 [88.9, 93.7] | 1.6 [0.8, 3.1] | 0.8 [0.3, 2.0] | 92.0 [89.3, 94.0] |
| Test, Q1–Q3 goals | BC | at end | 87.5 [84.4, 90.1] | 0.8 [0.3, 2.0] | 0.4 [0.1, 1.4] | 85.9 [82.7, 88.7] |
| Test, Q4 goals (held out) | PPO | once | 86.1 [82.9, 88.9] | 3.9 [2.5, 6.0] | 0.2 [0.0, 1.1] | 94.7 [92.4, 96.4] |
| Test, Q4 goals (held out) | PPO | at end | 67.0 [62.8, 70.9] | 0.0 [0.0, 0.7] | 0.0 [0.0, 0.7] | 92.6 [90.0, 94.5] |
| Test, Q4 goals (held out) | BC | once | 88.5 [85.4, 91.0] | 3.5 [2.2, 5.5] | 0.2 [0.0, 1.1] | 94.9 [92.7, 96.5] |
| Test, Q4 goals (held out) | BC | at end | 63.7 [59.4, 67.7] | 0.4 [0.1, 1.4] | 0.0 [0.0, 0.7] | 93.6 [91.1, 95.4] |

## Table 2: closed-loop success-once / success-at-end (%) for every k, correct-goal demos

| Goals | Demos from | none | k = 1 | k = 2 | k = 4 | k = 8 | wrong goal, k = 1–8 (once) | Teacher |
|---|---|---|---|---|---|---|---|---|
| Test, Q1–Q3 goals | PPO | 1 / 0 | 90 / 82 | 92 / 88 | 91 / 88 | 92 / 89 | 1.4–2.1 | 94 / 89 |
| Test, Q1–Q3 goals | BC | 1 / 0 | 91 / 84 | 92 / 88 | 92 / 88 | 90 / 84 | 1.4–1.8 | 92 / 86 |
| Test, Q4 goals (held out) | PPO | 0 / 0 | 86 / 63 | 89 / 67 | 86 / 67 | 90 / 65 | 3.3–3.9 | 95 / 93 |
| Test, Q4 goals (held out) | BC | 0 / 0 | 86 / 58 | 89 / 66 | 88 / 64 | 89 / 64 | 2.9–5.1 | 95 / 94 |

## Table 3: teacher-forced action MSE of best.pt (correct-goal / wrong-goal demos)

| Split | none | k = 1 | k = 2 | k = 4 | k = 8 |
|---|---|---|---|---|---|
| Validation (Q1–Q3) | 0.0045 | 0.0019 / 0.117 | 0.0017 / 0.128 | 0.0015 / 0.135 | 0.0017 / 0.127 |
| Test, Q1–Q3 goals | 0.0044 | 0.0020 / 0.125 | 0.0017 / 0.137 | 0.0016 / 0.142 | 0.0018 / 0.134 |
| Test, Q4 goals (held out) | 0.0076 | 0.0031 / 0.077 | 0.0029 / 0.082 | 0.0027 / 0.088 | 0.0041 / 0.085 |

## Table 4: transformer hyperparameters

| Setting | Value | Why |
|---|---|---|
| Layers / width / heads | 12 / 256 / 8 | GPT-2 configuration of Garg et al. (2022) |
| Parameters | 9.75M | follows from the above |
| Optimizer | AdamW, lr 0.0001, weight decay 0 | lr and batch size as in Garg et al. (2022) |
| Batch size | 64 | as in Garg et al. (2022) |
| Schedule | linear warmup 1000 steps, cosine to 10% | standard warmup + cosine decay; not tuned |
| Training steps | 100,000 | compute budget (~3.6 h on one L4); validation MSE still improving at the end |
| Context sizes k (uniform) | 0, 1, 2, 4, 8 | covers no context to 8 demos (401 tokens) |
| Snippet stride | every 4th step, 25 pairs per demo | fits k = 8 in the 1,024-token context |
| Training goals / demos per goal | 2,048 / 10 | Q1–Q3 goals; 2 teachers |
| Precision | bf16 autocast, gradient clip 1.0 | speed on an L4 GPU |
| Model selection | lowest mean validation MSE (correct, k = 0) | selected step 100,000 |

Training loss: 0.0775 at step 500, 0.00218 at step 100,000 (200 logged points).
Intervals are 95% Wilson score intervals with n = 512 episodes, treating episodes as independent; they do not
include variation across training seeds (one seed so far).

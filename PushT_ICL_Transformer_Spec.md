# Push-T ICL transformer specification

Updated November 20, 2026.

## 1. Scope and latest decision

Implement one GPT-2-style in-context learner to infer control policies from demonstrations. The first experiment uses numerical observations and single-action prediction in Push-T.

**Use the same learner architecture and the same trained checkpoint to evaluate different teacher policy types.** Train that checkpoint on a balanced mixture of teacher types, then freeze it and evaluate held-out teachers from each type. Candidate teachers are MLP behavioral cloning (BC), PPO, ACT, and Diffusion Policy. The exact teacher roster depends on obtaining competent teachers; architectural uniformity applies to the ICL learner, not to the teachers.

This decision supersedes the older plan to train separate BC-population and PPO-population learners in `PushT_ICL_Codex_Handoff.md` and the proposal. This specification does not itself update those older files.

- Each training prompt contains examples and query targets from **one teacher**.
- Different prompts can come from different teacher types.
- No explicit teacher ID, policy-type label, or reward parameters are supplied to the learner.
- At evaluation, only the context and query observation change. Learner weights stay fixed.
- Repeated training seeds produce independent replications. Within each replication, evaluate the same checkpoint on all teacher types.
- Keep the learner's MSE action head fixed for the primary comparison. Changing the learner head by teacher type would introduce another variable.

The central question is whether demonstrations help a frozen transformer infer an unfamiliar teacher's policy. This is a Garg-style control-policy experiment, not a comparison of pretrained language models or large VLAs.

## 2. Exact backbone

| Component | Specification |
|---|---|
| Model family | Decoder-only, GPT-2-style causal transformer |
| Transformer blocks | 12 |
| Hidden/embedding dimension | 256 |
| Attention heads | 8 |
| Dimension per head | 32 |
| Attention | Full causal multi-head self-attention, including the current token |
| Q, K, V projections | Each maps 256 to 256, with biases; implementation may fuse them |
| Attention score scaling | Divide QK-transpose scores by square root of 32 |
| Attention output projection | 256 to 256, with bias |
| Feedforward network | 256 to 1024 to 256, with biases |
| Feedforward activation | GPT-2 `gelu_new` |
| Normalization | Pre-LayerNorm in each block, plus final LayerNorm |
| LayerNorm epsilon | 1e-5 |
| Positional encoding | Learned absolute position embeddings |
| Maximum sequence length | 1024 tokens |
| Embedding dropout | 0.0 |
| Attention dropout | 0.0 |
| Residual dropout | 0.0 |
| Cross-attention | Disabled |
| KV cache for the initial implementation | Disabled |
| Language tokenizer | None |
| Pretrained language weights | None; random initialization |
| Output | Continuous actions, with a linear regression head |

Each block is:

```text
u = x + CausalSelfAttention(LayerNorm_1(x))
x_next = u + Linear_2(gelu_new(Linear_1(LayerNorm_2(u))))
```

All hidden states remain 256-dimensional. The two LayerNorms have separate learned parameters.

The 12-layer, 256-dimensional, 8-head backbone follows Garg's main model size. The policy input/output projections, 1024-token capacity, and experimental protocol below specify our adaptation; do not describe the entire configuration as an exact reproduction of Garg.

## 3. Environment-dependent dimensions

Define:

- `B`: batch size.
- `d_s`: dimension of the selected numerical observation vector.
- `d_a`: action dimension for the selected controller.
- `H`: predicted action horizon; **H = 1 for the primary experiment**.
- `N`: number of support observation/action pairs.

All teacher types in a comparison must use the same observation definition and action interface. Keep feature ordering, normalization, controller, and control frequency consistent.

For ManiSkill, inspect the actual selected environment and controller spaces. Do not assume Gym-PushT's five observation coordinates and two action coordinates. Use the single-environment spaces when working with vectorized environments, and flatten only the intended numerical fields in a stable documented order.

The recommended robot-arm option is ManiSkill `PushT-v1`. The older saved proposal still describes 2D Gym-PushT. This learner can accommodate either through `d_s` and `d_a`, but one run must use one coherent environment interface.

## 4. Numerical tokenization and sequence layout

Create three trainable projections:

```python
state_encoder  = nn.Linear(d_s, 256, bias=True)
action_encoder = nn.Linear(H * d_a, 256, bias=True)
action_head    = nn.Linear(256, H * d_a, bias=True)
```

One whole observation vector is one token. One whole action vector, or flattened action chunk, is one token. Coordinates are not separate tokens and are not converted into text.

For support pairs `(s_i, A_i)` and query observation `s_q`, feed:

```text
[E_s(s_1), E_a(A_1), ..., E_s(s_N), E_a(A_N), E_s(s_q)]
```

- Sequence length is `L = 2*N + 1`.
- `N = 64` gives 129 tokens.
- With 1024 available positions, at most 511 support pairs fit in this layout.
- Position indices are `0, ..., L-1`.
- Do not add BOS, EOS, separator tokens, or token-type embeddings in this initial version.
- Do not insert teacher labels.
- The correct query action is a target, not an input token.
- GPT2Model adds positional embeddings internally. Do not add them twice.

The saved project evaluates `k = 0, 1, 2, 4, 8` support **snippets**. A snippet can contain several pairs, so `k` is not generally `N`. Choose and record the snippet duration and sampling rate before running; ensure the resulting `N` fits the sequence limit. Concatenate the selected snippets' pairs for this baseline. Query observations must come from separate rollouts of the same teacher.

### Tensor contract

| Tensor | Shape |
|---|---|
| Support states | `[B, N, d_s]` |
| Support actions/chunks | `[B, N, H, d_a]` |
| Flattened support chunks | `[B, N, H*d_a]` |
| Query states | `[B, d_s]` |
| Embedded input sequence | `[B, 2*N+1, 256]` |
| Query targets | `[B, H, d_a]` |
| Predictions | `[B, H, d_a]` |

Initially use the same `N` within each batch, while varying it across batches. This avoids padding complexity. Zero-context batches have `N=0` and contain only the query-state token.

If variable-length padded batches are implemented later, use an attention mask and gather each example's actual query position. Do not read the last padded position as the query. Right padding is the straightforward choice for absolute position embeddings.

## 5. Backbone construction and initialization

Use Hugging Face `GPT2Model`, not `GPT2LMHeadModel`. Pass `inputs_embeds`, not text token IDs. Record and pin the PyTorch and Transformers versions used in the actual run.

```python
import torch
from torch import nn
from transformers import GPT2Config, GPT2Model

config = GPT2Config(
    vocab_size=1,
    n_positions=1024,
    n_embd=256,
    n_layer=12,
    n_head=8,
    n_inner=1024,
    activation_function="gelu_new",
    resid_pdrop=0.0,
    embd_pdrop=0.0,
    attn_pdrop=0.0,
    layer_norm_epsilon=1e-5,
    initializer_range=0.02,
    scale_attn_weights=True,
    scale_attn_by_inverse_layer_idx=False,
    reorder_and_upcast_attn=False,
    add_cross_attention=False,
    use_cache=False,
    bos_token_id=None,
    eos_token_id=None,
)

backbone = GPT2Model(config)
backbone.wte.weight.requires_grad_(False)

state_encoder = nn.Linear(d_s, 256)
action_encoder = nn.Linear(H * d_a, 256)
action_head = nn.Linear(256, H * d_a)

for layer in (state_encoder, action_encoder, action_head):
    nn.init.normal_(layer.weight, mean=0.0, std=0.02)
    nn.init.zeros_(layer.bias)
```

Keep GPT2Model's native initialization for the backbone, including its residual-projection initialization. The explicit normal initialization above applies to our three added linear layers. Set the experiment's random seed before construction.

`vocab_size=1` avoids allocating a large unused language vocabulary. Its one-row word embedding remains unused and frozen because all inputs use `inputs_embeds`. No word-prediction head or softmax is needed.

In the actual implementation, register the backbone and all three projections inside one `nn.Module`, move that module to the device, and optimize all its trainable parameters. Do not optimize only the backbone or accidentally exclude the numerical encoders/head.

The model should have approximately 9.75 million parameters for modest state/action dimensions. Print the actual total and trainable counts; they depend on `d_s`, `d_a`, and `H`.

## 6. Forward pass and loss

The following illustrates the forward computation. Inputs are already normalized and located on the model's device.

```python
# support_states:  [B, N, d_s]
# support_actions: [B, N, H, d_a]
# query_states:    [B, d_s]

state_tokens = state_encoder(support_states)                  # [B,N,256]
action_tokens = action_encoder(support_actions.flatten(2))    # [B,N,256]

paired = torch.stack((state_tokens, action_tokens), dim=2)      # [B,N,2,256]
context_tokens = paired.flatten(1, 2)                         # [B,2*N,256]
query_token = state_encoder(query_states).unsqueeze(1)         # [B,1,256]
tokens = torch.cat((context_tokens, query_token), dim=1)

if tokens.shape[1] > config.n_positions:
    raise ValueError("Context exceeds the configured token budget")

hidden = backbone(inputs_embeds=tokens).last_hidden_state
query_hidden = hidden[:, -1]  # Last position is query; batch has no padding
prediction = action_head(query_hidden).reshape(-1, H, d_a)

loss = torch.nn.functional.mse_loss(prediction, query_targets)
```

The prediction uses the query position after GPT2Model's final LayerNorm. The head predicts all `H*d_a` coordinates in one pass. It is not autoregressive over the coordinates or future actions within the output chunk.

Use **query-only MSE**, averaged over batch, horizon, and action dimensions. This is the baseline chosen for our separate support/query rollout design. Garg's original training averages losses over prompt prefixes; our query-only objective is an explicit adaptation.

Use a linear output without tanh. Compute MSE before clipping. When executing the policy, clip normalized predictions to their valid range and convert them to the environment's action representation.

These snippets specify construction and forward behavior, not a complete trainer. They have not been used to train or benchmark a model in this conversation.

## 7. Normalization and action semantics

- Compute state means and standard deviations from the training split only, using the same statistics for every teacher type. Use `(state - mean) / max(std, 1e-6)` per feature. Freeze these statistics for validation/test.
- Use one common action normalization across all teacher types. If the environment already exposes normalized controller commands, do not normalize them a second time unintentionally.
- For finite per-coordinate bounds `low` and `high`, a documented affine convention is `a_norm = 2*(a-low)/(high-low)-1`. Invert that same transform before stepping the environment.
- Keep ground-truth labels in the same representation as support actions and predictions.
- Freeze preprocessing at evaluation. Do not compute test-teacher-specific normalization statistics.
- Save state-field ordering, bounds, statistics, and controller mode with each checkpoint.

For the initial `H=1` comparison, every label is the next action actually selected by the teacher under its documented rollout settings. ACT/diffusion teachers may internally predict longer sequences; record their prediction and execution horizons. Their deployed behavior can also depend on history, temporal ensembling, or a cached chunk. If that behavior is not determined by the supplied observation, either standardize those settings or explicitly provide the needed history. Do not misinterpret hidden teacher state as an ICL failure.

## 8. One-checkpoint training protocol

For each training example:

1. Sample a teacher type uniformly among the selected types.
2. Sample a training teacher within that type uniformly.
3. Select support demonstrations from that teacher.
4. Select a query observation and target action from a separate rollout of that same teacher.
5. Construct the prompt and minimize query-action MSE.

This is one concrete way to balance the mixture. Prevent larger teacher datasets from dominating simply because they contain more frames. Vary context length during training, including zero context. Keep type/teacher balancing independent of context length.

At evaluation:

- Freeze one checkpoint per training seed and use it across all teacher types.
- Hold out entire teacher policies, not just trajectories from training teachers.
- Compare correct context, no context, and another teacher's context. For the main wrong-context control, prefer a different teacher of the same type, so policy type alone cannot explain the result.
- Use matched query states, rollout starts, support duration, and observation history where applicable.
- Report action error, success, overlap, and the teacher-performance gap, grouped by teacher type and context length.
- For stochastic teachers, record their sampling settings and interpret action MSE alongside success. A single sampled action need not be the only valid output.

The primary question is generalization to held-out teachers within policy types seen during training. Holding out a whole policy type is a separate transfer experiment. Do not silently merge those two evaluation settings.

The existing proposal uses three learner seeds. Each seed is a replication of the mixed-teacher experiment, not a teacher-specific model.

## 9. Controlled extensions and remaining choices

### Action chunking

After `H=1` is working, run `H=8` as a separate controlled experiment. Change both action projection widths to accommodate `H*d_a` and train a new mixed-teacher model. Within that experiment, the same checkpoint must again serve every teacher type. Execute one predicted action before replanning for the initial horizon comparison.

Construct chunks from valid teacher trajectories without crossing episode boundaries. For fixed-length output, sample query/support anchors with at least H valid future actions. Keep the amount of demonstrated information comparable; overlapping chunk labels can expose more future actions than their anchor count suggests.

### Vision plus robot proprioception

This is a later modality experiment. Replace the numerical observation encoder with an image-plus-proprioception encoder while retaining the transformer backbone settings. That encoder's exact architecture has not yet been fixed. Within the visual experiment, use one shared learner/checkpoint across teacher types. Do not claim the numerical checkpoint accepts images without adaptation.

### Learner action heads and backbones

Flow matching, a diffusion learner head, or a Llama-style backbone are separate ablations. Do not select different learner heads/backbones depending on which teacher is being evaluated in the primary comparison.

### Still to choose and log

Optimizer settings, learning-rate schedule, batch size, total updates, snippet duration, teacher counts, split sizes, and checkpoint-selection rule are not fixed by this architecture export. Determine them through a small pilot and save them in the run configuration. Do not present invented settings or unmeasured compute estimates as previously agreed decisions.

At minimum, validate output shapes, zero-context behavior, causal masking, absence of query-label leakage, correct action scaling, and closed-loop behavior before long training. Record package versions, data/teacher splits, random seeds, and model parameter counts. Pin one checkpoint-selection rule using validation data, never test-teacher performance.

## 10. Sources and instruction for the receiving Codex session

- Garg et al.: https://arxiv.org/abs/2208.01066
- Garg code repository: https://github.com/dtsip/in-context-learning
- GPT2Model/GPT2Config documentation: https://huggingface.co/docs/transformers/model_doc/gpt2
- ManiSkill observations: https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html
- ManiSkill Push-T expert/data release: https://huggingface.co/datasets/haosulab/ManiSkill_Demonstrations/tree/main/demos/PushT-v1/rl

Suggested instruction:

> Read this specification before implementing the learner. It supersedes the older separate-learner plan: the primary experiment trains one mixed-teacher transformer and evaluates the same checkpoint across teacher types. Preserve the exact backbone and token layout above. Inspect existing code first, derive state/action dimensions from the selected environment/controller, and clearly record any remaining implementation choices.

"""CPU checks required by PushT_ICL_Transformer_Spec.md section 9 (shapes, zero context, causal masking,
no query-label leakage, action scaling) plus prompt construction. Run: python -m pytest state_icl/tests -q
Data-based tests need a tiny collected split at $ICL_TEST_DATA (see README); they are skipped otherwise.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pusht_icl as P  # noqa: E402

D_S, D_A = len(P.STATE_COLS), P.ACT_DIM


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return P.ICLPolicy(D_S, D_A).eval()


def test_goal_fields_removed():
    assert D_S == 28 and not set(P.GOAL_COLS) & set(P.STATE_COLS)
    assert P.GOAL_COLS == (21, 22, 23)


def test_param_count_and_frozen_wte(model):
    total, trainable = P.count_params(model)
    assert 9.6e6 < total < 9.9e6
    assert total - trainable == model.backbone.wte.weight.numel() == 256


def test_shapes_and_zero_context(model):
    B, N = 3, 50
    ss, sa, q = torch.randn(B, N, D_S), torch.randn(B, N, D_A), torch.randn(B, D_S)
    assert model(ss, sa, q).shape == (B, 1, D_A)
    out0 = model(ss[:, :0], sa[:, :0], q)
    assert out0.shape == (B, 1, D_A)


def test_token_budget(model):
    with pytest.raises(ValueError):
        model(torch.randn(1, 512, D_S), torch.randn(1, 512, D_A), torch.randn(1, D_S))


def test_causal_masking(model):
    """Hidden state at position i must not depend on tokens after i."""
    torch.manual_seed(1)
    tok = torch.randn(1, 9, 256)
    h1 = model.backbone(inputs_embeds=tok).last_hidden_state
    tok2 = tok.clone(); tok2[:, 5:] += torch.randn(1, 4, 256)
    h2 = model.backbone(inputs_embeds=tok2).last_hidden_state
    assert torch.allclose(h1[:, :5], h2[:, :5], atol=1e-5)
    assert not torch.allclose(h1[:, 5:], h2[:, 5:], atol=1e-3)


def test_context_changes_prediction(model):
    torch.manual_seed(2)
    q = torch.randn(1, D_S)
    a = model(torch.randn(1, 25, D_S), torch.randn(1, 25, D_A), q)
    b = model(torch.randn(1, 25, D_S), torch.randn(1, 25, D_A), q)
    assert not torch.allclose(a, b)


DATA = os.environ.get("ICL_TEST_DATA")
needs_data = pytest.mark.skipif(not DATA, reason="set ICL_TEST_DATA to a tiny collected split folder")


@needs_data
def test_prompts_no_leakage_and_successful_support():
    path = Path(DATA) / "train_ppo.npz"
    stats = P.compute_stats([path])
    bank = P.Bank(path, torch.device("cpu"), stats)
    d = np.load(path)
    # learner states carry no goal: de-normalized states reproduce obs without the goal columns
    raw = bank.states * torch.tensor(stats["std"], dtype=torch.float32) + torch.tensor(stats["mean"], dtype=torch.float32)
    assert torch.allclose(raw, torch.as_tensor(d["obs"][..., list(P.STATE_COLS)]), atol=1e-4)
    assert torch.equal(bank.actions, torch.as_tensor(d["actions"])) and bank.actions.abs().max() <= 1.0
    gen = torch.Generator().manual_seed(0)
    ts = torch.arange(0, 100, 4)
    for k in (0, 1, 2):
        B = 64
        # re-run the sampler's index logic to check its choices
        g_state = gen.get_state()
        ss, sa, qs, qa = P.sample_prompts(bank, B, k, gen, ts)
        assert ss.shape == (B, k * len(ts), P.D_S if hasattr(P, "D_S") else D_S) and qa.shape == (B, D_A)
        gen.set_state(g_state)
        eligible = torch.nonzero((bank.n_good >= k) & (bank.E > k)).squeeze(1)
        g = eligible[torch.randint(len(eligible), (B,), generator=gen)]
        keys = torch.rand(B, bank.E, generator=gen)
        keys = torch.where(bank.success[g], keys, keys + 2.0)
        sup = keys.argsort(1)[:, :k]
        qkeys = torch.rand(B, bank.E, generator=gen)
        if k:
            qkeys = qkeys.scatter(1, sup, 2.0)
        qe = qkeys.argmin(1)
        for i in range(B):
            assert bank.success[g[i], sup[i]].all(), "support episode not successful"
            assert qe[i] not in sup[i], "query episode reused as support"
            assert len(set(sup[i].tolist())) == k
        # labels are the teacher's actions at the query states
        flat_s, flat_a = bank.states.reshape(-1, D_S), bank.actions.reshape(-1, D_A)
        for i in range(B):
            j = (flat_s - qs[i]).abs().sum(1).argmin()
            assert torch.allclose(flat_a[j], qa[i])

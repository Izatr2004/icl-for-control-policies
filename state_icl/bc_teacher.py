"""Load Nathan's goal-conditioned BC teacher (state_bc run 4, distilled from the PPO teacher).

Copied from state_bc/pusht_state_bc.py (BCPolicy, load_policy) so state_icl does not import the whole BC
module. Checkpoint format: {"config": {...}, "state_dict": {...}, "extra": {...}}; the normalization
statistics are buffers in the state dict, so the policy takes raw 31-number observations (goal included).
"""
import numpy as np
import torch
from torch import nn


class BCPolicy(nn.Module):
    """MLP teacher: normalized state -> action in [-1, 1] (identical to state_bc's BCPolicy)."""

    def __init__(self, state_dim, action_dim, state_mean, state_scale, hidden=256, depth=2, tanh_output=True):
        super().__init__()
        layers, width = [], state_dim
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.ReLU()]
            width = hidden
        layers.append(nn.Linear(width, action_dim))
        self.net = nn.Sequential(*layers)
        self.tanh_output = tanh_output
        self.register_buffer("state_mean", torch.as_tensor(state_mean, dtype=torch.float32))
        self.register_buffer("state_scale", torch.as_tensor(state_scale, dtype=torch.float32))

    def forward(self, raw_state):
        out = self.net((raw_state - self.state_mean) / self.state_scale)
        return torch.tanh(out) if self.tanh_output else out

    @torch.no_grad()
    def act(self, raw_state):
        return self(raw_state).clamp(-1.0, 1.0)


def load(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    policy = BCPolicy(cfg["state_dim"], cfg["action_dim"], np.zeros(cfg["state_dim"]), np.ones(cfg["state_dim"]),
                      hidden=cfg["hidden"], depth=cfg["depth"], tanh_output=cfg["tanh_output"])
    policy.load_state_dict(ckpt["state_dict"])
    assert cfg["state_dim"] == 31 and cfg["action_dim"] == 7, "expected the 31-number goal-conditioned state"
    return policy.to(device).eval()

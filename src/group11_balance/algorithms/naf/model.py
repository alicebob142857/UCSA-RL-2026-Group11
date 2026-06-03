"""Neural modules for Normalized Advantage Functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def build_mlp(input_dim: int, hidden_sizes: tuple[int, ...], output_dim: int) -> nn.Sequential:
    """Build a small ReLU MLP, allowing an empty hidden stack."""
    layers: list[nn.Module] = []
    last_dim = input_dim
    for size in hidden_sizes:
        layers.append(nn.Linear(last_dim, size))
        layers.append(nn.ReLU())
        last_dim = size
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class NAFNetwork(nn.Module):
    """Q network with a normalized quadratic advantage.

    Q(s, a) = V(s) - 1/2 * (a - mu(s))^T P(s) (a - mu(s)).
    The action maximizer is therefore mu(s), so continuous-action argmax is
    available without numerical optimization.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        mu_hidden_sizes: tuple[int, ...] = (),
        q_hidden_sizes: tuple[int, ...] = (128, 64),
        min_p: float = 1e-3,
        max_p: float = 100.0,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.mu_hidden_sizes = tuple(int(v) for v in mu_hidden_sizes)
        self.q_hidden_sizes = tuple(int(v) for v in q_hidden_sizes)
        self.min_p = float(min_p)
        self.max_p = float(max_p)

        tril_size = self.action_dim * (self.action_dim + 1) // 2
        self.mu_net = build_mlp(self.obs_dim, self.mu_hidden_sizes, self.action_dim)
        self.value_net = build_mlp(self.obs_dim, self.q_hidden_sizes, 1)
        self.l_net = build_mlp(self.obs_dim, self.q_hidden_sizes, tril_size)

        row, col = torch.tril_indices(self.action_dim, self.action_dim)
        self.register_buffer("tril_row", row)
        self.register_buffer("tril_col", col)
        self.register_buffer("diag_index", torch.arange(self.action_dim))

    def config(self) -> dict[str, Any]:
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "mu_hidden_sizes": list(self.mu_hidden_sizes),
            "q_hidden_sizes": list(self.q_hidden_sizes),
            "min_p": self.min_p,
            "max_p": self.max_p,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "NAFNetwork":
        values = dict(config)
        values["mu_hidden_sizes"] = tuple(values.get("mu_hidden_sizes", ()))
        values["q_hidden_sizes"] = tuple(values.get("q_hidden_sizes", (128, 64)))
        return cls(**values)

    def forward(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        raw_mu = self.mu_net(obs)
        mu = torch.clamp(raw_mu, -1.0, 1.0)
        value = self.value_net(obs)
        p = self._positive_matrix(obs)
        return {"mu": mu, "value": value, "p": p}

    def q_value(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        out = self.forward(obs)
        diff = (action - out["mu"]).unsqueeze(-1)
        p_diff = torch.bmm(out["p"], diff)
        advantage = -0.5 * torch.bmm(diff.transpose(1, 2), p_diff).squeeze(-1)
        return out["value"] + advantage

    def greedy_action(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)["mu"]

    def _positive_matrix(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        raw_l = self.l_net(obs)
        l_matrix = obs.new_zeros(batch_size, self.action_dim, self.action_dim)
        l_matrix[:, self.tril_row, self.tril_col] = raw_l

        diag_raw = l_matrix[:, self.diag_index, self.diag_index]
        diag = F.softplus(diag_raw) + self.min_p
        if self.max_p > 0:
            diag = torch.clamp(diag, max=float(np.sqrt(self.max_p)))
        l_matrix[:, self.diag_index, self.diag_index] = diag
        return torch.bmm(l_matrix, l_matrix.transpose(1, 2))


class NAFPolicy:
    """Lightweight inference wrapper used by evaluation and web demos."""

    def __init__(self, network: NAFNetwork, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.network = network.to(self.device)
        self.network.eval()

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "NAFPolicy":
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=device)
        network = NAFNetwork.from_config(checkpoint["model_config"])
        network.load_state_dict(checkpoint["model_state"])
        return cls(network, device=device)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        del deterministic
        obs_array = np.asarray(obs, dtype=np.float32)
        single = obs_array.ndim == 1
        if single:
            obs_array = obs_array[None, :]
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=self.device)
            action = self.network.greedy_action(obs_tensor).cpu().numpy().astype(np.float32)
        if single:
            return action[0], None
        return action, None

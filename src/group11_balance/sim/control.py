"""Classical-control helpers used as teachers for learning algorithms."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import linalg

from group11_balance.sim.dynamics import default_discrete_model
from group11_balance.sim.env import ENV_TO_MODEL


LQR_Q = np.diag([51.0, 51.0, 33.0, 131.0, 51.0, 51.0, 131.0, 131.0]).astype(np.float64)
LQR_R = 5e-4 * np.eye(2, dtype=np.float64)


@lru_cache(maxsize=1)
def discrete_lqr_gain() -> np.ndarray:
    """Compute the discrete LQR feedback matrix for the nominal simulator."""
    g, h = default_discrete_model()
    p = linalg.solve_discrete_are(g, h, LQR_Q, LQR_R)
    return np.linalg.solve(h.T @ p @ h + LQR_R, h.T @ p @ g)


def lqr_physical_action(state_env_order: np.ndarray, action_limit: float = 200.0) -> np.ndarray:
    """Return clipped 2D physical wheel action from an env-order state."""
    k = discrete_lqr_gain()
    model_state = np.asarray(state_env_order, dtype=np.float64)[ENV_TO_MODEL]
    action = -k @ model_state
    return np.clip(action, -action_limit, action_limit).astype(np.float32)


def lqr_common_normalized_action(state_env_order: np.ndarray, action_limit: float = 200.0) -> np.ndarray:
    """Return the 1D normalized common-mode action used by the PPO environment."""
    physical = lqr_physical_action(state_env_order, action_limit=action_limit)
    common = float(np.mean(physical))
    return np.asarray([np.clip(common / action_limit, -1.0, 1.0)], dtype=np.float32)

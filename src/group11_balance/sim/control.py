"""Classical-control helpers used as teachers for learning algorithms."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import linalg

from group11_balance.sim.dynamics import default_discrete_model
from group11_balance.sim.env import ENV_TO_MODEL
from group11_balance.sim.task import TASK_BALANCE, TASK_VELOCITY, validate_task, validate_target_wheel_velocity


LQR_Q = np.diag([51.0, 51.0, 33.0, 131.0, 51.0, 51.0, 131.0, 131.0]).astype(np.float64)
LQR_R = 5e-4 * np.eye(2, dtype=np.float64)


@lru_cache(maxsize=1)
def discrete_lqr_gain() -> np.ndarray:
    """Compute the discrete LQR feedback matrix for the nominal simulator."""
    g, h = default_discrete_model()
    p = linalg.solve_discrete_are(g, h, LQR_Q, LQR_R)
    return np.linalg.solve(h.T @ p @ h + LQR_R, h.T @ p @ g)


def lqr_error_state(
    state_env_order: np.ndarray,
    *,
    task: str = TASK_BALANCE,
    target_wheel_velocity: float = 0.0,
) -> np.ndarray:
    """Return the model-order LQR error state for the selected task."""
    task = validate_task(task)
    target_wheel_velocity = validate_target_wheel_velocity(target_wheel_velocity)
    model_state = np.asarray(state_env_order, dtype=np.float64)[ENV_TO_MODEL].copy()
    if task == TASK_VELOCITY:
        model_state[0] = 0.0
        model_state[1] = 0.0
        model_state[4] -= target_wheel_velocity
        model_state[5] -= target_wheel_velocity
    return model_state


def lqr_common_affine_policy(
    action_limit: float = 200.0,
    *,
    task: str = TASK_BALANCE,
    target_wheel_velocity: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Return env-order weights and bias for the normalized common-mode LQR law."""
    task = validate_task(task)
    target_wheel_velocity = validate_target_wheel_velocity(target_wheel_velocity)
    k = discrete_lqr_gain()
    weights_model_order = -np.mean(k, axis=0, dtype=np.float64) / float(action_limit)
    bias = 0.0
    if task == TASK_VELOCITY:
        bias = -float((weights_model_order[4] + weights_model_order[5]) * target_wheel_velocity)
        weights_model_order = weights_model_order.copy()
        weights_model_order[0] = 0.0
        weights_model_order[1] = 0.0
    weights_env_order = np.zeros_like(weights_model_order)
    weights_env_order[ENV_TO_MODEL] = weights_model_order
    return weights_env_order.astype(np.float32), bias


def lqr_physical_action(
    state_env_order: np.ndarray,
    action_limit: float = 200.0,
    *,
    task: str = TASK_BALANCE,
    target_wheel_velocity: float = 0.0,
) -> np.ndarray:
    """Return clipped 2D physical wheel action from an env-order state."""
    k = discrete_lqr_gain()
    model_state = lqr_error_state(
        state_env_order,
        task=task,
        target_wheel_velocity=target_wheel_velocity,
    )
    action = -k @ model_state
    return np.clip(action, -action_limit, action_limit).astype(np.float32)


def lqr_common_normalized_action(
    state_env_order: np.ndarray,
    action_limit: float = 200.0,
    *,
    task: str = TASK_BALANCE,
    target_wheel_velocity: float = 0.0,
) -> np.ndarray:
    """Return the 1D normalized common-mode action used by the PPO environment."""
    weights, bias = lqr_common_affine_policy(
        action_limit=action_limit,
        task=task,
        target_wheel_velocity=target_wheel_velocity,
    )
    state = np.asarray(state_env_order, dtype=np.float32)
    common = float(state @ weights + bias)
    return np.asarray([np.clip(common, -1.0, 1.0)], dtype=np.float32)

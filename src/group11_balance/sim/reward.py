"""Reward function for PPO balancing experiments."""

from __future__ import annotations

import numpy as np

from group11_balance.sim.task import TASK_BALANCE, TASK_VELOCITY, validate_task, validate_target_wheel_velocity


def balancing_reward(
    state: np.ndarray,
    action: np.ndarray,
    *,
    action_limit: float,
    failed: bool,
    previous_action: np.ndarray | None = None,
    task: str = TASK_BALANCE,
    target_wheel_velocity: float = 0.0,
) -> float:
    """Return a bounded reward for keeping the robot upright and calm."""
    if failed:
        return -12.0

    task = validate_task(task)
    target_wheel_velocity = validate_target_wheel_velocity(target_wheel_velocity)

    wheel_l, wheel_r, wheel_l_rate, wheel_r_rate, body, body_rate, pole, pole_rate = [
        float(v) for v in state
    ]
    u_l, u_r = [float(v) for v in action]

    upright_bonus = 0.85 * np.exp(-8.0 * body * body) + 0.70 * np.exp(-7.0 * pole * pole)
    still_bonus = 0.18 * np.exp(-(body_rate / 4.0) ** 2) + 0.16 * np.exp(-(pole_rate / 5.0) ** 2)
    alive_bonus = 0.20

    angle_cost = 0.42 * (body / 0.35) ** 2 + 0.38 * (pole / 0.45) ** 2
    rate_cost = 0.12 * (body_rate / 4.0) ** 2 + 0.10 * (pole_rate / 5.0) ** 2
    if task == TASK_VELOCITY:
        wheel_rate = 0.5 * (wheel_l_rate + wheel_r_rate)
        velocity_error = wheel_rate - target_wheel_velocity
        posture_gate = np.exp(-8.0 * body * body - 7.0 * pole * pole)
        speed_bonus = 0.35 * np.exp(-(velocity_error / 2.5) ** 2) * posture_gate
        wheel_cost = 0.30 * (velocity_error / 3.0) ** 2
        drift_cost = 0.0
    else:
        speed_bonus = 0.0
        wheel_cost = 0.015 * ((wheel_l_rate / 20.0) ** 2 + (wheel_r_rate / 20.0) ** 2)
        drift_cost = 0.0015 * ((wheel_l / 25.0) ** 2 + (wheel_r / 25.0) ** 2)
    action_cost = 0.025 * ((u_l / action_limit) ** 2 + (u_r / action_limit) ** 2)
    symmetry_cost = 0.04 * abs(u_l - u_r) / (2.0 * action_limit)

    smooth_cost = 0.0
    if previous_action is not None:
        du = (np.asarray(action, dtype=np.float64) - np.asarray(previous_action, dtype=np.float64)) / action_limit
        smooth_cost = 0.10 * float(np.dot(du, du))

    body_safe = np.deg2rad(15.0)
    pole_safe = np.deg2rad(20.0)
    danger_cost = (
        1.35 * (max(0.0, abs(body) - body_safe) / (0.35 - body_safe)) ** 2
        + 1.55 * (max(0.0, abs(pole) - pole_safe) / (0.45 - pole_safe)) ** 2
    )
    falling_cost = (
        0.38 * max(0.0, body * body_rate) / (0.35 * 4.0)
        + 0.52 * max(0.0, pole * pole_rate) / (0.45 * 5.0)
    )

    # Positive action convention in the model moves the base opposite to the
    # visual forward direction.  Penalize large actions that accelerate the base
    # in a direction that makes the pole fall farther away from upright.
    base_accel = -0.5 * (u_l + u_r)
    wrong_direction = max(0.0, -pole * base_accel) / (0.45 * action_limit)
    direction_cost = 0.22 * min(wrong_direction, 1.0)

    reward = (
        alive_bonus
        + upright_bonus
        + still_bonus
        + speed_bonus
        - angle_cost
        - rate_cost
        - wheel_cost
        - drift_cost
        - action_cost
        - symmetry_cost
        - smooth_cost
        - falling_cost
        - danger_cost
        - direction_cost
    )
    return float(np.clip(reward, -10.0, 1.8))

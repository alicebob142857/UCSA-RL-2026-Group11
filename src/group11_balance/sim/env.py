"""Gymnasium simulator for a two-stage balancing robot."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from group11_balance.sim.dynamics import RobotConstants, default_discrete_model
from group11_balance.sim.reward import balancing_reward
from group11_balance.sim.task import TASK_BALANCE, TASK_VELOCITY, validate_task, validate_target_wheel_velocity


ENV_TO_MODEL = np.array([0, 1, 4, 6, 2, 3, 5, 7], dtype=np.int64)
MODEL_TO_ENV = np.array([0, 1, 4, 5, 2, 6, 3, 7], dtype=np.int64)


class TwoStageBalanceEnv(gym.Env):
    """A compact continuous-control environment for PPO training.

    Public state order:
        theta_l, theta_r, theta_l_dot, theta_r_dot,
        body_angle, body_rate, pole_angle, pole_rate

    Internal model order:
        theta_l, theta_r, body_angle, pole_angle,
        theta_l_dot, theta_r_dot, body_rate, pole_rate
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        init_level: str = "easy",
        action_limit: float = 8000.0,
        constants: RobotConstants = RobotConstants(),
        sensor_noise: float = 0.0,
        task: str = TASK_BALANCE,
        target_wheel_velocity: float = 0.0,
    ):
        super().__init__()
        self.constants = constants
        self.dt = constants.dt
        self.action_limit = float(action_limit)
        self.sensor_noise = float(sensor_noise)
        self.task = validate_task(task)
        self.target_wheel_velocity = validate_target_wheel_velocity(target_wheel_velocity)
        self.g, self.h = default_discrete_model(constants)

        self.observation_space = self._make_observation_space()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        self.max_steps = 1000
        self.body_limit = np.deg2rad(30.0)
        self.pole_limit = np.deg2rad(45.0)
        self.wheel_position_limit = 20.0
        self.wheel_speed_limit = 30.0
        self.level = init_level
        self.state = np.zeros(8, dtype=np.float32)
        self.steps = 0
        self.previous_action = np.zeros(2, dtype=np.float32)

    def set_level(self, level: str) -> None:
        if level not in LEVELS:
            raise ValueError(f"unknown curriculum level {level!r}")
        self.level = level

    def set_task(self, task: str, target_wheel_velocity: float = 0.0) -> None:
        self.task = validate_task(task)
        self.target_wheel_velocity = validate_target_wheel_velocity(target_wheel_velocity)
        self.observation_space = self._make_observation_space()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        if "state" in options:
            state = np.asarray(options["state"], dtype=np.float32)
            if state.shape != (8,):
                raise ValueError("reset option 'state' must have shape (8,)")
        else:
            state = sample_state(self.level, self.np_random)
        self.state = state.astype(np.float32)
        self.steps = 0
        self.previous_action = np.zeros(2, dtype=np.float32)
        return self._observe(), {
            "level": self.level,
            "task": self.task,
            "target_wheel_velocity": self.target_wheel_velocity,
        }

    def step(self, action):
        normalized = float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], -1.0, 1.0))
        physical = np.array(
            [normalized * self.action_limit, normalized * self.action_limit],
            dtype=np.float32,
        )

        model_state = self.state[ENV_TO_MODEL].astype(np.float64)
        next_model_state = self.g @ model_state + self.h @ physical.astype(np.float64)
        self.state = next_model_state[MODEL_TO_ENV].astype(np.float32)
        self.steps += 1

        failed, reason = self._failure_reason(self.state)
        truncated = self.steps >= self.max_steps
        reward = balancing_reward(
            self.state,
            physical,
            action_limit=self.action_limit,
            failed=failed,
            previous_action=self.previous_action,
            task=self.task,
            target_wheel_velocity=self.target_wheel_velocity,
        )
        self.previous_action = physical
        info = {
            "physical_action": physical,
            "failure_reason": reason,
            "level": self.level,
            "task": self.task,
            "target_wheel_velocity": self.target_wheel_velocity,
        }
        return self._observe(), reward, failed, truncated, info

    def _failure_reason(self, state: np.ndarray) -> tuple[bool, str | None]:
        wheel_l, wheel_r, wheel_l_rate, wheel_r_rate, body, body_rate, pole, pole_rate = [
            float(v) for v in state
        ]
        wheel_center = 0.5 * (wheel_l + wheel_r)
        wheel_rate_center = 0.5 * (wheel_l_rate + wheel_r_rate)
        if abs(body) > self.body_limit:
            return True, "body_angle"
        if abs(pole) > self.pole_limit:
            return True, "pole_angle"
        if self.task == TASK_VELOCITY:
            velocity_error = wheel_rate_center - self.target_wheel_velocity
            if abs(velocity_error) > self.wheel_speed_limit:
                return True, "wheel_velocity_error"
        else:
            if abs(wheel_center) > self.wheel_position_limit:
                return True, "wheel_position"
            if abs(wheel_rate_center) > self.wheel_speed_limit:
                return True, "wheel_speed"
        if abs(body) > np.deg2rad(24.0) and body * body_rate > 0.0:
            return True, "body_falling_outward"
        if abs(pole) > np.deg2rad(34.0) and pole * pole_rate > 0.0:
            return True, "pole_falling_outward"
        return False, None

    def _observe(self) -> np.ndarray:
        obs = self.state.copy()
        if self.sensor_noise > 0:
            obs += self.np_random.normal(0.0, self.sensor_noise, size=obs.shape).astype(np.float32)
        return obs

    def _make_observation_space(self) -> spaces.Box:
        wheel_position_high = max(40.0, abs(self.target_wheel_velocity) * 12.0)
        high = np.array(
            [
                wheel_position_high,
                wheel_position_high,
                40.0,
                40.0,
                np.pi / 6,
                20.0,
                np.pi / 4,
                20.0,
            ],
            dtype=np.float32,
        )
        return spaces.Box(-high, high, dtype=np.float32)


LEVELS = {
    "easy": {
        "wheel": 0.006,
        "wheel_rate": 0.006,
        "body": 0.025,
        "body_rate": 0.15,
        "pole": 0.018,
        "pole_rate": 0.12,
    },
    "medium": {
        "wheel": 0.010,
        "wheel_rate": 0.010,
        "body": 0.045,
        "body_rate": 0.40,
        "pole": 0.030,
        "pole_rate": 0.25,
    },
    "hard": {
        "wheel": 0.015,
        "wheel_rate": 0.015,
        "body": 0.060,
        "body_rate": 0.55,
        "pole": 0.045,
        "pole_rate": 0.45,
    },
}


def sample_state(level: str, rng: np.random.Generator) -> np.ndarray:
    cfg = LEVELS[level]
    return np.array(
        [
            rng.uniform(-cfg["wheel"], cfg["wheel"]),
            rng.uniform(-cfg["wheel"], cfg["wheel"]),
            rng.uniform(-cfg["wheel_rate"], cfg["wheel_rate"]),
            rng.uniform(-cfg["wheel_rate"], cfg["wheel_rate"]),
            rng.uniform(-cfg["body"], cfg["body"]),
            rng.uniform(-cfg["body_rate"], cfg["body_rate"]),
            rng.uniform(-cfg["pole"], cfg["pole"]),
            rng.uniform(-cfg["pole_rate"], cfg["pole_rate"]),
        ],
        dtype=np.float32,
    )

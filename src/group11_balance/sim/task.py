"""Task definitions shared by simulation, training, visualization, and export."""

from __future__ import annotations


TASK_BALANCE = "balance"
TASK_VELOCITY = "velocity"
TASKS = (TASK_BALANCE, TASK_VELOCITY)

TARGET_WHEEL_VELOCITY_LIMIT = 20.0


def validate_task(task: str) -> str:
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; choose from {TASKS}")
    return task


def validate_target_wheel_velocity(value: float) -> float:
    target = float(value)
    if abs(target) > TARGET_WHEEL_VELOCITY_LIMIT:
        raise ValueError(
            "target_wheel_velocity must be within "
            f"+/-{TARGET_WHEEL_VELOCITY_LIMIT:g} rad/s"
        )
    return target

"""Serve an interactive browser demo for a trained NAF model."""

from __future__ import annotations

import argparse
from pathlib import Path

from group11_balance.algorithms.naf.model import NAFPolicy
from group11_balance.sim.task import TASKS, TASK_BALANCE, TASK_VELOCITY, validate_target_wheel_velocity
from group11_balance.visualization.policy_web_demo import serve_policy_demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/models/group11_naf.pt")
    parser.add_argument("--level", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--task", choices=TASKS, default=TASK_BALANCE)
    parser.add_argument("--target-wheel-velocity", dest="target_wheel_velocity", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8849)
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(f"model not found: {args.model}")
    if args.task == TASK_VELOCITY:
        validate_target_wheel_velocity(args.target_wheel_velocity)
    model = NAFPolicy.load(args.model)
    serve_policy_demo(
        model=model,
        algorithm_name="NAF",
        level=args.level,
        seed=args.seed,
        host=args.host,
        port=args.port,
        task=args.task,
        target_wheel_velocity=args.target_wheel_velocity,
    )


if __name__ == "__main__":
    main()

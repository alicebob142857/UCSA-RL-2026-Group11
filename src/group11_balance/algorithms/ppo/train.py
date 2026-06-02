"""Train a PPO controller with curriculum learning.

The terminal output is intentionally kept to a tqdm progress bar.  Detailed
training events, curriculum checks, evaluation states, and final metrics are
written to a log file so that long training runs stay readable.
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from tqdm.auto import tqdm

from group11_balance.sim.control import discrete_lqr_gain, lqr_common_normalized_action
from group11_balance.sim.env import ENV_TO_MODEL, LEVELS, TwoStageBalanceEnv


LEVEL_ORDER = ["easy", "medium", "hard"]


@dataclass
class TrainConfig:
    seed: int = 11
    total_steps: int = 300_000
    learning_rate: float = 1e-4
    gamma: float = 0.995
    n_steps: int = 2048
    batch_size: int = 256
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    log_std_init: float = -1.5
    target_kl: float = 0.03
    lr_schedule: str = "linear"
    net_arch: tuple[int, ...] = (128, 64, 32)
    start_level: str = "easy"
    max_level: str = "hard"
    curriculum: bool = True
    promotion_success_rate: float = 0.9
    promotion_patience: int = 2
    promotion_check_freq: int = 20_000
    promotion_eval_episodes: int = 20
    best_success_gate: float = 0.8
    eval_episodes: int = 20
    model_path: str = "outputs/models/group11_ppo.zip"
    eval_csv: str = "outputs/logs/group11_ppo_eval.csv"
    train_log: str = "outputs/logs/group11_ppo_train.log"
    lqr_warm_start: bool = True
    lqr_warm_start_steps: int = 2000
    lqr_warm_start_samples: int = 8192
    lqr_warm_start_batch: int = 512
    lqr_warm_start_lr: float = 1e-3
    lqr_exact_linear_init: bool = True
    lqr_trajectory_fraction: float = 0.65
    lqr_rollout_max_steps: int = 500
    bc_regularization: bool = True
    bc_steps_per_rollout: int = 24
    bc_samples_per_level: int = 512
    bc_batch: int = 256
    bc_lr: float = 3e-4
    bc_log_every_rollouts: int = 10
    device: str = "auto"


def configure_logger(path: str) -> logging.Logger:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("group11_ppo")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def read_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {} if data is None else dict(data)


def merge_config(config: dict[str, Any], args: argparse.Namespace) -> TrainConfig:
    values = TrainConfig().__dict__.copy()
    values.update(config)
    for key in values:
        override = getattr(args, key, None)
        if override is not None:
            values[key] = override
    if isinstance(values["net_arch"], list):
        values["net_arch"] = tuple(int(v) for v in values["net_arch"])
    return TrainConfig(**values)


def make_env(level: str, seed: int | None = None):
    env = TwoStageBalanceEnv(init_level=level)
    if seed is not None:
        env.reset(seed=seed)
    return Monitor(env)


def make_schedule(value: float, schedule: str):
    if schedule == "constant":
        return value
    if schedule == "linear":
        return lambda progress_remaining: progress_remaining * value
    raise ValueError(f"unknown learning-rate schedule: {schedule}")


def level_index(level: str) -> int:
    if level not in LEVEL_ORDER:
        raise ValueError(f"unknown level {level!r}; choose from {LEVEL_ORDER}")
    return LEVEL_ORDER.index(level)


def is_success(final_obs: np.ndarray, terminated: bool, steps: int) -> bool:
    body = float(final_obs[4])
    body_rate = float(final_obs[5])
    pole = float(final_obs[6])
    pole_rate = float(final_obs[7])
    wheel_center = 0.5 * float(final_obs[0] + final_obs[1])
    wheel_rate = 0.5 * float(final_obs[2] + final_obs[3])
    return (
        not terminated
        and steps >= 1000
        and abs(body) < np.deg2rad(20.0)
        and abs(pole) < np.deg2rad(25.0)
        and abs(body_rate) < 2.0
        and abs(pole_rate) < 3.0
        and abs(wheel_center) < 15.0
        and abs(wheel_rate) < 20.0
    )


def state_summary(obs: np.ndarray) -> str:
    names = [
        "theta_l",
        "theta_r",
        "theta_l_dot",
        "theta_r_dot",
        "body",
        "body_rate",
        "pole",
        "pole_rate",
    ]
    return ", ".join(f"{name}={float(value):+.4f}" for name, value in zip(names, obs))


def evaluate_policy(
    model: PPO,
    level: str,
    episodes: int,
    seed: int,
    logger: logging.Logger | None = None,
    tag: str = "eval",
) -> dict[str, float]:
    returns: list[float] = []
    lengths: list[int] = []
    successes = 0
    for ep in range(episodes):
        env = TwoStageBalanceEnv(init_level=level)
        obs, _ = env.reset(seed=seed + ep)
        total = 0.0
        terminated = False
        truncated = False
        step_count = 0
        last_info: dict[str, Any] = {}
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, last_info = env.step(action)
            total += float(reward)
            step_count += 1
        success = is_success(obs, terminated, step_count)
        returns.append(total)
        lengths.append(step_count)
        successes += int(success)
        if logger is not None:
            logger.info(
                "%s episode=%d level=%s return=%.3f length=%d success=%s "
                "terminated=%s truncated=%s reason=%s final_state=[%s]",
                tag,
                ep,
                level,
                total,
                step_count,
                success,
                terminated,
                truncated,
                last_info.get("failure_reason"),
                state_summary(obs),
            )
    return {
        "return_mean": float(np.mean(returns)),
        "length_mean": float(np.mean(lengths)),
        "success_rate": successes / episodes,
    }


def curriculum_levels(start_level: str, max_level: str) -> list[str]:
    start_idx = level_index(start_level)
    max_idx = level_index(max_level)
    if start_idx > max_idx:
        raise ValueError("start_level cannot be harder than max_level")
    return LEVEL_ORDER[start_idx : max_idx + 1]


def sample_teacher_states(
    levels: list[str],
    n_samples: int,
    seed: int,
    *,
    trajectory_fraction: float = 0.0,
    rollout_max_steps: int = 500,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    states = []
    counts = np.full(len(levels), n_samples // len(levels), dtype=np.int64)
    counts[: n_samples % len(levels)] += 1
    for level, count in zip(levels, counts):
        env = TwoStageBalanceEnv(init_level=level)
        trajectory_count = int(round(float(count) * np.clip(trajectory_fraction, 0.0, 1.0)))
        reset_count = int(count) - trajectory_count
        for _ in range(reset_count):
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            states.append(obs)
        while trajectory_count > 0:
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            for _ in range(max(1, rollout_max_steps)):
                states.append(obs.copy())
                trajectory_count -= 1
                if trajectory_count <= 0:
                    break
                action = lqr_common_normalized_action(obs)
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
    return np.asarray(states, dtype=np.float32)


def actor_mean_action(model: PPO, obs_tensor: torch.Tensor) -> torch.Tensor:
    features = model.policy.extract_features(obs_tensor, model.policy.pi_features_extractor)
    latent_pi = model.policy.mlp_extractor.forward_actor(features)
    return model.policy.action_net(latent_pi)


def assign_linear_lqr_actor(model: PPO, action_limit: float = 200.0) -> bool:
    """Exactly initialize a no-hidden-layer PPO actor as the common-mode LQR law."""
    if len(model.policy.mlp_extractor.policy_net) != 0:
        return False
    k = discrete_lqr_gain()
    weights_model_order = -np.mean(k, axis=0, dtype=np.float64) / float(action_limit)
    weights_env_order = np.zeros_like(weights_model_order)
    weights_env_order[ENV_TO_MODEL] = weights_model_order
    layer = model.policy.action_net
    if layer.weight.shape != (1, len(weights_env_order)):
        return False
    with torch.no_grad():
        layer.weight.copy_(torch.as_tensor(weights_env_order[None, :], dtype=layer.weight.dtype, device=layer.weight.device))
        layer.bias.zero_()
    return True


def clone_actor_from_lqr(
    model: PPO,
    levels: list[str],
    *,
    n_samples: int,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    trajectory_fraction: float = 0.0,
    rollout_max_steps: int = 500,
) -> tuple[float, int]:
    if steps <= 0 or n_samples <= 0:
        return 0.0, 0
    states = sample_teacher_states(
        levels,
        n_samples=n_samples,
        seed=seed,
        trajectory_fraction=trajectory_fraction,
        rollout_max_steps=rollout_max_steps,
    )
    targets = np.asarray([lqr_common_normalized_action(state) for state in states], dtype=np.float32)
    device = model.policy.device
    obs_tensor = torch.as_tensor(states, dtype=torch.float32, device=device)
    target_tensor = torch.as_tensor(targets, dtype=torch.float32, device=device)

    params = list(model.policy.mlp_extractor.policy_net.parameters()) + list(model.policy.action_net.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    batch = min(batch_size, len(states))
    last_loss = 0.0
    for _ in range(steps):
        idx = torch.randint(0, len(states), (batch,), device=device)
        pred = actor_mean_action(model, obs_tensor[idx])
        loss = torch.mean((pred - target_tensor[idx]) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    return last_loss, len(states)


def warm_start_actor_from_lqr(model: PPO, cfg: TrainConfig, logger: logging.Logger) -> None:
    """Pre-train PPO actor to imitate the matched LQR teacher."""
    if not cfg.lqr_warm_start or cfg.lqr_warm_start_steps <= 0:
        logger.info("LQR warm start disabled")
        return

    if cfg.lqr_exact_linear_init and assign_linear_lqr_actor(model):
        logger.info("LQR exact linear actor initialization finished")
        return

    teacher_levels = curriculum_levels(cfg.start_level, cfg.max_level) if cfg.curriculum else [cfg.start_level]
    last_loss, samples = clone_actor_from_lqr(
        model,
        teacher_levels,
        n_samples=cfg.lqr_warm_start_samples,
        steps=cfg.lqr_warm_start_steps,
        batch_size=cfg.lqr_warm_start_batch,
        lr=cfg.lqr_warm_start_lr,
        seed=cfg.seed + 700_000,
        trajectory_fraction=cfg.lqr_trajectory_fraction,
        rollout_max_steps=cfg.lqr_rollout_max_steps,
    )
    logger.info(
        "LQR warm start finished levels=%s samples=%d steps=%d batch=%d "
        "trajectory_fraction=%.3f rollout_max_steps=%d final_mse=%.8f",
        ",".join(teacher_levels),
        samples,
        cfg.lqr_warm_start_steps,
        min(cfg.lqr_warm_start_batch, samples),
        cfg.lqr_trajectory_fraction,
        cfg.lqr_rollout_max_steps,
        last_loss,
    )


class TqdmProgressCallback(BaseCallback):
    """Only terminal progress output used by training."""

    def __init__(self, total_steps: int):
        super().__init__()
        self.total_steps = int(total_steps)
        self.bar: tqdm | None = None
        self.last_seen_steps = 0

    def _on_training_start(self) -> None:
        self.bar = tqdm(total=self.total_steps, desc="PPO training", unit="step", dynamic_ncols=True)

    def _on_step(self) -> bool:
        if self.bar is not None:
            delta = self.num_timesteps - self.last_seen_steps
            if delta > 0:
                self.bar.update(delta)
                self.last_seen_steps = self.num_timesteps
        return True

    def _on_training_end(self) -> None:
        if self.bar is not None:
            remaining = max(0, self.total_steps - self.last_seen_steps)
            if remaining:
                self.bar.update(remaining)
            self.bar.close()


class CurriculumCallback(BaseCallback):
    """Promote the simulator reset difficulty when evaluation success is high."""

    def __init__(self, cfg: TrainConfig, logger: logging.Logger):
        super().__init__()
        self.cfg = cfg
        self.file_logger = logger
        self.current_level = cfg.start_level
        self.reached_level = cfg.start_level
        self.max_level_idx = level_index(cfg.max_level)
        self.last_check = 0
        self.success_streak = 0
        model_path = Path(cfg.model_path)
        self.best_model_path = model_path.with_name(f"{model_path.stem}_best.zip")
        self.best_level = cfg.start_level
        self.best_score: tuple[int, int, float, float] = (-1, -1, -1.0, -1.0)
        self.rollout_count = 0

    def _on_training_start(self) -> None:
        self.file_logger.info(
            "curriculum start level=%s max=%s threshold=%.3f patience=%d check_freq=%d "
            "eval_episodes=%d best_success_gate=%.3f bc_regularization=%s",
            self.cfg.start_level,
            self.cfg.max_level,
            self.cfg.promotion_success_rate,
            self.cfg.promotion_patience,
            self.cfg.promotion_check_freq,
            self.cfg.promotion_eval_episodes,
            self.cfg.best_success_gate,
            self.cfg.bc_regularization,
        )
        for level in curriculum_levels(self.cfg.start_level, self.cfg.max_level):
            metrics = evaluate_policy(
                self.model,
                level=level,
                episodes=self.cfg.promotion_eval_episodes,
                seed=self.cfg.seed + 90_000 + 1000 * level_index(level),
                logger=self.file_logger,
                tag=f"warm_start_eval_{level}",
            )
            solved = int(metrics["success_rate"] >= self.cfg.best_success_gate)
            score = (
                solved,
                level_index(level) if solved else -1,
                metrics["success_rate"],
                metrics["length_mean"],
            )
            self.file_logger.info(
                "warm-start candidate level=%s success=%.3f return=%.3f length=%.3f",
                level,
                metrics["success_rate"],
                metrics["return_mean"],
                metrics["length_mean"],
            )
            if score > self.best_score:
                self.best_score = score
                self.best_level = level
                self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(self.best_model_path)
                self.file_logger.info(
                    "saved warm-start checkpoint path=%s level=%s success=%.3f length=%.3f",
                    self.best_model_path,
                    self.best_level,
                    metrics["success_rate"],
                    metrics["length_mean"],
                )

    def _on_rollout_end(self) -> None:
        if not self.cfg.bc_regularization or self.cfg.bc_steps_per_rollout <= 0:
            return
        self.rollout_count += 1
        levels = [self.current_level]
        loss, samples = clone_actor_from_lqr(
            self.model,
            levels,
            n_samples=self.cfg.bc_samples_per_level * len(levels),
            steps=self.cfg.bc_steps_per_rollout,
            batch_size=self.cfg.bc_batch,
            lr=self.cfg.bc_lr,
            seed=self.cfg.seed + 800_000 + self.rollout_count,
            trajectory_fraction=self.cfg.lqr_trajectory_fraction,
            rollout_max_steps=self.cfg.lqr_rollout_max_steps,
        )
        if self.rollout_count % max(1, self.cfg.bc_log_every_rollouts) == 0:
            self.file_logger.info(
                "bc regularization rollout=%d levels=%s samples=%d steps=%d final_mse=%.8f",
                self.rollout_count,
                ",".join(levels),
                samples,
                self.cfg.bc_steps_per_rollout,
                loss,
            )

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_check < self.cfg.promotion_check_freq:
            return True
        self.last_check = self.num_timesteps
        metrics = evaluate_policy(
            self.model,
            level=self.current_level,
            episodes=self.cfg.promotion_eval_episodes,
            seed=self.cfg.seed + 100_000 + self.num_timesteps,
            logger=self.file_logger,
            tag=f"curriculum_check_step_{self.num_timesteps}",
        )
        self.file_logger.info(
            "curriculum check step=%d level=%s success=%.3f return=%.3f length=%.3f",
            self.num_timesteps,
            self.current_level,
            metrics["success_rate"],
            metrics["return_mean"],
            metrics["length_mean"],
        )
        solved = int(metrics["success_rate"] >= self.cfg.best_success_gate)
        score = (
            solved,
            level_index(self.current_level) if solved else -1,
            metrics["success_rate"],
            metrics["length_mean"],
        )
        if score > self.best_score:
            self.best_score = score
            self.best_level = self.current_level
            self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.best_model_path)
            self.file_logger.info(
                "saved best checkpoint path=%s level=%s success=%.3f length=%.3f",
                self.best_model_path,
                self.best_level,
                metrics["success_rate"],
                metrics["length_mean"],
            )
        if metrics["success_rate"] < self.cfg.promotion_success_rate:
            self.success_streak = 0
            return True
        self.success_streak += 1
        self.file_logger.info(
            "curriculum promotion streak=%d/%d level=%s",
            self.success_streak,
            self.cfg.promotion_patience,
            self.current_level,
        )
        if self.success_streak < self.cfg.promotion_patience:
            return True

        next_idx = level_index(self.current_level) + 1
        if next_idx > self.max_level_idx:
            self.file_logger.info("curriculum already at maximum level=%s", self.current_level)
            return True
        self.current_level = LEVEL_ORDER[next_idx]
        self.reached_level = self.current_level
        self.success_streak = 0
        self.training_env.env_method("set_level", self.current_level)
        self.file_logger.info("curriculum promoted to level=%s at step=%d", self.current_level, self.num_timesteps)
        return True


def write_eval_csv(path: str, row: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def train(cfg: TrainConfig) -> None:
    if cfg.start_level not in LEVELS:
        raise ValueError(f"start_level must be one of {sorted(LEVELS)}")
    if cfg.max_level not in LEVELS:
        raise ValueError(f"max_level must be one of {sorted(LEVELS)}")

    logger = configure_logger(cfg.train_log)
    logger.info("training config=%s", cfg)

    env = make_env(cfg.start_level, seed=cfg.seed)
    model = PPO(
        "MlpPolicy",
        env,
        seed=cfg.seed,
        learning_rate=make_schedule(cfg.learning_rate, cfg.lr_schedule),
        policy_kwargs={"net_arch": list(cfg.net_arch), "log_std_init": cfg.log_std_init},
        target_kl=cfg.target_kl,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        verbose=0,
        device=cfg.device,
    )
    warm_start_actor_from_lqr(model, cfg, logger)

    callbacks: list[BaseCallback] = [TqdmProgressCallback(cfg.total_steps)]
    curriculum_callback = None
    if cfg.curriculum:
        curriculum_callback = CurriculumCallback(cfg, logger)
        callbacks.append(curriculum_callback)

    model.learn(total_timesteps=cfg.total_steps, callback=CallbackList(callbacks), progress_bar=False)

    final_level = cfg.start_level
    if curriculum_callback is not None and curriculum_callback.best_model_path.exists():
        model = PPO.load(curriculum_callback.best_model_path, env=env, device=cfg.device)
        final_level = curriculum_callback.best_level
        logger.info(
            "restored best checkpoint path=%s level=%s score=%s",
            curriculum_callback.best_model_path,
            final_level,
            curriculum_callback.best_score,
        )
    elif curriculum_callback is not None:
        final_level = curriculum_callback.reached_level

    model_path = Path(cfg.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    logger.info("saved final model path=%s eval_level=%s", model_path, final_level)

    metrics = evaluate_policy(
        model,
        level=final_level,
        episodes=cfg.eval_episodes,
        seed=cfg.seed + 900_000,
        logger=logger,
        tag="final_eval",
    )
    row = {
        "algorithm": "PPO",
        "total_steps": cfg.total_steps,
        "net_arch": "-".join(str(v) for v in cfg.net_arch),
        "start_level": cfg.start_level,
        "final_eval_level": final_level,
        "curriculum": cfg.curriculum,
        **metrics,
    }
    write_eval_csv(cfg.eval_csv, row)
    logger.info("saved evaluation csv=%s row=%s", cfg.eval_csv, row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total-steps", dest="total_steps", type=int, default=None)
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=None)
    parser.add_argument("--start-level", dest="start_level", choices=LEVEL_ORDER, default=None)
    parser.add_argument("--max-level", dest="max_level", choices=LEVEL_ORDER, default=None)
    parser.add_argument("--curriculum", dest="curriculum", action="store_true", default=None)
    parser.add_argument("--no-curriculum", dest="curriculum", action="store_false")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-path", dest="model_path", default=None)
    parser.add_argument("--eval-csv", dest="eval_csv", default=None)
    parser.add_argument("--train-log", dest="train_log", default=None)
    parser.add_argument("--lqr-warm-start", dest="lqr_warm_start", action="store_true", default=None)
    parser.add_argument("--no-lqr-warm-start", dest="lqr_warm_start", action="store_false")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = merge_config(read_config(args.config), args)
    train(cfg)


if __name__ == "__main__":
    main()

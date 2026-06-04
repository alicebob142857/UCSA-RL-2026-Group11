"""Train a Normalized Advantage Functions controller with curriculum learning."""

from __future__ import annotations

import argparse
import csv
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from group11_balance.algorithms.naf.model import NAFNetwork, NAFPolicy
from group11_balance.sim.control import lqr_common_affine_policy, lqr_common_normalized_action
from group11_balance.sim.env import LEVELS, TwoStageBalanceEnv
from group11_balance.sim.task import TASK_BALANCE, TASK_VELOCITY, TASKS, validate_task, validate_target_wheel_velocity


LEVEL_ORDER = ["easy", "medium", "hard"]


@dataclass
class TrainConfig:
    seed: int = 11
    total_steps: int = 120_000
    learning_rate: float = 3e-4
    gamma: float = 0.995
    tau: float = 0.005
    batch_size: int = 256
    buffer_size: int = 200_000
    learning_starts: int = 2048
    train_freq: int = 1
    gradient_steps: int = 1
    max_grad_norm: float = 10.0
    reward_scale: float = 0.1
    loss: str = "huber"
    mu_net_arch: tuple[int, ...] = ()
    q_net_arch: tuple[int, ...] = (128, 64)
    min_p: float = 1e-3
    max_p: float = 100.0
    exploration_noise: float = 0.002
    exploration_noise_final: float = 0.000375
    exploration_fraction: float = 0.70
    start_level: str = "easy"
    max_level: str = "hard"
    curriculum: bool = True
    promotion_success_rate: float = 0.75
    promotion_patience: int = 1
    promotion_check_freq: int = 20_000
    promotion_eval_episodes: int = 20
    best_success_gate: float = 0.75
    eval_episodes: int = 20
    model_path: str = "outputs/models/group11_naf.pt"
    eval_csv: str = "outputs/logs/group11_naf_eval.csv"
    train_log: str = "outputs/logs/group11_naf_train.log"
    lqr_warm_start: bool = True
    lqr_exact_linear_init: bool = True
    lqr_warm_start_steps: int = 2000
    lqr_warm_start_samples: int = 12_288
    lqr_warm_start_batch: int = 512
    lqr_warm_start_lr: float = 1e-3
    lqr_trajectory_fraction: float = 0.15
    lqr_rollout_max_steps: int = 500
    prefill_steps: int = 10_000
    prefill_policy: str = "lqr"
    prefill_noise: float = 0.0005
    bc_regularization: bool = True
    bc_loss_weight: float = 0.02
    bc_decay_fraction: float = 0.80
    log_interval: int = 5000
    task: str = TASK_BALANCE
    target_wheel_velocity: float = 0.0
    action_limit: float = 8000.0
    device: str = "auto"


class ReplayBuffer:
    """Simple numpy replay buffer for off-policy NAF updates."""

    def __init__(self, obs_dim: int, action_dim: int, capacity: int, seed: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.pos = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos, 0] = float(reward)
        self.next_obs[self.pos] = next_obs
        self.dones[self.pos, 0] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        idx = self.rng.integers(0, self.size, size=int(batch_size))
        return {
            "obs": torch.as_tensor(self.obs[idx], dtype=torch.float32, device=device),
            "actions": torch.as_tensor(self.actions[idx], dtype=torch.float32, device=device),
            "rewards": torch.as_tensor(self.rewards[idx], dtype=torch.float32, device=device),
            "next_obs": torch.as_tensor(self.next_obs[idx], dtype=torch.float32, device=device),
            "dones": torch.as_tensor(self.dones[idx], dtype=torch.float32, device=device),
        }


class NAFAgent:
    def __init__(self, obs_dim: int, action_dim: int, cfg: TrainConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.online = NAFNetwork(
            obs_dim,
            action_dim,
            mu_hidden_sizes=cfg.mu_net_arch,
            q_hidden_sizes=cfg.q_net_arch,
            min_p=cfg.min_p,
            max_p=cfg.max_p,
        ).to(device)
        self.target = NAFNetwork(
            obs_dim,
            action_dim,
            mu_hidden_sizes=cfg.mu_net_arch,
            q_hidden_sizes=cfg.q_net_arch,
            min_p=cfg.min_p,
            max_p=cfg.max_p,
        ).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=cfg.learning_rate)

    def act(self, obs: np.ndarray, *, noise_std: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs[None, :], dtype=torch.float32, device=self.device)
            action = self.online.greedy_action(obs_tensor).cpu().numpy()[0]
        if noise_std > 0.0:
            if rng is None:
                action = action + np.random.normal(0.0, noise_std, size=action.shape)
            else:
                action = action + rng.normal(0.0, noise_std, size=action.shape)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def update(self, batch: dict[str, torch.Tensor], *, bc_weight: float, teacher_actions: torch.Tensor | None) -> dict[str, float]:
        with torch.no_grad():
            target_v = self.target(batch["next_obs"])["value"]
            target_q = self.cfg.reward_scale * batch["rewards"] + self.cfg.gamma * (1.0 - batch["dones"]) * target_v

        current_q = self.online.q_value(batch["obs"], batch["actions"])
        if self.cfg.loss == "mse":
            td_loss = torch.mean((current_q - target_q) ** 2)
        elif self.cfg.loss == "huber":
            td_loss = torch.nn.functional.smooth_l1_loss(current_q, target_q)
        else:
            raise ValueError(f"unknown loss {self.cfg.loss!r}")

        bc_loss = current_q.new_tensor(0.0)
        if teacher_actions is not None and bc_weight > 0.0:
            mu = self.online(batch["obs"])["mu"]
            bc_loss = torch.mean((mu - teacher_actions) ** 2)

        loss = td_loss + float(bc_weight) * bc_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()
        self.soft_update()
        return {
            "loss": float(loss.detach().cpu()),
            "td_loss": float(td_loss.detach().cpu()),
            "bc_loss": float(bc_loss.detach().cpu()),
            "q_mean": float(current_q.detach().mean().cpu()),
            "target_mean": float(target_q.detach().mean().cpu()),
        }

    def soft_update(self) -> None:
        with torch.no_grad():
            for target_param, online_param in zip(self.target.parameters(), self.online.parameters()):
                target_param.mul_(1.0 - self.cfg.tau).add_(online_param, alpha=self.cfg.tau)


def configure_logger(path: str) -> logging.Logger:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("group11_naf")
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
    values = asdict(TrainConfig())
    values.update(config)
    for key in values:
        override = getattr(args, key, None)
        if override is not None:
            values[key] = override
    for key in ("mu_net_arch", "q_net_arch"):
        if isinstance(values[key], list):
            values[key] = tuple(int(v) for v in values[key])
    values["task"] = validate_task(values["task"])
    values["target_wheel_velocity"] = validate_target_wheel_velocity(values["target_wheel_velocity"])
    values["action_limit"] = float(values["action_limit"])
    if values["action_limit"] <= 0.0:
        raise ValueError("action_limit must be positive")
    return TrainConfig(**values)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def level_index(level: str) -> int:
    if level not in LEVEL_ORDER:
        raise ValueError(f"unknown level {level!r}; choose from {LEVEL_ORDER}")
    return LEVEL_ORDER.index(level)


def curriculum_levels(start_level: str, max_level: str) -> list[str]:
    start_idx = level_index(start_level)
    max_idx = level_index(max_level)
    if start_idx > max_idx:
        raise ValueError("start_level cannot be harder than max_level")
    return LEVEL_ORDER[start_idx : max_idx + 1]


def linear_schedule(start: float, end: float, fraction: float, step: int, total_steps: int) -> float:
    duration = max(1.0, float(total_steps) * max(0.0, min(1.0, fraction)))
    progress = min(1.0, float(step) / duration)
    return float(start + progress * (end - start))


def decayed_bc_weight(cfg: TrainConfig, step: int) -> float:
    if not cfg.bc_regularization:
        return 0.0
    duration = max(1.0, float(cfg.total_steps) * max(0.0, min(1.0, cfg.bc_decay_fraction)))
    progress = min(1.0, float(step) / duration)
    return float(cfg.bc_loss_weight * (1.0 - progress))


def lqr_batch_actions(
    obs: np.ndarray | torch.Tensor,
    device: torch.device | None = None,
    *,
    task: str = TASK_BALANCE,
    target_wheel_velocity: float = 0.0,
    action_limit: float = 8000.0,
) -> torch.Tensor:
    weights, bias = lqr_common_affine_policy(
        action_limit=action_limit,
        task=task,
        target_wheel_velocity=target_wheel_velocity,
    )
    if isinstance(obs, torch.Tensor):
        weight_tensor = torch.as_tensor(weights, dtype=obs.dtype, device=obs.device)
        bias_tensor = torch.as_tensor(bias, dtype=obs.dtype, device=obs.device)
        return torch.clamp(obs @ weight_tensor + bias_tensor, -1.0, 1.0).unsqueeze(1)
    obs_np = np.asarray(obs, dtype=np.float32)
    actions = np.clip(obs_np @ weights + bias, -1.0, 1.0).astype(np.float32)
    return torch.as_tensor(actions[:, None], dtype=torch.float32, device=device or torch.device("cpu"))


def is_success(
    final_obs: np.ndarray,
    terminated: bool,
    steps: int,
    *,
    task: str = TASK_BALANCE,
    target_wheel_velocity: float = 0.0,
) -> bool:
    task = validate_task(task)
    target_wheel_velocity = validate_target_wheel_velocity(target_wheel_velocity)
    body = float(final_obs[4])
    body_rate = float(final_obs[5])
    pole = float(final_obs[6])
    pole_rate = float(final_obs[7])
    wheel_center = 0.5 * float(final_obs[0] + final_obs[1])
    wheel_rate = 0.5 * float(final_obs[2] + final_obs[3])
    posture_ok = (
        not terminated
        and steps >= 1000
        and abs(body) < np.deg2rad(20.0)
        and abs(pole) < np.deg2rad(25.0)
        and abs(body_rate) < 2.0
        and abs(pole_rate) < 3.0
    )
    if task == TASK_VELOCITY:
        return posture_ok and abs(wheel_rate - target_wheel_velocity) < 2.5
    return posture_ok and abs(wheel_center) < 15.0 and abs(wheel_rate) < 20.0


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
    policy: NAFAgent | NAFPolicy,
    level: str,
    episodes: int,
    seed: int,
    logger: logging.Logger | None = None,
    tag: str = "eval",
    task: str = TASK_BALANCE,
    target_wheel_velocity: float = 0.0,
    action_limit: float = 8000.0,
) -> dict[str, float]:
    returns: list[float] = []
    lengths: list[int] = []
    successes = 0
    for ep in range(episodes):
        env = TwoStageBalanceEnv(
            init_level=level,
            action_limit=action_limit,
            task=task,
            target_wheel_velocity=target_wheel_velocity,
        )
        obs, _ = env.reset(seed=seed + ep)
        total = 0.0
        terminated = False
        truncated = False
        step_count = 0
        last_info: dict[str, Any] = {}
        while not (terminated or truncated):
            if isinstance(policy, NAFAgent):
                action = policy.act(obs, noise_std=0.0)
            else:
                action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, last_info = env.step(action)
            total += float(reward)
            step_count += 1
        success = is_success(
            obs,
            terminated,
            step_count,
            task=task,
            target_wheel_velocity=target_wheel_velocity,
        )
        returns.append(total)
        lengths.append(step_count)
        successes += int(success)
        if logger is not None:
            logger.info(
                "%s episode=%d level=%s task=%s target_wheel_velocity=%.3f return=%.3f length=%d success=%s "
                "terminated=%s truncated=%s reason=%s final_state=[%s]",
                tag,
                ep,
                level,
                task,
                target_wheel_velocity,
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


def sample_teacher_states(
    levels: list[str],
    n_samples: int,
    seed: int,
    *,
    trajectory_fraction: float = 0.0,
    rollout_max_steps: int = 500,
    task: str = TASK_BALANCE,
    target_wheel_velocity: float = 0.0,
    action_limit: float = 8000.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    states = []
    counts = np.full(len(levels), n_samples // len(levels), dtype=np.int64)
    counts[: n_samples % len(levels)] += 1
    for level, count in zip(levels, counts):
        env = TwoStageBalanceEnv(
            init_level=level,
            action_limit=action_limit,
            task=task,
            target_wheel_velocity=target_wheel_velocity,
        )
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
                action = lqr_common_normalized_action(
                    obs,
                    action_limit=action_limit,
                    task=task,
                    target_wheel_velocity=target_wheel_velocity,
                )
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
    return np.asarray(states, dtype=np.float32)


def assign_linear_lqr_mu(
    agent: NAFAgent,
    *,
    task: str = TASK_BALANCE,
    target_wheel_velocity: float = 0.0,
    action_limit: float = 8000.0,
) -> bool:
    if len(agent.online.mu_net) != 1:
        return False
    layer = agent.online.mu_net[0]
    if not isinstance(layer, torch.nn.Linear):
        return False
    weights, bias = lqr_common_affine_policy(
        action_limit=action_limit,
        task=task,
        target_wheel_velocity=target_wheel_velocity,
    )
    if layer.weight.shape != (1, len(weights)):
        return False
    with torch.no_grad():
        layer.weight.copy_(torch.as_tensor(weights[None, :], dtype=layer.weight.dtype, device=layer.weight.device))
        layer.bias.fill_(bias)
    agent.target.load_state_dict(agent.online.state_dict())
    return True


def clone_mu_from_lqr(
    agent: NAFAgent,
    levels: list[str],
    *,
    n_samples: int,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    trajectory_fraction: float,
    rollout_max_steps: int,
    task: str = TASK_BALANCE,
    target_wheel_velocity: float = 0.0,
    action_limit: float = 8000.0,
) -> tuple[float, int]:
    if steps <= 0 or n_samples <= 0:
        return 0.0, 0
    states = sample_teacher_states(
        levels,
        n_samples=n_samples,
        seed=seed,
        trajectory_fraction=trajectory_fraction,
        rollout_max_steps=rollout_max_steps,
        task=task,
        target_wheel_velocity=target_wheel_velocity,
        action_limit=action_limit,
    )
    targets = lqr_batch_actions(
        states,
        device=agent.device,
        task=task,
        target_wheel_velocity=target_wheel_velocity,
        action_limit=action_limit,
    )
    obs_tensor = torch.as_tensor(states, dtype=torch.float32, device=agent.device)
    optimizer = torch.optim.Adam(agent.online.mu_net.parameters(), lr=lr)
    batch = min(batch_size, len(states))
    last_loss = 0.0
    for _ in range(steps):
        idx = torch.randint(0, len(states), (batch,), device=agent.device)
        pred = agent.online(obs_tensor[idx])["mu"]
        loss = torch.mean((pred - targets[idx]) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    agent.target.load_state_dict(agent.online.state_dict())
    return last_loss, len(states)


def warm_start_mu_from_lqr(agent: NAFAgent, cfg: TrainConfig, logger: logging.Logger) -> None:
    if not cfg.lqr_warm_start or cfg.lqr_warm_start_steps <= 0:
        logger.info("LQR warm start disabled")
        return
    if cfg.lqr_exact_linear_init and assign_linear_lqr_mu(
        agent,
        task=cfg.task,
        target_wheel_velocity=cfg.target_wheel_velocity,
        action_limit=cfg.action_limit,
    ):
        logger.info("LQR exact linear mu initialization finished")
        return

    teacher_levels = curriculum_levels(cfg.start_level, cfg.max_level) if cfg.curriculum else [cfg.start_level]
    last_loss, samples = clone_mu_from_lqr(
        agent,
        teacher_levels,
        n_samples=cfg.lqr_warm_start_samples,
        steps=cfg.lqr_warm_start_steps,
        batch_size=cfg.lqr_warm_start_batch,
        lr=cfg.lqr_warm_start_lr,
        seed=cfg.seed + 700_000,
        trajectory_fraction=cfg.lqr_trajectory_fraction,
        rollout_max_steps=cfg.lqr_rollout_max_steps,
        task=cfg.task,
        target_wheel_velocity=cfg.target_wheel_velocity,
        action_limit=cfg.action_limit,
    )
    logger.info(
        "LQR mu warm start finished levels=%s samples=%d steps=%d batch=%d "
        "trajectory_fraction=%.3f rollout_max_steps=%d final_mse=%.8f",
        ",".join(teacher_levels),
        samples,
        cfg.lqr_warm_start_steps,
        min(cfg.lqr_warm_start_batch, samples),
        cfg.lqr_trajectory_fraction,
        cfg.lqr_rollout_max_steps,
        last_loss,
    )


def prefill_replay(agent: NAFAgent, buffer: ReplayBuffer, cfg: TrainConfig, logger: logging.Logger) -> None:
    if cfg.prefill_steps <= 0:
        return
    rng = np.random.default_rng(cfg.seed + 600_000)
    levels = curriculum_levels(cfg.start_level, cfg.max_level) if cfg.curriculum else [cfg.start_level]
    counts = np.full(len(levels), cfg.prefill_steps // len(levels), dtype=np.int64)
    counts[: cfg.prefill_steps % len(levels)] += 1
    for level, count in zip(levels, counts):
        env = TwoStageBalanceEnv(
            init_level=level,
            action_limit=cfg.action_limit,
            task=cfg.task,
            target_wheel_velocity=cfg.target_wheel_velocity,
        )
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        for _ in range(int(count)):
            if cfg.prefill_policy == "lqr":
                action = lqr_common_normalized_action(
                    obs,
                    action_limit=cfg.action_limit,
                    task=cfg.task,
                    target_wheel_velocity=cfg.target_wheel_velocity,
                )
            elif cfg.prefill_policy == "agent":
                action = agent.act(obs, noise_std=0.0)
            elif cfg.prefill_policy == "random":
                action = rng.uniform(-1.0, 1.0, size=(1,)).astype(np.float32)
            else:
                raise ValueError(f"unknown prefill_policy {cfg.prefill_policy!r}")
            if cfg.prefill_noise > 0:
                action = np.clip(action + rng.normal(0.0, cfg.prefill_noise, size=action.shape), -1.0, 1.0).astype(np.float32)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            buffer.add(obs, action, reward, next_obs, terminated)
            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    logger.info(
        "prefilled replay steps=%d policy=%s noise=%.4f levels=%s buffer_size=%d",
        cfg.prefill_steps,
        cfg.prefill_policy,
        cfg.prefill_noise,
        ",".join(levels),
        buffer.size,
    )


def save_checkpoint(path: str | Path, agent: NAFAgent, cfg: TrainConfig, *, step: int, level: str, metrics: dict[str, float] | None = None) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "algorithm": "NAF",
            "step": int(step),
            "level": level,
            "metrics": metrics or {},
            "config": asdict(cfg),
            "model_config": agent.online.config(),
            "model_state": agent.online.state_dict(),
        },
        out,
    )


def load_policy_for_eval(path: str | Path, device: torch.device) -> NAFPolicy:
    return NAFPolicy.load(path, device=device)


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

    set_global_seed(cfg.seed)
    device = resolve_device(cfg.device)
    logger = configure_logger(cfg.train_log)
    logger.info("training config=%s", cfg)
    logger.info("device=%s", device)

    probe_env = TwoStageBalanceEnv(
        init_level=cfg.start_level,
        action_limit=cfg.action_limit,
        task=cfg.task,
        target_wheel_velocity=cfg.target_wheel_velocity,
    )
    obs_dim = int(probe_env.observation_space.shape[0])
    action_dim = int(probe_env.action_space.shape[0])
    agent = NAFAgent(obs_dim, action_dim, cfg, device)
    buffer = ReplayBuffer(obs_dim, action_dim, cfg.buffer_size, seed=cfg.seed + 500_000)

    warm_start_mu_from_lqr(agent, cfg, logger)
    prefill_replay(agent, buffer, cfg, logger)

    model_path = Path(cfg.model_path)
    best_model_path = model_path.with_name(f"{model_path.stem}_best{model_path.suffix}")
    best_level = cfg.start_level
    best_score: tuple[int, int, float, float] = (-1, -1, -1.0, -1.0)

    def maybe_save_best(level: str, step: int, metrics: dict[str, float], tag: str) -> None:
        nonlocal best_level, best_score
        solved = int(metrics["success_rate"] >= cfg.best_success_gate)
        score = (
            solved,
            level_index(level) if solved else -1,
            metrics["success_rate"],
            metrics["length_mean"],
        )
        if score > best_score:
            best_score = score
            best_level = level
            save_checkpoint(best_model_path, agent, cfg, step=step, level=level, metrics=metrics)
            logger.info(
                "saved %s checkpoint path=%s level=%s success=%.3f length=%.3f score=%s",
                tag,
                best_model_path,
                best_level,
                metrics["success_rate"],
                metrics["length_mean"],
                best_score,
            )

    eval_levels = curriculum_levels(cfg.start_level, cfg.max_level) if cfg.curriculum else [cfg.start_level]
    for level in eval_levels:
        metrics = evaluate_policy(
            agent,
            level=level,
            episodes=cfg.promotion_eval_episodes,
            seed=cfg.seed + 90_000 + 1000 * level_index(level),
            logger=logger,
            tag=f"warm_start_eval_{level}",
            task=cfg.task,
            target_wheel_velocity=cfg.target_wheel_velocity,
            action_limit=cfg.action_limit,
        )
        logger.info(
            "warm-start candidate level=%s success=%.3f return=%.3f length=%.3f",
            level,
            metrics["success_rate"],
            metrics["return_mean"],
            metrics["length_mean"],
        )
        maybe_save_best(level, 0, metrics, "warm-start")

    rng = np.random.default_rng(cfg.seed + 1000)
    env = TwoStageBalanceEnv(
        init_level=cfg.start_level,
        action_limit=cfg.action_limit,
        task=cfg.task,
        target_wheel_velocity=cfg.target_wheel_velocity,
    )
    obs, _ = env.reset(seed=cfg.seed)
    current_level = cfg.start_level
    reached_level = cfg.start_level
    max_level_idx = level_index(cfg.max_level)
    last_check = 0
    success_streak = 0
    episode_return = 0.0
    episode_length = 0
    recent_returns: list[float] = []
    recent_lengths: list[int] = []
    last_update: dict[str, float] = {}

    logger.info(
        "curriculum start level=%s max=%s threshold=%.3f patience=%d check_freq=%d "
        "eval_episodes=%d best_success_gate=%.3f bc_regularization=%s",
        cfg.start_level,
        cfg.max_level,
        cfg.promotion_success_rate,
        cfg.promotion_patience,
        cfg.promotion_check_freq,
        cfg.promotion_eval_episodes,
        cfg.best_success_gate,
        cfg.bc_regularization,
    )

    with tqdm(total=cfg.total_steps, desc="NAF training", unit="step", dynamic_ncols=True) as bar:
        for step in range(1, cfg.total_steps + 1):
            noise_std = linear_schedule(
                cfg.exploration_noise,
                cfg.exploration_noise_final,
                cfg.exploration_fraction,
                step,
                cfg.total_steps,
            )
            action = agent.act(obs, noise_std=noise_std, rng=rng)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            buffer.add(obs, action, reward, next_obs, terminated)
            episode_return += float(reward)
            episode_length += 1
            obs = next_obs

            if terminated or truncated:
                recent_returns.append(episode_return)
                recent_lengths.append(episode_length)
                recent_returns = recent_returns[-20:]
                recent_lengths = recent_lengths[-20:]
                obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
                episode_return = 0.0
                episode_length = 0

            if step >= cfg.learning_starts and buffer.size >= cfg.batch_size and step % cfg.train_freq == 0:
                for _ in range(cfg.gradient_steps):
                    batch = buffer.sample(cfg.batch_size, device)
                    bc_weight = decayed_bc_weight(cfg, step)
                    teacher_actions = (
                        lqr_batch_actions(
                            batch["obs"],
                            task=cfg.task,
                            target_wheel_velocity=cfg.target_wheel_velocity,
                            action_limit=cfg.action_limit,
                        )
                        if bc_weight > 0.0
                        else None
                    )
                    last_update = agent.update(batch, bc_weight=bc_weight, teacher_actions=teacher_actions)

            if step % cfg.log_interval == 0:
                logger.info(
                    "train step=%d level=%s noise=%.4f buffer=%d ep_return_mean=%.3f "
                    "ep_length_mean=%.3f loss=%s",
                    step,
                    current_level,
                    noise_std,
                    buffer.size,
                    float(np.mean(recent_returns)) if recent_returns else 0.0,
                    float(np.mean(recent_lengths)) if recent_lengths else 0.0,
                    last_update,
                )

            if cfg.curriculum and step - last_check >= cfg.promotion_check_freq:
                last_check = step
                metrics = evaluate_policy(
                    agent,
                    level=current_level,
                    episodes=cfg.promotion_eval_episodes,
                    seed=cfg.seed + 100_000 + step,
                    logger=logger,
                    tag=f"curriculum_check_step_{step}",
                    task=cfg.task,
                    target_wheel_velocity=cfg.target_wheel_velocity,
                    action_limit=cfg.action_limit,
                )
                logger.info(
                    "curriculum check step=%d level=%s success=%.3f return=%.3f length=%.3f",
                    step,
                    current_level,
                    metrics["success_rate"],
                    metrics["return_mean"],
                    metrics["length_mean"],
                )
                maybe_save_best(current_level, step, metrics, "best")
                if metrics["success_rate"] < cfg.promotion_success_rate:
                    success_streak = 0
                else:
                    success_streak += 1
                    logger.info(
                        "curriculum promotion streak=%d/%d level=%s",
                        success_streak,
                        cfg.promotion_patience,
                        current_level,
                    )
                    if success_streak >= cfg.promotion_patience:
                        next_idx = level_index(current_level) + 1
                        if next_idx > max_level_idx:
                            logger.info("curriculum already at maximum level=%s", current_level)
                        else:
                            current_level = LEVEL_ORDER[next_idx]
                            reached_level = current_level
                            success_streak = 0
                            env.set_level(current_level)
                            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
                            logger.info("curriculum promoted to level=%s at step=%d", current_level, step)
            bar.update(1)

    final_level = reached_level
    if best_model_path.exists():
        policy = load_policy_for_eval(best_model_path, device)
        final_level = best_level
        logger.info("restored best checkpoint path=%s level=%s score=%s", best_model_path, final_level, best_score)
    else:
        save_checkpoint(best_model_path, agent, cfg, step=cfg.total_steps, level=final_level)
        policy = load_policy_for_eval(best_model_path, device)

    if best_model_path.exists():
        # Preserve optimizer-free checkpoint format and make final path mirror PPO's restored-best behavior.
        try:
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(best_model_path, map_location=device)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, model_path)
    logger.info("saved final model path=%s eval_level=%s", model_path, final_level)

    metrics = evaluate_policy(
        policy,
        level=final_level,
        episodes=cfg.eval_episodes,
        seed=cfg.seed + 900_000,
        logger=logger,
        tag="final_eval",
        task=cfg.task,
        target_wheel_velocity=cfg.target_wheel_velocity,
        action_limit=cfg.action_limit,
    )
    row = {
        "algorithm": "NAF",
        "total_steps": cfg.total_steps,
        "mu_net_arch": "-".join(str(v) for v in cfg.mu_net_arch),
        "q_net_arch": "-".join(str(v) for v in cfg.q_net_arch),
        "task": cfg.task,
        "target_wheel_velocity": cfg.target_wheel_velocity,
        "action_limit": cfg.action_limit,
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
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--learning-starts", dest="learning_starts", type=int, default=None)
    parser.add_argument("--start-level", dest="start_level", choices=LEVEL_ORDER, default=None)
    parser.add_argument("--max-level", dest="max_level", choices=LEVEL_ORDER, default=None)
    parser.add_argument("--task", choices=TASKS, default=None)
    parser.add_argument("--target-wheel-velocity", dest="target_wheel_velocity", type=float, default=None)
    parser.add_argument("--action-limit", dest="action_limit", type=float, default=None)
    parser.add_argument("--curriculum", dest="curriculum", action="store_true", default=None)
    parser.add_argument("--no-curriculum", dest="curriculum", action="store_false")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-path", dest="model_path", default=None)
    parser.add_argument("--eval-csv", dest="eval_csv", default=None)
    parser.add_argument("--train-log", dest="train_log", default=None)
    parser.add_argument("--eval-episodes", dest="eval_episodes", type=int, default=None)
    parser.add_argument("--promotion-eval-episodes", dest="promotion_eval_episodes", type=int, default=None)
    parser.add_argument("--lqr-warm-start", dest="lqr_warm_start", action="store_true", default=None)
    parser.add_argument("--no-lqr-warm-start", dest="lqr_warm_start", action="store_false")
    parser.add_argument("--prefill-steps", dest="prefill_steps", type=int, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = merge_config(read_config(args.config), args)
    train(cfg)


if __name__ == "__main__":
    main()

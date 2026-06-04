"""Generate reproducible figures and summary tables for the Group 11 report."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from group11_balance.algorithms.naf.model import NAFPolicy
from group11_balance.sim.control import lqr_common_normalized_action
from group11_balance.sim.env import TwoStageBalanceEnv


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = Path(__file__).resolve().parent
FIG_DIR = REPORT_DIR / "figures"
DATA_DIR = REPORT_DIR / "data"

COLORS = {
    "linear": "#1F77B4",
    "mlp": "#D62728",
    "naf": "#2CA02C",
    "gray": "#7F8C8D",
    "gold": "#D4A017",
}

FIXED_INITIAL_STATE = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        np.deg2rad(3.0),
        0.0,
        np.deg2rad(-2.0),
        0.0,
    ],
    dtype=np.float32,
)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def read_eval_rows() -> list[dict[str, str]]:
    files = [
        "group11_ppo_eval.csv",
        "group11_ppo_mlp_eval.csv",
        "group11_naf_eval.csv",
        "group11_ppo_velocity_eval.csv",
        "group11_ppo_mlp_velocity_eval.csv",
        "group11_naf_velocity_eval.csv",
    ]
    rows: list[dict[str, str]] = []
    for name in files:
        path = ROOT / "outputs" / "logs" / name
        with path.open(newline="", encoding="utf-8") as f:
            row = next(csv.DictReader(f))
        row["source_file"] = name
        if row["algorithm"] == "PPO" and row.get("net_arch"):
            row["label"] = "MLP PPO"
        elif row["algorithm"] == "PPO":
            row["label"] = "Linear PPO"
        else:
            row["label"] = "NAF"
        rows.append(row)
    return rows


def write_summary(rows: list[dict[str, str]]) -> None:
    out = DATA_DIR / "experiment_summary.csv"
    fields = [
        "label",
        "task",
        "total_steps",
        "action_limit",
        "final_eval_level",
        "return_mean",
        "length_mean",
        "success_rate",
        "source_file",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def plot_final_metrics(rows: list[dict[str, str]]) -> None:
    labels = [f"{r['label']}\n{r['task']}" for r in rows]
    success = [float(r["success_rate"]) for r in rows]
    length = [float(r["length_mean"]) for r in rows]
    returns = [float(r["return_mean"]) for r in rows]
    colors = [
        COLORS["mlp"] if r["label"] == "MLP PPO" else COLORS["linear"] if r["label"] == "Linear PPO" else COLORS["naf"]
        for r in rows
    ]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), constrained_layout=True)
    metrics = [
        (success, "Success rate", (0, 1.08), "{:.2f}"),
        (length, "Mean episode length", (0, 1080), "{:.0f}"),
        (returns, "Mean return", (0, 1900), "{:.0f}"),
    ]
    for ax, (values, title, ylim, fmt) in zip(axes, metrics):
        bars = ax.bar(x, values, color=colors, width=0.72)
        ax.set_title(title)
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylim(*ylim)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (ylim[1] * 0.015), fmt.format(value), ha="center", va="bottom", fontsize=8)
    fig.suptitle("Final evaluation across algorithms and tasks", fontsize=13, fontweight="bold")
    fig.savefig(FIG_DIR / "01_final_metrics.png", bbox_inches="tight")
    plt.close(fig)


CHECK_RE = re.compile(
    r"curriculum check step=(?P<step>\d+) level=(?P<level>\w+) "
    r"success=(?P<success>[-\d.]+) return=(?P<return>[-\d.]+) length=(?P<length>[-\d.]+)"
)


def parse_curriculum_log(name: str) -> list[dict[str, float | str]]:
    points: list[dict[str, float | str]] = []
    text = (ROOT / "outputs" / "logs" / name).read_text(encoding="utf-8")
    for match in CHECK_RE.finditer(text):
        points.append(
            {
                "step": float(match.group("step")),
                "level": match.group("level"),
                "success": float(match.group("success")),
                "return": float(match.group("return")),
                "length": float(match.group("length")),
            }
        )
    return points


def write_curriculum_points() -> None:
    series = [
        ("Linear PPO", "balance", "group11_ppo_train.log"),
        ("MLP PPO", "balance", "group11_ppo_mlp_train.log"),
        ("NAF", "balance", "group11_naf_train.log"),
        ("Linear PPO", "velocity", "group11_ppo_velocity_train.log"),
        ("MLP PPO", "velocity", "group11_ppo_mlp_velocity_train.log"),
        ("NAF", "velocity", "group11_naf_velocity_train.log"),
    ]
    out = DATA_DIR / "curriculum_evaluation_points.csv"
    fields = ["label", "task", "source_file", "step", "level", "success", "return", "length"]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for label, task, log_name in series:
            for point in parse_curriculum_log(log_name):
                writer.writerow(
                    {
                        "label": label,
                        "task": task,
                        "source_file": log_name,
                        **point,
                    }
                )


def plot_curriculum_curves() -> None:
    series = [
        ("Linear PPO / balance", "group11_ppo_train.log", COLORS["linear"], "-"),
        ("MLP PPO / balance", "group11_ppo_mlp_train.log", COLORS["mlp"], "-"),
        ("Linear PPO / velocity", "group11_ppo_velocity_train.log", COLORS["linear"], "--"),
        ("MLP PPO / velocity", "group11_ppo_mlp_velocity_train.log", COLORS["mlp"], "--"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.2), sharex=True, constrained_layout=True)
    for label, log_name, color, line_style in series:
        points = parse_curriculum_log(log_name)
        steps = np.asarray([p["step"] for p in points], dtype=float)
        axes[0].plot(steps, [p["success"] for p in points], label=label, color=color, linestyle=line_style, linewidth=2)
        axes[1].plot(steps, [p["return"] for p in points], label=label, color=color, linestyle=line_style, linewidth=2)
    axes[0].set_ylabel("Success rate")
    axes[0].set_ylim(-0.05, 1.08)
    axes[0].axhline(0.8, color=COLORS["gray"], linewidth=1, linestyle=":", label="Promotion threshold")
    axes[0].set_title("Curriculum evaluation during PPO training")
    axes[1].set_ylabel("Mean return")
    axes[1].set_xlabel("Training steps")
    axes[0].legend(ncol=2, fontsize=8)
    fig.savefig(FIG_DIR / "02_ppo_curriculum_curves.png", bbox_inches="tight")
    plt.close(fig)


def plot_all_method_curriculum_curves() -> None:
    series = [
        ("Linear PPO", "group11_ppo_train.log", "group11_ppo_velocity_train.log", COLORS["linear"], "-"),
        ("MLP PPO", "group11_ppo_mlp_train.log", "group11_ppo_mlp_velocity_train.log", COLORS["mlp"], "-"),
        ("NAF", "group11_naf_train.log", "group11_naf_velocity_train.log", COLORS["naf"], "-"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 6.8), sharex="col", constrained_layout=True)
    for col, task in enumerate(["balance", "velocity"]):
        for label, balance_log, velocity_log, color, line_style in series:
            log_name = balance_log if task == "balance" else velocity_log
            points = parse_curriculum_log(log_name)
            steps = np.asarray([p["step"] for p in points], dtype=float)
            axes[0, col].plot(
                steps,
                [p["success"] for p in points],
                label=label,
                color=color,
                linestyle=line_style,
                marker="o",
                markersize=3,
                linewidth=1.8,
            )
            axes[1, col].plot(
                steps,
                [p["return"] for p in points],
                label=label,
                color=color,
                linestyle=line_style,
                marker="o",
                markersize=3,
                linewidth=1.8,
            )
        axes[0, col].axhline(0.8, color=COLORS["gray"], linewidth=1, linestyle=":", label="PPO promotion threshold")
        axes[0, col].set_ylim(-0.05, 1.08)
        axes[0, col].set_title(f"{task.capitalize()} task")
        axes[1, col].set_xlabel("Training steps")
    axes[0, 0].set_ylabel("Curriculum-eval success rate")
    axes[1, 0].set_ylabel("Curriculum-eval mean return")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.01), fontsize=8)
    fig.suptitle("Curriculum evaluation curves for all implemented methods", fontsize=13, fontweight="bold")
    fig.savefig(FIG_DIR / "06_all_method_curriculum_curves.png", bbox_inches="tight")
    plt.close(fig)


def load_rollout_models(task: str) -> list[tuple[str, object, float]]:
    suffix = "_velocity" if task == "velocity" else ""
    ppo_custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
    }
    return [
        (
            "Linear PPO",
            PPO.load(
                ROOT / "outputs" / "models" / f"group11_ppo{suffix}.zip",
                device="cpu",
                custom_objects=ppo_custom_objects,
            ),
            8000.0,
        ),
        (
            "MLP PPO",
            PPO.load(
                ROOT / "outputs" / "models" / f"group11_ppo_mlp{suffix}.zip",
                device="cpu",
                custom_objects=ppo_custom_objects,
            ),
            8000.0,
        ),
        (
            "NAF",
            NAFPolicy.load(ROOT / "outputs" / "models" / f"group11_naf{suffix}.pt", device="cpu"),
            8000.0,
        ),
    ]


def rollout_episode(
    model,
    *,
    task: str,
    action_limit: float,
    target_wheel_velocity: float,
) -> list[dict[str, float | str | bool]]:
    env = TwoStageBalanceEnv(
        init_level="hard",
        action_limit=action_limit,
        task=task,
        target_wheel_velocity=target_wheel_velocity,
    )
    obs, _ = env.reset(options={"state": FIXED_INITIAL_STATE.copy()})
    rows: list[dict[str, float | str | bool]] = []
    terminated = False
    truncated = False
    info: dict = {}
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        normalized = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        rows.append(
            {
                "step": env.steps,
                "time_s": env.steps * env.dt,
                "theta_l": float(obs[0]),
                "theta_r": float(obs[1]),
                "theta_l_dot": float(obs[2]),
                "theta_r_dot": float(obs[3]),
                "body_angle_rad": float(obs[4]),
                "body_rate_rad_s": float(obs[5]),
                "pole_angle_rad": float(obs[6]),
                "pole_rate_rad_s": float(obs[7]),
                "wheel_velocity_rad_s": 0.5 * float(obs[2] + obs[3]),
                "normalized_action": normalized,
                "physical_action_rad_s2": normalized * action_limit,
                "terminated": False,
                "truncated": False,
                "failure_reason": "",
            }
        )
        obs, _, terminated, truncated, info = env.step(action)
    rows.append(
        {
            "step": env.steps,
            "time_s": env.steps * env.dt,
            "theta_l": float(obs[0]),
            "theta_r": float(obs[1]),
            "theta_l_dot": float(obs[2]),
            "theta_r_dot": float(obs[3]),
            "body_angle_rad": float(obs[4]),
            "body_rate_rad_s": float(obs[5]),
            "pole_angle_rad": float(obs[6]),
            "pole_rate_rad_s": float(obs[7]),
            "wheel_velocity_rad_s": 0.5 * float(obs[2] + obs[3]),
            "normalized_action": np.nan,
            "physical_action_rad_s2": np.nan,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "failure_reason": info.get("failure_reason") or "",
        }
    )
    return rows


def generate_episode_response_data() -> dict[str, dict[str, list[dict[str, float | str | bool]]]]:
    results: dict[str, dict[str, list[dict[str, float | str | bool]]]] = {}
    out = DATA_DIR / "episode_time_responses.csv"
    fields = [
        "method",
        "task",
        "target_wheel_velocity",
        "action_limit",
        "step",
        "time_s",
        "theta_l",
        "theta_r",
        "theta_l_dot",
        "theta_r_dot",
        "body_angle_rad",
        "body_rate_rad_s",
        "pole_angle_rad",
        "pole_rate_rad_s",
        "wheel_velocity_rad_s",
        "normalized_action",
        "physical_action_rad_s2",
        "terminated",
        "truncated",
        "failure_reason",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for task, target in [("balance", 0.0), ("velocity", 2.0)]:
            results[task] = {}
            for method, model, action_limit in load_rollout_models(task):
                rows = rollout_episode(
                    model,
                    task=task,
                    action_limit=action_limit,
                    target_wheel_velocity=target,
                )
                results[task][method] = rows
                for row in rows:
                    writer.writerow(
                        {
                            "method": method,
                            "task": task,
                            "target_wheel_velocity": target,
                            "action_limit": action_limit,
                            **row,
                        }
                    )
    return results


def plot_episode_response(
    task: str,
    responses: dict[str, list[dict[str, float | str | bool]]],
) -> None:
    color_for = {"Linear PPO": COLORS["linear"], "MLP PPO": COLORS["mlp"], "NAF": COLORS["naf"]}
    fig, axes = plt.subplots(4, 1, figsize=(9.0, 8.4), sharex=True, constrained_layout=True)
    for method, rows in responses.items():
        time = np.asarray([float(row["time_s"]) for row in rows])
        body_deg = np.rad2deg([float(row["body_angle_rad"]) for row in rows])
        pole_deg = np.rad2deg([float(row["pole_angle_rad"]) for row in rows])
        wheel_velocity = np.asarray([float(row["wheel_velocity_rad_s"]) for row in rows])
        action = np.asarray([float(row["physical_action_rad_s2"]) for row in rows])
        color = color_for[method]
        axes[0].plot(time, body_deg, label=method, color=color, linewidth=1.8)
        axes[1].plot(time, pole_deg, label=method, color=color, linewidth=1.8)
        axes[2].plot(time, wheel_velocity, label=method, color=color, linewidth=1.8)
        axes[3].plot(time, action, label=method, color=color, linewidth=1.3)
    axes[0].axhline(0.0, color=COLORS["gray"], linewidth=1, linestyle=":")
    axes[1].axhline(0.0, color=COLORS["gray"], linewidth=1, linestyle=":")
    axes[2].axhline(2.0 if task == "velocity" else 0.0, color=COLORS["gray"], linewidth=1, linestyle=":")
    axes[3].axhline(0.0, color=COLORS["gray"], linewidth=1, linestyle=":")
    axes[0].set_ylabel("Body angle (deg)")
    axes[1].set_ylabel("Pole angle (deg)")
    axes[2].set_ylabel("Mean wheel speed\n(rad/s)")
    axes[3].set_ylabel("Action\n(rad/s²)")
    axes[3].set_xlabel("Time within one episode (s)")
    axes[0].legend(ncol=3, fontsize=8)
    title = "Balance" if task == "balance" else "Velocity tracking"
    fig.suptitle(
        f"{title} single-episode time response from the same fixed initial disturbance",
        fontsize=13,
        fontweight="bold",
    )
    filename = "07_balance_episode_response.png" if task == "balance" else "08_velocity_episode_response.png"
    fig.savefig(FIG_DIR / filename, bbox_inches="tight")
    plt.close(fig)


FINAL_EVAL_RE = re.compile(
    r"final_eval episode=\d+ .*? success=(?P<success>True|False) "
    r"terminated=(?P<terminated>True|False) truncated=(?P<truncated>True|False) reason=(?P<reason>\S+)"
)


def final_eval_outcomes(log_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    text = (ROOT / "outputs" / "logs" / log_name).read_text(encoding="utf-8")
    for match in FINAL_EVAL_RE.finditer(text):
        if match.group("success") == "True":
            key = "success"
        else:
            reason = match.group("reason")
            key = "other failure" if reason == "None" else reason
        counts[key] = counts.get(key, 0) + 1
    return counts


def plot_failure_reasons() -> None:
    series = [
        ("Linear PPO\nbalance", "group11_ppo_train.log", COLORS["linear"]),
        ("MLP PPO\nbalance", "group11_ppo_mlp_train.log", COLORS["mlp"]),
        ("NAF\nbalance", "group11_naf_train.log", COLORS["naf"]),
        ("Linear PPO\nvelocity", "group11_ppo_velocity_train.log", "#5DADE2"),
        ("MLP PPO\nvelocity", "group11_ppo_mlp_velocity_train.log", "#F1948A"),
        ("NAF\nvelocity", "group11_naf_velocity_train.log", "#58D68D"),
    ]
    outcome_order = [
        "success",
        "wheel_speed",
        "wheel_velocity_error",
        "body_falling_outward",
        "pole_falling_outward",
        "body_angle",
        "pole_angle",
        "other failure",
    ]
    outcome_colors = {
        "success": "#2CA02C",
        "wheel_speed": "#D4A017",
        "wheel_velocity_error": "#E67E22",
        "body_falling_outward": "#D62728",
        "pole_falling_outward": "#9467BD",
        "body_angle": "#8C564B",
        "pole_angle": "#E377C2",
        "other failure": "#7F8C8D",
    }
    x = np.arange(len(series))
    bottoms = np.zeros(len(series))
    fig, ax = plt.subplots(figsize=(10.5, 4.6), constrained_layout=True)
    for outcome in outcome_order:
        values = np.asarray([final_eval_outcomes(log).get(outcome, 0) for _, log, _ in series], dtype=float)
        if np.all(values == 0):
            continue
        ax.bar(x, values, bottom=bottoms, label=outcome.replace("_", " "), color=outcome_colors[outcome], width=0.68)
        bottoms += values
    ax.set_xticks(x, [label for label, _, _ in series])
    ax.set_ylabel("Episodes in final evaluation")
    ax.set_ylim(0, 21)
    ax.set_title("Final-evaluation outcomes (20 episodes per experiment)")
    ax.legend(ncol=2, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.savefig(FIG_DIR / "03_final_eval_outcomes.png", bbox_inches="tight")
    plt.close(fig)


def plot_action_scale() -> None:
    pole_deg = np.linspace(-5.0, 5.0, 401)
    limits = [200.0, 1000.0, 8000.0]
    colors = [COLORS["gray"], COLORS["gold"], COLORS["linear"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), constrained_layout=True)
    for limit, color in zip(limits, colors):
        actions = []
        for deg in pole_deg:
            state = np.zeros(8, dtype=np.float32)
            state[6] = np.deg2rad(deg)
            normalized = float(lqr_common_normalized_action(state, action_limit=limit)[0])
            actions.append(normalized * limit)
        axes[0].plot(pole_deg, actions, label=f"action_limit={limit:g}", color=color, linewidth=2)
    axes[0].set_xlabel("Pole angle (deg)")
    axes[0].set_ylabel("LQR teacher action (rad/s²)")
    axes[0].set_title("Action clipping changes feedback strength")
    axes[0].legend()

    limits_arr = np.asarray(limits)
    delta_speed = limits_arr * 0.01
    bars = axes[1].bar([str(int(x)) for x in limits_arr], delta_speed, color=colors, width=0.65)
    axes[1].set_xlabel("Action limit (rad/s²)")
    axes[1].set_ylabel("Max target-speed change per 10 ms (rad/s)")
    axes[1].set_title("Firmware execution scale")
    for bar, value in zip(bars, delta_speed):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(delta_speed) * 0.02, f"{value:g}", ha="center", va="bottom")
    fig.savefig(FIG_DIR / "04_action_scale_analysis.png", bbox_inches="tight")
    plt.close(fig)


def plot_system_overview() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 3.1))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 3.1)
    ax.axis("off")
    boxes = [
        (0.3, 1.2, 1.8, 0.8, "8-D state\nobservation", "#E8EEF5"),
        (2.5, 1.2, 1.8, 0.8, "PPO / NAF\npolicy", "#DCEFE2"),
        (4.7, 1.2, 1.8, 0.8, "Normalized\naction [-1, 1]", "#F7E8C6"),
        (6.9, 1.2, 1.8, 0.8, "Physical action\nu = a · limit", "#F5DCDC"),
        (9.1, 1.2, 2.0, 0.8, "Discrete model\nx[k+1]=Gx+Hu", "#E8EEF5"),
    ]
    for x, y, w, h, text, fill in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=fill, edgecolor="#5B6573", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)
    for x in [2.1, 4.3, 6.5, 8.7]:
        ax.annotate("", xy=(x + 0.35, 1.6), xytext=(x, 1.6), arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#5B6573"})
    ax.annotate("", xy=(1.2, 1.15), xytext=(10.1, 1.15), arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#5B6573", "connectionstyle": "arc3,rad=-0.28"})
    ax.text(5.65, 0.25, "next state and reward feed the online training loop", ha="center", va="center", color="#5B6573")
    ax.set_title("Shared simulation and online reinforcement-learning loop", fontsize=13, fontweight="bold", pad=10)
    fig.savefig(FIG_DIR / "05_system_overview.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    rows = read_eval_rows()
    write_summary(rows)
    write_curriculum_points()
    episode_responses = generate_episode_response_data()
    plot_final_metrics(rows)
    plot_curriculum_curves()
    plot_all_method_curriculum_curves()
    plot_episode_response("balance", episode_responses["balance"])
    plot_episode_response("velocity", episode_responses["velocity"])
    plot_failure_reasons()
    plot_action_scale()
    plot_system_overview()
    print(f"Generated report assets in {REPORT_DIR}")


if __name__ == "__main__":
    main()

"""Shared browser demo server for trained balance-control policies."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from group11_balance.sim.env import TwoStageBalanceEnv
from group11_balance.sim.task import TASK_BALANCE, TASK_VELOCITY, validate_target_wheel_velocity


STATE_NAMES = [
    "theta_l",
    "theta_r",
    "theta_l_dot",
    "theta_r_dot",
    "body_angle",
    "body_rate",
    "pole_angle",
    "pole_rate",
]

DEMO_MAX_STEPS = 10_000_000


def build_html(algorithm_name: str, initial_level: str) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Group11 {algorithm_name} Balance Demo</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f5f6f8; color: #202124; }}
    main {{ max-width: 1080px; margin: 22px auto; padding: 0 18px; }}
    canvas {{ width: 100%; aspect-ratio: 16 / 9; background: #fff; border: 1px solid #d7dce2; display: block; }}
    .bar {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin: 14px 0; }}
    button, select {{ border: 1px solid #aab2bd; background: white; padding: 8px 12px; cursor: pointer; }}
    label {{ display: inline-flex; gap: 6px; align-items: center; font-size: 14px; }}
    input[type="range"] {{ width: 140px; }}
    pre {{ background: #111827; color: #e5e7eb; padding: 12px; overflow: auto; font-size: 13px; }}
  </style>
</head>
<body>
<main>
  <h2>Group11 {algorithm_name} 二阶平衡车仿真测试</h2>
  <canvas id="scene" width="960" height="540"></canvas>
  <div class="bar">
    <button id="pause">Pause</button>
    <button id="reset">Reset</button>
    <button id="kickSmall">小扰动</button>
    <button id="kickLarge">大扰动</button>
    <label>难度
      <select id="level">
        <option value="easy">easy</option>
        <option value="medium">medium</option>
        <option value="hard">hard</option>
      </select>
    </label>
    <label>steps/frame <input id="speed" type="range" min="1" max="10" value="1"></label>
    <span id="speedText">1</span>
  </div>
  <pre id="stats">loading...</pre>
</main>
<script>
const canvas = document.getElementById("scene");
const ctx = canvas.getContext("2d");
const stats = document.getElementById("stats");
const pause = document.getElementById("pause");
const reset = document.getElementById("reset");
const level = document.getElementById("level");
const speed = document.getElementById("speed");
const speedText = document.getElementById("speedText");
let playing = true;
let current = null;

async function api(path) {{
  const res = await fetch(path);
  if (!res.ok) throw new Error(await res.text());
  current = await res.json();
  if (current.done) {{
    playing = false;
    pause.textContent = "Play";
  }}
  draw();
}}

function drawGround(ground, distanceM, followCamera) {{
  const w = canvas.width;
  const pxPerMeter = 220;
  const tickM = 0.1;
  const majorEvery = 5;
  const cameraLeftM = followCamera ? distanceM - w * 0.5 / pxPerMeter : -w * 0.5 / pxPerMeter;
  const firstTick = Math.floor(cameraLeftM / tickM) - 1;
  const lastTick = Math.ceil((cameraLeftM + w / pxPerMeter) / tickM) + 1;

  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(0, ground, w, canvas.height - ground);
  ctx.strokeStyle = "#e5e7eb";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, ground);
  ctx.lineTo(w, ground);
  ctx.stroke();

  ctx.font = "12px Arial, sans-serif";
  ctx.textBaseline = "top";
  for (let i = firstTick; i <= lastTick; i++) {{
    const worldM = i * tickM;
    const x = (worldM - cameraLeftM) * pxPerMeter;
    const major = i % majorEvery === 0;
    ctx.strokeStyle = major ? "#94a3b8" : "#cbd5e1";
    ctx.lineWidth = major ? 2 : 1;
    ctx.beginPath();
    ctx.moveTo(x, ground);
    ctx.lineTo(x, ground + (major ? 20 : 10));
    ctx.stroke();
    if (major) {{
      ctx.fillStyle = "#64748b";
      ctx.fillText(worldM.toFixed(1) + "m", x + 4, ground + 24);
    }}
  }}
}}

function drawRobot(state) {{
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);

  const thetaL = state[0], thetaR = state[1];
  const body = state[4], pole = state[6];
  const followCamera = current && current.task === "velocity";
  const x = followCamera ? w * 0.5 : w * 0.5 + Math.max(-260, Math.min(260, 15 * 0.5 * (thetaL + thetaR)));
  const ground = h * 0.74;
  drawGround(ground, Number(current.distance_m || 0), followCamera);

  const wheelR = 34;
  const leftX = x - 55, rightX = x + 55;
  const wheelY = ground - wheelR;

  ctx.fillStyle = "#374151";
  [[leftX, thetaL], [rightX, thetaR]].forEach(([cx, theta]) => {{
    ctx.beginPath();
    ctx.arc(cx, wheelY, wheelR, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#111827";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(cx, wheelY);
    ctx.lineTo(cx + Math.sin(theta) * wheelR * 0.72, wheelY - Math.cos(theta) * wheelR * 0.72);
    ctx.stroke();
  }});

  const baseX = x;
  const baseY = wheelY - 18;
  const chassisW = 118;
  const chassisH = 24;

  ctx.save();
  ctx.translate(baseX, baseY + 10);
  ctx.rotate(body);
  ctx.fillStyle = "#dbeafe";
  ctx.strokeStyle = "#2563eb";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.rect(-chassisW / 2, -chassisH / 2, chassisW, chassisH);
  ctx.fill();
  ctx.stroke();
  ctx.restore();

  const poleLen = 275;
  const poleEndX = baseX + Math.sin(pole) * poleLen;
  const poleEndY = baseY - Math.cos(pole) * poleLen;

  ctx.lineWidth = 10;
  ctx.strokeStyle = "#dc2626";
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(baseX, baseY);
  ctx.lineTo(poleEndX, poleEndY);
  ctx.stroke();
  ctx.lineCap = "butt";

  ctx.fillStyle = "#111827";
  ctx.beginPath();
  ctx.arc(baseX, baseY, 8, 0, Math.PI * 2);
  ctx.fill();
}}

function draw() {{
  if (!current) return;
  drawRobot(current.state);
  stats.textContent = JSON.stringify(current, null, 2);
}}

async function loop() {{
  if (playing) {{
    const n = Number(speed.value);
    speedText.textContent = String(n);
    for (let i = 0; i < n; i++) await api("/api/step");
  }}
  requestAnimationFrame(loop);
}}

async function playAndApi(path) {{
  playing = true;
  pause.textContent = "Pause";
  await api(path);
}}

pause.onclick = () => {{
  playing = !playing;
  pause.textContent = playing ? "Pause" : "Play";
}};
reset.onclick = () => playAndApi("/api/reset?level=" + encodeURIComponent(level.value));
level.onchange = () => playAndApi("/api/reset?level=" + encodeURIComponent(level.value));
document.getElementById("kickSmall").onclick = () => playAndApi("/api/disturb?body_angle=0.015&pole_angle=0.015&body_rate=0.08&pole_rate=0.06");
document.getElementById("kickLarge").onclick = () => playAndApi("/api/disturb?body_angle=0.04&pole_angle=0.035&body_rate=0.25&pole_rate=0.18");

level.value = "{initial_level}";
api("/api/reset?level=" + encodeURIComponent(level.value)).then(loop);
</script>
</body>
</html>
"""


class DemoState:
    def __init__(self, model, level: str, seed: int, task: str, target_wheel_velocity: float):
        self.model = model
        self.task = task
        self.target_wheel_velocity = validate_target_wheel_velocity(target_wheel_velocity)
        self.env = make_demo_env(level, task, self.target_wheel_velocity)
        self.seed = seed
        self.obs, _ = self.env.reset(seed=seed)
        self.reward = 0.0
        self.done = False
        self.lock = threading.Lock()

    def reset(self, level: str) -> dict:
        with self.lock:
            self.env = make_demo_env(level, self.task, self.target_wheel_velocity)
            self.obs, _ = self.env.reset(seed=self.seed)
            self.reward = 0.0
            self.done = False
            return self.snapshot()

    def step(self) -> dict:
        with self.lock:
            if self.done:
                return self.snapshot()
            action, _ = self.model.predict(self.obs, deterministic=True)
            self.obs, reward, terminated, truncated, info = self.env.step(action)
            self.reward += float(reward)
            self.done = bool(terminated)
            snap = self.snapshot()
            snap["last_info"] = serializable(info)
            snap["truncated"] = bool(truncated)
            return snap

    def disturb(self, params: dict[str, list[str]]) -> dict:
        with self.lock:
            delta = np.zeros(8, dtype=np.float32)
            delta[4] = query_float(params, "body_angle", "body")
            delta[5] = query_float(params, "body_rate")
            delta[6] = query_float(params, "pole_angle", "pole")
            delta[7] = query_float(params, "pole_rate")
            self.env.state = (self.env.state + delta).astype(np.float32)
            self.obs = self.env._observe()
            self.done = False
            return self.snapshot()

    def snapshot(self) -> dict:
        wheel_position = 0.5 * float(self.obs[0] + self.obs[1])
        wheel_velocity = 0.5 * float(self.obs[2] + self.obs[3])
        distance_m = wheel_position * self.env.constants.wheel_radius_m
        return {
            "state_names": STATE_NAMES,
            "state": [round(float(v), 5) for v in self.obs],
            "steps": self.env.steps,
            "level": self.env.level,
            "task": self.task,
            "wheel_position_rad": round(wheel_position, 5),
            "distance_m": round(distance_m, 5),
            "target_wheel_velocity": round(self.target_wheel_velocity, 5),
            "wheel_velocity": round(wheel_velocity, 5),
            "wheel_velocity_error": round(wheel_velocity - self.target_wheel_velocity, 5),
            "total_reward": round(self.reward, 4),
            "done": self.done,
        }


def serializable(info: dict) -> dict:
    result = {}
    for key, value in info.items():
        if isinstance(value, np.ndarray):
            result[key] = [float(v) for v in value]
        else:
            result[key] = value
    return result


def make_demo_env(level: str, task: str, target_wheel_velocity: float) -> TwoStageBalanceEnv:
    env = TwoStageBalanceEnv(
        init_level=level,
        task=task,
        target_wheel_velocity=target_wheel_velocity,
    )
    env.max_steps = DEMO_MAX_STEPS
    return env


def query_float(params: dict[str, list[str]], *names: str, default: float = 0.0) -> float:
    for name in names:
        if name in params:
            return float(params[name][0])
    return float(default)


def make_handler(state: DemoState, html: str):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self.send_text(html, "text/html; charset=utf-8")
                elif parsed.path == "/api/reset":
                    level = parse_qs(parsed.query).get("level", ["easy"])[0]
                    self.send_json(state.reset(level))
                elif parsed.path == "/api/step":
                    self.send_json(state.step())
                elif parsed.path == "/api/disturb":
                    self.send_json(state.disturb(parse_qs(parsed.query)))
                else:
                    self.send_error(404)
            except Exception as exc:  # noqa: BLE001
                self.send_error(500, str(exc))

        def send_text(self, body: str, content_type: str):
            data = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, payload: dict):
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args):
            return

    return Handler


def serve_policy_demo(
    *,
    model,
    algorithm_name: str,
    level: str,
    seed: int,
    host: str,
    port: int,
    task: str = TASK_BALANCE,
    target_wheel_velocity: float = 0.0,
) -> None:
    state = DemoState(
        model,
        level=level,
        seed=seed,
        task=task,
        target_wheel_velocity=target_wheel_velocity,
    )
    html = build_html(algorithm_name, initial_level=level)
    server = ThreadingHTTPServer((host, port), make_handler(state, html))
    url = f"http://{host}:{port}/"
    print(f"Serving {algorithm_name} demo at {url}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()

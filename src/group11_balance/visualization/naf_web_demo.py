"""Serve an interactive browser demo for a trained NAF model."""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from group11_balance.algorithms.naf.model import NAFPolicy
from group11_balance.sim.env import TwoStageBalanceEnv


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


HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Group11 NAF Balance Demo</title>
  <style>
    body { margin: 0; font-family: Arial, sans-serif; background: #f5f6f8; color: #202124; }
    main { max-width: 1080px; margin: 22px auto; padding: 0 18px; }
    canvas { width: 100%; aspect-ratio: 16 / 9; background: #fff; border: 1px solid #d7dce2; display: block; }
    .bar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin: 14px 0; }
    button, select { border: 1px solid #aab2bd; background: white; padding: 8px 12px; cursor: pointer; }
    label { display: inline-flex; gap: 6px; align-items: center; font-size: 14px; }
    input[type="range"] { width: 140px; }
    pre { background: #111827; color: #e5e7eb; padding: 12px; overflow: auto; font-size: 13px; }
  </style>
</head>
<body>
<main>
  <h2>Group11 NAF 二阶平衡车仿真测试</h2>
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

async function api(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(await res.text());
  current = await res.json();
  draw();
}

function drawRobot(state) {
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#e5e7eb";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, h * 0.74);
  ctx.lineTo(w, h * 0.74);
  ctx.stroke();

  const thetaL = state[0], thetaR = state[1];
  const body = state[4], pole = state[6];
  const x = w * 0.5 + Math.max(-260, Math.min(260, 15 * 0.5 * (thetaL + thetaR)));
  const ground = h * 0.74;
  const wheelR = 34;
  const leftX = x - 55, rightX = x + 55;
  const wheelY = ground - wheelR;

  ctx.fillStyle = "#374151";
  [leftX, rightX].forEach(cx => {
    ctx.beginPath();
    ctx.arc(cx, wheelY, wheelR, 0, Math.PI * 2);
    ctx.fill();
  });

  const baseX = x;
  const baseY = wheelY - 20;
  const bodyLen = 150;
  const poleLen = 190;
  const bodyEndX = baseX + Math.sin(body) * bodyLen;
  const bodyEndY = baseY - Math.cos(body) * bodyLen;
  const poleEndX = bodyEndX + Math.sin(body + pole) * poleLen;
  const poleEndY = bodyEndY - Math.cos(body + pole) * poleLen;

  ctx.lineWidth = 13;
  ctx.strokeStyle = "#2563eb";
  ctx.beginPath();
  ctx.moveTo(baseX, baseY);
  ctx.lineTo(bodyEndX, bodyEndY);
  ctx.stroke();

  ctx.lineWidth = 8;
  ctx.strokeStyle = "#dc2626";
  ctx.beginPath();
  ctx.moveTo(bodyEndX, bodyEndY);
  ctx.lineTo(poleEndX, poleEndY);
  ctx.stroke();

  ctx.fillStyle = "#111827";
  ctx.beginPath();
  ctx.arc(baseX, baseY, 8, 0, Math.PI * 2);
  ctx.fill();
}

function draw() {
  if (!current) return;
  drawRobot(current.state);
  stats.textContent = JSON.stringify(current, null, 2);
}

async function loop() {
  if (playing) {
    const n = Number(speed.value);
    speedText.textContent = String(n);
    for (let i = 0; i < n; i++) await api("/api/step");
  }
  requestAnimationFrame(loop);
}

pause.onclick = () => {
  playing = !playing;
  pause.textContent = playing ? "Pause" : "Play";
};
reset.onclick = () => api("/api/reset?level=" + encodeURIComponent(level.value));
level.onchange = () => api("/api/reset?level=" + encodeURIComponent(level.value));
document.getElementById("kickSmall").onclick = () => api("/api/disturb?body=0.06&pole=0.04&body_rate=0.4&pole_rate=0.25");
document.getElementById("kickLarge").onclick = () => api("/api/disturb?body=0.13&pole=0.08&body_rate=0.9&pole_rate=0.6");

api("/api/reset?level=easy").then(loop);
</script>
</body>
</html>
"""


class DemoState:
    def __init__(self, model_path: str, level: str, seed: int):
        self.model = NAFPolicy.load(model_path)
        self.env = TwoStageBalanceEnv(init_level=level)
        self.seed = seed
        self.obs, _ = self.env.reset(seed=seed)
        self.reward = 0.0
        self.done = False
        self.lock = threading.Lock()

    def reset(self, level: str) -> dict:
        with self.lock:
            self.env = TwoStageBalanceEnv(init_level=level)
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
            self.done = bool(terminated or truncated)
            snap = self.snapshot()
            snap["last_info"] = serializable(info)
            return snap

    def disturb(self, params: dict[str, list[str]]) -> dict:
        with self.lock:
            delta = np.zeros(8, dtype=np.float32)
            delta[4] = float(params.get("body", ["0"])[0])
            delta[5] = float(params.get("body_rate", ["0"])[0])
            delta[6] = float(params.get("pole", ["0"])[0])
            delta[7] = float(params.get("pole_rate", ["0"])[0])
            self.env.state = (self.env.state + delta).astype(np.float32)
            self.obs = self.env._observe()
            self.done = False
            return self.snapshot()

    def snapshot(self) -> dict:
        return {
            "state_names": STATE_NAMES,
            "state": [round(float(v), 5) for v in self.obs],
            "steps": self.env.steps,
            "level": self.env.level,
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


def make_handler(state: DemoState):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self.send_text(HTML, "text/html; charset=utf-8")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/models/group11_naf.pt")
    parser.add_argument("--level", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8849)
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(f"model not found: {args.model}")
    state = DemoState(args.model, level=args.level, seed=args.seed)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    url = f"http://{args.host}:{args.port}/"
    print(f"Serving NAF demo at {url}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()

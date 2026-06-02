# UCSA-RL-2026-Group11

本仓库为国科大 2026 年强化学习课程大作业 Group 11 项目仓库，主题是二阶平衡车强化学习控制。

当前仓库已完成 **共享仿真环境** 与 **PPO 仿真训练模块** 的初始版本。其他模型、真实小车数据、实验对比、报告与展示材料仍待组员继续补全。

## 当前仓库结构

```text
UCSA-RL-2026-Group11/
├── README.md
├── pyproject.toml
├── uv.lock
├── configs/
│   └── ppo_curriculum.yaml
├── src/
│   └── group11_balance/
│       ├── __init__.py
│       ├── check_env.py
│       ├── train_ppo.py                  # PPO 兼容入口
│       ├── sim/                          # 共享仿真环境
│       │   ├── __init__.py
│       │   ├── control.py                # LQR 教师控制器与 baseline
│       │   ├── dynamics.py
│       │   ├── env.py
│       │   └── reward.py
│       ├── algorithms/
│       │   └── ppo/
│       │       ├── __init__.py
│       │       └── train.py
│       └── visualization/
│           ├── __init__.py
│           └── web_demo.py
└── outputs/
    ├── logs/
    └── models/
```

## 模块状态

| 模块 | 状态 | 说明 |
|---|---|---|
| 共享仿真环境 | 已完成初版 | `TwoStageBalanceEnv`，供不同算法复用 |
| 动力学模型 | 已完成初版 | 使用离散线性状态空间模型 |
| Reward 设计 | 已完成初版 | 奖励直立、稳定和平滑控制，惩罚摔倒和过大动作 |
| LQR baseline | 已完成初版 | 用于验证仿真环境，也用于 PPO 训练前 warm start |
| PPO 仿真训练 | 已完成初版 | 使用 Stable-Baselines3 PPO，默认采用线性 actor |
| PPO 课程学习 | 已完成初版 | 支持 `easy -> medium -> hard` 难度提升 |
| PPO 网页可视化 | 已完成初版 | 支持加载模型、浏览器显示仿真、手动加入扰动 |
| NAF 模型 | 待补全 | 可由负责 NAF 的同学加入代码与说明 |
| 其他 RL 模型 | 待补全 | 如 SAC、TD3、DDPG、Q-learning 等 |
| 真实小车数据 | 待补全 | 包括采集方式、数据格式、系统辨识等 |
| 实验结果对比 | 待补全 | 统一整理 reward 曲线、成功率、rollout 效果 |
| 报告 / PPT | 待补全 | 放置课程展示材料与分工说明 |

## 快速开始

安装依赖：

```bash
UV_CACHE_DIR=.uv-cache uv sync
```

检查仿真环境：

```bash
MPLCONFIGDIR=.mpl-cache UV_CACHE_DIR=.uv-cache uv run python -m group11_balance.check_env
```

## 共享仿真环境

仿真环境模块位于：

```text
src/group11_balance/sim/
```

主要文件：

- `dynamics.py`：构建二阶平衡车线性动力学模型；
- `env.py`：定义 Gymnasium 仿真环境；
- `control.py`：定义 LQR 教师控制器，可用于 baseline 和 PPO warm start；
- `reward.py`：定义平衡任务奖励函数。

后续其他算法，例如 NAF、SAC、TD3、DDPG，应优先复用这里的 `TwoStageBalanceEnv`，避免每个算法重复写一套环境。

强化学习训练不是从固定离线数据集中读取样本，而是和仿真环境在线交互：

```text
当前状态 state
    ↓
算法策略输出动作 action
    ↓
仿真环境计算 next_state 和 reward
    ↓
算法根据交互轨迹更新策略
```

## 状态、动作与动力学

状态顺序：

```text
theta_l, theta_r, theta_l_dot, theta_r_dot,
body_angle, body_rate, pole_angle, pole_rate
```

含义：

- `theta_l` / `theta_r`：左右轮角度；
- `theta_l_dot` / `theta_r_dot`：左右轮角速度；
- `body_angle`：车身倾角；
- `body_rate`：车身角速度；
- `pole_angle`：摆杆倾角；
- `pole_rate`：摆杆角速度。

仿真环境使用离散线性状态空间模型：

```text
x[k+1] = G x[k] + H u[k]
```

其中：

- `x[k]` 是当前 8 维状态；
- `u[k]` 是当前控制动作；
- `G` 描述系统自然演化；
- `H` 描述控制动作对系统的影响。

当前 PPO 策略输出 1 维归一化动作：

```text
a ∈ [-1, 1]
```

环境内部映射成左右轮同向控制：

```text
[u_l, u_r] = [200 * a, 200 * a]
```

该设计用于降低训练难度，使策略优先学习前后方向平衡，而不是学习左右轮差速转向。

## PPO 模块

PPO 训练模块位于：

```text
src/group11_balance/algorithms/ppo/train.py
```

同时保留兼容入口：

```text
src/group11_balance/train_ppo.py
```

因此下面两个命令等价：

```bash
uv run python -m group11_balance.train_ppo ...
uv run python -m group11_balance.algorithms.ppo.train ...
```

PPO 为在线强化学习算法，因此我们首先根据动力学参数构造仿真环境，让 PPO 在仿真环境中不断采样状态、动作、奖励，并根据课程学习逐步提高初始状态扰动难度。

当前版本默认采用 **线性 PPO actor + LQR 精确初始化**。

这样设计的原因是：平衡车在小角度附近可以用线性状态空间模型近似，而 LQR 本身就是线性反馈控制律。如果用较深的 MLP 去近似这个线性控制律，训练时容易因为动作探索和函数近似误差把稳定控制器“学偏”；改成线性 actor 后，PPO 仍然是 PPO，只是策略结构更贴合当前仿真动力学。

这不是使用真实小车数据，也不是离线 RL。训练数据仍然来自 PPO 与仿真环境的在线交互；LQR 只用于给 actor 一个稳定的初始控制律。

训练时终端只显示 `tqdm` 进度条。详细信息会写入日志文件：

```text
outputs/logs/group11_ppo_train.log
```

日志中包含：

- 训练配置；
- curriculum 每次评估结果；
- 每个评估 episode 的最终状态；
- 是否成功；
- 失败原因；
- 最终评估指标。

训练过程中会根据课程评估保存 best checkpoint。训练结束后，`outputs/models/group11_ppo.zip` 会被替换为验证表现最好的策略，而不是简单保存最后一步策略。

短训练测试：

```bash
MPLCONFIGDIR=.mpl-cache UV_CACHE_DIR=.uv-cache uv run python -m group11_balance.algorithms.ppo.train \
  --config configs/ppo_curriculum.yaml \
  --total-steps 2000 \
  --no-curriculum \
  --model-path outputs/models/smoke_ppo.zip \
  --eval-csv outputs/logs/smoke_ppo_eval.csv \
  --train-log outputs/logs/smoke_ppo_train.log
```

正式训练：

```bash
MPLCONFIGDIR=.mpl-cache UV_CACHE_DIR=.uv-cache uv run python -m group11_balance.algorithms.ppo.train \
  --config configs/ppo_curriculum.yaml
```

正式训练默认输出：

```text
outputs/models/group11_ppo.zip
outputs/logs/group11_ppo_eval.csv
outputs/logs/group11_ppo_train.log
```

## PPO 课程学习配置

配置文件：

```text
configs/ppo_curriculum.yaml
```

当前训练难度：

```text
easy -> medium -> hard
```

课程学习逻辑：

1. 先在 easy 或配置指定的初始难度上训练；
2. 每隔固定步数评估当前策略；
3. 如果成功率连续达到阈值，则提升到下一难度；
4. 最高提升到配置中的 `max_level`。

当前关键配置：

```yaml
net_arch: []
lqr_warm_start: true
lqr_exact_linear_init: true
lqr_warm_start_steps: 1
curriculum: true
promotion_success_rate: 0.8
promotion_patience: 1
promotion_check_freq: 20000
promotion_eval_episodes: 20
best_success_gate: 0.8
bc_regularization: false
```

当前配置下，PPO 会先保存 warm-start 后在 `easy/medium/hard` 中表现最好的 checkpoint，然后继续按课程学习训练。训练结束时会恢复最高难度且成功率达到 `best_success_gate` 的 checkpoint。

最近一次验证结果：

```text
final_eval_level = hard
success_rate     = 0.85
length_mean      = 853.2
return_mean      = 1530.22
```

对应输出文件：

```text
outputs/models/group11_ppo.zip
outputs/logs/group11_ppo_eval.csv
outputs/logs/group11_ppo_train.log
```

## PPO 网页可视化

训练完成后，可以启动网页仿真器，观察当前 PPO 模型控制效果：

```bash
MPLCONFIGDIR=.mpl-cache UV_CACHE_DIR=.uv-cache uv run python -m group11_balance.visualization.web_demo \
  --model outputs/models/group11_ppo.zip \
  --level easy \
  --port 8848
```

然后在浏览器打开：

```text
http://127.0.0.1:8848/
```

网页支持：

- 播放 / 暂停；
- reset；
- 切换 `easy / medium / hard` 初始难度；
- 添加小扰动；
- 添加大扰动；
- 查看当前状态、步数、累计 reward 和是否失败。

这个工具用于训练后直观测试模型抗干扰能力。若模型刚训练很少步，遇到扰动后摔倒是正常现象。

## 后续待补全文档

以下内容请后续协作者按实际进度补充。

### NAF 模型

待补充：

- 模型原理；
- 代码入口；
- 训练命令；
- 输出文件；
- 实验结果。

### 其他模型

待补充：

- 计划使用的模型；
- 与 PPO / NAF 的区别；
- 训练配置；
- 实验结果。

### 数据采集与真实小车

待补充：

- 小车连接方式；
- WiFi / 串口数据协议；
- 真实数据保存位置；
- 数据格式；
- 是否用于系统辨识。

### 实验对比

待补充：

- LQR baseline；
- PPO 结果；
- NAF 结果；
- 其他模型结果；
- 成功率、平均回报、平均 episode 长度；
- 曲线图与可视化截图。

### 报告与展示

待补充：

- 小组分工；
- 文献综述；
- 方法介绍；
- 实验设置；
- 实验结果；
- 结论与不足。

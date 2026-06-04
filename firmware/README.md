# 固件烧录与硬件准备指南

本指南面向**零嵌入式基础**的同学. 按步骤操作即可完成固件烧录和 WiFi 模块配置.

> 所有需要的文件都已经在本 repo 中, 不需要去别的地方找.

---

## 目录

1. [你需要准备什么](#1-你需要准备什么)
2. [安装 USB 驱动 (CH9102)](#2-安装-usb-驱动-ch9102)
3. [固件模式选择](#3-固件模式选择)
4. [安装 WiFi 模块](#4-安装-wifi-模块)
5. [配置 WiFi 模块](#5-配置-wifi-模块)
6. [验证通信](#6-验证通信)
7. [接入导出的 C 策略](#7-接入导出的-c-策略)
8. [常见问题](#8-常见问题)
9. [文件清单](#9-文件清单)

---

## 1. 你需要准备什么

| 物品 | 说明 |
|------|------|
| WHEELTEC B585 二阶平衡车 | 就是发的那辆小车 |
| DT06 WiFi 模块 | 用来替换小车上的蓝牙模块 (课程应该有发) |
| Type-C 数据线 | 连接小车和电脑, 用于烧录固件 |
| Windows 电脑 | mcuisp 烧录工具只有 Windows 版 |
| 笔记本电脑 (任意系统) | 连接小车 WiFi 跑 Python 代码 |

---

## 2. 安装 USB 驱动 (CH9102)

小车通过 Type-C 口连接电脑时, 电脑需要 CH9102 芯片的驱动才能识别串口.

1. 进入 `firmware/tools/CH_Driver/CH343SER/`
2. 双击运行 `Driver/SETUP.EXE` 安装驱动, 一路点下一步即可
3. 安装完成后, 用 Type-C 线连接小车和电脑
4. 打开 **设备管理器** (Win+X → 设备管理器), 在 "端口 (COM 和 LPT)" 下应该能看到一个新的 COM 口 (例如 COM3)
5. **记住这个 COM 口号**, 烧录时要用

> 如果设备管理器里看不到 COM 口: 换一根数据线试试 (有些线只能充电不能传数据), 或者重启电脑.

---

## 3. 固件模式选择

本项目推荐先采集实车数据, 所以默认需要小车使用板载 LQR 自行平衡:

| 模式 | 用途 | 说明 |
|------|------|------|
| `Control_mode = 0` | 实车数据采集 | 小车自己用 LQR 平衡, PC 只通过 WiFi 接收状态 |
| `Control_mode = 1` | PC 发命令测试 | PC 通过 WiFi 发送 `(u_L, u_R)`, 受 WiFi 延迟影响较大 |

如果小车当前出厂固件能自行平衡并上报 WiFi 状态, **采集数据时可以先不烧录**.

仓库里的 hex 文件统一放在 `firmware/hex/` 目录下. 其中 `firmware/hex/MiniBalance.hex` 是 `Control_mode = 1` 测试固件, 用于 PC 端发命令实验, 不建议作为数据采集默认固件.

如果需要从源码重新编译采集固件, 确认 `firmware/keil/MiniBalance/Inc/control.h` 中:

```c
#define Control_mode  0
```

下面步骤只适用于需要烧录 `firmware/hex/MiniBalance.hex` 做 PC 发命令测试的情况.

### 步骤

1. **打开小车电源开关** (烧录时小车需要通电)
2. 用 **Type-C 数据线**连接小车的 "USB 一键下载" 口和电脑
3. 双击运行 `firmware/tools/mcuisp.exe`
4. 在 mcuisp 界面中:
   - 顶部菜单栏确认 **Port** 选择了正确的 COM 口 (第 2 步记住的那个)
   - 波特率 (bps) 保持默认 `460800` 即可
   - 点击 **"..."按钮** (在 "系统下载的目标文件" 旁边), 选择 `firmware/hex/MiniBalance.hex`
   - 确认勾选了 **"编程前重装文件"**
   - 底部下拉框选择 **"DTR的低电平复位, RTS高电平进BootLoader"**
5. 点击 **"开始编程(P)"**
6. 等待进度条走完, 提示 "成功" 即烧录完成

> 烧录完成后按一下小车上的**复位按键** (在主板上, 标注为 "复位按键" 的那个小按钮).

### mcuisp 截图参考

详细截图见 `firmware/docs/平衡车补充操作步骤.pdf` 第 2 页.

---

## 4. 安装 WiFi 模块

小车出厂装的是**蓝牙模块**, 我们需要换成 **DT06 WiFi 模块**.

1. **关闭小车电源**
2. 找到主板上的 **"蓝牙模块接口"** (在主板边缘, 标注为 "蓝牙模块接口")
3. **拔下蓝牙模块** — 直接垂直向上拔即可, 注意不要硬掰
4. **插上 DT06 WiFi 模块** — 引脚对齐插入同一个接口, WiFi 模块和蓝牙模块外形相似
5. 确认插紧, 引脚没有歪斜

> WiFi 模块和蓝牙模块的接口是一样的, 形状也类似, 不会插反.

---

## 5. 配置 WiFi 模块

WiFi 模块出厂**一般已经配置好了**, 默认参数:

| 参数 | 默认值 |
|------|--------|
| WiFi 名称 | `Minibalance_XXXXXX` (XXXXXX 是设备号) |
| WiFi 密码 | `12345678` |
| 通信协议 | TCP |
| 端口 | `6390` |
| IP 地址 | `192.168.4.1` |

### 如果需要修改配置 (通常不需要)

1. 打开小车电源, 让 WiFi 模块上电
2. 用电脑/手机连接 WiFi: `Minibalance_XXXXXX`, 密码 `12345678`
3. 打开浏览器, 输入 `192.168.4.1`
4. 进入 **MODULE → Networks**
5. 设置:
   - Socket Type: **TCP Server**
   - TCP Server Local Port: **6390**
6. 点击 **SAVE**
7. 给 WiFi 模块重新上电 (关闭再打开小车电源)

---

## 6. 验证通信

完成固件烧录和 WiFi 模块安装后, 建议先做两层验证:

1. 电脑已连接小车 WiFi, 能访问 `192.168.4.1`.
2. 小车上电后串口/WiFi 端能持续收到形如 `{1.23:4.56:...}` 的状态帧.

当前仓库的 Python 代码主要包含仿真训练、可视化和策略导出, 未随仓库提供旧版的 `wifi_test.py`、`wifi_lqr.py` 或 `collect_real.py` 运行脚本. 如果需要做实车通信脚本, 应基于固件发出的文本状态帧和 TCP `192.168.4.1:6390` 接口单独补充.

固件端需要关注两个模式:

- `Control_mode = 0`: 小车板载控制, 适合实车状态采集和上板策略测试.
- `Control_mode = 1`: PC 端发送控制命令, 适合通信链路和延迟测试.

## Keil 工程入口与控制逻辑

Keil 工程文件:

```text
firmware/keil/MDK-ARM/MiniBalance.uvprojx
```

工程入口文件:

```text
firmware/keil/Core/Src/main.c
```

`main.c` 中的 `main()` 是程序入口, 主要负责 HAL 初始化、外设初始化和主循环显示/通信逻辑. 平衡控制的 5ms 闭环不在 `main()` 主循环里实现.

实际控制文件:

```text
firmware/keil/MiniBalance/control.c
```

该文件实现了 HAL 外部中断回调:

```c
void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
```

控制流程在该回调中完成: 读取编码器和姿态角, 根据 `Control_mode` 计算 `u_L/u_R`, 转换为左右轮目标速度, 再通过 PI 控制和 `Set_Pwm()` 输出 PWM. 因此接入 LQR 替换、线性反馈策略或 Q-Learning/SARSA 导出的 tabular 策略时, 主要修改 `firmware/keil/MiniBalance/control.c`, 而不是 `main.c`.

---

## 7. 接入导出的 C 策略

当前仓库使用统一导出脚本 `group11_balance.deploy.export_to_c`. 该脚本只会生成 C 头文件, 不会自动修改已经烧录到小车里的固件. 策略真正上板需要复制头文件、重新编译 Keil 工程并烧录新固件.

当前源码采用插拔式开关. 策略选择在:

```text
firmware/keil/MiniBalance/Inc/control.h
```

当前已加入的固件源码配置:

```c
#define Control_mode       0
#define BOARD_POLICY_MODE  BOARD_POLICY_SB3
```

如需恢复最稳妥的原始 LQR 数据采集模式, 将 `BOARD_POLICY_MODE` 改为 `BOARD_POLICY_LQR` 后重新编译烧录.

含义:

| 宏 | 含义 | 需要额外头文件 |
|----|------|----------------|
| `BOARD_POLICY_LQR` | 原始实车 LQR | 不需要 |
| `BOARD_POLICY_LINEAR` | 调用导出的 `linear_predict()` | `linear_policy.h`，当前仓库未提供对应导出脚本 |
| `BOARD_POLICY_TABULAR` | 调用导出的 `tabular_predict()` | `tabular_policy.h`，当前仓库未提供对应导出脚本 |
| `BOARD_POLICY_SB3` | 调用 Group 11 导出策略的 `sb3_predict()` 兼容入口 | `sb3_policy.h` + `group11_*.h` |

### Group 11 策略上板

`export_to_c.py` 支持当前仓库中的 PPO 与 NAF 模型, 只导出 deterministic actor 推理代码, 不导出 value/Q 网络、优化器、采样逻辑或 replay buffer. 默认按训练脚本的 common-mode normalized action 导出: actor 输出 1 维 `[-1, 1]`, C 端映射为 `[u, u]`.

在 repo 根目录导出头文件:

```bash
uv run python -m group11_balance.deploy.export_to_c \
  --algo PPO \
  --model outputs/models/group11_ppo.zip \
  --output outputs/firmware/group11_ppo_policy.h
```

NAF 模型示例:

```bash
uv run python -m group11_balance.deploy.export_to_c \
  --algo NAF \
  --model outputs/models/group11_naf.pt \
  --output outputs/firmware/group11_naf_policy.h
```

若要直接编译进 Keil 工程, 将导出的 `group11_*.h` 复制到 `firmware/keil/MiniBalance/`, 并在 `firmware/keil/MiniBalance/sb3_policy.h` 中选择要 include 的策略头文件.

然后修改 `firmware/keil/MiniBalance/Inc/control.h`:

```c
#define Control_mode       0
#define BOARD_POLICY_MODE  BOARD_POLICY_SB3
```

### 控制逻辑位置

`firmware/keil/MiniBalance/control.c` 中已经预留分支:

```c
#if BOARD_POLICY_MODE == BOARD_POLICY_LINEAR
    linear_predict(state, action);
#elif BOARD_POLICY_MODE == BOARD_POLICY_TABULAR
    tabular_predict(state, action);
#elif BOARD_POLICY_MODE == BOARD_POLICY_SB3
    sb3_predict(state, action);
#else
    /* original LQR */
#endif
```

因此常规接入只需要生成/复制头文件并修改 `BOARD_POLICY_MODE`, 不需要重写中断控制流程.

### 编译烧录

1. 打开 `firmware/keil/MDK-ARM/MiniBalance.uvprojx`.
2. 确认 `sb3_policy.h` 和它 include 的 `group11_*.h` 已存在.
3. 重新编译工程.
4. 将新生成的 hex 文件复制或移动到 `firmware/hex/`, 再从该目录选择并烧录.
5. 若需要恢复原始 LQR, 将 `BOARD_POLICY_MODE` 改回 `BOARD_POLICY_LQR` 后重新编译烧录.

---

## 8. 常见问题

### Q: mcuisp 找不到 COM 口?
- 确认 CH9102 驱动已安装
- 换一根 Type-C 数据线 (有些线只能充电)
- 确认小车电源已打开
- 重启电脑后重试

### Q: 烧录失败 / 超时?
- 确认选择了正确的 COM 口
- 确认底部下拉框选的是 "DTR的低电平复位, RTS高电平进BootLoader"
- 关闭其他可能占用串口的软件
- 按住复位键, 点开始编程, 然后松开复位键

### Q: WiFi 搜不到?
- 确认 WiFi 模块已正确插入 (不是蓝牙模块)
- 确认小车电源已打开
- WiFi 模块上电后需要等几秒才会广播

### Q: WiFi/TCP 连接超时?
- 确认笔记本已连接小车的 WiFi
- 确认没有其他设备同时连接
- 确认连接地址为 `192.168.4.1:6390`
- 尝试关闭小车电源, 等 5 秒, 重新开启

### Q: 我用的是 Mac/Linux, 怎么烧录?
- mcuisp 只有 Windows 版. 借一台 Windows 电脑烧录一次即可, 之后的 Python 开发在任何系统上都能做
- 烧录只需要做一次, 后续不需要再烧

---

## 9. 文件清单

```
firmware/
├── README.md                                # 本文件
├── hex/                                     # 统一存放可烧录的 .hex 文件
│   ├── MiniBalance.hex                      # PC 发命令测试固件 (Control_mode=1)
│   ├── MiniBalance.pc_control.hex           # PC 控制测试固件备份
│   └── MiniBalance_group11_*.hex            # Group 11 不同策略/实验对应的固件
├── keil/
│   ├── MDK-ARM/MiniBalance.uvprojx          # Keil 工程入口
│   ├── Core/                                # STM32 HAL 初始化与 main
│   ├── MiniBalance/                         # 平衡控制、策略头文件与传感器逻辑
│   ├── HAREWARE/                            # 电机、编码器、OLED、USART3 等外设逻辑
│   └── Drivers/                             # STM32 HAL/CMSIS 依赖
├── tools/
│   ├── mcuisp.exe                           # 烧录工具 (Windows)
│   ├── mcuispConfig.ini                     # mcuisp 配置
│   └── CH_Driver/                           # USB 串口驱动
└── docs/
    ├── 平衡车补充操作步骤.pdf                  # 烧录截图 + 主板标注图
    └── V5.5_wifi等新增功能说明.pdf             # WiFi 模块配置详细说明
```

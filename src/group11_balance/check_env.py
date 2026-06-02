"""Small environment sanity check."""

from __future__ import annotations

import numpy as np

from group11_balance.sim.env import TwoStageBalanceEnv


def main() -> None:
    env = TwoStageBalanceEnv(init_level="easy")
    obs, info = env.reset(seed=0)
    print("initial obs:", np.round(obs, 5), info)
    total = 0.0
    terminated = False
    truncated = False
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))
        total += reward
        if terminated or truncated:
            break
    print("after 20 zero-action steps:")
    print("obs:", np.round(obs, 5))
    print("total_reward:", round(total, 3))
    print("done:", terminated, truncated, info)


if __name__ == "__main__":
    main()

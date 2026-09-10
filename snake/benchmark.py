"""测量环境与随机动作的 CPU 吞吐，不把该数值冒充神经网络训练速度。"""

import argparse
import json
import time
from pathlib import Path

from snake.agents import RandomPolicy
from snake.envs import SingleSnakeEnv
from snake.experiment import source_manifest


def measure(steps: int, seed: int) -> dict:
    env, policy = SingleSnakeEnv(), RandomPolicy()
    policy.reset_episode(seed)
    observation, _ = env.reset(seed=seed)
    resets, reset_seconds = 1, 0.0
    started = time.perf_counter()
    for _ in range(steps):
        observation, _, terminated, truncated, _ = env.step(policy.act(observation))
        if terminated or truncated:
            reset_started = time.perf_counter()
            observation, _ = env.reset()
            reset_seconds += time.perf_counter() - reset_started
            resets += 1
    elapsed = time.perf_counter() - started
    env.close()
    return {"steps": steps, "resets": resets, "seconds": elapsed,
            "steps_per_second": steps / elapsed, "reset_seconds": reset_seconds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1000:
        parser.error("吞吐测量至少 1000 步")
    if args.output.exists():
        parser.error("输出已存在，请使用新文件名")
    measure(1000, 33)  # 排除导入与首次初始化；正式测量使用另一个随机源。
    result = {"mode": "CPU environment + random policy, no renderer, no optimizer",
              "warmup_steps": 1000, "measurement": measure(args.steps, 34), **source_manifest()}
    assert result["measurement"]["resets"] >= 3
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["measurement"], indent=2))


if __name__ == "__main__":
    main()

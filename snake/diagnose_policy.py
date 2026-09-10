"""在一个明确指定的验证回合保存完整轨迹并识别重复观测。"""

import argparse
import hashlib
import json
from pathlib import Path

from snake.agents.dqn import DQNPolicy
from snake.envs import SingleSnakeEnv
from snake.envs.single_snake import decode_observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1005)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1000 <= args.seed <= 1019:
        parser.error("阶段二诊断只接受验证种子 1000–1019")
    if args.output.exists():
        parser.error("保留旧诊断，请使用新文件名")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    policy = DQNPolicy.load(args.checkpoint)
    env = SingleSnakeEnv()
    observation, _ = env.reset(seed=args.seed)
    seen = {}
    first_repeat = None
    rows = []
    while True:
        key = observation.tobytes()
        if key in seen and first_repeat is None:
            first_repeat = {"first_step": seen[key], "repeated_step": env.steps, "period": env.steps - seen[key]}
        seen.setdefault(key, env.steps)
        body, food, direction = decode_observation(observation)
        action = policy.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        rows.append({"before_step": env.steps - 1, "body": body, "food": food, "direction": direction,
                     "action": action, "reward": reward, "terminated": terminated, "truncated": truncated})
        if terminated or truncated:
            break
    result = {"env_seed": args.seed, "checkpoint": str(args.checkpoint.resolve()),
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "model_step": policy.metadata["step"], "first_repeated_observation": first_repeat,
              "result": info, "steps": rows}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    env.close()
    print(json.dumps({key: value for key, value in result.items() if key != "steps"}, indent=2))


if __name__ == "__main__":
    main()

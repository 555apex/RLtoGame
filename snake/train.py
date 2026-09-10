"""DQN 命令入口；运行目录不可覆盖，正式测试不参与训练。"""

import argparse
from datetime import datetime
from pathlib import Path

from snake.experiment import ROOT
from snake.training import load_dqn_config, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/dqn.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--steps", type=int, help="累计总环境步数；默认使用配置中的 300000")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path, help="完整 resume.pt；恢复到新运行目录")
    args = parser.parse_args()
    config = load_dqn_config(args.config)
    if args.steps is not None:
        if args.steps <= 0:
            parser.error("steps 必须为正整数")
        config["total_steps"] = args.steps
    output = args.output or ROOT / "runs" / f"{config['algorithm']}_seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    result = train(config, args.seed, args.device, output, args.resume)
    print(f"Completed {result['step']} steps. Results: {output.resolve()}")


if __name__ == "__main__":
    main()

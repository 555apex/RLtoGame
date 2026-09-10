"""CPU/CUDA 使用同一 DQN 交互与更新流程，先验证可用性再比较吞吐。"""

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from snake.training import Trainer, load_dqn_config


def measure(device: str, steps: int, config: dict) -> dict:
    trainer = Trainer(config, 77, device)
    for _ in range(2000):
        trainer.interact()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    updates = trainer.agent.updates
    started = time.perf_counter()
    for _ in range(steps):
        trainer.interact()
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    result = {"device": device, "warmup_steps": 2000, "measured_steps": steps,
              "updates": trainer.agent.updates - updates, "seconds": seconds,
              "steps_per_second": steps / seconds, "last_update": trainer.last_metrics,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0}
    trainer.env.close()
    del trainer
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=6000)
    args = parser.parse_args()
    if args.steps < 1000 or args.output.exists():
        parser.error("至少测量 1000 步，且输出文件不能已存在")
    config = load_dqn_config()
    result = {"torch": str(torch.__version__), "cuda_build": torch.version.cuda,
              "cuda_available": torch.cuda.is_available(), "config": config, "trials": [],
              "scope": "2000-step warmup then DQN interaction + replay + optimizer; no validation/checkpoint I/O"}
    if torch.cuda.is_available():
        result["gpu"] = torch.cuda.get_device_name(0)
        result["cuda_arch_list"] = torch.cuda.get_arch_list()
    for device in ("cpu", "cuda"):
        try:
            result["trials"].append(measure(device, args.steps, config))
        except Exception as error:
            result["trials"].append({"device": device, "error": repr(error)})
    valid = [r for r in result["trials"] if "steps_per_second" in r]
    if not valid:
        raise RuntimeError("没有设备通过训练吞吐验证")
    result["selected_device"] = max(valid, key=lambda r: r["steps_per_second"])["device"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

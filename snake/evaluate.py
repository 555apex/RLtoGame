"""统一评估入口：阶段一只运行验证集，正式测试需显式登记开关。"""

import argparse
import csv
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from snake.experiment import ROOT, load_protocol, policy_seed, run_episode, source_manifest


def summarize(rows: list[dict]) -> dict:
    result = {}
    for name in sorted({row["policy"] for row in rows}):
        group = [row for row in rows if row["policy"] == name]
        food = [row["food"] for row in group]
        result[name] = {"episodes": len(group), "mean_food": statistics.mean(food),
                        "episode_food_std": statistics.stdev(food) if len(food) > 1 else 0.0,
                        "mean_length": statistics.mean(row["length"] for row in group),
                        "mean_occupancy": statistics.mean(row["occupancy"] for row in group),
                        "mean_steps": statistics.mean(row["steps"] for row in group),
                        "mean_return": statistics.mean(row["return"] for row in group),
                        "collisions": sum("collision" in row["end_reason"] for row in group),
                        "successes": sum(row["success"] for row in group),
                        "truncations": sum(row["truncated"] for row in group)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=["random", "heuristic", "all", "dqn", "double_dqn"], default=None)
    parser.add_argument("--checkpoint", type=Path, help="DQN 策略模型 initial.pt / best.pt / last.pt")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--allow-final-test", action="store_true",
                        help="仅在方案冻结后使用；阶段一不要访问 test")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    requested_policy = args.policy
    args.policy = args.policy or ("dqn" if args.checkpoint else "all")
    if (args.policy in ("dqn", "double_dqn")) != (args.checkpoint is not None):
        parser.error("DQN 评估必须提供 --checkpoint，其他策略不能使用模型文件")
    if args.split == "test" and not args.allow_final_test:
        parser.error("最终测试尚需冻结方案；请使用 validation")
    config = load_protocol()
    output = args.output or ROOT / "runs" / datetime.now().strftime("baselines_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    start, end = config["splits"][args.split]
    policies = ["random", "heuristic"] if args.policy == "all" else [args.policy]
    policy_instance = None
    checkpoint_info = None
    if args.checkpoint:
        from snake.agents.dqn import DQNPolicy
        policy_instance = DQNPolicy.load(args.checkpoint, args.device)
        actual_algorithm = policy_instance.metadata["config"]["algorithm"]
        if requested_policy is not None and requested_policy != actual_algorithm:
            parser.error("--policy 与存档中的算法不一致")
        policies = [actual_algorithm]
        checkpoint_info = {"path": str(args.checkpoint.resolve()),
                           "sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                           "metadata": policy_instance.metadata, "device": args.device}
    manifest = {"run_id": output.name, "created_utc": datetime.now(timezone.utc).isoformat(),
                "split": args.split, "env_seeds": list(range(start, end + 1)),
                "policies": policies, "protocol": config, "checkpoint": checkpoint_info, **source_manifest()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows = []
    started = time.perf_counter()
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = None
        for name in policies:
            for env_seed in range(start, end + 1):
                seed = policy_seed(config["policy_seed_base"], env_seed)
                try:
                    row = run_episode(name, env_seed, seed, policy_instance=policy_instance)
                except Exception as error:
                    (output / "failure.json").write_text(json.dumps({"policy": name, "env_seed": env_seed,
                        "policy_seed": seed, "error": repr(error)}, indent=2), encoding="utf-8")
                    raise  # 不把技术异常静默转成正常输局，也不挑选其他种子替换。
                row["split"] = args.split
                if writer is None:
                    writer = csv.DictWriter(file, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                file.flush()
                rows.append(row)
    unit = ("episode; one frozen trained model, training seed in manifest" if args.checkpoint
            else "episode; fixed policies, no training seeds")
    summary = {"split": args.split, "wall_seconds": time.perf_counter() - started,
               "statistics_unit": unit, "policies": summarize(rows)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved to: {output.resolve()}")


if __name__ == "__main__":
    main()

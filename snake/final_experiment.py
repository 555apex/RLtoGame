"""先冻结，再测试：使选模依据、模型文件和最终评测结果可以逐项审计。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from snake.agents.dqn import DQNPolicy
from snake.evaluate import summarize
from snake.experiment import ROOT, load_protocol, policy_seed, run_episode
from snake.run_experiments import MATRIX, read_json, sha256, verify_run

DATA = ROOT / "reports/data/final_experiment_v1"
CORE = ("agents/dqn.py", "agents/replay.py", "agents/baselines.py", "agents/__init__.py",
        "envs/single_snake.py", "envs/__init__.py", "training.py", "experiment.py", "evaluate.py",
        "final_experiment.py", "configs/single_snake.json", "configs/dqn.json", "configs/double_dqn.json")
LOGS = ("manifest.json", "summary.json", "source_snapshot.json", "episodes.jsonl", "updates.jsonl",
        "validation.jsonl", "validation_episodes.jsonl", "events.jsonl")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def tensor_hash(weights: dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(weights.items()):
        digest.update(name.encode())
        digest.update(tensor.cpu().numpy().tobytes())
    return digest.hexdigest()


def audit_training(directory: Path, summary: dict) -> None:
    """从逐局日志复算预算、验证均值及选模点，避免只信 summary。"""
    episodes = read_rows(directory / "episodes.jsonl")
    if sum(row["steps"] for row in episodes) + summary["partial_episode"]["steps"] != 300000:
        raise ValueError("训练回合与预算对账失败")
    for row in episodes:
        if not 0 <= row["env_seed"] <= 999 or abs(sum(row["reward_components"].values()) - row["return"]) > 1e-7:
            raise ValueError("训练种子或奖励分量异常")
    rows = read_rows(directory / "validation_episodes.jsonl")
    validation = read_rows(directory / "validation.jsonl")
    if len(rows) != 620 or [r["step"] for r in validation] != list(range(0, 300001, 10000)):
        raise ValueError("验证次数不完整")
    for record in validation:
        group = [r for r in rows if r["step"] == record["step"]]
        if sorted(r["env_seed"] for r in group) != list(range(1000, 1020)):
            raise ValueError("验证种子不匹配")
        if statistics.mean(r["food"] for r in group) != record["mean_food"]:
            raise ValueError("验证均值不能由逐局记录还原")
    best = max(validation, key=lambda row: row["mean_food"])
    if (best["step"], best["mean_food"]) != (summary["best_step"], summary["best_validation_food"]):
        raise ValueError("选模点未遵守验证均值最大、同分保留较早版本")


def freeze(output: Path) -> None:
    matrix = read_json(MATRIX)
    items, controls = [], {}
    # 全部校验通过后才写冻结目录。阶段二 DQN 0 保留自己的源码快照和时间。
    for run in matrix["runs"]:
        manifest, summary = verify_run(run)
        directory = ROOT / run["path"]
        audit_training(directory, summary)
        for variant in ("best", "initial"):
            path = directory / "checkpoints" / f"{variant}.pt"
            payload = torch.load(path, map_location="cpu", weights_only=True)
            expected_step = summary["best_step"] if variant == "best" else 0
            if (payload["config"] != manifest["config"] or payload["training_seed"] != run["seed"]
                    or payload["step"] != expected_step):
                raise ValueError("策略存档的配置、种子或步数与日志不符")
            weight_hash = tensor_hash(payload["online"])
            item = {"id": f"{run['algorithm']}_seed{run['seed']}_{variant}",
                    "method": run["algorithm"] if variant == "best" else "initial",
                    "training_seed": run["seed"], "variant": variant,
                    "path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path),
                    "tensor_sha256": weight_hash, "step": payload["step"],
                    "validation_mean_food": summary["best_validation_food"] if variant == "best" else None,
                    "run": run["path"], "config_sha256": manifest["config_sha256"]}
            if variant == "best":
                items.append(item)
            elif run["seed"] not in controls:
                controls[run["seed"]] = item
            elif controls[run["seed"]]["tensor_sha256"] != weight_hash:
                raise ValueError("同种子初始模型不一致，不能共用无学习对照")
    # 展示模型在测试前由验证集选定，同分沿矩阵顺序保留；不展示测试冠军。
    demo = max(items, key=lambda item: item["validation_mean_food"])["id"]
    payload = {"created_utc": utc_now(), "status": "frozen_before_test", "matrix": matrix,
               "matrix_sha256": sha256(MATRIX), "protocol": load_protocol(), "demo_model_id": demo,
               "test_seeds": list(range(10000, 10020)), "models": items + list(controls.values()),
               "baselines": ["random", "heuristic"], "expected_test_episodes": 220,
               "core_sha256": {name: sha256(ROOT / name) for name in CORE},
               "selection_uses_test": False, "technical_failures": []}
    output.mkdir(parents=True, exist_ok=False)
    archives = output / "training"
    archives.mkdir()
    for run in matrix["runs"]:
        target = archives / f"{run['algorithm']}_seed{run['seed']}"
        target.mkdir()
        for name in LOGS:
            shutil.copy2(ROOT / run["path"] / name, target / name)
    write_json(output / "freeze.json", payload)
    print(f"Frozen 6 best models + 3 shared initial controls: {output}")


def verify_freeze(directory: Path) -> dict:
    frozen = read_json(directory / "freeze.json")
    if frozen["protocol"] != load_protocol() or frozen["matrix_sha256"] != sha256(MATRIX):
        raise ValueError("冻结后协议或矩阵改变")
    for name, digest in frozen["core_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"冻结后代码改变: {name}")
    for model in frozen["models"]:
        if sha256(ROOT / model["path"]) != model["sha256"]:
            raise ValueError(f"冻结后权重改变: {model['id']}")
    return frozen


def aggregate(rows: list[dict]) -> dict:
    per_model = []
    for model_id in dict.fromkeys(row["model_id"] for row in rows):
        group = [row for row in rows if row["model_id"] == model_id]
        method = group[0]["policy"]
        per_model.append({"model_id": model_id, "method": method,
                          "training_seed": group[0]["training_seed"], **summarize(group)[method]})
    learned = {}
    for method in ("initial", "dqn", "double_dqn"):
        means = [row["mean_food"] for row in per_model if row["method"] == method]
        if len(means) != 3:
            raise ValueError("跨训练种子统计必须恰好包含三个模型")
        learned[method] = {"training_seeds": 3, "episodes_per_model": 20,
                           "mean_food": statistics.mean(means), "seed_mean_food_std": statistics.stdev(means)}
    return {"per_model": per_model, "across_training_seeds": learned,
            "statistics_unit": "mean of 20 test episodes for each training seed; sample std over 3 such means"}


def evaluate_frozen(directory: Path) -> None:
    frozen = verify_freeze(directory)
    output = directory / "test"
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    write_json(output / "manifest.json", {"started_utc": utc_now(), "split": "test",
               "freeze_sha256": sha256(directory / "freeze.json"), "test_seeds": frozen["test_seeds"],
               "configurations_changed": False})
    rows, timings = [], {}
    candidates = frozen["models"] + [{"id": name, "method": name, "training_seed": None}
                                       for name in frozen["baselines"]]
    with (output / "episodes.csv").open("w", encoding="utf-8", newline="") as file:
        writer = None
        for model in candidates:
            model_start = time.perf_counter()
            policy = DQNPolicy.load(ROOT / model["path"]) if "path" in model else None
            for env_seed in frozen["test_seeds"]:
                try:
                    row = run_episode(model["method"], env_seed,
                                      policy_seed(frozen["protocol"]["policy_seed_base"], env_seed),
                                      policy_instance=policy)
                except Exception as error:
                    write_json(output / "failure.json", {"model_id": model["id"], "env_seed": env_seed,
                               "error": repr(error), "utc": utc_now()})
                    raise
                row.update(model_id=model["id"], training_seed=model["training_seed"], split="test")
                if writer is None:
                    writer = csv.DictWriter(file, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                file.flush()
                rows.append(row)
            timings[model["id"]] = time.perf_counter() - model_start
            print(f"Evaluated {model['id']}", flush=True)
    if len(rows) != frozen["expected_test_episodes"]:
        raise ValueError("最终测试局数不完整")
    result = {**aggregate(rows), "episodes": len(rows), "completed_utc": utc_now(),
              "wall_seconds": time.perf_counter() - started, "model_wall_seconds": timings,
              "episodes_sha256": sha256(output / "episodes.csv"), "technical_failures": []}
    write_json(output / "summary.json", result)
    print(json.dumps(result["across_training_seeds"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "test"])
    parser.add_argument("--output", type=Path, default=DATA)
    args = parser.parse_args()
    (freeze if args.action == "freeze" else evaluate_frozen)(args.output)


if __name__ == "__main__":
    main()

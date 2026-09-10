"""归档真实单种子 DQN 日志，重建阶段二 Markdown 报告及图表。"""

import argparse
import ast
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from snake.experiment import ROOT

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def draw(history, updates, diagnosis, assets):
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                         "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    assets.mkdir(parents=True, exist_ok=True)
    steps = np.array([row["step"] for row in history]) / 10000
    food = [row["mean_food"] for row in history]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, layout="constrained")
    axes[0].plot(steps, food, "o-", color="#205A92", markersize=4, label="DQN / 训练种子 0")
    axes[0].axhline(0.15, color="#656B73", linestyle="--", label="随机基线：0.15")
    best = int(np.argmax(food))
    axes[0].annotate(f"验证最优：{food[best]:.2f}", (steps[best], food[best]),
                     xytext=(-95, 12), textcoords="offset points", arrowprops={"arrowstyle": "-"})
    axes[0].set(ylabel="平均食物数 / 局", ylim=(-0.05, max(food) * 1.2 + 0.1))
    axes[0].legend(frameon=False, loc="upper left")
    axes[1].plot(steps, [r["truncations"] * 5 for r in history], "s-", color="#AD7814", markersize=4)
    axes[1].set(xlabel="累计环境步数 / 万步", ylabel="验证截断比例 / %", ylim=(-3, 105))
    for ax in axes:
        ax.grid(axis="y", color="#E2E5E9")
    fig.suptitle("DQN 验证表现与时间截断\n单个训练种子 · 每次 20 局 · 每 1 万步验证 · 未平滑", fontsize=14)
    fig.savefig(assets / "dqn_learning.png", dpi=160)
    plt.close(fig)

    x = np.array([r["step"] for r in updates]) / 10000
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, layout="constrained")
    axes[0].plot(x, [r["loss"] for r in updates], color="#205A92", linewidth=1.2)
    axes[0].set(ylabel="平均 Huber 损失")
    axes[1].plot(x, [r["mean_abs_td_error"] for r in updates], color="#AD7814", linewidth=1.2)
    axes[1].set(xlabel="累计环境步数 / 万步", ylabel="平均绝对 TD 误差")
    for ax in axes:
        ax.grid(axis="y", color="#E2E5E9")
    fig.suptitle("DQN 更新诊断\n每 1000 环境步内更新的均值 · 无额外平滑 · 损失不是游戏成绩", fontsize=14)
    fig.savefig(assets / "dqn_updates.png", dpi=160)
    plt.close(fig)

    cycle = diagnosis["first_repeated_observation"]
    segment = diagnosis["steps"][cycle["first_step"]:cycle["repeated_step"] + 1]
    heads = [r["body"][0] for r in segment]
    food_row, food_col = segment[0]["food"]
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    ax.plot([p[1] for p in heads], [p[0] for p in heads], "o-", color="#205A92", markersize=4)
    for index in range(0, len(heads) - 1, 4):
        a, b = heads[index:index + 2]
        ax.annotate("", xy=(b[1], b[0]), xytext=(a[1], a[0]),
                    arrowprops={"arrowstyle": "->", "color": "#205A92", "lw": 2})
    ax.scatter([food_col], [food_row], marker="*", s=180, color="#AD7814", label="未吃到的食物")
    ax.scatter([heads[0][1]], [heads[0][0]], marker="s", s=95, facecolors="white", edgecolors="#205A92", label="重复状态的起点")
    ax.set(xlim=(-0.5, 9.5), ylim=(9.5, -0.5), xticks=range(10), yticks=range(10), xlabel="列", ylabel="行")
    ax.set_aspect("equal")
    ax.grid(color="#E2E5E9")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title(f"验证种子 {diagnosis['env_seed']}：蛇头循环轨迹\n第 {cycle['first_step']}–{cycle['repeated_step']} 步 · 周期 {cycle['period']} 步")
    fig.savefig(assets / "dqn_loop.png", dpi=160)
    plt.close(fig)


def extract_target(source: str) -> str:
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "dqn_targets")
    return "\n".join(source.splitlines()[node.lineno - 2:node.end_lineno])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="首次从训练目录归档日志；重建时不需要模型或原 runs 目录")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "reports/data/dqn_seed0_training")
    args = parser.parse_args()
    if args.run:
        args.data_dir.mkdir(parents=True, exist_ok=True)
        for name in ("manifest.json", "summary.json", "source_snapshot.json", "episodes.jsonl", "updates.jsonl",
                     "validation.jsonl", "validation_episodes.jsonl", "events.jsonl"):
            destination, source = args.data_dir / name, args.run / name
            if destination.exists() and destination.read_bytes() != source.read_bytes():
                raise ValueError(f"已有不同历史数据：{destination}，请换归档目录")
            if not destination.exists():
                shutil.copy2(source, destination)
    manifest, summary = read_json(args.data_dir / "manifest.json"), read_json(args.data_dir / "summary.json")
    history, updates = read_rows(args.data_dir / "validation.jsonl"), read_rows(args.data_dir / "updates.jsonl")
    episodes, validations = read_rows(args.data_dir / "episodes.jsonl"), read_rows(args.data_dir / "validation_episodes.jsonl")
    snapshot = read_json(args.data_dir / "source_snapshot.json")
    assert manifest["resume"] is None, "当前报告模板适用于完整单次种子 0 运行"
    assert not summary["final_test_used"] and summary["status"] == "completed"
    assert [r["step"] for r in history] == list(range(0, summary["step"] + 1, 10000))
    for item in history:
        rows = [r for r in validations if r["step"] == item["step"]]
        assert sorted(r["env_seed"] for r in rows) == list(range(1000, 1020))
        assert abs(np.mean([r["food"] for r in rows]) - item["mean_food"]) < 1e-10
    assert all(0 <= row["env_seed"] <= 999 for row in episodes)
    assert sum(row["steps"] for row in episodes) + summary["partial_episode"]["steps"] == summary["step"]
    assert all(abs(row["return"] - sum(row["reward_components"].values())) < 1e-7 for row in episodes)
    assert len(episodes) == summary["completed_episodes"]
    best = max(history, key=lambda row: row["mean_food"])
    assert best["step"] == summary["best_step"]
    for name in ("agents/dqn.py", "agents/replay.py", "training.py", "envs/single_snake.py"):
        assert hashlib.sha256(snapshot[name].encode("utf-8")).hexdigest() == manifest["sources"]["files"][name]
    diagnosis = read_json(ROOT / "reports/data/dqn_seed0_loop1005.json")
    repeat = diagnosis["first_repeated_observation"]
    for i in range(repeat["repeated_step"], len(diagnosis["steps"])):
        for key in ("body", "food", "direction", "action"):
            assert diagnosis["steps"][i][key] == diagnosis["steps"][i - repeat["period"]][key]
    assert diagnosis["checkpoint_sha256"] == summary["checkpoint_sha256"]["best.pt"]
    draw(history, updates, diagnosis, ROOT / "reports/assets")
    template = (ROOT / "reports/stage2_template.md").read_text(encoding="utf-8")
    comparison = []
    for label, step in (("DQN 初始网络", 0), ("DQN 验证最优", best["step"]), ("DQN 末次网络", summary["step"])):
        group = [r for r in validations if r["step"] == step]
        foods = [r["food"] for r in group]
        comparison.append(f"| {label} | {step} | {np.mean(foods):.2f} | {np.std(foods, ddof=1):.2f} | {np.mean([r['steps'] for r in group]):.1f} | {sum('collision' in r['end_reason'] for r in group)}/20 | {sum(r['truncated'] for r in group)}/20 |")
    tokens = {"{{RESULT_ROWS}}": "\n".join(comparison), "{{TD_CODE}}": extract_target(snapshot["agents/dqn.py"]),
              "{{BEST_FOOD}}": f"{best['mean_food']:.2f}", "{{BEST_STEP}}": str(best["step"]),
              "{{LAST_FOOD}}": f"{history[-1]['mean_food']:.2f}", "{{TRAIN_SECONDS}}": f"{summary['training_seconds']:.2f}",
              "{{WALL_SECONDS}}": f"{summary['segment_wall_seconds']:.2f}", "{{EPISODES}}": str(len(episodes)),
              "{{UPDATES}}": str(summary["updates"]), "{{MEMORY_MIB}}": f"{summary['process_memory_bytes'] / 2**20:.1f}",
              "{{SOURCE_HASH}}": manifest["sources"]["source_sha256"], "{{BEST_HASH}}": summary["checkpoint_sha256"]["best.pt"]}
    for key, value in tokens.items():
        template = template.replace(key, value)
    assert "{{" not in template
    (ROOT / "reports/stage2_report.md").write_text(template, encoding="utf-8")
    (ROOT / "reports/data/stage2_report_qa.json").write_text(json.dumps({
        "training_steps": summary["step"], "episodes": len(episodes), "validation_rows": len(validations),
        "source_hashes_verified": True, "seed_splits_verified": True, "return_components_verified": True,
        "training_step_accounting_verified": True, "loop_period_verified_until_episode_end": repeat["period"]}, indent=2), encoding="utf-8")
    print("Stage 2 report and data reconciliation complete.")


if __name__ == "__main__":
    main()

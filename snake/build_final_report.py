"""从冻结归档和逐局记录重建单蛇报告；不需要权重，不运行新测试局。"""

from __future__ import annotations

import ast
import csv
import hashlib
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from snake.final_experiment import DATA, aggregate, audit_training, read_rows, write_json
from snake.run_experiments import read_json, sha256
from snake.experiment import ROOT

ASSETS = ROOT / "reports/assets"
NAMES = {"dqn": "DQN", "double_dqn": "Double DQN", "initial": "初始网络",
         "random": "均匀随机", "heuristic": "曼哈顿启发式"}


def load_test_rows() -> list[dict]:
    with (DATA / "test/episodes.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for key in ("food", "length", "steps", "env_seed", "policy_seed"):
            row[key] = int(row[key])
        row["training_seed"] = int(row["training_seed"]) if row["training_seed"] else None
        for key in ("occupancy", "return", "reward_step_cost", "reward_food", "reward_collision", "reward_full_board"):
            row[key] = float(row[key])
        for key in ("terminated", "truncated", "success"):
            if row[key] not in ("True", "False"):
                raise ValueError("CSV 中布尔字段不合法")
            row[key] = row[key] == "True"
    return rows


def validate_evidence(rows: list[dict], frozen: dict, stored: dict) -> dict:
    assert len(rows) == 220 and sha256(DATA / "test/episodes.csv") == stored["episodes_sha256"]
    manifest = read_json(DATA / "test/manifest.json")
    assert manifest["freeze_sha256"] == sha256(DATA / "freeze.json")
    assert frozen["created_utc"] < manifest["started_utc"] < stored["completed_utc"]
    expected = {m["id"]: (m["method"], m["training_seed"]) for m in frozen["models"]}
    expected.update({name: (name, None) for name in frozen["baselines"]})
    assert set(row["model_id"] for row in rows) == set(expected)
    for model, identity in expected.items():
        group = [r for r in rows if r["model_id"] == model]
        assert sorted(r["env_seed"] for r in group) == frozen["test_seeds"]
        assert all((r["policy"], r["training_seed"]) == identity and r["split"] == "test" for r in group)
    for row in rows:
        assert row["length"] == row["food"] + 3 and row["occupancy"] == row["length"] / 100
        assert row["terminated"] != row["truncated"] and 1 <= row["steps"] <= 2000
        assert not row["truncated"] or row["steps"] == 2000
        assert row["reward_food"] == row["food"]
        assert abs(row["reward_step_cost"] + 0.001 * row["steps"]) < 1e-7
        assert row["reward_collision"] == -int("collision" in row["end_reason"])
        assert row["reward_full_board"] == 10 * int(row["success"])
        assert abs(sum(row[k] for k in row if k.startswith("reward_")) - row["return"]) < 1e-7
    assert aggregate(rows)["per_model"] == stored["per_model"]
    assert aggregate(rows)["across_training_seeds"] == stored["across_training_seeds"]
    for item in frozen["matrix"]["runs"]:
        directory = DATA / "training" / f"{item['algorithm']}_seed{item['seed']}"
        audit_training(directory, read_json(directory / "summary.json"))
        archived = read_json(directory / "manifest.json")
        snapshot = read_json(directory / "source_snapshot.json")
        for name, digest in archived["sources"]["files"].items():
            assert hashlib.sha256(snapshot[name].encode()).hexdigest() == digest, name
    return {"status": "passed", "training_steps": 1800000, "training_runs": 6,
            "validation_episodes": 3720, "test_episodes": 220,
            "checks": ["budget and reward reconciliation", "exact seed lists", "model identities",
                       "frozen before test", "archived source hashes", "statistics recomputed from CSV"],
            "freeze_sha256": sha256(DATA / "freeze.json"), "episodes_sha256": sha256(DATA / "test/episodes.csv")}


def save_figure(figure, name: str) -> None:
    figure.savefig(ASSETS / name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_results(summary: dict) -> None:
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"], "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    figures, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, layout="constrained")
    for column, method in enumerate(("dqn", "double_dqn")):
        for seed, color in zip(range(3), ("#247ba0", "#ee964b", "#7b5ea7")):
            directory = DATA / "training" / f"{method}_seed{seed}"
            validation = read_rows(directory / "validation.jsonl")
            x = [row["step"] / 1000 for row in validation]
            axes[0, column].plot(x, [r["mean_food"] for r in validation], color=color, label=f"种子 {seed}")
            axes[1, column].plot(x, [r["truncations"] / 20 for r in validation], color=color)
        axes[0, column].set(title=NAMES[method], ylabel="验证平均食物数", ylim=(0, None))
        axes[0, column].legend()
        axes[1, column].set(xlabel="累计训练环境步（千）", ylabel="验证截断比例", ylim=(0, 1.05))
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    maximum = max(ax.get_ylim()[1] for ax in axes[0])
    for ax in axes[0]:
        ax.set_ylim(0, maximum)  # 同一指标使用同一比例尺，避免夸大曲线差别。
    save_figure(figures, "final_training.png")

    figures, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    model_rows = summary["per_model"]
    means = summary["across_training_seeds"]
    for index, method in enumerate(("initial", "dqn", "double_dqn")):
        axes[0].errorbar(index, means[method]["mean_food"], yerr=means[method]["seed_mean_food_std"],
                         fmt="o", color="#184e77", capsize=6, markersize=8)
        points = [r["mean_food"] for r in model_rows if r["method"] == method]
        axes[0].scatter(np.array([index] * 3) + [-0.09, 0, 0.09], points, color="#ee964b", s=30, zorder=3)
    axes[0].set(xticks=range(3), xticklabels=[NAMES[x] for x in means],
                ylabel="测试平均食物数", title="学习前后：大点±训练种子样本标准差", ylim=(0, None))
    axes[0].text(0.02, 0.95, "橙点：每个训练种子的 20 局均值", transform=axes[0].transAxes, va="top")
    methods = ("random", "heuristic", "dqn", "double_dqn")
    heights = [next(r["mean_food"] for r in model_rows if r["method"] == method) if method not in means
               else means[method]["mean_food"] for method in methods]
    axes[1].bar(range(4), heights, color=["#adb5bd", "#44836b", "#247ba0", "#8f68ad"])
    for i, value in enumerate(heights):
        axes[1].text(i, value + 0.25, f"{value:.2f}", ha="center")
    axes[1].set(xticks=range(4), xticklabels=[NAMES[x] for x in methods], ylabel="测试平均食物数",
                title="四组规定方法：各自汇总均值", ylim=(0, max(heights) * 1.16 + 0.5))
    save_figure(figures, "final_test_food.png")

    learned = [r for r in model_rows if r["method"] in ("dqn", "double_dqn")]
    figure, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    labels = [f"{NAMES[r['method']]} / {r['training_seed']}" for r in learned]
    bottom = np.zeros(6)
    for key, name, color in (("collisions", "碰撞", "#c76b58"), ("truncations", "时间截断", "#d4a853"),
                             ("successes", "满盘成功", "#44836b")):
        values = np.array([r[key] / 20 * 100 for r in learned])
        ax.barh(labels, values, left=bottom, label=name, color=color)
        bottom += values
    ax.set(xlim=(0, 100), xlabel="每模型 20 局测试的比例（%）", title="终局原因：碰撞与截断必须分开")
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    save_figure(figure, "final_outcomes.png")


def code_excerpt(source: str, name: str) -> str:
    node = next(n for n in ast.parse(source).body if getattr(n, "name", None) == name)
    start = min([node.lineno] + [n.lineno for n in node.decorator_list])
    return "\n".join(source.splitlines()[start - 1:node.end_lineno])


def build_findings(summary: dict) -> None:
    rows = {(r["method"], r["training_seed"]): r for r in summary["per_model"]}
    lines = []
    for seed in range(3):
        initial, dqn, double = [rows[method, seed]["mean_food"] for method in ("initial", "dqn", "double_dqn")]
        lines.append(f"| {seed} | {initial:.2f} | {dqn:.2f} | {double:.2f} | "
                     f"{dqn-initial:+.2f} | {double-initial:+.2f} | {double-dqn:+.2f} |")
    text = """# 单蛇结果解读与答辩用结论

数据来源：[冻结后的逐局测试](data/final_experiment_v1/test/episodes.csv)。本页由报告构建脚本生成配对表，每项都是同一训练种子下的 20 局食物均值。

| 训练种子 | 初始 | DQN | Double DQN | DQN−初始 | Double−初始 | Double−DQN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
""" + "\n".join(lines) + """

**在本次三个训练种子上，更新后的模型都比对应初始模型吃到更多食物。** 这提供了有限范围内的学习增益证据；也高于均匀随机基线的 0.15 个/局，但远低于启发式的 19.10 个/局。不能据三个种子和二十种环境就声称普遍泛化可靠。

**Double DQN 没有体现出平均食物数优势。** 两者跨种子均值均为 0.85；相同训练种子配对差值为 +0.05、−0.15、+0.10，方向不一致。本实验不支持“改用 Double DQN 就能提高当前单蛇成绩”，也不证明两种算法在所有设置中等效。没有做显著性检验或更多预算探索。

**两种方法都频繁出现低食物数的长时间存活。** 各自 60 局中均有 38 局截断、22 局碰撞，满盘次数为 0。截断比例 63.3%，应结合完整轨迹判断循环；不能仅凭达到时间上限推断每局都是同一种循环。保留的 DQN 验证轨迹确实出现了 24 步周期。

**验证领先没有直接变成测试领先。** 验证均值最高的 Double DQN 种子 1 为 1.75，冻结后测试为 0.80；DQN 种子 0 验证为 1.30，测试为 0.60。模型是在反复使用的验证集上选出的，结果可能有选择偏差；环境种子不同也引入波动。单凭这个差值不能量化过拟合程度。

答辩时可说：“我实现并验证了两种 TD 更新，完成了规定预算和独立测试。三个种子都出现相对初始模型的觅食提升，但绝对成绩较低，Double DQN 本次没有提高平均食物数。循环与碰撞说明策略仍有局限。奖励、探索或表示的改进需要单独预登记消融，本轮没有看测试后补训。”

上述内容是实验解释材料，学生需独立理解后用自己的话表达；个人完成情况按实际记录填写。
"""
    (ROOT / "reports/single_snake_findings.md").write_text(text, encoding="utf-8")


def main() -> None:
    frozen, summary = read_json(DATA / "freeze.json"), read_json(DATA / "test/summary.json")
    qa = validate_evidence(load_test_rows(), frozen, summary)
    plot_results(summary)
    build_findings(summary)
    model_rows = summary["per_model"]
    stats = summary["across_training_seeds"]
    per_seed = "\n".join(f"| {NAMES[r['method']]} | {r['training_seed'] if r['training_seed'] is not None else '—'} | "
        f"{r['mean_food']:.2f} | {r['episode_food_std']:.2f} | {r['mean_length']:.2f} | "
        f"{100*r['mean_occupancy']:.2f}% | {r['mean_steps']:.1f} | {r['mean_return']:.3f} | "
        f"{r['collisions']}/20 | {r['truncations']}/20 | {r['successes']}/20 |" for r in model_rows)
    across = "\n".join(f"| {NAMES[name]} | {s['mean_food']:.3f} | {s['seed_mean_food_std']:.3f} |"
                        for name, s in stats.items())
    run_rows, hashes = [], []
    for run in frozen["matrix"]["runs"]:
        directory = DATA / "training" / f"{run['algorithm']}_seed{run['seed']}"
        result, manifest = read_json(directory / "summary.json"), read_json(directory / "manifest.json")
        run_rows.append(f"| {NAMES[run['algorithm']]} | {run['seed']} | {result['best_step']} | "
                        f"{result['best_validation_food']:.2f} | {result['updates']} | "
                        f"{result['training_seconds']:.1f} | {result['segment_wall_seconds']:.1f} | "
                        f"{result['process_memory_bytes']/1024**2:.1f} |")
        hashes.append({"algorithm": run["algorithm"], "training_seed": run["seed"],
                       "directory": run["path"], "files": result["checkpoint_sha256"],
                       "config_sha256": manifest["config_sha256"], "source_sha256": manifest["sources"]["source_sha256"]})
    write_json(ROOT / "reports/data/weights_manifest.json", {"publication": "local only; user approval pending",
               "runs": hashes, "freeze_sha256": sha256(DATA / "freeze.json")})
    source = read_json(DATA / "training/double_dqn_seed0/source_snapshot.json")["agents/dqn.py"]
    values = {"PER_SEED": per_seed, "ACROSS": across, "RUNS": "\n".join(run_rows),
              "DQN_CODE": code_excerpt(source, "dqn_targets"), "DDQN_CODE": code_excerpt(source, "double_dqn_targets"),
              "FREEZE_HASH": sha256(DATA / "freeze.json"), "FREEZE_TIME": frozen["created_utc"],
              "TEST_TIME": read_json(DATA / "test/manifest.json")["started_utc"],
              "DEMO_MODEL": frozen["demo_model_id"],
              "DQN_MEAN": f"{stats['dqn']['mean_food']:.3f}", "DQN_STD": f"{stats['dqn']['seed_mean_food_std']:.3f}",
              "DDQN_MEAN": f"{stats['double_dqn']['mean_food']:.3f}", "DDQN_STD": f"{stats['double_dqn']['seed_mean_food_std']:.3f}",
              "INITIAL": f"{stats['initial']['mean_food']:.3f}",
              "RANDOM": f"{next(r['mean_food'] for r in model_rows if r['method']=='random'):.2f}",
              "HEURISTIC": f"{next(r['mean_food'] for r in model_rows if r['method']=='heuristic'):.2f}"}
    template = (ROOT / "reports/single_snake_template.md").read_text(encoding="utf-8")
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)
    assert "{{" not in template
    (ROOT / "reports/single_snake_report.md").write_text(template, encoding="utf-8")
    write_json(ROOT / "reports/data/final_report_qa.json", qa)
    print("Final report and 3 figures rebuilt; 220 test episodes and 1800000 training steps reconciled.")


if __name__ == "__main__":
    main()

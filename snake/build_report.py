"""从保存的逐局记录重建阶段报告和静态图；默认不重新运行实验。"""

import argparse
import csv
import hashlib
import inspect
import json
import os
import statistics
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from snake.agents import HeuristicPolicy
from snake.envs import SingleSnakeEnv
from snake.experiment import ROOT


def figures(rows, output):
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                         "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4.5), layout="constrained")
    for policy, label, marker, color in [("random", "随机策略", "o", "#205A92"),
                                         ("heuristic", "启发式", "^", "#AD7814")]:
        group = [r for r in rows if r["policy"] == policy]
        ax.scatter([int(r["env_seed"]) - 1000 for r in group], [int(r["food"]) for r in group],
                   label=label, marker=marker, color=color, s=55)
    ax.set(title="两种固定策略的逐局食物数", xlabel="验证种子（1000 + 横轴值）", ylabel="食物数 / 局")
    ax.set_xticks(range(20))
    ax.set_ylim(bottom=-1)
    ax.grid(axis="y", color="#E2E5E9")
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.suptitle("同一验证列表 · 每种策略 20 局 · 点形区分策略 · 未使用正式测试集", fontsize=11)
    fig.savefig(output / "baseline_food.png", dpi=160)
    plt.close(fig)

    env = SingleSnakeEnv()
    obs, _ = env.reset(seed=1000)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for ax, values, title in zip(axes[:2], [obs[:100], obs[100:200]],
                                  ["有序蛇身地图 / 100 维", "食物地图 / 100 维"]):
        grid = values.reshape(10, 10)
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("snake_blue", ["#FFFFFF", "#A9C9E5"])
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=max(float(grid.max()), 0.01))
        for row, col in zip(*np.nonzero(grid)):
            ax.text(col, row, f"{grid[row, col]:.2f}", ha="center", va="center", fontsize=9, color="#263445")
        ax.set(title=title, xlabel="列", ylabel="行", xticks=range(10), yticks=range(10))
    axes[2].axis("off")
    axes[2].text(0.05, 0.8, "朝向 / 4 维\n\n上　右　下　左\n0　  1　  0　  0\n\n按行展开后拼接：\n100 + 100 + 4 = 204\n\n蛇头：0.01\n蛇尾：0.03", va="top", linespacing=1.7)
    fig.suptitle("初始观测编码示意 · 环境种子 1000 · float32", fontsize=14)
    fig.savefig(output / "observation.png", dpi=160)
    plt.close(fig)
    env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "reports/data/baselines_validation_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    rows = list(csv.DictReader((args.data_dir / "episodes.csv").open(encoding="utf-8")))
    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((args.data_dir / "summary.json").read_text(encoding="utf-8"))
    for name in ("agents/baselines.py", "envs/single_snake.py"):
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != manifest["files"][name]:
            raise ValueError(f"{name} 已偏离实验时源码，请使用原版本生成历史报告或重新评估")
    if manifest["split"] != "validation" or len(rows) != 40:
        raise ValueError("阶段一报告需要两种固定策略各 20 局的验证记录")
    for name in ("random", "heuristic"):
        group = [row for row in rows if row["policy"] == name]
        assert sorted(int(row["env_seed"]) for row in group) == list(range(1000, 1020))
        assert abs(statistics.mean(int(row["food"]) for row in group) - summary["policies"][name]["mean_food"]) < 1e-10
    for row in rows:
        assert int(row["length"]) == int(row["food"]) + 3
        assert abs(float(row["return"]) - sum(float(row[k]) for k in row if k.startswith("reward_"))) < 1e-8
    figures(rows, args.output_dir / "assets")
    random, heuristic = summary["policies"]["random"], summary["policies"]["heuristic"]
    table = "\n".join(f"| {label} | {s['episodes']} | {s['mean_food']:.2f} | {s['episode_food_std']:.2f} | {s['mean_length']:.2f} | {s['mean_occupancy']:.2%} | {s['mean_steps']:.1f} | {s['mean_return']:.3f} | {s['collisions']}/20 | {s['successes']}/20 | {s['truncations']}/20 |"
                      for label, s in [("随机", random), ("启发式", heuristic)])
    snippet = inspect.getsource(HeuristicPolicy.act)
    report = r'''# 单蛇强化学习实验报告：阶段一环境与基线

## 摘要

本阶段实现并验证了固定 10×10 单蛇环境，以及随机、启发式两种无需训练的策略。在预先登记的验证种子 1000–1019 上，每种策略运行 20 局：随机平均食物数为 **RANDOM_FOOD**，启发式为 **HEURISTIC_FOOD**。这是固定策略基线结果，尚未实施 DQN 或证明强化学习收益。

完成证据包括环境与策略测试、原始逐局记录、配置和源码哈希、复现命令、示意图与失败轨迹。本报告是后续综合报告的单蛇阶段章节；学生姓名、学号、实际个人贡献由学生填写，未虚构成员信息。

## 1. 任务建模与评价口径

棋盘不穿墙、无障碍，初始蛇为 `(5,5),(5,4),(5,3)`，朝右。动作 `0/1/2` 分别为直行/左转/右转。食物从所有空格中均匀采样。进食增长一格，撞墙或身体终止；未进食时允许进入本步腾出的尾格。长度达到 100 时立即成功，不再生成食物。每局 2000 步为外部截断；终止优先。

观测由有序蛇身地图、食物地图与上/右/下/左 one-hot 朝向组成，按行展平为 204 维 float32。身体先后顺序决定尾格何时腾空，不能仅靠普通占用图替代。

![初始观测编码](assets/observation.png)

图中非零值展示头到尾的编码；食物图与方向向量单独拼接。该观测保留底层游戏状态，2000 步外部采样预算不属于限时获胜任务，因此未编码剩余时间。

即时奖励及优化目标定义为：

$$
r_{t+1}=-0.001+I_{\rm eat}-I_{\rm collision}+10I_{\rm full},\qquad
J(\pi)=\mathbb E_\pi\left[\sum_{t=0}^{T-1}\gamma^t r_{t+1}\right].
$$

当前固定策略没有优化上述目标。评估主指标为单局食物数；辅助指标为最终长度、占用率、步数、不折扣奖励和、碰撞率、满盘成功数、截断率。终止或截断后的奖励不再累计。技术异常会单独写入失败文件并中止，不静默删除。

## 2. 环境与软件架构

```mermaid
flowchart LR
    C[协议与种子配置] --> E[SingleSnakeEnv]
    E -->|204 维观测| P[随机 / 启发式策略]
    P -->|动作 0 / 1 / 2| E
    E -->|奖励与诊断信息| L[逐局日志]
    E --> V[Pygame 桌面与 RGB 渲染]
    L --> R[统计与 Markdown 报告]
```

图中策略只读取观测，诊断信息走日志通路；渲染器不改变游戏逻辑。环境每次返回新数组，避免后续训练时污染经验池。碰撞终局保留最后合法棋盘，最后动作与结束原因描述碰撞尝试，终局观测仍属于声明的空间。

```mermaid
flowchart TD
    A[reset：固定出生与种子生成食物] --> B[策略读取观测]
    B --> C[选择相对动作]
    C --> D[从旧状态计算候选头和碰撞集合]
    D --> E{发生碰撞?}
    E -->|是| F[碰撞终止，奖励含碰撞项]
    E -->|否| G[移动身体；进食则增长与补食物]
    G --> H{满盘或时间耗尽?}
    H -->|满盘| I[成功终止]
    H -->|仅时间耗尽| J[时间截断]
    H -->|否| B
```

本图是阶段一交互流程，尚未加入网络更新。未来 DQN 会在交互记录后添加经验回放与梯度更新，不修改既有基础规则。

## 3. 基线算法与实际代码

随机策略从三个动作等概率采样，包括危险动作；动作随机源与环境随机源分离。启发式只考虑本步可达的安全格，选择到当前食物曼哈顿距离最小者：

$$
d((r,c),(r_f,c_f))=|r-r_f|+|c-c_f|.
$$

并列按直行、左转、右转；无安全动作时直行。以下片段直接从当前实现提取：

```python
HEURISTIC_CODE
```

启发式排除立即碰撞是其算法定义。随机策略和未来基础 DQN 不额外屏蔽危险动作，因此报告不会将启发式的提升单独归因于距离选择；它同时使用了一步避险规则。

## 4. 实验设计、结果与局限

两个策略在相同的 20 个验证环境种子上各运行一次。随机策略的每局动作种子由配置中的固定基数和环境种子派生。不同策略改变身体和空格集合，因此相同环境种子并不保证之后食物的绝对坐标序列相同。

| 策略 | 局数 | 平均食物数 | 局间样本标准差 | 平均长度 | 平均占用率 | 平均步数 | 平均奖励和 | 碰撞 | 满盘 | 截断 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
RESULT_TABLE

上表标准差描述同一固定策略在 20 个回合之间的波动，**不是跨训练种子的标准差**。本阶段没有网络训练、模型选择或独立训练重复；不做强显著性声明。单次耗时仅用于本机复现参考。

![两种固定策略逐局食物数](assets/baseline_food.png)

图中每个点代表一个验证回合，圆点为随机、三角为启发式。分布表明启发式在本验证列表上取得更高食物数，但这并不保证满盘，也不代表其在任意场景或对任何学习算法都占优。正式测试种子 10000–10019 未使用。

### 失败案例与机制

种子 1000 的启发式在第 66 步发生身体碰撞，此前吃到 10 个食物。轨迹见 TRACE_LINK。第 65 步之后，蛇头位于 `(4,7)`，朝右，食物在 `(3,2)`；直行、左转、右转的候选头分别为 `(4,8)`、`(3,7)`、`(5,7)`，三者都是本步不会腾空的身体格。按协议只能回退为直行，因此下一步碰撞。

![失败时保留的最后合法棋盘](PREVIEW_LINK)

画面中的头部白点指向右侧，尾部小圆点在上方；候选头所在的三个邻格都被身体围住。策略只最小化到当前食物的距离，没有判断未来可达空间或完整逃生路径；这段回放提供了对“局部避险不能保证长期安全”的具体解释。

这解释了为什么“当前能安全接近食物”不等于长期安全。第一阶段只确认现象与规则逻辑，尚未用受控实验区分长期规划、探索和奖励设计的影响。

## 5. 验证与复现

规则测试涵盖生成食物、增长、相对转向、尾格腾空、墙体/自身碰撞、满盘、终止/截断优先级、观察副本、随机重现和环境空间。基线测试涵盖同分顺序、全危险回退和未屏蔽的随机动作。接口检查通过不替代这些规则案例。

运行环境：Python PYTHON_VERSION，Gymnasium GYM_VERSION，NumPy NUMPY_VERSION，Pygame-ce PYGAME_VERSION；本阶段使用 CPU，无网络参数更新。精确依赖在 `requirements-lock.txt`。基础环境吞吐另见 `data/benchmark.json`，不把不含优化器的吞吐用作 DQN 训练时长保证。

从仓库根目录执行：

```powershell
snake/.venv/Scripts/python -m pytest -c snake/pytest.ini snake/checks
snake/.venv/Scripts/python -m snake.evaluate --split validation
snake/.venv/Scripts/python -m snake.build_report
snake/.venv/Scripts/python -m snake.demo --policy heuristic --seed 1000
```

原始来源：DATA_DIRECTORY。实验来源哈希 `SOURCE_HASH`，配置哈希 `CONFIG_HASH`；manifest 内保存每个源码文件哈希和配置快照。图表由 `build_report.py` 从 CSV 重建，数值与 summary 对账。未来若环境或算法源码变化，应保留旧运行并创建新目录，不能覆写旧实验来源。

## 6. 后续工作与个人贡献

下一阶段补充张量、MLP、梯度、Q 值与 TD 目标讲义，手写 DQN 核心。Double DQN 对照、三种子训练、验证模型选择、最终测试和正式学习曲线均未开展。

待验证的问题：DQN 是否在相同规则下超过随机基线？是否学到比一步启发式更好的长期行为？Double DQN 在受控预算下是否稳定改善？这些问题须以真实训练结果回答。

环境、基线、测试、材料和报告由 AI 辅助编写与运行。学生应独立阅读、完成复盘练习、复现实验，并记录实际修改与理解；当前不声明学生已经独立实现或完成答辩。正式报告的成员、学号、分工与个人贡献待据实填写。

## 参考资料

- 课程依据：仓库 `outputs/强化学习大作业概述与开发指南.md` 第 2.1、4、6、7 节。
- [Gymnasium 环境接口](https://gymnasium.farama.org/api/env/) 与 [自定义环境](https://gymnasium.farama.org/introduction/create_custom_env/)。
- [Gymnasium 时间限制](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)。
- [Pygame-ce 官方文档](https://pyga.me/docs/)。
'''
    substitutions = {"RANDOM_FOOD": f"{random['mean_food']:.2f}",
                     "HEURISTIC_FOOD": f"{heuristic['mean_food']:.2f}", "RESULT_TABLE": table,
                     "HEURISTIC_CODE": snippet.rstrip(), "PYTHON_VERSION": manifest["python"],
                     "GYM_VERSION": manifest["dependencies"]["gymnasium"],
                     "NUMPY_VERSION": manifest["dependencies"]["numpy"],
                     "PYGAME_VERSION": manifest["dependencies"]["pygame-ce"],
                     "SOURCE_HASH": manifest["source_sha256"], "CONFIG_HASH": manifest["config_sha256"],
                     "DATA_DIRECTORY": args.data_dir.resolve().as_posix(),
                     "TRACE_LINK": "[逐步轨迹](" + Path(os.path.relpath(ROOT / "reports/data/heuristic_seed1000.jsonl", args.output_dir)).as_posix() + ")",
                     "PREVIEW_LINK": Path(os.path.relpath(ROOT / "reports/assets/demo_preview.png", args.output_dir)).as_posix()}
    for key, value in substitutions.items():
        report = report.replace(key, value)
    (args.output_dir / "stage1_report.md").write_text(report, encoding="utf-8")
    print(f"Report: {args.output_dir / 'stage1_report.md'}")


if __name__ == "__main__":
    main()

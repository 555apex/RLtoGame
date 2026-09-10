# 实验数据与日志说明

这些文件是报告的原始证据与可重建数据，不是模拟成绩。新读者优先阅读 [完整报告](../single_snake_report.md) 和 [逐种子解读](../single_snake_findings.md)，再按本页追溯数据。

## 最先查看哪几个文件

| 路径 | 用途 |
| --- | --- |
| [final_experiment_v1/test/episodes.csv](final_experiment_v1/test/episodes.csv) | 220 局独立测试逐局数据 |
| [final_experiment_v1/test/summary.json](final_experiment_v1/test/summary.json) | 各模型统计与跨三个训练种子的汇总 |
| [final_experiment_v1/freeze.json](final_experiment_v1/freeze.json) | 测试前的配置、模型、核心代码哈希与选模记录 |
| [final_experiment_v1/training](final_experiment_v1/training) | 六次训练的日志、配置和实际源码快照 |
| [weights_manifest.json](weights_manifest.json) | initial / best / last / resume 的文件校验值；不包含模型本体 |
| [checks_stage3.xml](checks_stage3.xml) | 82 项自动检查的 JUnit 记录 |
| [final_report_qa.json](final_report_qa.json) | 预算、种子、奖励、来源与统计对账结果 |

## 一次训练目录里有什么

六个归档目录按 `dqn_seed0/1/2` 和 `double_dqn_seed0/1/2` 命名。原始运行目录在开发机器 `snake/runs`，这里是可提交的精简副本；不复制权重、回放池或虚拟环境。

| 文件 | 粒度与主要字段 |
| --- | --- |
| `manifest.json` | 一次运行：算法、训练种子、配置、协议、设备、依赖、源码哈希、创建时间 |
| `source_snapshot.json` | 此运行真实使用的 Python 源文件内容，与 manifest 中各文件哈希对应 |
| `episodes.jsonl` | 每个**已结束训练回合**一行：累计 step、episode、env_seed、epsilon、food、steps、return、终局和奖励分量 |
| `updates.jsonl` | 每 1000 环境步窗口：更新总次数、窗口更新次数、平均损失、裁剪前梯度范数、Q 值和绝对 TD 误差 |
| `validation.jsonl` | 初始及每 10000 步的验证汇总，每点固定 20 局 |
| `validation_episodes.jsonl` | 具体验证回合，每行含对应训练 step 和环境种子 |
| `events.jsonl` | 启动、恢复、保存和完成事件；发生技术异常时单独记录失败 |
| `summary.json` | 预算、更新次数、最优选模步数、耗时、内存、末尾未结束回合、存档校验值 |

JSONL 每行是独立 JSON 对象，适合逐行读取。`episodes.jsonl` 不包含预算截止时尚未结束的回合，该部分在 `summary.json/partial_episode`：**已结束回合步数之和 + partial_episode.steps = 300000**。

训练 summary 中 `final_test_used: false` 表示训练及选模未使用测试集；最终测试另存于 `final_experiment_v1/test`，不能把这项误读为整个项目未运行测试。

## 逐局测试 CSV 字段

| 字段 | 含义 |
| --- | --- |
| `policy` | random / heuristic / initial / dqn / double_dqn |
| `model_id`、`training_seed` | 冻结模型身份及训练种子；固定基线没有训练种子 |
| `env_seed`、`policy_seed` | 环境随机源与独立动作随机源；测试环境为 10000–10019 |
| `food`、`length`、`occupancy` | 食物数、最终长度、长度除以 100；长度为食物数 + 3 |
| `steps`、`return` | 回合动作数、未折扣累计奖励 |
| `terminated`、`truncated` | 真正终止与外部时间截断，二者分开 |
| `success`、`end_reason` | 是否满盘，以及碰撞/截断/满盘原因 |
| `reward_step_cost`、`reward_food`、`reward_collision`、`reward_full_board` | 各奖励分量的回合累计值，加总等于 return |

CSV 布尔值是 `True/False` 字符串。用 Python 读取时不要直接写 `bool("False")`，应明确比较字符串或转换类型。

## 为什么是 220 局

- 六个 best 模型，每模型 20 局：120 局。
- 三个初始网络，每模型 20 局：60 局。同种子的 DQN 与 Double DQN 初始张量相同，共用对照，不重复算六个独立模型。
- 随机和启发式，各 20 局：40 局。

先计算每个训练模型的 20 局均值，再计算三个模型均值的平均与样本标准差。`episode_food_std` 表示回合间波动；`seed_mean_food_std` 表示训练种子均值的波动，不能混用。标准差不是 95% 置信区间。

## 如何查看自己的日志

在 Python 中读取一份新训练的验证记录，例如将以下内容保存为 snake 下的个人练习脚本并在仓库根目录执行，或在虚拟环境 Python 交互窗口执行：

```python
import json
from pathlib import Path

run = Path("snake/runs/my_first_dqn")
rows = [json.loads(line) for line in
        (run / "validation.jsonl").read_text(encoding="utf-8").splitlines()]
for row in rows:
    print(row["step"], row["mean_food"], row["truncations"])
```

若要观察自己的训练曲线，可接着运行：

```python
import os
os.environ.setdefault("MPLCONFIGDIR", "snake/.cache/matplotlib")
import matplotlib.pyplot as plt

plt.plot([r["step"] for r in rows], [r["mean_food"] for r in rows], marker="o")
plt.xlabel("Training environment steps")
plt.ylabel("Validation mean food")
plt.tight_layout()
plt.savefig(run / "my_validation_curve.png", dpi=150)
plt.close()
```

短练习只有少数验证点，不能当作正式训练曲线。示例仅画你指定的运行，不修改冻结数据。

## 重建正式报告

在仓库根目录、安装完整依赖后执行：

```powershell
snake/.venv/Scripts/python -m snake.build_final_report
```

脚本从归档检查 180 万步预算、3720 条验证与 220 条测试记录、种子拆分、奖励和源码快照哈希，随后生成报告与图表；不加载权重，不启动新训练，也不重新访问测试环境。

部分 manifest 和轨迹保留开发机器的绝对路径，用于记录来源，不要求读者使用相同用户名或目录。可执行命令使用仓库相对路径。`.gitattributes` 固定源码与配置为 LF 换行，同时对原始实验数据关闭换行转换，保留其实际字节及 SHA-256。

## 历史数据与当前数据的关系

- `baselines_validation_v1` 是阶段一验证基线，不是最终测试。
- `dqn_seed0_training` 是阶段二 DQN 种子 0 原始归档；正式矩阵复用该运行，没有再次训练。
- `dqn_seed0_best_validation` 与 `..._v2` 保留一次评估元数据修正前后记录，成绩一致，不当作两次独立实验。
- `dqn_device_benchmark.json` 为初测，`dqn_device_benchmark_isolated.json` 为无测试进程重叠的复测；设备选型采用后者。
- `*_seed1000.jsonl` 与 `dqn_seed0_loop1005.json` 是验证环境下的演示/诊断轨迹，不是测试调参依据。
- `delivery_qa.json` 是原开发机器交付检查记录，包含本地权重与 GIF 验证；源码克隆不包含这些二进制，不能直接重跑完整交付检查。

主报告以 `final_experiment_v1` 为正式实验依据，历史记录用于解释真实开发过程，不删掉失败来美化结果。

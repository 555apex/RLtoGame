# 单蛇强化学习

自行实现的 Gymnasium 贪吃蛇环境，配套随机 / 启发式策略、手写 DQN / Double DQN、中文教学和可复现的实验材料。

[返回首页](../README.md) · [实验报告](reports/single_snake_report.md) · [数据字典](reports/data/README.md) · [答辩练习](learning/03_exercises_and_defense.md)

## 当前能力与边界

- 固定 10×10 棋盘，204 维完整游戏观测，三个相对动作，规则与渲染分离。
- 支持训练、评估和演示；完整存档可恢复回合中途的环境、经验池、优化器与随机状态。
- 六次正式训练共 180 万环境步，冻结后测试 220 局，82 项自动检查通过。
- 测试食物数：DQN **0.850 ± 0.218**，Double DQN **0.850 ± 0.229**；初始网络 0.10，随机 0.15，启发式 19.10。± 为三个训练模型均值的样本标准差。
- 三个训练种子都有学习增益，但绝对成绩较低，本次 Double DQN 未提高平均食物数。双蛇尚未实现。

## 先选一条体验路线

| 路线 | 需要什么 | 运行后能看到什么 |
| --- | --- | --- |
| 直接玩游戏 / 看基线 | 阶段一依赖 | 桌面棋盘、动作、奖励分量、得分和结束原因 |
| 阅读真实实验 | 克隆仓库即可阅读 Markdown / PNG / 日志 | 六次训练、220 局测试及失败分析 |
| 自行训练网络 | 完整训练依赖 | 新运行目录、模型存档、训练与验证日志 |
| 看学习策略实时游玩 | 完整依赖 + 自己训练或取得的模型 | 模型自主决策，可暂停逐步检查 |

**源码仓库不包含模型二进制。** 原开发机器的 `snake/runs/*/checkpoints/*.pt` 已存在，但新克隆没有这些文件。模型附件尚未发布；校验清单在 [weights_manifest.json](reports/data/weights_manifest.json)。阶段报告与演示稿中引用的原始模型路径属于开发机器记录。

## 安装与快速体验

以下命令在仓库根目录的 Windows PowerShell 执行，实测 Python 3.13.6。显式调用虚拟环境 Python，不必激活环境或修改 PowerShell 执行策略。

```powershell
python -m venv snake/.venv
snake/.venv/Scripts/python -m pip install -r snake/requirements-stage1-lock.txt
snake/.venv/Scripts/python -m snake.demo --policy heuristic --seed 1000
```

其他策略：

```powershell
snake/.venv/Scripts/python -m snake.demo --policy human --seed 1000
snake/.venv/Scripts/python -m snake.demo --policy random --seed 1000
```

空格暂停/继续，N 单步，R 同种子重置，Esc 退出。人工模式初始暂停，↑ / ← / → 分别执行一次直行 / 左转 / 右转，方向相对蛇头。策略模式按 N 执行模型选择的下一动作。

![学习策略的状态面板示例](reports/assets/double_dqn_demo_preview.png)

窗口显示食物数、长度、步数、占用率、动作、即时奖励及各分量。终局保留最后合法棋盘，碰撞由结束原因和最后动作解释。图片来自已训练 Double DQN 的验证演示，不代表启发式或平均测试成绩。

## 安装完整训练环境

在已有虚拟环境基础上执行：

```powershell
snake/.venv/Scripts/python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
snake/.venv/Scripts/python -m pip install -r snake/requirements-lock.txt
snake/.venv/Scripts/python -m pip check
```

该锁定环境已在 Windows / Python 3.13.6 验证。PyTorch 2.7.1+cu128 可在本机 RTX 5070 Ti 上更新；小型单环境 MLP 的 CPU 单线程实测更快，因此正式训练用 CPU。CUDA 构建也能执行 CPU 训练。其他操作系统和依赖组合未验证。

## 训练 → 评估 → 实时演示

先跑一个 2000 步练习，确认全链路；它用于学习命令，不代表高质量策略：

```powershell
snake/.venv/Scripts/python -m snake.train --seed 0 --device cpu --steps 2000 --output snake/runs/my_first_dqn
snake/.venv/Scripts/python -m snake.evaluate --checkpoint snake/runs/my_first_dqn/checkpoints/best.pt --split validation
snake/.venv/Scripts/python -m snake.demo --checkpoint snake/runs/my_first_dqn/checkpoints/best.pt --seed 1000
```

训练过程中每次验证会打印当前步数、平均食物数和选中的 best 步数。练习模型可能几乎不会觅食。默认完整训练为 300000 环境步，开发机器单次运行段约 102–143 秒，排除初始导入和 Trainer 构造；其他机器用时不同。

训练 Double DQN 并评估：

```powershell
snake/.venv/Scripts/python -m snake.train --config snake/configs/double_dqn.json --seed 0 --device cpu --output snake/runs/my_double_dqn
snake/.venv/Scripts/python -m snake.evaluate --checkpoint snake/runs/my_double_dqn/checkpoints/best.pt --split validation
```

继续前面的短练习：

```powershell
snake/.venv/Scripts/python -m snake.train --seed 0 --steps 4000 --resume snake/runs/my_first_dqn/checkpoints/resume.pt --output snake/runs/my_first_dqn_resumed
```

`--steps` 是累计目标，不是追加步数；输出目录必须是新的，避免覆盖旧证据。恢复时保留算法和超参数，只能改变总步数。

| 模型文件 | 用途 |
| --- | --- |
| `initial.pt` | 未更新网络，检查学习前行为 |
| `best.pt` | 验证平均食物数最大，同分保留较早模型 |
| `last.pt` | 训练末次网络，不一定优于 best |
| `resume.pt` | 完整续训状态，约 83 MB；用于 `--resume` |

`initial/best/last.pt` 用于评估和演示的 `--checkpoint`，不能与 `resume.pt` 混用。所有运行日志字段见 [实验数据说明](reports/data/README.md)。

## 可视化、报告与导出

已有正式图表可以直接阅读；以下命令只读归档，不训练、不访问测试环境，也不需要模型二进制：

```powershell
snake/.venv/Scripts/python -m snake.build_final_report
```

输出为 `snake/reports/single_snake_report.md`、结果解读和 `reports/assets/final_*.png`。它重建的是**已归档正式实验**，不会自动读取你刚刚训练的新练习目录。

各命令的参数可以随时查询：

```powershell
snake/.venv/Scripts/python -m snake.train --help
snake/.venv/Scripts/python -m snake.evaluate --help
```

新练习的 `validation.jsonl` 记录每个验证点的食物均值，`updates.jsonl` 记录更新诊断；具体读取示例见数据字典。不要把新练习结果混入原正式报告。

导出 GIF 与动作轨迹：

```powershell
snake/.venv/Scripts/python -m snake.demo --policy heuristic --seed 1000 --headless --max-frames 120 --gif snake/runs/my_heuristic.gif
```

同时产生同名 JSONL，记录初始观测、逐步动作、奖励和结束原因。将 `--policy heuristic` 换成 `--checkpoint 你的模型路径` 可导出学习策略。`--max-frames` 是预览长度，达到上限记为 `preview_limit`，不会伪造环境的时间截断。仓库中的静态图可直接查看，历史 GIF 可按 [演示讲解稿](learning/04_demo_walkthrough.md) 重建。

## 检查与正式复现

```powershell
snake/.venv/Scripts/python -m pytest -c snake/pytest.ini snake/checks
snake/.venv/Scripts/python -m snake.learning.dqn_math
snake/.venv/Scripts/python -m snake.evaluate --split validation
```

完整检查需要安装训练依赖。规则检查包括腾空尾格、满盘、2000 步截断、奖励叠加、复制观测与同种子复现；算法检查包含手算目标、验证隔离与恢复。开发机器的自动化结果不代替学生亲自操作桌面的验收。

正式训练矩阵为 `configs/experiment_matrix.json`。对于**新的克隆目录**，完整复现实验可以先创建 DQN 种子 0 的运行，再执行余下矩阵：

```powershell
# 原开发机器已有这些运行，请勿重复执行第一条
snake/.venv/Scripts/python -m snake.train --seed 0 --device cpu --output snake/runs/dqn_seed0_stage2
snake/.venv/Scripts/python -m snake.run_experiments
snake/.venv/Scripts/python -m snake.final_experiment freeze --output snake/runs/my_reproduction
snake/.venv/Scripts/python -m snake.final_experiment test --output snake/runs/my_reproduction
```

矩阵入口会校验并复用完成的运行，拒绝覆盖未完成目录。新运行与历史记录保留各自源码标识，不承诺跨平台逐位复现。原正式数据已存在于 `reports/data/final_experiment_v1`，所以新复现必须使用新的冻结/测试输出目录。

训练环境种子 0–999、验证 1000–1019、测试 10000–10019。正式测试前冻结配置、模型和源码；之后不根据测试调参。奖励等练习使用独立配置与输出目录，并遵守基础评测规则。当前基础协议会拒绝只改配置、不改规则实现的不一致设置。

## 源码阅读地图

| 文件 / 目录 | 职责 | 建议先理解 |
| --- | --- | --- |
| `envs/single_snake.py` | 游戏规则、Gymnasium 接口、204 维编码 | 一步动作如何改变棋盘 |
| `envs/rendering.py` | 独立 Pygame 渲染 | 画面如何对应观测和 info |
| `agents/baselines.py` | 随机、启发式 | 仅读观测，危险动作筛选区别 |
| `agents/dqn.py` | MLP、两种 TD 目标、更新与模型加载 | 公式到张量代码 |
| `agents/replay.py` | 环形经验池 | 为什么复制经验与随机采样 |
| `training.py` | 训练、独立验证、完整恢复 | 三类随机源与步数调度 |
| `final_experiment.py` | 冻结、最终测试与三种子统计 | 测试为何不能选模 |
| `checks/` | 环境、算法和接口检查 | 怎样证明关键机制正确 |
| `learning/` | 中文讲义、手算和复盘 | 从 Python 逐步进入 RL |
| `reports/` | 报告、图表、日志、来源与校验 | 成绩能否由原始记录还原 |

按 [第一课](learning/01_rl_and_snake.md) → [第二课](learning/02_tensors_and_dqn.md) → [第三课](learning/03_double_dqn_and_experiments.md) → [答辩复盘](learning/03_exercises_and_defense.md) 阅读。阶段一、二报告保留当时状态；当前结论以完整单蛇报告为准。

## 常见问题

- **找不到模型文件**：新克隆没有原开发机器的 runs；先训练自己的模型，或等待模型附件发布。
- **输出目录已存在**：换新名称，保留之前记录；不要把旧日志和新日志混合。
- **缺少 torch / gymnasium**：确认命令使用 `snake/.venv/Scripts/python`，并按对应路线安装依赖。
- **窗口不动**：人工模式初始暂停；按方向键单步或空格继续。策略已经终局时按 R 重置。
- **模型一直绕圈**：查看食物数与截断原因。当前模型确实存在这种失败，不能用更长存活时间代替觅食成绩。
- **完整交付审计提示缺少权重或 GIF**：`snake.check_delivery` 面向保有原始权重和录像的开发机器；新克隆可运行 pytest 与报告重建检查。

代码、实验执行与文档使用 AI 辅助。个人贡献按实际学习、修改和复现情况填写。更新概要见 [CHANGELOG](../CHANGELOG.md)，详细过程见 [PROGRESS](../PROGRESS.md)。

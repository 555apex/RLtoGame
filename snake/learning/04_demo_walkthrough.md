# 桌面演示与失败案例讲解稿

演示使用验证环境种子，不根据测试成绩挑选录像。先读报告的四方法测试结果，明确这一段只说明具体行为，不能替代统计结论。

## 1. 先操作游戏：状态、动作和奖励

在仓库根目录执行：

```powershell
snake/.venv/Scripts/python -m snake.demo --policy human --seed 1000
```

人工模式开始时暂停。按 ↑ 直行、← 左转、→ 右转，方向相对蛇头；空格切换运行/暂停，N 单步直行，R 重置同一种子，Esc 退出。解释：蛇头朝右时，左转是向棋盘上方前进，不是向左边移动。

讲解词：“每次动作前进一格。普通移动扣 0.001，吃食物时再加 1。身体图有顺序，尾部本步离开，因此进入腾空尾格可以合法。窗口的奖励分量帮助我核对这些规则。”

## 2. 展示冻结的 Double DQN 模型

```powershell
snake/.venv/Scripts/python -m snake.demo --checkpoint snake/runs/double_dqn_seed1_stage3/checkpoints/best.pt --seed 1000 --fps 5
```

该模型是测试前按验证均值选出的展示模型：训练种子 1，第 220000 步；不是训练最后一次，也不是测试成绩最高的模型。此验证回合吃到 3 个食物，在第 37 步身体碰撞。按空格暂停，再按 N 逐步观察末尾动作。屏幕终局保留最后合法棋盘，碰撞尝试由“最后动作”和“结束原因”说明。

讲解词：“它表现出一定觅食能力，但会撞到自己的身体。最终测试食物均值只有 0.80；这一局的 3 个食物不能当作平均能力。”

无需开窗口也能查看随源码提供的 [静态预览](../reports/assets/double_dqn_demo_preview.png) 和 [完整动作轨迹](../reports/data/double_dqn_seed1_seed1000.jsonl)。开发机器还保存了 `reports/assets/double_dqn_seed1_seed1000.gif`，该二进制不包含在源码克隆中。新克隆需先训练或取得对应模型，再导出自己的版本并使用新文件名：

```powershell
snake/.venv/Scripts/python -m snake.demo --checkpoint snake/runs/double_dqn_seed1_stage3/checkpoints/best.pt --seed 1000 --headless --max-frames 120 --gif snake/runs/my_double_demo.gif
```

会同时生成 `my_double_demo.jsonl`。其中 reset 行记录初始观测、环境种子和模型哈希；step 行包含动作、奖励、终止/截断和下一观测；demo_end 说明结束原因。`max-frames` 是预览上限，达到上限记为 `preview_limit`，不会伪造任务截断。

## 3. 展示 DQN 的循环失败

```powershell
snake/.venv/Scripts/python -m snake.demo --checkpoint snake/runs/dqn_seed0_stage2/checkpoints/best.pt --seed 1005 --fps 12
```

从第 7 步开始，每 24 步回到相同观测；食物一直没有吃到。现场不必等完整 2000 步，展示一个周期后打开 [循环图](../reports/assets/dqn_loop.png)，并说明 [完整诊断](../reports/data/dqn_seed0_loop1005.json) 记录了全部 2000 步、0 个食物和时间截断。

讲解词：“它在相同观测下重复相同动作，且没有触发新食物采样，所以形成循环。存活时间很长，任务分数却没有改善。这是保留失败案例的价值。”

## 4. 对照启发式与实际结果

```powershell
snake/.venv/Scripts/python -m snake.demo --policy heuristic --seed 1000 --fps 8
```

该验证回合 66 步、10 个食物后身体碰撞。启发式只考虑一步安全和到食物的距离，也可能把自己围住。其正式测试均值为 19.10，明显高于目前网络，但本次仍无满盘成功。

最后打开 [结果解读](../reports/single_snake_findings.md)，指出 DQN 与 Double DQN 的均值都为 0.85，相同训练种子的差值方向不一致。回答“改进为什么没有更强”时，先讲算法改变了什么，再陈述实际结果与待验证假设，避免把设计动机当作实验结论。

## 5. 学生亲自验收

- [ ] 能独立启动上述命令，暂停、单步、重置并关闭窗口。
- [ ] 能解释一次转移的动作含义和奖励分量。
- [ ] 能从 JSONL 找到最后一个 step，说明观测属于动作之前还是之后。
- [ ] 能区分验证演示、验证选模和独立测试。
- [ ] 能说明 AI 辅助的内容和自己实际完成的复现工作。

这些项目需要学生实际操作后勾选；自动化的无窗口验证不能代替你的亲自验收。

"""刻意让两张网络偏好的动作不同，避免错误实现也通过目标值检查。"""

import ast
import copy
import json
from pathlib import Path

import torch

from snake.agents.dqn import DQNAgent, double_dqn_targets, dqn_targets
from snake.training import load_dqn_config


def test_double_target_separates_selection_evaluation():
    online = torch.tensor([[4., 2., 1.]] * 3, requires_grad=True)
    target = torch.tensor([[1., 8., 3.]] * 3, requires_grad=True)
    rewards = torch.tensor([1., 1., 1.])
    ended = torch.tensor([False, True, False])  # 第三项代表时间截断，仍 bootstrap。
    actual = double_dqn_targets(rewards, ended, online, target, 0.9)
    torch.testing.assert_close(actual, torch.tensor([1.9, 1., 1.9]))
    torch.testing.assert_close(dqn_targets(rewards, ended, target, 0.9), torch.tensor([8.2, 1., 8.2]))
    assert not actual.requires_grad


def test_configs_differ_only_in_algorithm_and_share_initial_weights():
    dqn = load_dqn_config()
    double = load_dqn_config(Path("snake/configs/double_dqn.json"))
    assert {k for k in dqn if dqn[k] != double[k]} == {"algorithm"}
    for seed in (0, 1, 2):
        first = DQNAgent(dqn, seed)
        second = DQNAgent(double, seed)
        for k, v in first.online.state_dict().items():
            assert torch.equal(v, second.online.state_dict()[k])


def test_existing_dqn_run_remains_algorithmically_compatible():
    # 读取本项目阶段二真实存档，只执行所需定义；不运行任何归档入口或实验。
    snapshot = json.loads(Path("snake/reports/data/dqn_seed0_training/source_snapshot.json").read_text(encoding="utf-8"))
    old = ast.parse(snapshot["agents/dqn.py"])
    keep = [node for node in old.body if isinstance(node, (ast.Import, ast.ImportFrom)) or
            getattr(node, "name", "") in ("QNetwork", "dqn_targets", "DQNAgent")]
    namespace = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "stage2_archived_dqn", "exec"), namespace)
    torch.set_num_threads(1)
    config = load_dqn_config()
    archived, current = namespace["DQNAgent"](config, 0), DQNAgent(config, 0)
    generator = torch.Generator().manual_seed(17)
    batch = {"observations": torch.rand(64, 204, generator=generator),
             "next_observations": torch.rand(64, 204, generator=generator),
             "rewards": torch.randn(64, generator=generator),
             "actions": torch.randint(3, (64,), generator=generator),
             "terminated": torch.arange(64) % 3 == 0}
    for _ in range(5):
        assert archived.update(copy.deepcopy(batch)) == current.update(copy.deepcopy(batch))
        for k, v in archived.online.state_dict().items():
            assert torch.equal(v, current.online.state_dict()[k])
    # 学习循环和环境逻辑不变，因此复用已完成的 seed 0，无需额外消耗 300000 步。
    previous = ast.parse(snapshot["training.py"])
    present = ast.parse(Path("snake/training.py").read_text(encoding="utf-8"))
    classes = [next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Trainer")
               for tree in (previous, present)]
    assert ast.dump(classes[0]) == ast.dump(classes[1])

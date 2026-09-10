"""从公式到一次梯度更新的手写 DQN，不使用现成 RL 训练器。"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, hidden_sizes=(128, 128)):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(204, hidden_sizes[0]), nn.ReLU(),
                                    nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU(),
                                    nn.Linear(hidden_sizes[1], 3))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.layers(observation)


@torch.no_grad()
def dqn_targets(rewards: torch.Tensor, terminated: torch.Tensor,
                next_q_values: torch.Tensor, gamma: float) -> torch.Tensor:
    """y = r + gamma * (1 - terminated) * max Q_target(s', a')。

    truncated 不进入掩码：外部时间预算用尽时，底层游戏仍有后续价值。
    """
    next_values = next_q_values.max(dim=1).values
    return rewards + gamma * (~terminated).float() * next_values


@torch.no_grad()
def double_dqn_targets(rewards: torch.Tensor, terminated: torch.Tensor,
                       online_next_q: torch.Tensor, target_next_q: torch.Tensor,
                       gamma: float) -> torch.Tensor:
    """在线网络选动作，目标网络评估该动作；其余目标语义与 DQN 一致。"""
    next_actions = online_next_q.argmax(dim=1, keepdim=True)
    next_values = target_next_q.gather(1, next_actions).squeeze(1)
    return rewards + gamma * (~terminated).float() * next_values


def epsilon_at(step: int, config: dict) -> float:
    fraction = min(max(step, 0) / config["epsilon_decay_steps"], 1.0)
    return config["epsilon_start"] + fraction * (config["epsilon_end"] - config["epsilon_start"])


class DQNAgent:
    def __init__(self, config: dict, seed: int, device: str = "cpu"):
        self.config = copy.deepcopy(config)
        self.device = torch.device(device)
        torch.manual_seed(seed)
        self.online = QNetwork(config["hidden_sizes"]).to(self.device)
        self.target = copy.deepcopy(self.online).eval()
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config["learning_rate"])
        self.exploration_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        self.updates = 0

    def act(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        # 纯贪心评估不消耗探索 RNG；危险动作仍然属于同一个三动作空间。
        if epsilon > 0 and self.exploration_rng.random() < epsilon:
            return int(self.exploration_rng.integers(3))
        with torch.no_grad():
            tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            return int(self.online(tensor.unsqueeze(0)).argmax(dim=1).item())

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        with torch.no_grad():
            if self.config["algorithm"] == "double_dqn":
                targets = double_dqn_targets(batch["rewards"], batch["terminated"],
                                             self.online(batch["next_observations"]),
                                             self.target(batch["next_observations"]), self.config["gamma"])
            else:
                targets = dqn_targets(batch["rewards"], batch["terminated"],
                                      self.target(batch["next_observations"]), self.config["gamma"])
        # 每个样本只更新实际执行动作的 Q 值；gather 返回 [B,1]，压成 [B] 对齐目标。
        predictions = self.online(batch["observations"]).gather(1, batch["actions"].unsqueeze(1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(predictions, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("TD 损失非有限值，停止训练并保留异常记录")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.online.parameters(), self.config["max_grad_norm"],
                                            error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        return {"loss": float(loss.detach().item()), "grad_norm_before_clip": float(grad_norm),
                "mean_q": float(predictions.detach().mean().item()),
                "mean_abs_td_error": float((targets - predictions.detach()).abs().mean().item())}

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def state_dict(self) -> dict:
        return {"online": self.online.state_dict(), "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(), "updates": self.updates,
                "exploration_rng": copy.deepcopy(self.exploration_rng.bit_generator.state)}

    def load_state_dict(self, state: dict) -> None:
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = state["updates"]
        self.exploration_rng.bit_generator.state = copy.deepcopy(state["exploration_rng"])


class DQNPolicy:
    """只推理的策略接口：加载轻量模型、回合重置、读取观测返回动作。"""

    def __init__(self, network: QNetwork, device: str = "cpu"):
        self.device = torch.device(device)
        self.network = network.to(self.device).eval()

    @classmethod
    def load(cls, checkpoint: Path | str, device: str = "cpu") -> "DQNPolicy":
        from snake.experiment import load_protocol
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("kind") != "snake_dqn_policy_v1" or payload["protocol"] != load_protocol():
            raise ValueError("演示/评估需要本项目导出的 DQN 策略存档（initial/best/last.pt）")
        if payload["config"]["algorithm"] not in ("dqn", "double_dqn"):
            raise ValueError("模型中的算法类型不受支持")
        torch.set_num_threads(payload["config"]["torch_threads"])
        network = QNetwork(payload["config"]["hidden_sizes"])
        network.load_state_dict(payload["online"])
        policy = cls(network, device)
        policy.metadata = {key: value for key, value in payload.items() if key != "online"}
        return policy

    def reset_episode(self, seed: int, agent_id: str = "snake") -> None:
        pass  # 当前 MLP 无回合记忆，评估时不使用 epsilon 探索。

    @torch.inference_mode()
    def act(self, observation: np.ndarray) -> int:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        return int(self.network(tensor.unsqueeze(0)).argmax(dim=1).item())

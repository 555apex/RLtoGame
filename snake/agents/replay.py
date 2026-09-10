"""手写固定容量环形经验池，存副本、独立随机源、可完整恢复。"""

from __future__ import annotations

import copy

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int, observation_size: int = 204):
        if capacity <= 0:
            raise ValueError("经验池容量必须为正数")
        self.capacity = capacity
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)
        # 分列数组比保存许多 Python 元组紧凑；赋值时 NumPy 复制观测值。
        self.arrays = {
            "observations": np.zeros((capacity, observation_size), dtype=np.float32),
            "next_observations": np.zeros((capacity, observation_size), dtype=np.float32),
            "actions": np.zeros(capacity, dtype=np.int64),
            "rewards": np.zeros(capacity, dtype=np.float32),
            "terminated": np.zeros(capacity, dtype=np.bool_),
            "truncated": np.zeros(capacity, dtype=np.bool_),
        }

    def __len__(self) -> int:
        return self.size

    def add(self, observation, action, reward, next_observation, terminated, truncated) -> None:
        values = (observation, next_observation, action, reward, terminated, truncated)
        for array, value in zip(self.arrays.values(), values):
            array[self.position] = value
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device | str) -> dict[str, torch.Tensor]:
        if self.size < batch_size:
            raise ValueError("有效经验数量不足以采样一个 batch")
        indices = self.rng.choice(self.size, size=batch_size, replace=False)
        return {key: torch.as_tensor(value[indices], device=device) for key, value in self.arrays.items()}

    def state_dict(self) -> dict:
        return {"capacity": self.capacity, "position": self.position, "size": self.size,
                "rng": copy.deepcopy(self.rng.bit_generator.state),
                # 只保存有效段；用 Tensor 序列化，使 torch.load(weights_only=True) 可安全加载。
                "arrays": {key: torch.from_numpy(value[:self.size].copy()) for key, value in self.arrays.items()}}

    def load_state_dict(self, state: dict) -> None:
        if state["capacity"] != self.capacity:
            raise ValueError("恢复经验池的容量必须与配置一致")
        self.position, self.size = state["position"], state["size"]
        if not 0 <= self.size <= self.capacity or not 0 <= self.position < self.capacity:
            raise ValueError("经验池存档索引无效")
        for key, values in state["arrays"].items():
            self.arrays[key][:self.size] = values.cpu().numpy()
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])

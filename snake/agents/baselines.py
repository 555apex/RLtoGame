"""无需训练的两种策略：为后续判断学习是否有效提供参照。"""

import numpy as np

from snake.envs.single_snake import ACTION_TURNS, BOARD_SIZE, DIRECTIONS, decode_observation


class RandomPolicy:
    def __init__(self):
        self.rng = np.random.default_rng()

    def reset_episode(self, seed: int, agent_id: str = "snake") -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray) -> int:
        # 故意不排除危险动作；否则比较的已不是均匀随机策略。
        return int(self.rng.integers(3))


class HeuristicPolicy:
    def reset_episode(self, seed: int, agent_id: str = "snake") -> None:
        pass  # 确定性无记忆策略，无需随机源。

    def act(self, observation: np.ndarray) -> int:
        body, food, direction = decode_observation(observation)
        if not body or food is None:
            return 0
        candidates = []
        for action, turn in enumerate(ACTION_TURNS):
            dr, dc = DIRECTIONS[(direction + turn) % 4]
            head = (body[0][0] + dr, body[0][1] + dc)
            eating = head == food
            occupied = body if eating else body[:-1]
            if (0 <= head[0] < BOARD_SIZE and 0 <= head[1] < BOARD_SIZE
                    and head not in occupied):
                distance = abs(head[0] - food[0]) + abs(head[1] - food[1])
                candidates.append((distance, action))
        # 元组排序自然实现同距离时 0→1→2 的协议；全危险时仍返回直行。
        return min(candidates)[1] if candidates else 0


def make_policy(name: str):
    if name == "random":
        return RandomPolicy()
    if name == "heuristic":
        return HeuristicPolicy()
    raise ValueError(f"阶段一仅支持 random / heuristic，收到 {name!r}")

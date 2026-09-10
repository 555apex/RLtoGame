"""单蛇基础协议：先写清状态转移，再接入学习算法。"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

BOARD_SIZE = 10
CAPACITY = BOARD_SIZE**2
# 顺时针排列，方便相对转向；坐标始终是 (行, 列)。
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
ACTION_TURNS = (0, -1, 1)  # 0 直行，1 左转，2 右转
INITIAL_BODY = ((5, 5), (5, 4), (5, 3))


def encode_observation(body: list[tuple[int, int]], food: tuple[int, int] | None,
                       direction: int) -> np.ndarray:
    """前 100 维蛇身，中间 100 维食物，最后 4 维朝向；地图按行展开。"""
    observation = np.zeros(204, dtype=np.float32)
    for order, (row, col) in enumerate(body, start=1):
        observation[row * BOARD_SIZE + col] = order / CAPACITY
    if food is not None:
        observation[CAPACITY + food[0] * BOARD_SIZE + food[1]] = 1
    observation[2 * CAPACITY + direction] = 1
    return observation


def decode_observation(observation: np.ndarray) -> tuple[list[tuple[int, int]], tuple[int, int] | None, int]:
    """策略只能从观测恢复信息，不能读取 env.body 或 info 中的诊断字段。"""
    observation = np.asarray(observation, dtype=np.float32)
    if observation.shape != (204,):
        raise ValueError("单蛇策略需要 shape=(204,) 的观测")
    body_map = observation[:CAPACITY]
    occupied = np.flatnonzero(body_map)
    ordered = occupied[np.argsort(body_map[occupied])]
    body = [divmod(int(index), BOARD_SIZE) for index in ordered]
    food_cells = np.flatnonzero(observation[CAPACITY:2 * CAPACITY])
    food = divmod(int(food_cells[0]), BOARD_SIZE) if len(food_cells) else None
    return body, food, int(observation[2 * CAPACITY:].argmax())


class SingleSnakeEnv(gym.Env):
    """固定 10×10 基础环境。max_steps 可缩短以测试边界，正式协议为 2000。"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    def __init__(self, render_mode: str | None = None, max_steps: int = 2000):
        super().__init__()
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"不支持的 render_mode: {render_mode}")
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
            raise ValueError("max_steps 必须为正整数")
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(204,), dtype=np.float32)
        self.body: list[tuple[int, int]] = []
        self.food: tuple[int, int] | None = None
        self.direction = 1
        self.steps = 0
        self.food_eaten = 0
        self.last_action: int | None = None
        self.last_reward = 0.0
        self.end_reason = "running"
        self._finished = True
        self._renderer = None
        self.reward_components = self._empty_components()

    @staticmethod
    def _empty_components() -> dict[str, float]:
        return {"step_cost": 0.0, "food": 0.0, "collision": 0.0, "full_board": 0.0}

    def _spawn_food(self) -> tuple[int, int] | None:
        # 从全部空格均匀选一次。避免“反复猜坐标”在满盘时永不结束。
        occupied = set(self.body)
        empty = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
                 if (r, c) not in occupied]
        return empty[int(self.np_random.integers(len(empty)))] if empty else None

    def _observation(self) -> np.ndarray:
        # 每次创建新数组；环境继续移动不会偷偷改写 replay buffer 中的历史。
        return encode_observation(self.body, self.food, self.direction)

    def _info(self) -> dict:
        return {"raw_score": self.food_eaten, "length": len(self.body),
                "occupancy": len(self.body) / CAPACITY, "env_frames": self.steps,
                "reward_components": self.reward_components.copy(),
                "is_success": self.end_reason == "full_board",
                "end_reason": self.end_reason}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.body = list(INITIAL_BODY)
        self.direction, self.steps, self.food_eaten = 1, 0, 0
        self.last_action, self.last_reward = None, 0.0
        self.reward_components = self._empty_components()
        self.end_reason, self._finished = "running", False
        self.food = self._spawn_food()
        if self.render_mode == "human":
            self.render()
        return self._observation(), self._info()

    def step(self, action: int):
        if self._finished:
            raise RuntimeError("回合尚未开始或已经结束，请先 reset()")
        if not self.action_space.contains(action):
            raise ValueError("动作必须为 0（直行）、1（左转）或 2（右转）")
        self.last_action = int(action)
        next_direction = (self.direction + ACTION_TURNS[int(action)]) % 4
        dr, dc = DIRECTIONS[next_direction]
        head = (self.body[0][0] + dr, self.body[0][1] + dc)
        eating = head == self.food
        # 不进食则尾格本步腾空。一定要在插入新头之前决定碰撞集合。
        occupied = self.body if eating else self.body[:-1]
        wall = not (0 <= head[0] < BOARD_SIZE and 0 <= head[1] < BOARD_SIZE)
        collision = wall or head in occupied
        self.steps += 1
        self.reward_components = self._empty_components()
        self.reward_components["step_cost"] = -0.001
        terminated = False
        if collision:
            # 保留最后合法棋盘供终局观察和回放；碰撞尝试由 action/end_reason 表示。
            terminated = True
            self.end_reason = "wall_collision" if wall else "body_collision"
            self.reward_components["collision"] = -1.0
        else:
            self.direction = next_direction
            self.body.insert(0, head)
            if eating:
                self.food_eaten += 1
                self.reward_components["food"] = 1.0
                if len(self.body) == CAPACITY:
                    terminated = True
                    self.end_reason = "full_board"
                    self.reward_components["full_board"] = 10.0
                    self.food = None
                else:
                    self.food = self._spawn_food()
            else:
                self.body.pop()
        # 真正终止优先：最后一步撞墙/满盘不会被改成普通超时。
        truncated = self.steps >= self.max_steps and not terminated
        if truncated:
            self.end_reason = "time_limit"
        self._finished = terminated or truncated
        self.last_reward = sum(self.reward_components.values())
        observation, info = self._observation(), self._info()
        if self.render_mode == "human":
            self.render()
        return observation, self.last_reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from .rendering import SnakeRenderer
            self._renderer = SnakeRenderer(human=self.render_mode == "human")
        return self._renderer.draw(self._observation(), self._info(), self.last_action, self.last_reward)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

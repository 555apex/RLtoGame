"""只负责绘图，不决定游戏规则；rgb_array 模式不创建桌面窗口。"""

import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

from .single_snake import BOARD_SIZE, DIRECTIONS, decode_observation


class SnakeRenderer:
    WIDTH, HEIGHT = 940, 630

    def __init__(self, human: bool = False):
        pygame.font.init()
        self.human = human
        self.screen = None
        self.surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.font = pygame.font.SysFont("microsoftyahei,simhei,arial", 18)
        self.title_font = pygame.font.SysFont("microsoftyahei,simhei,arial", 26, bold=True)
        if human:
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("RLtoGame | Single Snake")

    def draw(self, observation: np.ndarray, info: dict, action: int | None,
             reward: float, status: str = "", policy_name: str = "") -> np.ndarray:
        surface = self.surface
        surface.fill("#F5F7FA")
        body, food, direction = decode_observation(observation)
        self._text("SNAKE / 强化学习实验室", (24, 16), title=True)
        self._text("10 × 10 · 观测 204 维 · 相对动作 3 种", (24, 53))
        left, top, cell = 24, 92, 48
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                rect = pygame.Rect(left + col * cell, top + row * cell, cell, cell)
                pygame.draw.rect(surface, "#FFFFFF", rect)
                pygame.draw.rect(surface, "#DCE2E9", rect, width=1)
        for index, (row, col) in reversed(list(enumerate(body))):
            rect = pygame.Rect(left + col * cell + 3, top + row * cell + 3, cell - 6, cell - 6)
            pygame.draw.rect(surface, "#205A92" if index == 0 else "#A9C9E5", rect, border_radius=7)
            if index == 0:
                dr, dc = DIRECTIONS[direction]
                cx, cy = rect.center
                pygame.draw.circle(surface, "#FFFFFF", (cx + dc * 10, cy + dr * 10), 5)
            elif index == len(body) - 1:
                pygame.draw.circle(surface, "#205A92", rect.center, 4)
        if food is not None:
            row, col = food
            pygame.draw.circle(surface, "#D19118", (left + col * cell + cell // 2,
                                                     top + row * cell + cell // 2), 12)
        actions = ("直行", "左转", "右转")
        lines = [f"策略：{policy_name or '环境预览'}", f"状态：{status or info['end_reason']}",
                 f"食物数：{info['raw_score']}    长度：{info['length']}",
                 f"步数：{info['env_frames']}    占用：{info['occupancy']:.0%}",
                 f"上一步动作：{'尚未动作' if action is None else actions[action]}",
                 f"即时奖励：{reward:+.3f}", "奖励分量（上一步）"]
        labels = {"step_cost": "步成本", "food": "进食", "collision": "碰撞", "full_board": "满盘"}
        lines += [f"  {labels[key]}：{value:+.3f}" for key, value in info['reward_components'].items()]
        lines += ["", "空格：暂停 / 继续", "N：单步    R：同种子重置", "人工：↑直行  ←左转  →右转", "Esc：退出；尾部用圆点标识"]
        for index, line in enumerate(lines):
            self._text(line, (536, 94 + index * 29))
        self._text("蓝色蛇头上的白点表示朝向；金色圆点为食物。", (24, 589))
        if self.screen is not None:
            self.screen.blit(surface, (0, 0))
            pygame.display.flip()
        return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2)).copy()

    def _text(self, text: str, position: tuple[int, int], title: bool = False):
        font = self.title_font if title else self.font
        self.surface.blit(font.render(text, True, "#263445"), position)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.screen = None

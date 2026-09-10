"""实验公共记录：种子、逐局指标、版本与源码标识。"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np

from snake.agents import make_policy
from snake.envs import SingleSnakeEnv

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "single_snake.json"


def load_protocol() -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    # 阶段一是固定协议；拒绝“改了配置文件但环境仍跑旧规则”的隐性不一致。
    expected = {"protocol": "single_snake_v1", "board_size": 10, "max_steps": 2000,
                "initial_body": [[5, 5], [5, 4], [5, 3]], "initial_direction": "right",
                "actions": ["straight", "left", "right"],
                "observation": "ordered_body_100_food_100_direction_up_right_down_left_4",
                "rewards": {"step_cost": -0.001, "food": 1.0, "collision": -1.0, "full_board": 10.0}}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"{key} 与固定基础环境不一致；改变协议需要同时修改环境与测试")
    intervals = list(config["splits"].values())
    for i, (start, end) in enumerate(intervals):
        if start > end or any(max(start, a) <= min(end, b) for a, b in intervals[i + 1:]):
            raise ValueError("环境种子拆分不合法或存在交集")
    return config


def policy_seed(base: int, env_seed: int) -> int:
    # 环境 RNG 与动作 RNG 分开，前一局消耗的动作数不会改变后一局的动作序列。
    return int(np.random.SeedSequence([base, env_seed]).generate_state(1)[0])


def run_episode(policy_name: str, env_seed: int, seed: int, *, policy_instance=None) -> dict:
    policy = make_policy(policy_name) if policy_instance is None else policy_instance
    policy.reset_episode(seed)
    env = SingleSnakeEnv()
    observation, _ = env.reset(seed=env_seed)
    totals = dict.fromkeys(("step_cost", "food", "collision", "full_board"), 0.0)
    total_return = 0.0
    try:
        while True:
            action = policy.act(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_return += reward
            for name, value in info["reward_components"].items():
                totals[name] += value
            if terminated or truncated:
                break
    finally:
        env.close()
    return {"policy": policy_name, "env_seed": env_seed, "policy_seed": seed,
            "food": info["raw_score"], "length": info["length"], "occupancy": info["occupancy"],
            "steps": info["env_frames"], "return": total_return,
            "terminated": terminated, "truncated": truncated,
            "success": info["is_success"], "end_reason": info["end_reason"],
            **{f"reward_{key}": value for key, value in totals.items()}}


def source_manifest() -> dict:
    files = sorted([p for p in ROOT.rglob("*.py") if not any(
        part.startswith(".") or part in ("runs", "__pycache__") for part in p.relative_to(ROOT).parts)])
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"source_sha256": digest, "files": hashes,
            "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
            "python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(),
            "dependencies": {name: importlib.metadata.version(name) for name in
                             ("gymnasium", "numpy", "pygame-ce", "matplotlib", "pillow", "pytest")}}

"""单环境 DQN：采样、更新、独立验证、完整存档。"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import torch

from snake.agents.dqn import DQNAgent, epsilon_at
from snake.agents.replay import ReplayBuffer
from snake.envs import SingleSnakeEnv
from snake.experiment import ROOT, load_protocol, source_manifest


def load_dqn_config(path: Path | str = ROOT / "configs/dqn.json") -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config["algorithm"] not in ("dqn", "double_dqn"):
        raise ValueError("algorithm 必须为 dqn 或 double_dqn")
    for key in ("replay_capacity", "batch_size", "learning_starts", "train_frequency",
                "target_update_interval", "epsilon_decay_steps", "total_steps", "validation_interval",
                "checkpoint_interval", "log_interval", "torch_threads"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} 必须为正整数")
    if not config["replay_capacity"] >= config["learning_starts"] >= config["batch_size"]:
        raise ValueError("需要 replay_capacity >= learning_starts >= batch_size")
    if (not 0 <= config["gamma"] < 1 or not 0 <= config["epsilon_end"] <= config["epsilon_start"] <= 1
            or config["learning_rate"] <= 0 or config["max_grad_norm"] <= 0):
        raise ValueError("折扣、探索或优化参数范围不合法")
    if len(config["hidden_sizes"]) != 2 or any(type(n) is not int or n <= 0 for n in config["hidden_sizes"]):
        raise ValueError("网络需要两个正整数隐藏层宽度")
    return config


def save_torch(payload: dict, path: Path) -> None:
    """同目录临时文件写完再替换，避免中断时把原存档写坏。"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def cpu_weights(network) -> dict:
    return {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}


def snapshot_environment(env: SingleSnakeEnv) -> dict:
    fields = ("body", "food", "direction", "steps", "food_eaten", "last_action", "last_reward",
              "end_reason", "_finished", "reward_components")
    return {"fields": {name: copy.deepcopy(getattr(env, name)) for name in fields},
            "rng": copy.deepcopy(env.np_random.bit_generator.state)}


def restore_environment(env: SingleSnakeEnv, state: dict) -> np.ndarray:
    env.reset(seed=0)
    for name, value in state["fields"].items():
        setattr(env, name, copy.deepcopy(value))
    env.np_random.bit_generator.state = copy.deepcopy(state["rng"])
    return env._observation()


def validation_episodes(agent: DQNAgent, seeds: list[int]) -> list[dict]:
    """新环境、epsilon=0、no_grad；不调用任何训练 RNG。"""
    env = SingleSnakeEnv()
    was_training = agent.online.training
    agent.online.eval()
    rows = []
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            reward_sum = 0.0
            while True:
                observation, reward, terminated, truncated, info = env.step(agent.act(observation))
                reward_sum += reward
                if terminated or truncated:
                    break
            rows.append({"env_seed": seed, "food": info["raw_score"], "length": info["length"],
                         "occupancy": info["occupancy"], "steps": info["env_frames"], "return": reward_sum,
                         "terminated": terminated, "truncated": truncated, "end_reason": info["end_reason"],
                         "success": info["is_success"]})
    finally:
        agent.online.train(was_training)
        env.close()
    return rows


class Trainer:
    def __init__(self, config: dict, seed: int, device: str = "cpu"):
        self.config, self.seed = copy.deepcopy(config), seed
        self.protocol = load_protocol()
        torch.set_num_threads(config["torch_threads"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random.seed(seed)
        self.agent = DQNAgent(config, seed, device)
        replay_seed = int(np.random.SeedSequence([seed, 2]).generate_state(1)[0])
        self.replay = ReplayBuffer(config["replay_capacity"], replay_seed)
        self.env_rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))
        self.env = SingleSnakeEnv()
        self.step = self.episodes = 0
        self.best_score, self.best_step = -float("inf"), 0
        self.best_policy = None
        self.initial_policy = self.policy_payload()
        self.validation_history = []
        self.training_seconds = 0.0
        self.last_metrics = None
        self._reset_episode()

    def _reset_episode(self) -> None:
        low, high = self.protocol["splits"]["train"]
        self.env_seed = int(self.env_rng.integers(low, high + 1))
        self.observation, _ = self.env.reset(seed=self.env_seed)
        self.episode_return = 0.0
        self.episode_components = {name: 0.0 for name in ("step_cost", "food", "collision", "full_board")}

    def policy_payload(self) -> dict:
        return {"kind": "snake_dqn_policy_v1", "online": cpu_weights(self.agent.online),
                "config": copy.deepcopy(self.config), "protocol": copy.deepcopy(self.protocol),
                "training_seed": self.seed, "step": self.step}

    def interact(self) -> tuple[dict | None, dict | None]:
        """恰好一个环境动作；更新与同步都按已完成环境步数调度。"""
        epsilon = epsilon_at(self.step, self.config)
        action = self.agent.act(self.observation, epsilon)
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        self.replay.add(self.observation, action, reward, next_obs, terminated, truncated)
        self.observation = next_obs
        self.step += 1
        self.episode_return += reward
        for name, value in info["reward_components"].items():
            self.episode_components[name] += value
        metrics = None
        if len(self.replay) >= self.config["learning_starts"] and self.step % self.config["train_frequency"] == 0:
            metrics = self.agent.update(self.replay.sample(self.config["batch_size"], self.agent.device))
            self.last_metrics = metrics
        if self.step % self.config["target_update_interval"] == 0:
            self.agent.sync_target()
        episode = None
        if terminated or truncated:
            self.episodes += 1
            episode = {"step": self.step, "episode": self.episodes, "env_seed": self.env_seed,
                       "epsilon": epsilon, "return": self.episode_return, "food": info["raw_score"],
                       "length": info["length"], "occupancy": info["occupancy"], "steps": info["env_frames"],
                       "terminated": terminated, "truncated": truncated, "end_reason": info["end_reason"],
                       "success": info["is_success"], "reward_components": self.episode_components.copy()}
            self._reset_episode()
        return episode, metrics

    def validate(self) -> tuple[dict, list[dict], bool]:
        low, high = self.protocol["splits"]["validation"]
        rows = validation_episodes(self.agent, list(range(low, high + 1)))
        score = float(np.mean([r["food"] for r in rows]))
        summary = {"step": self.step, "mean_food": score, "mean_steps": float(np.mean([r["steps"] for r in rows])),
                   "collisions": sum("collision" in r["end_reason"] for r in rows),
                   "truncations": sum(r["truncated"] for r in rows), "successes": sum(r["success"] for r in rows),
                   "training_seconds": self.training_seconds}
        improved = score > self.best_score  # 同分保留较早 checkpoint，包括步数 0 的初始网络。
        if improved:
            self.best_score, self.best_step = score, self.step
            self.best_policy = self.policy_payload()
        self.validation_history.append(summary)
        return summary, rows, improved

    def state_dict(self) -> dict:
        return {"kind": "snake_dqn_training_v1", "config": self.config, "protocol": self.protocol,
                "seed": self.seed, "step": self.step, "episodes": self.episodes,
                "agent": self.agent.state_dict(), "replay": self.replay.state_dict(),
                "env_schedule_rng": copy.deepcopy(self.env_rng.bit_generator.state),
                "environment": snapshot_environment(self.env), "env_seed": self.env_seed,
                "episode_return": self.episode_return, "episode_components": self.episode_components,
                "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state_all() if self.agent.device.type == "cuda" else [],
                "best_score": self.best_score, "best_step": self.best_step, "best_policy": self.best_policy,
                "initial_policy": self.initial_policy, "validation_history": self.validation_history,
                "training_seconds": self.training_seconds, "last_metrics": self.last_metrics}

    def load_state_dict(self, state: dict) -> None:
        if state.get("kind") != "snake_dqn_training_v1" or state["protocol"] != self.protocol:
            raise ValueError("恢复文件种类或环境协议不匹配")
        for key, value in self.config.items():
            if key != "total_steps" and value != state["config"][key]:
                raise ValueError(f"恢复时不可修改 {key}，请另行登记新实验")
        if self.seed != state["seed"]:
            raise ValueError("恢复时训练种子必须一致")
        self.agent.load_state_dict(state["agent"])
        self.replay.load_state_dict(state["replay"])
        self.env_rng.bit_generator.state = copy.deepcopy(state["env_schedule_rng"])
        self.observation = restore_environment(self.env, state["environment"])
        for name in ("step", "episodes", "env_seed", "episode_return", "episode_components", "best_score",
                     "best_step", "best_policy", "initial_policy", "validation_history", "training_seconds", "last_metrics"):
            setattr(self, name, copy.deepcopy(state[name]))
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        if state["cuda_rng"] and self.agent.device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])


def train(config: dict, seed: int, device: str, output: Path, resume: Path | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()
    trainer = Trainer(config, seed, device)
    parent = None
    if resume is not None:
        trainer.load_state_dict(torch.load(resume, map_location="cpu", weights_only=True))
        parent = {"checkpoint": str(resume.resolve()), "sha256": hashlib.sha256(resume.read_bytes()).hexdigest(),
                  "step": trainer.step, "resume_mode": "complete simulator and RNG state"}
    if config["total_steps"] <= trainer.step:
        trainer.env.close()
        raise ValueError("总预算必须大于当前步数；--steps 指累计目标而非追加步数")
    hardware = {"device": device, "cpu_threads": torch.get_num_threads(), "torch": str(torch.__version__),
                "cuda_build": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "logical_cpus": psutil.cpu_count(), "system_ram_bytes": psutil.virtual_memory().total,
                "cudnn_deterministic": True, "cross_device_bitwise_reproducibility": False}
    manifest = {"run_id": output.name, "created_utc": datetime.now(timezone.utc).isoformat(),
                "training_seed": seed, "config": config, "protocol": trainer.protocol,
                "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                "hardware": hardware, "resume": parent, "final_test_used": False, "sources": source_manifest()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "source_snapshot.json").write_text(json.dumps({name: (ROOT / name).read_text(encoding="utf-8")
         for name in manifest["sources"]["files"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    save_torch(trainer.initial_policy, checkpoints / "initial.pt")
    if trainer.best_policy is not None:
        save_torch(trainer.best_policy, checkpoints / "best.pt")
    files = {name: (output / f"{name}.jsonl").open("w", encoding="utf-8")
             for name in ("episodes", "updates", "validation", "validation_episodes", "events")}

    def log(name, row):
        files[name].write(json.dumps(row) + "\n")
        files[name].flush()

    def validate():
        started = time.perf_counter()
        summary, rows, improved = trainer.validate()
        summary["validation_seconds"] = time.perf_counter() - started
        log("validation", summary)
        for row in rows:
            log("validation_episodes", {"step": trainer.step, **row})
        if improved:
            save_torch(trainer.best_policy, checkpoints / "best.pt")
        print(json.dumps({"step": trainer.step, "validation_food": summary["mean_food"],
                          "best_food": trainer.best_score, "best_step": trainer.best_step}), flush=True)

    started_run = time.perf_counter()
    wall_base = trainer.training_seconds
    try:
        if not trainer.validation_history:
            validate()
        log("events", {"event": "resume" if resume else "start", "step": trainer.step, "parent": parent})
        batch_metrics = []
        last_tick = time.perf_counter()
        while trainer.step < config["total_steps"]:
            episode, metrics = trainer.interact()
            if episode is not None:
                log("episodes", episode)
            if metrics is not None:
                batch_metrics.append(metrics)
            if trainer.step % config["log_interval"] == 0 and batch_metrics:
                log("updates", {"step": trainer.step, "updates": trainer.agent.updates,
                                "epsilon": epsilon_at(trainer.step, config), "window_updates": len(batch_metrics),
                                **{key: float(np.mean([m[key] for m in batch_metrics])) for key in batch_metrics[0]}})
                batch_metrics.clear()
            if trainer.step % config["validation_interval"] == 0:
                trainer.training_seconds += time.perf_counter() - last_tick
                validate()
                last_tick = time.perf_counter()
            if trainer.step % config["checkpoint_interval"] == 0:
                trainer.training_seconds += time.perf_counter() - last_tick
                save_torch(trainer.state_dict(), checkpoints / "resume.pt")
                log("events", {"event": "checkpoint", "step": trainer.step})
                last_tick = time.perf_counter()
        trainer.training_seconds += time.perf_counter() - last_tick
        if trainer.validation_history[-1]["step"] != trainer.step:
            validate()
        save_torch(trainer.policy_payload(), checkpoints / "last.pt")
        save_torch(trainer.state_dict(), checkpoints / "resume.pt")
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoints.glob("*.pt")}
        memory = psutil.Process().memory_info()
        result = {"status": "completed", "step": trainer.step, "updates": trainer.agent.updates,
                  "completed_episodes": trainer.episodes, "best_step": trainer.best_step,
                  "best_validation_food": trainer.best_score, "validation_history": trainer.validation_history,
                  "training_seconds": trainer.training_seconds, "segment_wall_seconds": time.perf_counter() - started_run,
                  "segment_training_seconds": trainer.training_seconds - wall_base,
                  "partial_episode": {"env_seed": trainer.env_seed, "steps": trainer.env.steps,
                                      "food": trainer.env.food_eaten, "saved_for_exact_resume": True},
                  "checkpoint_sha256": hashes, "final_test_used": False,
                  "process_memory_bytes": getattr(memory, "peak_wset", memory.rss),
                  "process_memory_metric": "peak working set" if hasattr(memory, "peak_wset") else "final RSS",
                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0}
        (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        log("events", {"event": "completed", "step": trainer.step})
        return result
    except BaseException as error:
        log("events", {"event": "failed", "step": trainer.step, "error": repr(error)})
        (output / "failure.json").write_text(json.dumps({"step": trainer.step, "error": repr(error),
            "recovery": "use last completed resume.pt; interrupted update is not a valid transition"}, indent=2), encoding="utf-8")
        raise
    finally:
        trainer.env.close()
        for file in files.values():
            file.close()

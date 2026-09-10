"""桌面人工/策略演示，以及无窗口的 GIF 与逐步轨迹导出。"""

import argparse
import hashlib
import json
import time
from pathlib import Path

from snake.agents import make_policy
from snake.envs import SingleSnakeEnv
from snake.envs.rendering import SnakeRenderer, pygame
from snake.experiment import load_protocol, policy_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=["human", "random", "heuristic", "dqn", "double_dqn"], default=None)
    parser.add_argument("--checkpoint", type=Path, help="加载 DQN initial/best/last.pt")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-frames", type=int, default=300,
                        help="无窗口预览动作上限；不是训练或评测终止规则")
    parser.add_argument("--gif", type=Path)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    requested_policy = args.policy
    args.policy = args.policy or ("dqn" if args.checkpoint else "heuristic")
    if (args.policy in ("dqn", "double_dqn")) != (args.checkpoint is not None):
        parser.error("DQN 演示必须提供 --checkpoint，其他策略不能使用模型文件")
    if args.fps <= 0 or args.max_frames <= 0:
        parser.error("fps / max-frames 必须为正整数")
    if args.headless and args.policy == "human":
        parser.error("人工操作需要桌面窗口")
    if args.gif and not args.headless:
        parser.error("GIF 导出请使用 --headless，防止交互预览无限积累帧")
    trace_path = args.trace or (args.gif.with_suffix(".jsonl") if args.gif else None)
    for path in (args.gif, trace_path):
        if path is not None:
            if path.exists():
                parser.error(f"文件已存在，换一个输出名以保留之前记录：{path}")
            path.parent.mkdir(parents=True, exist_ok=True)
    config = load_protocol()
    seed = policy_seed(config["policy_seed_base"], args.seed)
    checkpoint_info = None
    if args.checkpoint:
        from snake.agents.dqn import DQNPolicy
        policy = DQNPolicy.load(args.checkpoint, args.device)
        args.policy = policy.metadata["config"]["algorithm"]
        if requested_policy is not None and requested_policy != args.policy:
            parser.error("--policy 与存档中的算法不一致")
        checkpoint_info = {"path": str(args.checkpoint.resolve()),
                           "sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                           "step": policy.metadata["step"]}
    else:
        policy = None if args.policy == "human" else make_policy(args.policy)
    env = SingleSnakeEnv()
    renderer = SnakeRenderer(human=not args.headless)
    trace = trace_path.open("w", encoding="utf-8") if trace_path else None
    frames = []
    count = 0
    episode = 0

    def record(data):
        if trace:
            trace.write(json.dumps(data, ensure_ascii=False) + "\n")
            trace.flush()

    def reset():
        if policy is not None:
            policy.reset_episode(seed)
        obs, info = env.reset(seed=args.seed)
        record({"event": "reset", "episode": episode, "env_seed": args.seed,
                "policy_seed": seed, "policy": args.policy, "checkpoint": checkpoint_info,
                "observation": obs.tolist(), "info": info})
        return obs, info

    observation, info = reset()
    done, paused, reward, last_action = False, args.policy == "human", 0.0, None
    last_tick = time.perf_counter()
    running = True
    try:
        frame = renderer.draw(observation, info, None, 0.0, "准备", args.policy)
        if args.gif:
            from PIL import Image
            frames.append(Image.fromarray(frame))
        while running:
            advance = args.headless
            requested_action = 0
            if not args.headless:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_SPACE:
                            paused = not paused
                        elif event.key == pygame.K_n:
                            advance = True
                        elif event.key == pygame.K_r:
                            episode += 1
                            observation, info = reset()
                            done, reward, last_action = False, 0.0, None
                            last_tick = time.perf_counter()
                        elif event.key in (pygame.K_UP, pygame.K_LEFT, pygame.K_RIGHT) and policy is None:
                            requested_action = {pygame.K_UP: 0, pygame.K_LEFT: 1, pygame.K_RIGHT: 2}[event.key]
                            advance = True
                if not paused and time.perf_counter() - last_tick >= 1 / args.fps:
                    advance = True
            if not running:
                break
            if advance and not done:
                last_action = requested_action if policy is None else policy.act(observation)
                observation, reward, terminated, truncated, info = env.step(last_action)
                count += 1
                done = terminated or truncated
                last_tick = time.perf_counter()
                record({"event": "step", "episode": episode, "action": last_action,
                        "reward": reward, "terminated": terminated, "truncated": truncated,
                        "observation": observation.tolist(), "info": info})
            status = info["end_reason"] if done else ("暂停" if paused else "运行")
            frame = renderer.draw(observation, info, last_action, reward, status, args.policy)
            if args.gif:
                frames.append(Image.fromarray(frame))
            if args.headless and (done or count >= args.max_frames):
                break
            if not args.headless:
                time.sleep(0.01)
        reason = info["end_reason"] if done else ("preview_limit" if args.headless else "user_closed")
        record({"event": "demo_end", "reason": reason, "total_actions": count})
        if args.gif:
            frames[0].save(args.gif, save_all=True, append_images=frames[1:],
                           duration=max(10, round(1000 / args.fps)), loop=0)
        print(json.dumps({"preview_end": reason, "actions": count, "info": info}, ensure_ascii=True))
    finally:
        if trace:
            trace.close()
        renderer.close()
        env.close()


if __name__ == "__main__":
    main()

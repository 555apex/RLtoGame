"""交付审计：本地链接、冻结文件、依赖锁、示例录像与逐步轨迹。"""

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from snake.agents.dqn import DQNPolicy
from snake.envs import SingleSnakeEnv
from snake.envs.rendering import SnakeRenderer
from snake.experiment import ROOT
from snake.final_experiment import DATA, read_rows, verify_freeze, write_json
from snake.run_experiments import read_json, sha256, verify_run


def main() -> None:
    frozen = verify_freeze(DATA)
    for item in frozen["matrix"]["runs"]:
        verify_run(item)
    markdown = [ROOT.parent / name for name in ("README.md", "AGENTS.md", "PROGRESS.md")]
    markdown += [p for p in ROOT.rglob("*.md") if not any(
        part.startswith(".") or part == "runs" for part in p.relative_to(ROOT).parts)]
    count = 0
    for document in markdown:
        text = document.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if target.startswith(("http:", "https:", "#")):
                continue
            target = target.split("#", 1)[0].strip("<>")
            if not (document.parent / target).exists():
                raise FileNotFoundError(f"{document}: {target}")
            count += 1
    trajectory = ROOT / "reports/data/double_dqn_seed1_seed1000.jsonl"
    rows = read_rows(trajectory)
    header, steps = rows[0], [r for r in rows if r["event"] == "step"]
    model = next(m for m in frozen["models"] if m["id"] == frozen["demo_model_id"])
    assert header["checkpoint"]["sha256"] == model["sha256"]
    policy = DQNPolicy.load(ROOT / model["path"])
    env = SingleSnakeEnv()
    obs, _ = env.reset(seed=header["env_seed"])
    np.testing.assert_array_equal(obs, np.array(header["observation"], dtype=np.float32))
    for row in steps:
        assert policy.act(obs) == row["action"]
        obs, reward, ended, truncated, info = env.step(row["action"])
        np.testing.assert_array_equal(obs, np.array(row["observation"], dtype=np.float32))
        assert (reward, ended, truncated, info) == (row["reward"], row["terminated"], row["truncated"], row["info"])
    assert len(steps) == 37 and info["raw_score"] == 3 and info["end_reason"] == "body_collision"
    with Image.open(ROOT / "reports/assets/double_dqn_seed1_seed1000.gif") as gif:
        assert gif.n_frames == len(steps) + 1
    renderer = SnakeRenderer(human=False)
    frame = renderer.draw(obs, info, steps[-1]["action"], reward, info["end_reason"], "double_dqn")
    Image.fromarray(frame).save(ROOT / "reports/assets/double_dqn_demo_preview.png")
    renderer.close()
    env.close()
    result = {"status": "passed", "markdown_files": len(markdown), "local_links": count,
              "frozen_models": len(frozen["models"]), "completed_runs": 6,
              "demo_actions_replayed": len(steps), "demo_gif_frames": len(steps) + 1,
              "demo_trace_sha256": sha256(trajectory), "test_used_for_tuning": False,
              "desktop_student_acceptance": "pending"}
    write_json(ROOT / "reports/data/delivery_qa.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

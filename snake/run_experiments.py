"""顺序执行已登记的六次训练；已完成记录须通过校验才能复用。"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from snake.experiment import ROOT, load_protocol
from snake.training import load_dqn_config

MATRIX = ROOT / "configs/experiment_matrix.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_run(item: dict) -> tuple[dict, dict]:
    directory = ROOT / item["path"]
    manifest, summary = read_json(directory / "manifest.json"), read_json(directory / "summary.json")
    expected = load_dqn_config(ROOT / "configs" / f"{item['algorithm']}.json")
    if (manifest["config"] != expected or manifest["training_seed"] != item["seed"]
            or manifest["protocol"] != load_protocol() or manifest["hardware"]["device"] != "cpu"
            or summary["status"] != "completed" or summary["step"] != 300000
            or summary["updates"] != 74751 or summary["final_test_used"]):
        raise ValueError(f"训练记录与预登记不符: {directory}")
    for name, digest in summary["checkpoint_sha256"].items():
        if sha256(directory / "checkpoints" / name) != digest:
            raise ValueError(f"权重校验失败: {directory / name}")
    return manifest, summary


def main() -> None:
    matrix = read_json(MATRIX)
    for item in matrix["runs"]:
        directory = ROOT / item["path"]
        if not directory.exists():
            if item["reuse"]:
                raise FileNotFoundError("预登记复用的阶段二记录缺失；请恢复原始记录")
            command = [sys.executable, "-m", "snake.train", "--config",
                       str(ROOT / "configs" / f"{item['algorithm']}.json"), "--seed", str(item["seed"]),
                       "--device", matrix["device"], "--output", str(directory)]
            subprocess.run(command, cwd=ROOT.parent, check=True)
        verify_run(item)
        print(f"Verified {item['algorithm']} seed {item['seed']}: 300000 steps", flush=True)


if __name__ == "__main__":
    main()

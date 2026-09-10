"""验证统计单位与冻结保护，不读取真实保留测试环境。"""

import json

import pytest

from snake.final_experiment import aggregate, verify_freeze


def test_statistics_use_three_model_means_not_sixty_episodes():
    rows = []
    for method in ("initial", "dqn", "double_dqn"):
        for seed in range(3):
            for episode in range(20):
                food = seed * 2 + episode % 2
                rows.append({"policy": method, "model_id": f"{method}_{seed}", "training_seed": seed,
                             "food": food, "length": food + 3, "occupancy": (food + 3) / 100,
                             "steps": 10, "return": food - 1.01, "end_reason": "wall_collision",
                             "success": False, "truncated": False})
    result = aggregate(rows)["across_training_seeds"]["dqn"]
    assert result["mean_food"] == 2.5 and result["seed_mean_food_std"] == 2.0


def test_statistics_reject_missing_training_seed():
    with pytest.raises(ValueError, match="三个"):
        aggregate([])


def test_freeze_rejects_changed_model_before_evaluation(tmp_path, monkeypatch):
    from snake import final_experiment as module
    (tmp_path / "model.pt").write_bytes(b"original")
    (tmp_path / "matrix.json").write_text("{}")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "MATRIX", tmp_path / "matrix.json")
    payload = {"protocol": module.load_protocol(), "matrix_sha256": module.sha256(tmp_path / "matrix.json"),
               "core_sha256": {}, "models": [{"id": "example", "path": "model.pt",
                                               "sha256": module.sha256(tmp_path / "model.pt")}]}
    (tmp_path / "freeze.json").write_text(json.dumps(payload))
    verify_freeze(tmp_path)
    (tmp_path / "model.pt").write_bytes(b"modified")
    with pytest.raises(ValueError, match="权重"):
        verify_freeze(tmp_path)

"""验证公式与学习状态，而不是仅检查训练程序能够启动。"""

import copy
import json

import numpy as np
import pytest
import torch

from snake.agents.dqn import DQNAgent, DQNPolicy, dqn_targets, epsilon_at
from snake.agents.replay import ReplayBuffer
from snake.training import Trainer, load_dqn_config, save_torch, snapshot_environment, validation_episodes


@pytest.fixture(params=["dqn", "double_dqn"])
def config(request):
    c = load_dqn_config()
    c["algorithm"] = request.param
    c.update(replay_capacity=128, batch_size=8, learning_starts=16,
             target_update_interval=32, validation_interval=80, checkpoint_interval=80,
             total_steps=160, log_interval=16)
    torch.set_num_threads(1)
    return c


def test_td_target_terminal_truncated_and_no_gradient():
    next_q = torch.tensor([[1., 3., 2.], [1., 3., 2.], [1., 3., 2.]], requires_grad=True)
    # 普通转移和截断转移的 terminated 都是 False；真终止目标只剩 r。
    targets = dqn_targets(torch.tensor([1., 1., 1.]), torch.tensor([False, True, False]), next_q, 0.9)
    torch.testing.assert_close(targets, torch.tensor([3.7, 1., 3.7]))
    assert not targets.requires_grad


def test_replay_copies_ring_buffer_and_restores_rng(tmp_path):
    replay = ReplayBuffer(4, seed=9)
    obs = np.ones(204, dtype=np.float32)
    for i in range(6):
        replay.add(obs * i, i % 3, float(i), obs * (i + 1), i == 4, i == 5)
    obs[:] = 100
    assert len(replay) == 4 and replay.position == 2
    assert set(replay.arrays["rewards"]) == {2, 3, 4, 5}
    assert replay.arrays["observations"].max() == 5
    path = tmp_path / "replay.pt"
    torch.save(replay.state_dict(), path)
    restored = ReplayBuffer(4, seed=0)
    restored.load_state_dict(torch.load(path, weights_only=True))
    a, b = replay.sample(4, "cpu"), restored.sample(4, "cpu")
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    assert a["truncated"].any() and a["terminated"].any()


def test_network_update_frozen_target_and_sync(config):
    trainer = Trainer(config, 0)
    for _ in range(20):
        trainer.interact()
    before_online = copy.deepcopy(trainer.agent.online.state_dict())
    before_target = copy.deepcopy(trainer.agent.target.state_dict())
    trainer.agent.update(trainer.replay.sample(config["batch_size"], "cpu"))
    assert any(not torch.equal(before_online[k], v) for k, v in trainer.agent.online.state_dict().items())
    for key, value in trainer.agent.target.state_dict().items():
        torch.testing.assert_close(value, before_target[key], rtol=0, atol=0)
    assert all(p.grad is None and not p.requires_grad for p in trainer.agent.target.parameters())
    trainer.agent.sync_target()
    for key, value in trainer.agent.target.state_dict().items():
        torch.testing.assert_close(value, trainer.agent.online.state_dict()[key], rtol=0, atol=0)


def test_epsilon_schedule(config):
    assert epsilon_at(0, config) == 1
    assert epsilon_at(50000, config) == pytest.approx(0.525)
    assert epsilon_at(100000, config) == pytest.approx(0.05)
    assert epsilon_at(300000, config) == pytest.approx(0.05)


def test_greedy_act_does_not_consume_exploration_rng(config):
    agent = DQNAgent(config, 0)
    before = copy.deepcopy(agent.exploration_rng.bit_generator.state)
    agent.act(np.zeros(204, dtype=np.float32), 0)
    assert before == agent.exploration_rng.bit_generator.state


def test_validation_preserves_all_training_state(config):
    trainer = Trainer(config, 4)
    for _ in range(20):
        trainer.interact()
    environment = snapshot_environment(trainer.env)
    observation = trainer.observation.copy()
    rngs = [copy.deepcopy(r.bit_generator.state) for r in
            (trainer.env_rng, trainer.agent.exploration_rng, trainer.replay.rng)]
    torch_rng = torch.get_rng_state().clone()
    weights = copy.deepcopy(trainer.agent.online.state_dict())
    validation_episodes(trainer.agent, [1000, 1001])
    assert environment == snapshot_environment(trainer.env)
    np.testing.assert_array_equal(observation, trainer.observation)
    for before, rng in zip(rngs, (trainer.env_rng, trainer.agent.exploration_rng, trainer.replay.rng)):
        assert before == rng.bit_generator.state
    assert torch.equal(torch_rng, torch.get_rng_state())
    for key, value in trainer.agent.online.state_dict().items():
        assert torch.equal(weights[key], value)
    assert trainer.agent.online.training


def test_exact_cpu_resume_mid_episode(config, tmp_path):
    original = Trainer(config, 0)
    for _ in range(73):
        original.interact()
    path = tmp_path / "resume.pt"
    save_torch(original.state_dict(), path)
    expected = [original.interact() for _ in range(50)]
    restored = Trainer(config, 0)
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    actual = [restored.interact() for _ in range(50)]
    assert actual == expected
    assert snapshot_environment(original.env) == snapshot_environment(restored.env)
    assert original.replay.rng.bit_generator.state == restored.replay.rng.bit_generator.state
    for key, value in original.agent.online.state_dict().items():
        torch.testing.assert_close(value, restored.agent.online.state_dict()[key], rtol=0, atol=0)


def test_policy_load_matches_online(config, tmp_path):
    trainer = Trainer(config, 0)
    for _ in range(20):
        trainer.interact()
    path = tmp_path / "policy.pt"
    save_torch(trainer.policy_payload(), path)
    policy = DQNPolicy.load(path)
    with torch.no_grad():
        obs = torch.from_numpy(trainer.observation)
        torch.testing.assert_close(policy.network(obs), trainer.agent.online(obs), rtol=0, atol=0)
    assert policy.act(trainer.observation) == trainer.agent.act(trainer.observation)
    assert policy.metadata["step"] == 20


def test_validation_tie_keeps_earlier_model(config, monkeypatch):
    trainer = Trainer(config, 0)
    monkeypatch.setattr("snake.training.validation_episodes", lambda *_: [
        {"food": 2, "steps": 10, "end_reason": "wall_collision", "truncated": False, "success": False}])
    assert trainer.validate()[2]
    trainer.step = 100
    assert not trainer.validate()[2] and trainer.best_step == 0


def test_resume_rejects_hyperparameter_change(config):
    trainer = Trainer(config, 0)
    different = copy.deepcopy(config)
    different["gamma"] = 0.5
    with pytest.raises(ValueError, match="gamma"):
        Trainer(different, 0).load_state_dict(trainer.state_dict())


def test_training_end_to_end_and_resume(config, tmp_path):
    from snake.training import train
    first_config = dict(config, total_steps=80)
    first = tmp_path / "first"
    result = train(first_config, 0, "cpu", first)
    assert result["step"] == 80 and result["updates"] == 17
    assert not result["final_test_used"]
    second = tmp_path / "second"
    restored = train(config, 0, "cpu", second, first / "checkpoints/resume.pt")
    assert restored["step"] == 160 and restored["updates"] == 37
    assert json.loads((second / "manifest.json").read_text())["resume"]["step"] == 80
    for name in ("initial.pt", "best.pt", "last.pt", "resume.pt"):
        assert (second / "checkpoints" / name).is_file()


def test_training_stores_final_observation_before_reset(config, monkeypatch):
    from snake.envs.single_snake import decode_observation
    trainer = Trainer(config, 0)
    trainer.env.max_steps = 1
    monkeypatch.setattr(trainer.agent, "act", lambda *_: 0)
    episode, _ = trainer.interact()
    assert episode["truncated"] and not episode["terminated"]
    assert trainer.replay.arrays["truncated"][0]
    assert decode_observation(trainer.replay.arrays["next_observations"][0])[0][0] == (5, 6)
    assert decode_observation(trainer.observation)[0][0] == (5, 5)


def test_dqn_demo_entrypoint(config, monkeypatch, tmp_path):
    import sys
    from snake import demo
    trainer = Trainer(config, 0)
    model, trace = tmp_path / "policy.pt", tmp_path / "demo.jsonl"
    save_torch(trainer.policy_payload(), model)
    monkeypatch.setattr(sys, "argv", ["demo", "--checkpoint", str(model), "--headless",
                                      "--max-frames", "2", "--trace", str(trace)])
    demo.main()
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["policy"] == config["algorithm"] and rows[0]["checkpoint"]["step"] == 0
    assert len([r for r in rows if r["event"] == "step"]) == 2


def test_dqn_evaluation_entrypoint_records_training_seed(config, monkeypatch, tmp_path):
    import sys
    from snake import evaluate
    trainer = Trainer(config, 0)
    model, output = tmp_path / "policy.pt", tmp_path / "evaluation"
    save_torch(trainer.policy_payload(), model)
    monkeypatch.setattr(sys, "argv", ["evaluate", "--checkpoint", str(model), "--output", str(output)])
    evaluate.main()
    summary = json.loads((output / "summary.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert summary["policies"][config["algorithm"]]["episodes"] == 20 and summary["split"] == "validation"
    assert "frozen trained model" in summary["statistics_unit"]
    assert manifest["checkpoint"]["metadata"]["training_seed"] == 0

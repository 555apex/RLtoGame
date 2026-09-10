import numpy as np

from snake.agents import HeuristicPolicy, RandomPolicy
from snake.envs.single_snake import encode_observation
from snake.experiment import load_protocol, policy_seed, run_episode


def test_random_uniform_and_reproducible():
    observation = encode_observation([(0, 9), (0, 8), (0, 7)], (9, 0), 1)
    a, b = RandomPolicy(), RandomPolicy()
    a.reset_episode(5)
    b.reset_episode(5)
    first = [a.act(observation) for _ in range(9000)]
    assert first == [b.act(observation) for _ in range(9000)]
    # 在墙角仍可选择撞墙动作；这里检查没有偷偷加危险屏蔽。
    counts = np.bincount(first, minlength=3)
    assert np.all((counts > 2700) & (counts < 3300))


def test_heuristic_straight_tie_and_input_unchanged():
    observation = encode_observation([(5, 5), (5, 4), (5, 3)], (4, 6), 1)
    original = observation.copy()
    assert HeuristicPolicy().act(observation) == 0
    np.testing.assert_array_equal(observation, original)


def test_heuristic_avoids_wall():
    observation = encode_observation([(0, 9), (0, 8), (0, 7)], (9, 9), 1)
    assert HeuristicPolicy().act(observation) == 2


def test_heuristic_can_enter_tail():
    observation = encode_observation([(1, 1), (1, 0), (2, 0), (2, 1)], (3, 1), 1)
    assert HeuristicPolicy().act(observation) == 2


def test_no_safe_action_returns_straight():
    observation = encode_observation([(0, 0), (0, 1), (1, 1), (1, 0), (2, 0)], (9, 9), 3)
    assert HeuristicPolicy().act(observation) == 0


def test_left_before_right_when_tied():
    # 直行被身体挡住；左/右都离食物 3 格，必须选左。
    observation = encode_observation([(5, 5), (5, 4), (4, 4), (3, 4), (3, 5),
                                      (3, 6), (4, 6), (5, 6), (6, 6)], (5, 7), 1)
    assert HeuristicPolicy().act(observation) == 1


def test_episode_logs_reconcile_and_reproduce():
    config = load_protocol()
    seed = policy_seed(config["policy_seed_base"], 1000)
    a = run_episode("heuristic", 1000, seed)
    assert a == run_episode("heuristic", 1000, seed)
    assert a["length"] == a["food"] + 3
    assert abs(a["return"] - sum(a[key] for key in a if key.startswith("reward_"))) < 1e-8
    assert a["terminated"] != a["truncated"]

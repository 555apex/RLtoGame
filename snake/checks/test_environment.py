"""用可手算的棋盘验证规则，避免凭“看起来能玩”判断环境正确。"""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from snake.envs import SingleSnakeEnv
from snake.envs.single_snake import decode_observation


def position(env, body, food, direction=1):
    """仅测试使用：构造精确状态，不开放到训练 reset 接口。"""
    env.reset(seed=0)
    env.body = list(body)
    env.food = food
    env.direction = direction
    env.food_eaten = len(body) - 3


def test_gymnasium_api():
    env = SingleSnakeEnv()
    check_env(env, skip_render_check=True)
    env.close()


def test_initial_encoding_and_independent_arrays():
    env = SingleSnakeEnv()
    obs, info = env.reset(seed=7)
    original = obs.copy()
    assert obs.dtype == np.float32 and env.observation_space.contains(obs)
    body, food, direction = decode_observation(obs)
    assert body == [(5, 5), (5, 4), (5, 3)] and direction == 1
    assert food not in body and info["length"] == 3
    assert obs[55] == np.float32(0.01) and obs[53] == np.float32(0.03)
    env.step(0)
    np.testing.assert_array_equal(obs, original)
    obs[:] = 0
    assert len(env.body) >= 3


@pytest.mark.parametrize("seed", range(20))
def test_food_never_on_body(seed):
    env = SingleSnakeEnv()
    env.reset(seed=seed)
    assert env.food not in env.body


@pytest.mark.parametrize("action,head,direction", [(0, (5, 6), 1), (1, (4, 5), 0), (2, (6, 5), 2)])
def test_relative_turns(action, head, direction):
    env = SingleSnakeEnv()
    position(env, [(5, 5), (5, 4), (5, 3)], (0, 0))
    _, reward, terminated, truncated, _ = env.step(action)
    assert env.body == [head, (5, 5), (5, 4)]
    assert env.direction == direction
    assert reward == -0.001 and not terminated and not truncated


def test_eat_grows_and_reward_is_additive():
    env = SingleSnakeEnv()
    position(env, [(5, 5), (5, 4), (5, 3)], (5, 6))
    _, reward, terminated, truncated, info = env.step(0)
    assert env.body == [(5, 6), (5, 5), (5, 4), (5, 3)]
    assert reward == pytest.approx(0.999)
    assert info["raw_score"] == 1 and env.food not in env.body
    assert not terminated and not truncated


def test_enter_vacating_tail_is_legal():
    env = SingleSnakeEnv()
    position(env, [(1, 1), (1, 0), (2, 0), (2, 1)], (0, 0))
    _, _, terminated, _, _ = env.step(2)
    assert not terminated
    assert env.body == [(2, 1), (1, 1), (1, 0), (2, 0)]


def test_body_collision():
    env = SingleSnakeEnv()
    position(env, [(1, 1), (1, 0), (2, 0), (2, 1), (2, 2)], (0, 0))
    _, reward, terminated, truncated, info = env.step(2)
    assert terminated and not truncated
    assert reward == pytest.approx(-1.001) and info["end_reason"] == "body_collision"


def test_collision_keeps_valid_terminal_observation():
    env = SingleSnakeEnv(max_steps=1)
    position(env, [(0, 9), (0, 8), (0, 7)], (9, 0))
    obs, reward, terminated, truncated, info = env.step(0)
    assert terminated and not truncated  # 第 1 步碰撞优先于第 1 步预算。
    assert env.observation_space.contains(obs)
    assert info["end_reason"] == "wall_collision" and reward == pytest.approx(-1.001)


def test_full_board_last_food_and_no_spawn(monkeypatch):
    path = [(r, c) for r in range(10) for c in (range(10) if r % 2 == 0 else range(9, -1, -1))]
    env = SingleSnakeEnv(max_steps=1)
    position(env, list(reversed(path[:-1])), path[-1], direction=3)
    def forbidden_spawn():
        raise AssertionError("满盘不能再生成食物")
    monkeypatch.setattr(env, "_spawn_food", forbidden_spawn)
    obs, reward, terminated, truncated, info = env.step(0)
    assert terminated and not truncated and info["is_success"]
    assert reward == pytest.approx(10.999)
    assert info["raw_score"] == 97 and info["length"] == 100 and env.food is None
    assert np.count_nonzero(obs[:100]) == 100 and np.count_nonzero(obs[100:200]) == 0


def test_time_limit_has_no_failure_reward():
    env = SingleSnakeEnv(max_steps=1)
    position(env, [(5, 5), (5, 4), (5, 3)], (0, 0))
    obs, reward, terminated, truncated, info = env.step(0)
    assert not terminated and truncated and reward == -0.001
    assert info["end_reason"] == "time_limit"
    assert decode_observation(obs)[0][0] == (5, 6)  # 保存截断前观测，不能自动 reset。


def test_eating_on_time_limit():
    env = SingleSnakeEnv(max_steps=1)
    position(env, [(5, 5), (5, 4), (5, 3)], (5, 6))
    _, reward, terminated, truncated, info = env.step(0)
    assert truncated and not terminated and info["length"] == 4
    assert reward == pytest.approx(0.999)


def test_default_2000_steps_no_hunger_death():
    env = SingleSnakeEnv()
    position(env, [(5, 5), (5, 4), (5, 3)], (0, 0))
    for step in range(1, 2001):
        _, reward, terminated, truncated, info = env.step(2)
        assert not terminated and reward == -0.001
        assert truncated == (step == 2000)
    assert info["env_frames"] == 2000 and info["raw_score"] == 0


def test_seed_and_action_reproducibility():
    a, b = SingleSnakeEnv(), SingleSnakeEnv()
    np.testing.assert_array_equal(a.reset(seed=8)[0], b.reset(seed=8)[0])
    for action in [0, 2, 0, 2, 0, 2, 0, 2]:
        out_a, out_b = a.step(action), b.step(action)
        np.testing.assert_array_equal(out_a[0], out_b[0])
        assert out_a[1:] == out_b[1:]
        if out_a[2] or out_a[3]:
            break
    assert len({SingleSnakeEnv().reset(seed=seed)[0][100:200].tobytes() for seed in range(10)}) > 1


@pytest.mark.parametrize("action", [-1, 3, 0.5, "0"])
def test_illegal_actions_do_not_advance(action):
    env = SingleSnakeEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(action)
    assert env.steps == 0


def test_reset_required_after_end():
    env = SingleSnakeEnv(max_steps=1)
    with pytest.raises(RuntimeError):
        env.step(0)
    env.reset(seed=0)
    env.step(0)
    with pytest.raises(RuntimeError):
        env.step(0)


def test_rgb_renderer_and_close():
    env = SingleSnakeEnv(render_mode="rgb_array")
    env.reset(seed=0)
    rgb = env.render()
    assert rgb.shape == (630, 940, 3) and rgb.dtype == np.uint8
    env.close()
    env.close()

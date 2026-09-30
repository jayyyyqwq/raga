import numpy as np

from raaga_env.env import RaagaEnv
from raaga_env.jugalbandi_env import JugalbandiEnv


def test_raaga_env_state_round_trip():
    env = RaagaEnv(raga="yaman")
    env.reset(seed=1)
    for action in [11, 9, 7, 4, 40]:
        env.step(action)

    snapshot = env.get_state()

    restored = RaagaEnv(raga="bhairav")  # deliberately wrong raga, must be overwritten
    restored.set_state(snapshot)

    assert restored.get_state() == snapshot


def test_raaga_env_state_round_trip_produces_identical_future_trajectory():
    env = RaagaEnv(raga="yaman")
    env.reset(seed=1)
    for action in [11, 9, 7]:
        env.step(action)
    snapshot = env.get_state()

    actions = [4, 40, 16, 28]

    obs_a, rewards_a = [], []
    for action in actions:
        obs, reward, *_ = env.step(action)
        obs_a.append(obs.copy())
        rewards_a.append(reward)

    replay = RaagaEnv(raga="yaman")
    replay.reset(seed=1)
    replay.set_state(snapshot)

    obs_b, rewards_b = [], []
    for action in actions:
        obs, reward, *_ = replay.step(action)
        obs_b.append(obs.copy())
        rewards_b.append(reward)

    assert rewards_a == rewards_b
    for a, b in zip(obs_a, obs_b):
        assert np.array_equal(a, b)


def test_jugalbandi_env_state_round_trip_includes_drift():
    env = JugalbandiEnv(initial_dial=0.0)
    env.reset(seed=2)
    env.set_dial(0.75)  # switch to bhairav, enter grace period
    env.set_call([2, 6, 9, 11])
    env.step(1)  # forbidden in bhairav, but grace-reduced

    snapshot = env.get_state()
    assert snapshot["drift"]["active_raga_name"] == "bhairav"
    assert snapshot["drift"]["grace_steps_remaining"] > 0

    restored = JugalbandiEnv(initial_dial=0.0)
    restored.reset(seed=2)
    restored.set_state(snapshot)

    assert restored.get_state() == snapshot
    assert restored.drift.active_raga_name == "bhairav"
    assert restored.raga_name == "bhairav"
    assert restored.call_phrase == [2, 6, 9, 11]

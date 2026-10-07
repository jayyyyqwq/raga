import pytest
import numpy as np
from raaga_env.env import RaagaEnv
from raaga_env.jugalbandi_env import JugalbandiEnv


# ── Base env ──────────────────────────────────────────────────────────

def test_reset_obs_shape():
    env = RaagaEnv(raga="yaman")
    obs, info = env.reset()
    assert obs.shape == (14,)
    assert np.all((obs >= 0.0) & (obs <= 1.0))

def test_vadi_gives_positive_reward():
    env = RaagaEnv(raga="yaman")
    env.reset()
    # action = note (4=Ga, vadi of Yaman), duration 0 → action index 4
    _, reward, _, _, _ = env.step(4)
    assert reward > 0

def test_forbidden_note_penalty():
    env = RaagaEnv(raga="yaman")
    env.reset()
    # natural Ma = note 5, duration 0 → action 5
    _, reward, _, _, info = env.step(5)
    assert reward == -2.0
    assert info["forbidden_note_count"] == 1

def test_episode_terminates_at_length():
    env = RaagaEnv(raga="yaman", episode_length=16)
    env.reset()
    steps = 0
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        done = term or trunc
        steps += 1
    assert steps == 16

def test_obs_always_in_bounds():
    env = RaagaEnv(raga="yaman")
    env.reset()
    for _ in range(50):
        obs, _, _, _, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs), f"Obs out of bounds: {obs}"

def test_pakad_detection():
    env = RaagaEnv(raga="yaman")
    env.reset()
    # Ni(11) Dha(9) Pa(7) is a pakad of Yaman
    env.step(11)
    env.step(9)
    _, _, _, _, info = env.step(7)
    assert info["pakad_completions"] >= 1


# Phase 1.1 acceptance: cross-register pakads span mandra (0-11) and madhya
# (12-23) pitches and are only reachable with the 96-action space — they were
# the reason 96 was chosen over 48 (updatedplan.md §9.1, CLASSICAL_MUSIC_AUDIT.md).

def test_cross_register_pakad_reachable_yaman():
    env = RaagaEnv(raga="yaman")
    env.reset()
    pakad = [11, 14, 16]  # mandra Ni, madhya Re, madhya Ga
    info = None
    for note in pakad:
        _, _, _, _, info = env.step(note)  # duration 0: action == note (< 24)
    assert info["pakad_completions"] >= 1

def test_cross_register_pakad_reachable_bhairav():
    env = RaagaEnv(raga="bhairav")
    env.reset()
    pakad = [13, 12, 11, 12]  # madhya re, madhya Sa, mandra Ni, madhya Sa
    info = None
    for note in pakad:
        _, _, _, _, info = env.step(note)
    assert info["pakad_completions"] >= 1

def test_bhairav_forbidden():
    env = RaagaEnv(raga="bhairav")
    env.reset()
    # natural Re = note 2, forbidden in Bhairav → action 2
    _, reward, _, _, _ = env.step(2)
    assert reward == -2.0


# ── Jugalbandi env ────────────────────────────────────────────────────

def test_jugalbandi_obs_shape():
    env = JugalbandiEnv()
    obs, _ = env.reset()
    assert obs.shape == (22,)

def test_dial_switches_raga():
    env = JugalbandiEnv(initial_dial=0.0)
    env.reset()
    assert env.drift.active_raga_name == "yaman"
    switched = env.set_dial(0.75)
    assert switched is True
    assert env.drift.active_raga_name == "bhairav"

def test_grace_period_reduces_penalty():
    env = JugalbandiEnv(initial_dial=0.0)
    env.reset()
    env.set_dial(0.75)  # switch to bhairav, enter grace period
    # natural Re (note 2) is forbidden in Bhairav
    _, reward, _, _, _ = env.step(2)
    # grace factor 0.2 → penalty should be -0.4 not -2.0
    assert -1.0 < reward < 0.0, f"Expected reduced penalty, got {reward}"

def test_set_call_updates_tension():
    env = JugalbandiEnv()
    env.reset()
    env.set_call([2, 6, 9, 11])   # ends on Ni (far from Ga, vadi of Yaman)
    assert env.call_tension > 0.0


# ── ML retrain fix (2026-10, docs/RETRAIN_PLAN.md): call_phrase defaults to
# "no call yet", distinguishably from a real (even all-Sa) call ────────────

def test_fresh_env_has_no_call_phrase():
    env = JugalbandiEnv()
    env.reset()
    assert env.call_phrase == []
    assert env.call_tension == 0.0
    assert env.call_echoed is False


def test_set_call_resets_call_echoed():
    env = JugalbandiEnv()
    env.reset()
    env.call_echoed = True
    env.set_call([0, 2, 4, 7])
    assert env.call_echoed is False


def test_state_round_trip_preserves_call_echoed():
    env = JugalbandiEnv()
    env.reset()
    env.set_call([0, 2, 4, 7])
    env.call_echoed = True
    state = env.get_state()

    restored = JugalbandiEnv()
    restored.set_state(state)
    assert restored.call_echoed is True
    assert restored.call_phrase == [0, 2, 4, 7]


def test_set_state_tolerates_snapshots_without_call_echoed():
    """state_json produced before this fix existed won't have the key."""
    env = JugalbandiEnv()
    env.reset()
    state = env.get_state()
    del state["call_echoed"]

    restored = JugalbandiEnv()
    restored.set_state(state)  # must not raise KeyError
    assert restored.call_echoed is False


def test_no_call_is_requested_on_the_terminal_step():
    """A call requested on the last step can never be answered (the episode
    is over), but eval.metrics counted it — capping call_echo_rate at 7/8."""
    env = JugalbandiEnv(episode_length=64)
    env.reset(seed=0)
    requested = []
    for i in range(64):
        _, _, term, _, info = env.step(4)
        if info["call_requested"]:
            requested.append(i)
    assert term
    assert 63 not in requested
    assert requested == [7, 15, 23, 31, 39, 47, 55]


def test_set_call_tension_uses_the_stored_phrase():
    env = JugalbandiEnv(initial_dial=0.0)
    env.reset(seed=0)
    env.set_call([0, 2, 4, 11, 4])  # 5th note is dropped from call_phrase
    assert env.call_phrase == [0, 2, 4, 11]
    assert env.call_tension == pytest.approx(5 / 6.0)  # Ni->Ga: 7 up = 5 around



def test_reset_starts_from_initial_dial_not_the_previous_episodes_dial():
    env = JugalbandiEnv(initial_dial=0.0)
    env.reset(seed=0)
    env.set_dial(1.0)  # switch to Bhairav mid-episode
    env.reset(seed=1)
    assert env.drift.active_raga_name == "yaman"


def test_reset_dial_option_overrides_initial_dial():
    env = JugalbandiEnv(initial_dial=0.0)
    env.reset(seed=0, options={"dial": 1.0})
    assert env.drift.active_raga_name == "bhairav"
    assert not env.drift.in_grace_period  # starting in a raga is not a switch


def test_call_tension_is_circular():
    env = JugalbandiEnv(initial_dial=0.0)  # Yaman, vadi Ga (4)
    env.reset(seed=0)
    env.set_call([0, 0, 0, 4])
    assert env.call_tension == 0.0
    env.set_call([0, 0, 0, 10])  # ni: 6 semitones either way — maximal
    assert env.call_tension == pytest.approx(1.0)

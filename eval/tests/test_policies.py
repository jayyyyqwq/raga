import random

from raaga_env.jugalbandi_env import JugalbandiEnv

from eval.policies import (
    SAFE_SET_SWARAS,
    random_uniform_policy,
    random_valid_policy,
    safe_set_cycle_policy,
    scripted_oracle_policy,
)
from eval.rollout import DriftSchedule, rollout


def test_random_valid_policy_never_hits_a_forbidden_note_in_yaman():
    traj = rollout(random_valid_policy(random.Random(0)), seed=0, initial_dial=0.0, episode_length=64)
    forbidden_hits = sum(1 for s in traj.steps if "forbidden_note" in s.reward_breakdown)
    assert forbidden_hits == 0


def test_random_valid_policy_never_hits_a_forbidden_note_across_a_drift_switch():
    schedule = DriftSchedule(switches=((20, 1.0),))
    traj = rollout(
        random_valid_policy(random.Random(1)), seed=0, initial_dial=0.0,
        drift_schedule=schedule, episode_length=64,
    )
    forbidden_hits = sum(1 for s in traj.steps if "forbidden_note" in s.reward_breakdown)
    assert forbidden_hits == 0, "random-valid must track the *currently* active raga, not the starting one"


def test_random_valid_policy_produces_actions_in_range():
    env = JugalbandiEnv()
    obs, _ = env.reset()
    policy = random_valid_policy(random.Random(2))
    for _ in range(200):
        action = policy(list(obs), {})
        assert 0 <= action < env.action_space.n


# ── random_uniform_policy: the absolute floor, no validity awareness ───────

def test_random_uniform_policy_produces_actions_in_range():
    policy = random_uniform_policy(random.Random(0))
    for _ in range(200):
        action = policy([0.0] * 22, {})
        assert 0 <= action < 96


def test_random_uniform_policy_does_hit_forbidden_notes_sometimes():
    """Sanity check that it's genuinely not validity-aware — distinguishes
    it from random_valid_policy over enough steps."""
    traj = rollout(random_uniform_policy(random.Random(0)), seed=0, initial_dial=0.0, episode_length=64)
    forbidden_hits = sum(1 for s in traj.steps if "forbidden_note" in s.reward_breakdown)
    assert forbidden_hits > 0


# ── safe_set_cycle_policy: names the F13 degenerate policy explicitly ──────

def test_safe_set_cycle_policy_never_hits_a_forbidden_note_in_either_raga():
    for initial_dial in (0.0, 1.0):
        traj = rollout(safe_set_cycle_policy(), seed=0, initial_dial=initial_dial, episode_length=64)
        forbidden_hits = sum(1 for s in traj.steps if "forbidden_note" in s.reward_breakdown)
        assert forbidden_hits == 0


def test_safe_set_cycle_policy_stays_within_the_safe_set():
    traj = rollout(safe_set_cycle_policy(), seed=0, initial_dial=0.0, episode_length=32)
    for step in traj.steps:
        note = step.action % 24
        assert note % 12 in SAFE_SET_SWARAS


# ── scripted_oracle_policy: the ceiling, calibrates drift_adaptation_speed ──

def test_scripted_oracle_policy_completes_a_pakad_immediately_after_switching():
    schedule = DriftSchedule(switches=((20, 1.0),))
    traj = rollout(
        scripted_oracle_policy(schedule, random.Random(0)),
        seed=0, initial_dial=0.0, drift_schedule=schedule, episode_length=64,
    )
    # Bhairav's shortest pakad is 4 notes; it must complete within a few
    # steps of the switch, not eventually by luck.
    completion_steps = [
        s.step for s in traj.steps if "pakad_completion" in s.reward_breakdown and s.step >= 20
    ]
    assert completion_steps, "scripted-oracle should complete the new-raga pakad shortly after the switch"
    assert completion_steps[0] <= 24  # switch at 20 + 4-note pakad


def test_scripted_oracle_policy_never_hits_a_forbidden_note():
    schedule = DriftSchedule(switches=((30, 1.0),))
    traj = rollout(
        scripted_oracle_policy(schedule, random.Random(1)),
        seed=0, initial_dial=0.0, drift_schedule=schedule, episode_length=64,
    )
    forbidden_hits = sum(1 for s in traj.steps if "forbidden_note" in s.reward_breakdown)
    assert forbidden_hits == 0

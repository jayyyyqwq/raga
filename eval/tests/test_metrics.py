import json
import random

from eval.metrics import (
    GRACE_PERIOD_STEPS,
    adaptation_success_rate_at_k,
    compute_all_metrics,
    drift_adaptation_speed,
    jugalbandi_coherence,
    pakad_rate,
    post_switch_violation_decay,
    safe_set_occupancy,
    valid_raga_adherence,
)
from eval.policies import random_valid_policy, safe_set_cycle_policy, scripted_oracle_policy
from eval.rollout import DriftSchedule, rollout


def _switched_trajectory(switch_step: int = 20, policy=None, seed: int = 0):
    schedule = DriftSchedule(switches=((switch_step, 1.0),))
    policy = policy or safe_set_cycle_policy()
    return rollout(policy, seed=seed, initial_dial=0.0, drift_schedule=schedule, episode_length=64)


# ── valid_raga_adherence ─────────────────────────────────────────────────

def test_valid_raga_adherence_safe_set_cycle_is_perfect_in_every_segment():
    """safe_set_cycle never violates a forbidden-note rule in either raga —
    exactly the degenerate case this metric alone can't catch (F13); paired
    with safe_set_occupancy it should."""
    traj = _switched_trajectory(policy=safe_set_cycle_policy())
    result = valid_raga_adherence([traj])
    assert result.pre_switch == 1.0
    assert result.grace == 1.0
    assert result.post_grace == 1.0
    assert result.overall == 1.0


def test_valid_raga_adherence_splits_by_segment():
    traj = _switched_trajectory(switch_step=10, policy=random_valid_policy(random.Random(0)))
    result = valid_raga_adherence([traj])
    # random-valid never violates the currently-active raga either, so this
    # mainly checks the split doesn't crash and covers all three segments
    # (pre_switch steps 0-9, grace 10-12, post_grace 13-63).
    assert 0.0 <= result.pre_switch <= 1.0
    assert 0.0 <= result.grace <= 1.0
    assert 0.0 <= result.post_grace <= 1.0


def test_valid_raga_adherence_no_switch_episode_is_all_pre_switch():
    traj = rollout(safe_set_cycle_policy(), seed=0, initial_dial=0.0, episode_length=32)
    assert traj.switch_steps == ()
    result = valid_raga_adherence([traj])
    assert result.pre_switch == 1.0


# ── drift_adaptation_speed ───────────────────────────────────────────────

def test_drift_adaptation_speed_measures_steps_to_first_new_raga_pakad():
    schedule = DriftSchedule(switches=((20, 1.0),))
    traj = rollout(
        scripted_oracle_policy(schedule, random.Random(0)),
        seed=0, initial_dial=0.0, drift_schedule=schedule, episode_length=64,
    )
    result = drift_adaptation_speed([traj])
    assert result.n_switched_episodes == 1
    assert result.censoring_rate == 0.0
    assert result.median_steps is not None
    assert 0 <= result.median_steps <= 8  # bhairav's shortest pakad is 4 notes


def test_drift_adaptation_speed_censors_episodes_that_never_adapt():
    traj = _switched_trajectory(policy=safe_set_cycle_policy())  # never plays a pakad
    result = drift_adaptation_speed([traj])
    assert result.n_switched_episodes == 1
    assert result.censoring_rate == 1.0
    assert result.median_steps is None


def test_drift_adaptation_speed_ignores_episodes_with_no_switch():
    traj = rollout(safe_set_cycle_policy(), seed=0, initial_dial=0.0, episode_length=32)
    result = drift_adaptation_speed([traj])
    assert result.n_switched_episodes == 0


# ── adaptation_success_rate_at_k ─────────────────────────────────────────

def test_adaptation_success_rate_at_k_is_monotonically_nondecreasing():
    schedule = DriftSchedule(switches=((20, 1.0),))
    traj = rollout(
        scripted_oracle_policy(schedule, random.Random(0)),
        seed=0, initial_dial=0.0, drift_schedule=schedule, episode_length=64,
    )
    result = adaptation_success_rate_at_k([traj], ks=(5, 10, 20))
    assert result[5] <= result[10] <= result[20]
    assert result[20] == 1.0  # the oracle always adapts eventually, well within 20 steps


def test_adaptation_success_rate_at_k_zero_for_a_policy_that_never_adapts():
    traj = _switched_trajectory(policy=safe_set_cycle_policy())
    result = adaptation_success_rate_at_k([traj], ks=(5, 10, 20))
    assert result == {5: 0.0, 10: 0.0, 20: 0.0}


# ── post_switch_violation_decay ──────────────────────────────────────────

def test_post_switch_violation_decay_is_zero_for_a_policy_with_no_violations():
    traj = _switched_trajectory(policy=safe_set_cycle_policy())
    result = post_switch_violation_decay([traj], horizon=8)
    assert all(rate == 0.0 for rate in result.values())
    assert set(result.keys()) == set(range(8))


# ── pakad_rate, jugalbandi_coherence, safe_set_occupancy ─────────────────

def test_pakad_rate_counts_completions_per_episode():
    traj_a = rollout(safe_set_cycle_policy(), seed=0, initial_dial=0.0, episode_length=32)  # 0 pakads
    schedule = DriftSchedule(switches=((20, 1.0),))
    traj_b = rollout(
        scripted_oracle_policy(schedule, random.Random(0)),
        seed=0, initial_dial=0.0, drift_schedule=schedule, episode_length=64,
    )
    result = pakad_rate([traj_a, traj_b])
    assert result > 0  # traj_b contributes at least one pakad completion


def test_jugalbandi_coherence_is_nan_with_no_call_events():
    """A 3-step episode never reaches CALL_EVERY=8, so there are no
    call-response pairs to average over."""
    traj = rollout(lambda o, i: 12, seed=0, initial_dial=0.0, episode_length=3)
    result = jugalbandi_coherence([traj])
    assert result != result  # NaN != NaN


def test_safe_set_occupancy_is_one_for_safe_set_cycle_policy():
    traj = rollout(safe_set_cycle_policy(), seed=0, initial_dial=0.0, episode_length=32)
    assert safe_set_occupancy([traj]) == 1.0


def test_safe_set_occupancy_is_zero_when_never_in_the_safe_set():
    # Ma# (6) is valid in Yaman and outside the safe set {Sa,Ga,Pa,Ni}.
    traj = rollout(lambda o, i: 18, seed=0, initial_dial=0.0, episode_length=16)
    assert safe_set_occupancy([traj]) == 0.0


# ── compute_all_metrics: the full bundle is JSON-serialisable ───────────

def test_compute_all_metrics_report_is_json_serialisable():
    schedule = DriftSchedule(switches=((20, 1.0),))
    trajs = [
        rollout(
            scripted_oracle_policy(schedule, random.Random(s)),
            seed=s, initial_dial=0.0, drift_schedule=schedule, episode_length=64,
        )
        for s in range(5)
    ]
    report = compute_all_metrics(trajs)
    encoded = json.dumps(report.to_dict())
    assert "drift_adaptation_speed" in encoded

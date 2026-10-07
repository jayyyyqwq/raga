from raaga_env.jugalbandi_env import JugalbandiEnv

from eval.rollout import DriftSchedule, Trajectory, rollout, rollout_from_state


def _cycling_policy(actions: list[int]):
    """A scripted, deterministic policy: ignores obs/info, cycles a fixed
    action list. Used to test harness determinism, independent of any
    trained model."""
    counter = {"i": 0}

    def policy(observation, info) -> int:
        action = actions[counter["i"] % len(actions)]
        counter["i"] += 1
        return action

    return policy


def test_rollout_is_deterministic_across_invocations():
    actions = [11, 9, 7, 4, 40, 16, 28]
    schedule = DriftSchedule(switches=((20, 0.75),))

    traj_a = rollout(_cycling_policy(actions), seed=7, drift_schedule=schedule, arm="hidden")
    traj_b = rollout(_cycling_policy(actions), seed=7, drift_schedule=schedule, arm="hidden")

    assert traj_a.total_reward == traj_b.total_reward
    assert len(traj_a.steps) == len(traj_b.steps)
    for step_a, step_b in zip(traj_a.steps, traj_b.steps):
        assert step_a == step_b


def test_rollout_runs_full_episode_length():
    actions = [0, 4, 7, 11]
    traj = rollout(_cycling_policy(actions), seed=1, episode_length=64)
    assert isinstance(traj, Trajectory)
    assert len(traj.steps) == 64


def test_drift_schedule_switches_raga_mid_episode():
    actions = [0, 4, 7, 11]
    schedule = DriftSchedule(switches=((10, 0.75),))
    traj = rollout(_cycling_policy(actions), seed=3, drift_schedule=schedule, episode_length=32)

    assert traj.steps[9].info["active_raga"] == "yaman"
    assert traj.steps[10].info["active_raga"] == "bhairav"


def test_trajectory_records_its_own_switch_steps():
    actions = [0, 4, 7, 11]
    schedule = DriftSchedule(switches=((20, 0.75),))
    traj = rollout(_cycling_policy(actions), seed=0, drift_schedule=schedule, episode_length=32)
    assert traj.switch_steps == (20,)


def test_trajectory_switch_steps_empty_when_no_schedule():
    actions = [0, 4, 7, 11]
    traj = rollout(_cycling_policy(actions), seed=0, episode_length=32)
    assert traj.switch_steps == ()


# ── call_phrase_fn (eval harness used to never call env.set_call() at all —
# jugalbandi_coherence/call_echo_rate were silently NaN/0 for every policy
# ever run through this module) ─────────────────────────────────────────

_JUGALBANDI_KEYS = {"tension_resolve", "direction_contrast", "call_echo"}


def _fixed_call_phrase(raga, rng):
    return [0, 2, 4, 7]  # fixed, ignores raga/rng — just needs to be raga-valid


def _random_call_phrase(raga, rng):
    return [rng.choice(sorted(raga["valid_notes"])) for _ in range(4)]


def test_rollout_without_call_phrase_fn_never_submits_a_call():
    """Default/backward-compatible behaviour: omitting call_phrase_fn (every
    call site before this parameter existed) must still never call
    env.set_call() — no jugalbandi reward term can appear without a call."""
    actions = [0, 4, 7, 11]
    traj = rollout(_cycling_policy(actions), seed=0, episode_length=32)
    assert not any(_JUGALBANDI_KEYS & s.reward_breakdown.keys() for s in traj.steps)


def test_rollout_with_call_phrase_fn_submits_calls_on_the_env_cadence():
    actions = [0, 2, 4, 7]  # Sa Re Ga Pa — stays in Yaman, no forbidden notes
    traj = rollout(
        _cycling_policy(actions), seed=0, episode_length=32, call_phrase_fn=_fixed_call_phrase,
    )
    # CALL_EVERY=8, so call_requested fires at step indices 7, 15, 23, 31 —
    # by the next step after each, the call must be visible in the obs
    # (obs[8:12] are the call-phrase slots, JugalbandiEnv's own encoding).
    first_call_requested = next(s.step for s in traj.steps if s.info.get("call_requested"))
    after = traj.steps[first_call_requested + 1]
    assert any(after.observation[8:12]), "call phrase never reached obs after call_requested fired"


def test_rollout_call_sequence_is_deterministic_for_the_same_seed():
    actions = [0, 2, 4, 7]
    traj_a = rollout(
        _cycling_policy(actions), seed=5, episode_length=32, call_phrase_fn=_random_call_phrase,
    )
    traj_b = rollout(
        _cycling_policy(actions), seed=5, episode_length=32, call_phrase_fn=_random_call_phrase,
    )
    assert traj_a.steps == traj_b.steps


def test_rollout_from_state_submits_no_call_unless_given_a_call_fn():
    """Default: no call injection — only an explicit call_phrase_fn (the
    source episode's own generator) starts submitting calls."""
    env = JugalbandiEnv(initial_dial=0.0)
    env.reset(seed=0)
    for _ in range(10):
        env.step(4)
    snapshot = env.get_state()

    traj = rollout_from_state(
        _cycling_policy([2]), state=snapshot, episode_length=64, max_extra_steps=10,
    )
    assert not any(_JUGALBANDI_KEYS & s.reward_breakdown.keys() for s in traj.steps)


# ── rollout_from_state (Phase 2.3 / Claim B: per-step scoring needs a way to
# restart from a mid-episode snapshot instead of a fresh reset) ────────────

def test_rollout_from_state_continues_from_the_snapshot_not_from_scratch():
    env = JugalbandiEnv(initial_dial=0.0)
    env.reset(seed=0)
    for note in (11, 9, 7):  # a real pakad — leaves step_count=3, drought reset
        env.step(note)
    snapshot = env.get_state()
    assert snapshot["step_count"] == 3

    traj = rollout_from_state(
        _cycling_policy([4]), state=snapshot, episode_length=64, max_extra_steps=5,
    )
    assert len(traj.steps) == 5
    assert traj.steps[0].step == 3  # continues numbering from the snapshot, not 0
    assert traj.steps[-1].step == 7


def test_rollout_from_state_caps_at_episode_length():
    env = JugalbandiEnv(initial_dial=0.0, episode_length=64)
    env.reset(seed=0)
    for _ in range(60):
        env.step(4)
    snapshot = env.get_state()
    assert snapshot["step_count"] == 60

    traj = rollout_from_state(
        _cycling_policy([4]), state=snapshot, episode_length=64, max_extra_steps=20,
    )
    assert len(traj.steps) == 4  # only 4 steps remain before episode_length=64


def test_rollout_from_state_is_deterministic_for_a_deterministic_policy():
    env = JugalbandiEnv(initial_dial=0.75)
    env.reset(seed=0)
    for note in (13, 12, 11):
        env.step(note)
    snapshot = env.get_state()

    actions = [12, 17, 20, 16]
    traj_a = rollout_from_state(
        _cycling_policy(actions), state=snapshot, episode_length=64, max_extra_steps=8,
    )
    traj_b = rollout_from_state(
        _cycling_policy(actions), state=snapshot, episode_length=64, max_extra_steps=8,
    )
    assert traj_a.total_reward == traj_b.total_reward
    assert traj_a.steps == traj_b.steps


def test_rollout_from_state_matches_a_manual_env_replay():
    """The whole point: rollout_from_state's reward must agree with hand-
    stepping the same restored env — this is what makes per-step training
    reward trustworthy against Phase 0.3's single source of truth."""
    env = JugalbandiEnv(initial_dial=0.0)
    env.reset(seed=0)
    for note in (11, 9, 7, 4):
        env.step(note)
    snapshot = env.get_state()

    manual_env = JugalbandiEnv(episode_length=64)
    manual_env.set_state(snapshot)
    expected_rewards = []
    for note in (4, 4, 4):
        _, reward, *_ = manual_env.step(note)
        expected_rewards.append(reward)

    traj = rollout_from_state(
        _cycling_policy([4]), state=snapshot, episode_length=64, max_extra_steps=3,
    )
    assert [s.reward for s in traj.steps] == expected_rewards



def test_rollout_from_state_applies_a_switch_inside_the_horizon():
    import random as _random
    env = JugalbandiEnv(initial_dial=0.0)
    env.reset(seed=0)
    for _ in range(10):
        env.step(4)
    traj = rollout_from_state(
        _cycling_policy([4]), state=env.get_state(), episode_length=64, max_extra_steps=8,
        drift_schedule=DriftSchedule(switches=((13, 1.0),)),
    )
    assert traj.switch_steps == (13,)
    assert traj.steps[-1].info["active_raga"] == "bhairav"
    assert traj.steps[0].info["active_raga"] == "yaman"


def test_rollout_from_state_injects_calls_when_given_a_call_fn():
    import random as _random
    env = JugalbandiEnv(initial_dial=0.0)
    env.reset(seed=0)
    for _ in range(4):
        env.step(4)
    traj = rollout_from_state(
        _cycling_policy([4]), state=env.get_state(), episode_length=64, max_extra_steps=12,
        call_phrase_fn=_random_call_phrase, call_rng=_random.Random(0),
    )
    assert any(s.observation[8:12] != (0.0, 0.0, 0.0, 0.0) for s in traj.steps)

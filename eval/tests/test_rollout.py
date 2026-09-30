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

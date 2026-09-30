# Phase 3 (revised for Claim B / Phase 2.3) acceptance tests for
# training/train_grpo.py's per-step data pipeline.
#
# GRPOTrainer.train() itself needs the GPU-only stack (unsloth/trl/torch),
# which is deliberately not installed locally (requirements-train.txt is
# Colab/GPU-only — see requirements-train.txt's header comment). What's
# tested here is everything up to that boundary: drift scheduling, dataset
# snapshotting, and — critically — that the reward function scores a sampled
# action exactly the way hand-replaying the same restored state would,
# since Phase 0.3's whole point is that training reward and eval reward must
# go through the same code path (here: JugalbandiEnv.set_state +
# eval.rollout.rollout_from_state).

import json
import random

import pytest

from eval.policies import random_valid_policy
from eval.rollout import rollout_from_state
from raaga_env.jugalbandi_env import JugalbandiEnv
from raaga_env.prompting import Arm
from training.train_grpo import (
    DRIFT_WINDOW,
    EPISODE_LENGTH,
    MC_HORIZON,
    NO_SWITCH_CONTROL_PROB,
    STEP_PARSE_FAILURE_PENALTY,
    build_dataset,
    make_step_reward_fn,
    sample_drift_schedule,
)


# ── 3.3: drift scheduling, with the no-switch control condition (unchanged) ─

def test_sample_drift_schedule_respects_control_condition_rate():
    rng = random.Random(0)
    n = 5000
    no_switch = 0
    for _ in range(n):
        initial_dial, schedule = sample_drift_schedule(rng)
        assert initial_dial in (0.0, 1.0)
        if not schedule.switches:
            no_switch += 1
        else:
            (step, new_dial), = schedule.switches
            assert DRIFT_WINDOW[0] <= step <= DRIFT_WINDOW[1]
            assert new_dial != initial_dial
            assert new_dial in (0.0, 1.0)

    rate = no_switch / n
    assert abs(rate - NO_SWITCH_CONTROL_PROB) < 0.02, f"no-switch rate {rate} drifted from {NO_SWITCH_CONTROL_PROB}"


# ── 3.1/2.3: dataset is built in-process, one row per decision point ───────

def test_build_dataset_produces_expected_columns():
    rng = random.Random(1)
    rows = build_dataset(80, Arm.HIDDEN, rng)
    assert len(rows) == 80
    for row in rows:
        assert set(row.keys()) == {"prompt", "state_json", "mc_seed"}
        assert isinstance(row["prompt"], str)
        assert "hidden" not in row["prompt"].lower()  # sanity: arm-gating actually engaged
        state = json.loads(row["state_json"])
        assert 0 <= state["step_count"] < EPISODE_LENGTH


def test_build_dataset_spans_both_pre_and_post_switch_snapshots():
    rng = random.Random(1)
    rows = build_dataset(200, Arm.HIDDEN, rng)
    steps = [json.loads(r["state_json"])["step_count"] for r in rows]
    assert min(steps) < 16   # some snapshots taken well before the drift window
    assert max(steps) > 48   # some snapshots taken well after it


def test_build_dataset_snapshots_include_feedback_except_at_step_zero():
    rng = random.Random(1)
    rows = build_dataset(200, Arm.HIDDEN, rng)
    at_step_zero = [r for r in rows if json.loads(r["state_json"])["step_count"] == 0]
    after_step_zero = [r for r in rows if json.loads(r["state_json"])["step_count"] > 0]
    assert at_step_zero and after_step_zero  # both must occur over 200 samples

    for row in at_step_zero:
        assert "Last action" not in row["prompt"]
    for row in after_step_zero:
        assert "Last action" in row["prompt"]


def test_build_dataset_oracle_arm_names_the_opening_raga():
    rng = random.Random(2)
    rows = build_dataset(10, Arm.ORACLE, rng)
    for row in rows:
        assert ("yaman" in row["prompt"].lower()) or ("bhairav" in row["prompt"].lower())


# ── 2.3/3.2 acceptance: step_reward() must agree with a direct hand-replay ──
# of the same restored state — this is what "the reward function predicts
# analytically" means here: both training and eval go through
# JugalbandiEnv.set_state + eval.rollout.rollout_from_state (Phase 0.3/0.4).

def test_step_reward_matches_a_manual_state_replay():
    env = JugalbandiEnv(initial_dial=0.75)
    env.reset(seed=0)
    for note in (13, 12, 11):  # walk into the middle of an episode
        env.step(note)
    state = env.get_state()
    seed = 12345
    action = 4  # Ga

    manual_env = JugalbandiEnv(episode_length=EPISODE_LENGTH)
    manual_env.set_state(state)
    _, immediate, terminated, truncated, _ = manual_env.step(action)
    expected = immediate
    if not (terminated or truncated):
        continuation = rollout_from_state(
            random_valid_policy(random.Random(seed)),
            state=manual_env.get_state(),
            episode_length=EPISODE_LENGTH,
            max_extra_steps=MC_HORIZON,
            arm=Arm.HIDDEN.value,
        )
        expected += continuation.total_reward

    reward_fn = make_step_reward_fn(Arm.HIDDEN)
    got = reward_fn(
        prompts=["dummy"],
        completions=[str(action)],
        state_json=[json.dumps(state)],
        mc_seed=[seed],
    )
    assert got == [expected]


def test_step_reward_is_deterministic_for_the_same_mc_seed():
    env = JugalbandiEnv(initial_dial=0.0)
    env.reset(seed=0)
    state = env.get_state()
    reward_fn = make_step_reward_fn(Arm.DIAL)

    a = reward_fn(prompts=["p"], completions=["4"], state_json=[json.dumps(state)], mc_seed=[7])
    b = reward_fn(prompts=["p"], completions=["4"], state_json=[json.dumps(state)], mc_seed=[7])
    assert a == b


def test_step_reward_reflects_the_same_action_becoming_valid_after_a_switch():
    """The whole premise of Claim B: whether an action is good depends on
    the (invisible-in-text, for DIAL/HIDDEN) current regime. Natural Ma (5)
    is forbidden in Yaman but valid in Bhairav — the reward for playing it
    must differ between a still-Yaman state and a just-switched-to-Bhairav
    state, even though the two states look identical to a DIAL/HIDDEN-arm
    prompt reader. This is the signal training needs for the feedback
    channel to be learnable at all."""
    still_yaman = JugalbandiEnv(initial_dial=0.0)
    still_yaman.reset(seed=0)
    state_yaman = still_yaman.get_state()

    switched_to_bhairav = JugalbandiEnv(initial_dial=0.0)
    switched_to_bhairav.reset(seed=0)
    switched_to_bhairav.set_dial(1.0)
    state_bhairav = switched_to_bhairav.get_state()

    reward_fn = make_step_reward_fn(Arm.HIDDEN)
    reward_in_yaman = reward_fn(
        prompts=["p"], completions=["5"], state_json=[json.dumps(state_yaman)], mc_seed=[1],
    )[0]
    reward_in_bhairav = reward_fn(
        prompts=["p"], completions=["5"], state_json=[json.dumps(state_bhairav)], mc_seed=[1],
    )[0]
    assert reward_in_bhairav > reward_in_yaman


def test_step_reward_penalises_malformed_completion_without_crashing():
    env = JugalbandiEnv()
    env.reset(seed=0)
    state_json = json.dumps(env.get_state())
    reward_fn = make_step_reward_fn(Arm.DIAL)

    got = reward_fn(
        prompts=["p1", "p2"],
        completions=["not a number", "4"],
        state_json=[state_json, state_json],
        mc_seed=[1, 1],
    )
    assert got[0] == STEP_PARSE_FAILURE_PENALTY
    assert got[1] != STEP_PARSE_FAILURE_PENALTY


# ── proxy for "a 20-step training run completes without crashing" ──────────
# GRPOTrainer.train() needs the GPU stack; this exercises everything TRL
# would actually call into on the data side — build_dataset() then
# reward_funcs() over the *entire* simulated batch — without it.

def test_full_pipeline_runs_without_gpu_stack_for_20_steps_worth_of_data():
    rng = random.Random(3)
    batch, num_generations = 4, 4
    steps = 20
    rows = build_dataset(steps * batch, Arm.DIAL, rng)
    assert len(rows) == steps * batch

    reward_fn = make_step_reward_fn(Arm.DIAL)
    # Simulate GRPO sampling num_generations completions per prompt: a mix
    # of well-formed and malformed ones, to exercise both reward paths.
    prompts = [r["prompt"] for r in rows for _ in range(num_generations)]
    state_json = [r["state_json"] for r in rows for _ in range(num_generations)]
    mc_seed = [r["mc_seed"] for r in rows for _ in range(num_generations)]
    completions = []
    for i in range(len(prompts)):
        completions.append("garbage" if i % 5 == 0 else str(rng.randint(0, 95)))

    rewards = reward_fn(
        prompts=prompts, completions=completions, state_json=state_json, mc_seed=mc_seed,
    )
    assert len(rewards) == len(prompts)
    assert all(isinstance(r, float) for r in rewards)

    switch_present = any(
        json.loads(sj)["drift"]["steps_since_switch"] < 999 for sj in state_json
    )
    assert switch_present, "logged episodes must show drift events occurring"

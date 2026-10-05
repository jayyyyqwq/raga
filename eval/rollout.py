# In-process evaluation harness — no HTTP (updatedplan.md Phase 0.3).
#
# WHY this exists: training and ad-hoc scripts drove the env through the HTTP
# server, so a metric definition (e.g. what counts as "adherence") could
# silently differ between the script that produced a figure and the script
# that produced the next one. Every downstream number — baselines, trained
# arms, ablations, listening-study stimuli — must come through rollout() and
# metrics() so a change to a metric definition applies everywhere at once.

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from raaga_env.jugalbandi_env import JugalbandiEnv

# Type of eval.policies.sample_call_phrase, injected rather than imported
# directly — eval.policies imports DriftSchedule FROM this module, so a
# top-level import the other way would be circular. This module stays
# knowing nothing about *how* a call phrase gets generated, same as it
# already knows nothing about prompts/arms/models for Policy itself.
CallPhraseFn = Callable[[dict, random.Random], list[int]]


class Policy(Protocol):
    """A policy is any callable mapping (observation, info) to a legal action
    index. Arm-specific prompt rendering (Phase 2) and model inference belong
    inside the policy's closure, not in this harness — this module knows
    nothing about prompts, arms, or models, only about stepping the env."""

    def __call__(self, observation: np.ndarray, info: dict) -> int: ...


@dataclass(frozen=True)
class DriftSchedule:
    """When to move the raga dial during an episode. `switches` maps a step
    index to the dial value to set immediately before that step is taken."""

    switches: tuple[tuple[int, float], ...] = ()

    def as_dict(self) -> dict[int, float]:
        return dict(self.switches)


@dataclass(frozen=True)
class StepRecord:
    step: int
    observation: tuple[float, ...]
    action: int
    reward: float
    reward_breakdown: dict
    info: dict


@dataclass(frozen=True)
class Trajectory:
    seed: int
    arm: str | None
    steps: tuple[StepRecord, ...]
    total_reward: float
    # Step indices where the raga dial changed during this trajectory (empty
    # = no switch). Carried on the Trajectory itself, rather than requiring
    # every metric function to be handed the DriftSchedule separately, since
    # timing-based metrics (drift_adaptation_speed, post_switch_violation_
    # decay, the pre/grace/post-grace adherence split — Phase 4.3) all need
    # to know this and a trajectory should be self-describing.
    switch_steps: tuple[int, ...] = ()


def _run_steps(
    env: JugalbandiEnv,
    policy: Policy,
    *,
    obs: np.ndarray,
    info: dict,
    n_steps: int,
    step_offset: int,
    drift_schedule_by_step: dict[int, float] | None = None,
    call_phrase_fn: CallPhraseFn | None = None,
    call_rng: random.Random | None = None,
) -> list[StepRecord]:
    """Shared step loop for rollout() and rollout_from_state() — the one
    place a StepRecord gets built, so both entry points score identically.

    call_phrase_fn/call_rng: when both given, submits a fresh call via
    env.set_call() on the env's own CALL_EVERY cadence (info["call_requested"]
    firing after env.step()) — same ordering as train_grpo.py's build_dataset
    (break on terminated first, then check call_requested). Left as None by
    rollout_from_state() deliberately: that function continues an already-
    in-progress training snapshot for a short Monte-Carlo horizon, and must
    not start injecting calls that build_dataset() itself didn't put there —
    changing what call is active mid-continuation would make per-step
    training reward disagree with a hand-replay of the same state (Phase 0.3)."""
    schedule = drift_schedule_by_step or {}
    steps: list[StepRecord] = []
    for i in range(n_steps):
        step_idx = step_offset + i
        if step_idx in schedule:
            env.set_dial(schedule[step_idx])
            obs = env._get_obs()

        action = policy(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        breakdown = dict(info.get("reward_breakdown", {}))
        steps.append(
            StepRecord(
                step=step_idx,
                observation=tuple(float(x) for x in obs),
                action=int(action),
                reward=float(reward),
                reward_breakdown=breakdown,
                info={k: v for k, v in info.items() if k != "reward_breakdown"},
            )
        )
        if terminated or truncated:
            break

        if call_phrase_fn is not None and info.get("call_requested"):
            env.set_call(call_phrase_fn(env.raga, call_rng))
            obs = env._get_obs()
    return steps


def rollout(
    policy: Policy,
    *,
    seed: int,
    drift_schedule: DriftSchedule | None = None,
    arm: str | None = None,
    episode_length: int = 64,
    initial_dial: float = 0.0,
    call_phrase_fn: CallPhraseFn | None = None,
) -> Trajectory:
    """Run one full episode in-process, from a fresh reset, and return every
    step's observation, action, reward and breakdown.

    call_phrase_fn: pass eval.policies.sample_call_phrase (or compatible) to
    have this episode submit a fresh call on the env's own CALL_EVERY cadence
    — without it, call_phrase stays empty all episode and every jugalbandi
    metric (eval/metrics.py's jugalbandi_coherence, call_echo_rate) is
    permanently undefined for whatever this call produces, same as every
    policy run through this harness before this parameter existed. Omit it
    deliberately for tests that don't care about call-response at all.

    Determinism contract: for a fixed seed, a fixed drift_schedule and a
    deterministic policy, two calls return bit-identical trajectories. The
    env's own randomness (gymnasium's seeded np_random) is the only source of
    *environment* stochasticity, and it is seeded here. If call_phrase_fn is
    given, the call sequence is a second, independent but equally
    seed-deterministic source — seeded from `seed` too (random.Random(seed),
    a separate instance from gymnasium's np_random, so it doesn't interact
    with it) — so the determinism contract still holds exactly.
    """
    env = JugalbandiEnv(initial_dial=initial_dial, episode_length=episode_length)
    obs, info = env.reset(seed=seed)
    steps = _run_steps(
        env, policy, obs=obs, info=info, n_steps=episode_length, step_offset=0,
        drift_schedule_by_step=drift_schedule.as_dict() if drift_schedule else None,
        call_phrase_fn=call_phrase_fn,
        call_rng=random.Random(seed) if call_phrase_fn is not None else None,
    )
    total_reward = sum(s.reward for s in steps)
    switch_steps = tuple(step for step, _ in drift_schedule.switches) if drift_schedule else ()
    return Trajectory(
        seed=seed, arm=arm, steps=tuple(steps), total_reward=total_reward, switch_steps=switch_steps,
    )


def rollout_from_state(
    policy: Policy,
    *,
    state: dict,
    episode_length: int,
    max_extra_steps: int,
    arm: str | None = None,
) -> Trajectory:
    """rollout()'s sibling for continuing from a mid-episode snapshot instead
    of a fresh reset — restores a JugalbandiEnv to `state` (Phase 0.4) and
    runs `policy` for up to max_extra_steps more steps, capped so it never
    runs past `episode_length` total.

    Used by per-step GRPO training (Phase 2.3 / Claim B) to compute a
    Monte-Carlo return: apply the sampled action, then continue with a
    reference policy for a short horizon, so one action gets more than a
    single step's worth of credit assignment without generating a full blind
    episode (a single blind completion can't react to feedback about steps
    it hasn't taken yet, which is incompatible with Claim B). `total_reward`
    here is only the reward accumulated during *this* call — the return from
    the snapshot forward, not the whole episode's cumulative reward, which is
    what a scoring function comparing different first actions from the same
    state actually needs.
    """
    env = JugalbandiEnv(episode_length=episode_length)
    env.set_state(state)
    obs = env._get_obs()
    info: dict = {}

    n_steps = max(0, min(max_extra_steps, episode_length - state["step_count"]))
    steps = _run_steps(
        env, policy, obs=obs, info=info, n_steps=n_steps, step_offset=state["step_count"],
    )
    total_reward = sum(s.reward for s in steps)
    return Trajectory(seed=-1, arm=arm, steps=tuple(steps), total_reward=total_reward)

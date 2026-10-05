# Reusable reference policies for the eval.rollout harness.
#
# random_valid_policy is used two ways by per-step GRPO training (Phase
# 2.3 / Claim B): to generate the reference trajectories build_dataset()
# snapshots prompts from, and as the Monte-Carlo continuation policy after a
# sampled action. It's also updatedplan.md Phase 4.1's `random-valid`
# baseline ("Honest floor. Without it, a policy that learned only Yaman's
# scale looks impressive") — same policy, same reason to have exactly one
# definition of it.
#
# random_uniform_policy, safe_set_cycle_policy and scripted_oracle_policy are
# Phase 4.1's other three named baselines: the absolute floor, the named
# degenerate-policy detector (F13), and the ceiling that calibrates what
# drift_adaptation_speed can physically be.
#
# sample_call_phrase is used the same two-places-one-definition way as
# random_valid_policy above: by training (train_grpo.py's build_dataset,
# which used to define its own copy — moved here 2026-10 once
# eval.rollout.rollout() needed the exact same generator, see that
# module's call_rng) and by eval.rollout.rollout(), which injects a call
# every CALL_EVERY steps so jugalbandi_coherence/call_echo_rate
# (eval/metrics.py) have anything to measure at all — eval.rollout never
# called env.set_call() before this, which meant those two metrics were
# silently NaN/0 for every policy ever run through the eval harness,
# baselines included, regardless of how good a policy actually was at
# answering a call.

from __future__ import annotations

import random

from raaga_env.ragas import RAGAS, raga_from_dial

from .rollout import DriftSchedule

ACTION_SPACE_N = 96

# {Sa, Ga, Pa, Ni} — penalised in neither raga's forbidden-note set (F13).
# A policy confined to these never triggers a forbidden-note violation in
# either raga, so it scores identically before and after every drift event
# while demonstrating exactly zero adaptation.
SAFE_SET_SWARAS = (0, 4, 7, 11)


def random_uniform_policy(rng: random.Random = random):
    """Uniform over all 96 actions, with no awareness of which notes are
    even legal. Phase 4.1's absolute floor — every other policy should beat
    this by a wide margin, or something upstream is broken."""

    def policy(observation, info) -> int:
        return rng.randint(0, ACTION_SPACE_N - 1)

    return policy


def random_valid_policy(rng: random.Random = random):
    """Uniformly samples a note valid in the *currently active* raga (read
    directly off the obs vector's dial dim, so this works from any obs
    regardless of which arm rendered the prompt the model saw — the policy
    itself always has full information about the env it's stepping), a
    random duration, and a random register."""

    def policy(observation, info) -> int:
        raga = RAGAS[raga_from_dial(observation[15])]
        swara = rng.choice(sorted(raga["valid_notes"]))
        duration = rng.randint(0, 3)
        register = rng.choice((0, 12))
        note = register + swara
        return note + duration * 24

    return policy


def sample_call_phrase(raga: dict, rng: random.Random = random) -> list[int]:
    """A plausible human call phrase: 4 swaras drawn uniformly from the
    currently-active raga's valid notes (same sampling approach as
    random_valid_policy, minus register/duration — calls are swara-only,
    see JugalbandiEnv.set_call's contract).

    Deliberately raga-valid, not uniform over 0-11: a real human partner's
    phrase is musically coherent too, and training/evaluating against
    garbage calls (forbidden swaras with no raga-grammar meaning) would
    reward treating the call as noise instead of as something to listen to.

    Documented caveat (ML retrain finding, 2026-10; docs/RETRAIN_PLAN.md):
    because this is raga-valid, the call phrase's note choices become a
    soft, implicit signal about the active raga for every arm the call
    renders in (raaga_env.prompting), including HIDDEN. HIDDEN's guarantee
    is still "no raga name or dial value is ever written in words" — that's
    unchanged and still enforced by test_prompting.py's leak tests — but it
    is no longer a guarantee that *zero* information about the raga reaches
    the prompt. Treated as acceptable and realistic (a real accompanist
    gives exactly this kind of implicit signal) rather than worked around.
    """
    return [rng.choice(sorted(raga["valid_notes"])) for _ in range(4)]


def safe_set_cycle_policy():
    """Cycles Sa, Ga, Pa, Ni (madhya register, sixteenth notes) forever.
    Names the F13 degenerate-policy risk explicitly so it's detected by
    `safe_set_occupancy`, not discovered by a reviewer: a trained policy
    that converges here looks flawless on valid_raga_adherence while
    demonstrating zero adaptation to drift."""
    cycle = [12 + s for s in SAFE_SET_SWARAS]  # madhya register absolute pitches
    counter = {"i": 0}

    def policy(observation, info) -> int:
        note = cycle[counter["i"] % len(cycle)]
        counter["i"] += 1
        return note  # duration index 0

    return policy


def scripted_oracle_policy(schedule: DriftSchedule, rng: random.Random = random):
    """Plays random-valid normally, but the instant the dial switches, it
    "cheats" by immediately queuing the new raga's shortest pakad (reading
    the post-switch raga straight off the observation, not inferring it) —
    Phase 4.1's ceiling, used to calibrate what drift_adaptation_speed can
    physically be. Requires the DriftSchedule up front because it needs to
    recognise the switch step exactly, which a real policy under evaluation
    would not be given."""
    switch_steps = frozenset(step for step, _ in schedule.switches)
    fallback = random_valid_policy(rng)
    state = {"i": 0, "queue": []}

    def policy(observation, info) -> int:
        step = state["i"]
        state["i"] += 1

        if step in switch_steps:
            raga = RAGAS[raga_from_dial(observation[15])]
            shortest_pakad, _multiplier = min(raga["pakads"], key=lambda p: len(p[0]))
            state["queue"] = list(shortest_pakad)

        if state["queue"]:
            note = state["queue"].pop(0)
            return note  # duration index 0

        return fallback(observation, info)

    return policy

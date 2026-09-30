# The fixed, shared evaluation episode set (updatedplan.md Phase 4.2).
#
# WHY fixed and shared: every policy (baselines and trained arms alike) must
# be compared on identical conditions — same starting raga, same drift
# timing — or a difference in scores could just be a difference in which
# episodes each policy happened to get. Pairing is worth roughly a factor of
# three in statistical power here (Phase 4.2) and costs nothing once this
# list is generated once and reused everywhere.
#
# Unlike training's sample_drift_schedule() (train_grpo.py), which includes
# a 20% no-switch control condition so the policy being trained can't just
# learn "a switch always happens", every evaluation episode here contains
# exactly one switch: the metrics this set exists to support
# (drift_adaptation_speed, post_switch_violation_decay, the pre/grace/
# post-grace adherence split) are only defined for episodes that actually
# switch. A policy's behaviour on non-switching episodes is already
# what valid_raga_adherence measures over the "pre-switch" segment of every
# episode here, from step 0 to the switch.

from __future__ import annotations

import random
from dataclasses import dataclass

DRIFT_WINDOW = (16, 48)   # matches training/train_grpo.py's DRIFT_WINDOW
EPISODE_LENGTH = 64
N_EVAL_EPISODES = 200
# Fixed once, never regenerated casually — changing this constant changes
# every downstream number. If it must change, say so explicitly in whatever
# changes it, and expect every existing result to need re-running.
EVAL_MASTER_SEED = 20260910


@dataclass(frozen=True)
class EvalEpisode:
    episode_id: int
    seed: int
    initial_dial: float
    switch_step: int
    switch_dial: float


def generate_eval_episodes(
    n: int = N_EVAL_EPISODES,
    master_seed: int = EVAL_MASTER_SEED,
) -> tuple[EvalEpisode, ...]:
    """Deterministically generates the shared eval set: n episodes, split as
    evenly as possible between starting in Yaman and starting in Bhairav,
    each with one switch step drawn uniformly from DRIFT_WINDOW. Calling
    this twice with the same arguments returns identical episodes — that
    determinism, not a saved file, is this set's source of truth (consistent
    with how eval/fingerprint.py hashes source files rather than maintaining
    a redundant serialized copy)."""
    rng = random.Random(master_seed)
    episodes = []
    for i in range(n):
        initial_dial = 0.0 if i % 2 == 0 else 1.0  # alternate for an exact 50/50 split
        switch_step = rng.randint(*DRIFT_WINDOW)
        switch_dial = 1.0 if initial_dial < 0.5 else 0.0
        episodes.append(
            EvalEpisode(
                episode_id=i,
                seed=rng.randint(0, 2**31 - 1),
                initial_dial=initial_dial,
                switch_step=switch_step,
                switch_dial=switch_dial,
            )
        )
    return tuple(episodes)


EVAL_EPISODES: tuple[EvalEpisode, ...] = generate_eval_episodes()

# Metric formulas for the Jugalbandi experiment (updatedplan.md Phase 4.3).
#
# WHY this module: openenv.yaml's rubric names these metrics but never
# defined them precisely, which is itself one of the findings that made
# earlier numbers untrustworthy (F12). Every formula here is written down
# once and operates only on eval.rollout.Trajectory, so a baseline, a
# trained arm, and an ablation all get the same definition automatically —
# changing a formula here changes it everywhere at once (Phase 0.3's
# single-source-of-truth principle, extended to metrics).

from __future__ import annotations

import statistics
from dataclasses import dataclass

from raaga_env.jugalbandi_env import CALL_EVERY

from .rollout import Trajectory

GRACE_PERIOD_STEPS = 3  # matches raaga_env.drift.GRACE_PERIOD_STEPS
DEFAULT_SUCCESS_KS = (5, 10, 20)
DEFAULT_DECAY_HORIZON = 16

# {Sa, Ga, Pa, Ni} — duplicated from eval.policies.SAFE_SET_SWARAS rather
# than imported, to avoid metrics.py depending on policies.py for a single
# tuple of four ints; both are defined once each and both trace to F13.
SAFE_SET_SWARAS = (0, 4, 7, 11)


def _offset_since_switch(step: int, switch_steps: tuple[int, ...]) -> int | None:
    """Steps since the most recent switch at or before `step`, or None if
    no switch has happened yet by this step."""
    prior = [s for s in switch_steps if s <= step]
    return step - max(prior) if prior else None


def _segment(offset: int | None) -> str:
    if offset is None:
        return "pre_switch"
    if offset < GRACE_PERIOD_STEPS:
        return "grace"
    return "post_grace"


# ── valid_raga_adherence, split pre-switch / grace / post-grace ────────────

@dataclass(frozen=True)
class AdherenceSplit:
    pre_switch: float
    grace: float
    post_grace: float
    overall: float


def valid_raga_adherence(trajectories: list[Trajectory]) -> AdherenceSplit:
    """Fraction of steps with no forbidden-note violation, split three ways
    (openenv.yaml's rubric names this metric; Phase 4.3 requires the split
    because a single average hides the entire drift-adaptation effect)."""
    counts = {"pre_switch": [0, 0], "grace": [0, 0], "post_grace": [0, 0]}
    for t in trajectories:
        for s in t.steps:
            segment = _segment(_offset_since_switch(s.step, t.switch_steps))
            valid, total = counts[segment]
            counts[segment][1] = total + 1
            if "forbidden_note" not in s.reward_breakdown:
                counts[segment][0] = valid + 1

    def rate(segment: str) -> float:
        valid, total = counts[segment]
        return valid / total if total else float("nan")

    total_valid = sum(c[0] for c in counts.values())
    total_steps = sum(c[1] for c in counts.values())
    return AdherenceSplit(
        pre_switch=rate("pre_switch"),
        grace=rate("grace"),
        post_grace=rate("post_grace"),
        overall=total_valid / total_steps if total_steps else float("nan"),
    )


# ── drift_adaptation_speed, right-censored ──────────────────────────────────

@dataclass(frozen=True)
class AdaptationSpeed:
    median_steps: float | None  # None only if every switched episode is censored
    censoring_rate: float
    n_switched_episodes: int
    observed_steps: tuple[int, ...]  # uncensored observations, for further stats


def drift_adaptation_speed(trajectories: list[Trajectory]) -> AdaptationSpeed:
    """Steps from switch to the first new-raga pakad. Right-censored when no
    pakad occurs before the episode ends — dropping non-adapting episodes
    instead of censoring them is the easiest way to manufacture an
    unrealistically fast number (Phase 4.3), so censored episodes are
    counted, never discarded. Reports a median + censoring rate rather than
    a Kaplan-Meier curve (Phase 4.3 permits either)."""
    switched = [t for t in trajectories if t.switch_steps]
    observed: list[int] = []
    censored = 0
    for t in switched:
        switch = t.switch_steps[0]  # eval protocol: exactly one switch per episode
        first_pakad = next(
            (s.step for s in t.steps if s.step >= switch and "pakad_completion" in s.reward_breakdown),
            None,
        )
        if first_pakad is None:
            censored += 1
        else:
            observed.append(first_pakad - switch)

    n = len(switched)
    return AdaptationSpeed(
        median_steps=statistics.median(observed) if observed else None,
        censoring_rate=censored / n if n else float("nan"),
        n_switched_episodes=n,
        observed_steps=tuple(observed),
    )


def adaptation_success_rate_at_k(
    trajectories: list[Trajectory], ks: tuple[int, ...] = DEFAULT_SUCCESS_KS
) -> dict[int, float]:
    """Fraction of switched episodes that complete a new-raga pakad within K
    steps of the switch, for each K — report the curve, not one number."""
    switched = [t for t in trajectories if t.switch_steps]
    result: dict[int, float] = {}
    for k in ks:
        if not switched:
            result[k] = float("nan")
            continue
        successes = 0
        for t in switched:
            switch = t.switch_steps[0]
            if any(
                switch <= s.step < switch + k and "pakad_completion" in s.reward_breakdown
                for s in t.steps
            ):
                successes += 1
        result[k] = successes / len(switched)
    return result


def post_switch_violation_decay(
    trajectories: list[Trajectory], horizon: int = DEFAULT_DECAY_HORIZON
) -> dict[int, float]:
    """Forbidden-note rate as a function of steps since switch, averaged
    across episodes — the real adaptation figure, more informative than any
    scalar (Phase 4.3)."""
    switched = [t for t in trajectories if t.switch_steps]
    counts = {offset: [0, 0] for offset in range(horizon)}
    for t in switched:
        switch = t.switch_steps[0]
        for s in t.steps:
            offset = s.step - switch
            if 0 <= offset < horizon:
                forbidden, total = counts[offset]
                counts[offset][1] = total + 1
                if "forbidden_note" in s.reward_breakdown:
                    counts[offset][0] = forbidden + 1
    return {
        offset: (c[0] / c[1] if c[1] else float("nan")) for offset, c in counts.items()
    }


def pakad_rate(trajectories: list[Trajectory]) -> float:
    """Pakad completions per episode."""
    if not trajectories:
        return float("nan")
    total = sum(1 for t in trajectories for s in t.steps if "pakad_completion" in s.reward_breakdown)
    return total / len(trajectories)


def jugalbandi_coherence(trajectories: list[Trajectory]) -> float:
    """Average jugalbandi (call-response) reward per call-response pair.
    The jugalbandi reward layer (reward.py) is tension_resolve +
    direction_contrast; a "call-response pair" is one human call phrase
    (info["call_requested"] fires every CALL_EVERY steps) and the agent's
    response to it."""
    jugalbandi_keys = ("tension_resolve", "direction_contrast")
    total_jugalbandi_reward = sum(
        s.reward_breakdown.get(key, 0.0)
        for t in trajectories
        for s in t.steps
        for key in jugalbandi_keys
    )
    total_call_events = sum(
        1 for t in trajectories for s in t.steps if s.info.get("call_requested")
    )
    return total_jugalbandi_reward / total_call_events if total_call_events else float("nan")


def action_validity_rate(n_valid: int, n_total: int) -> float:
    """Fraction of completions parsing to a legal action with no clamping
    (Phase 1.2). Only meaningful for LLM-generated completions — scripted
    policies never fail to parse (they return an int directly, no text
    involved), so this is reported as exactly 1.0 for every baseline in
    this phase rather than computed through this function. Kept as a
    standalone function, not derived from Trajectory, because a Trajectory
    only exists for actions that *did* parse — the denominator this metric
    needs lives upstream of that, in a real completion-parsing loop."""
    return n_valid / n_total if n_total else float("nan")


def safe_set_occupancy(trajectories: list[Trajectory]) -> float:
    """Fraction of steps inside {Sa, Ga, Pa, Ni} (F13). Flags the
    degenerate policy that scores well on valid_raga_adherence while
    demonstrating zero adaptation to drift."""
    total = sum(len(t.steps) for t in trajectories)
    if not total:
        return float("nan")
    in_safe_set = sum(
        1 for t in trajectories for s in t.steps if (s.action % 24) % 12 in SAFE_SET_SWARAS
    )
    return in_safe_set / total


@dataclass(frozen=True)
class MetricReport:
    n_episodes: int
    mean_reward: float
    adherence: AdherenceSplit
    adaptation_speed: AdaptationSpeed
    adaptation_success_at_k: dict[int, float]
    post_switch_violation_decay: dict[int, float]
    pakad_rate: float
    jugalbandi_coherence: float
    safe_set_occupancy: float
    action_validity_rate: float  # 1.0 unless the caller overrides it

    def to_dict(self) -> dict:
        return {
            "n_episodes": self.n_episodes,
            "mean_reward": self.mean_reward,
            "valid_raga_adherence": {
                "pre_switch": self.adherence.pre_switch,
                "grace": self.adherence.grace,
                "post_grace": self.adherence.post_grace,
                "overall": self.adherence.overall,
            },
            "drift_adaptation_speed": {
                "median_steps": self.adaptation_speed.median_steps,
                "censoring_rate": self.adaptation_speed.censoring_rate,
                "n_switched_episodes": self.adaptation_speed.n_switched_episodes,
            },
            "adaptation_success_rate_at_k": {
                str(k): v for k, v in self.adaptation_success_at_k.items()
            },
            "post_switch_violation_decay": {
                str(k): v for k, v in self.post_switch_violation_decay.items()
            },
            "pakad_rate": self.pakad_rate,
            "jugalbandi_coherence": self.jugalbandi_coherence,
            "safe_set_occupancy": self.safe_set_occupancy,
            "action_validity_rate": self.action_validity_rate,
        }


def compute_all_metrics(
    trajectories: list[Trajectory], action_validity_rate_override: float = 1.0
) -> MetricReport:
    """Every Phase 4.3 metric, bundled into one report. This is the function
    every result file in eval/results/ is built from."""
    if not trajectories:
        raise ValueError("compute_all_metrics() requires at least one trajectory")

    return MetricReport(
        n_episodes=len(trajectories),
        mean_reward=sum(t.total_reward for t in trajectories) / len(trajectories),
        adherence=valid_raga_adherence(trajectories),
        adaptation_speed=drift_adaptation_speed(trajectories),
        adaptation_success_at_k=adaptation_success_rate_at_k(trajectories),
        post_switch_violation_decay=post_switch_violation_decay(trajectories),
        pakad_rate=pakad_rate(trajectories),
        jugalbandi_coherence=jugalbandi_coherence(trajectories),
        safe_set_occupancy=safe_set_occupancy(trajectories),
        action_validity_rate=action_validity_rate_override,
    )

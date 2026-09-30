# Reward function — all 5 layers.
# Each layer returns a named breakdown entry so we can log and plot per-component.
# WHY separate file: this is the most iterated part of the project.
# Judges will ask about every shaping decision here.
#
# Note encoding: notes are absolute pitches (0-11 = mandra, 12-23 = madhya).
# All swara-identity checks use  note % 12  to stay register-agnostic.
# Phrase/sequence checks (pakads, aaroha) use absolute pitch — octave matters there.

from .ragas import match_pakad

# Drought penalties grow with steps-since-last-played and used to be
# unbounded, which meant a long-enough drought outweighed the flat
# forbidden-note penalty (-2.0) — the environment ended up punishing
# "haven't played your vadi in a while" harder than an actual rule
# violation (updatedplan.md finding F11, fixed here in Phase 5.1). Floored
# so neither drought term can ever be worse than the mildest hard-rule
# penalty (aaroha_violation, -1.0); pakad drought is floored softer still,
# since neglecting a phrase is a lesser lapse than neglecting the vadi.
VADI_DROUGHT_FLOOR = -1.0
PAKAD_DROUGHT_FLOOR = -0.5


def compute_reward(
    note: int,
    duration: int,
    note_history: list[int],
    dur_history: list[int],
    tala_position: int,
    direction: int,
    raga: dict,
    tala: dict,
    pakad_drought: int,
    vadi_drought: int,
    grace_factor: float = 1.0,       # 0.2 during drift grace period
    call_phrase: list[int] | None = None,
    call_tension: float = 0.0,
) -> tuple[float, dict]:
    """
    Returns (total_reward, breakdown_dict).

    grace_factor: multiply hard penalties by this during grace period.
    call_phrase:  human's last 4-note call; enables jugalbandi rewards.
    """
    breakdown: dict[str, float] = {}
    swara = note % 12

    # ── LAYER 1: HARD RULES ─────────────────────────────────────────────
    if swara in raga["forbidden_notes"]:
        penalty = -2.0 * grace_factor
        breakdown["forbidden_note"] = penalty
        return penalty, breakdown

    # Direction-sensitive aaroha violation (e.g. Pa ascending in Yaman).
    # aaroha_swaras is the swara-level set; aaroha list is for phrase matching only.
    if direction == 1 and swara not in raga["aaroha_swaras"] and swara in raga["valid_notes"]:
        penalty = -1.0 * grace_factor
        breakdown["aaroha_violation"] = penalty
        return penalty, breakdown

    if swara not in raga["valid_notes"]:
        penalty = -1.5 * grace_factor
        breakdown["out_of_raga"] = penalty
        return penalty, breakdown

    # ── LAYER 2: SOFT RULES ──────────────────────────────────────────────
    reward = 0.0
    breakdown["valid_note"] = 0.2
    reward += 0.2

    if swara == raga["vadi"]:
        breakdown["vadi_emphasis"] = 0.4
        reward += 0.4
    if swara == raga["samvadi"]:
        breakdown["samvadi_emphasis"] = 0.3
        reward += 0.3

    if vadi_drought > 8:
        p = max(-0.05 * (vadi_drought - 8), VADI_DROUGHT_FLOOR)
        breakdown["vadi_drought_penalty"] = p
        reward += p

    # Smoothness: penalise large melodic leaps (absolute distance, so cross-octave leaps are naturally larger).
    if note_history:
        interval = abs(note - note_history[-1])
        if interval > raga["max_smooth_interval"]:
            p = -0.2 * (interval - raga["max_smooth_interval"])
            breakdown["large_leap_penalty"] = p
            reward += p

    # Repetition: prevent degenerate "always play Sa" policy.
    if len(note_history) >= 3 and all(n % 12 == swara for n in note_history[-3:]):
        breakdown["repetition_penalty"] = -0.3
        reward += -0.3

    if swara == raga["vadi"] and duration >= 2:
        breakdown["vadi_held"] = 0.15
        reward += 0.15

    # ── LAYER 3: SEQUENCE-LEVEL (PAKAD + TALA) ──────────────────────────
    # pakads are (phrase: list[int], reward_multiplier: float) tuples.
    # Implicit curriculum: short prefixes yield 0.5, full canonical phrases yield 1.0-1.2.
    match = match_pakad(note_history, note, raga)
    if match is not None:
        _phrase, multiplier = match
        r = 1.0 * multiplier
        breakdown["pakad_completion"] = r
        reward += r

    if pakad_drought > 12:
        p = max(-0.03 * (pakad_drought - 12), PAKAD_DROUGHT_FLOOR)
        breakdown["pakad_drought_penalty"] = p
        reward += p

    if tala_position == tala["sam"] and swara == raga["vadi"]:
        breakdown["sam_vadi_landing"] = 0.8
        reward += 0.8
    if tala_position == tala["sam"] and swara in raga["weak_notes"]:
        breakdown["weak_sam_landing"] = -0.4
        reward += -0.4
    if tala_position in tala["strong_beats"] and swara in {raga["vadi"], raga["samvadi"]}:
        breakdown["strong_beat_emphasis"] = 0.2
        reward += 0.2

    # ── LAYER 4: JUGALBANDI (CALL-RESPONSE) ─────────────────────────────
    if call_phrase:
        if call_tension > 0.5 and swara == raga["vadi"]:
            r = 0.5 * call_tension
            breakdown["tension_resolve"] = r
            reward += r

        if len(call_phrase) >= 2:
            call_dir = 1 if call_phrase[-1] > call_phrase[0] else 2
            if direction != 0 and direction != call_dir:
                breakdown["direction_contrast"] = 0.3
                reward += 0.3

    breakdown["total"] = reward
    return reward, breakdown

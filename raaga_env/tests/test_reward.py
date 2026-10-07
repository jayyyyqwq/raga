import pytest
from raaga_env.ragas import RAGAS, TALAS
from raaga_env.reward import (
    LEAP_PENALTY_FLOOR,
    PAKAD_DROUGHT_FLOOR,
    VADI_DROUGHT_FLOOR,
    compute_reward,
    move_direction,
)

YAMAN = RAGAS["yaman"]
TEENTAAL = TALAS["teentaal"]

BASE = dict(
    duration=0,
    note_history=[0],
    dur_history=[2],
    tala_position=4,
    raga=YAMAN,
    tala=TEENTAAL,
    pakad_drought=0,
    vadi_drought=0,
)


def test_forbidden_note_returns_minus_two():
    r, b = compute_reward(note=5, **BASE)  # natural Ma
    assert r == -2.0
    assert "forbidden_note" in b

def test_valid_note_positive():
    r, b = compute_reward(note=0, **BASE)  # Sa
    assert r > 0

def test_vadi_bonus():
    r, b = compute_reward(note=4, **BASE)  # Ga = vadi
    assert "vadi_emphasis" in b
    assert b["vadi_emphasis"] == 0.4

def test_pakad_completion_bonus():
    # Feed [11, 9] history then play 7 → completes Ni Dha Pa pakad
    r, b = compute_reward(note=7, **{**BASE, "note_history": [11, 9]})
    assert "pakad_completion" in b
    assert b["pakad_completion"] == 1.0

def test_sam_vadi_landing():
    r, b = compute_reward(note=4, **{**BASE, "tala_position": 0})  # beat 0 = sam, note = vadi
    assert "sam_vadi_landing" in b

def test_grace_factor_reduces_penalty():
    r, b = compute_reward(note=5, **{**BASE, "grace_factor": 0.2})
    assert r == -0.4   # -2.0 * 0.2

def test_repetition_penalty():
    r, b = compute_reward(note=0, **{**BASE, "note_history": [0, 0, 0]})
    assert "repetition_penalty" in b

def test_jugalbandi_tension_resolve():
    r, b = compute_reward(
        note=4,   # vadi
        **{**BASE, "call_phrase": [2, 9, 11, 9], "call_tension": 0.8}
    )
    assert "tension_resolve" in b

def test_jugalbandi_direction_contrast():
    # call goes up (2→9), agent moving down (Ga→Re) should get contrast reward
    r, b = compute_reward(
        note=2,
        **{**BASE, "note_history": [4], "call_phrase": [2, 4, 6, 9], "call_tension": 0.3}
    )
    assert "direction_contrast" in b


# ── ML retrain fix (2026-10, docs/RETRAIN_PLAN.md): LAYER 4 must be fully
# inert without a real call, and the new call_echo bonus must be bounded ───

def test_jugalbandi_layer_inactive_without_a_call():
    """call_phrase=[] ('no call yet') must not trigger ANY layer-4 term,
    even one that would otherwise match — this is the exact bug that let
    [0,0,0,0] (a truthy phantom default) corrupt every training episode."""
    r, b = compute_reward(
        note=4,  # vadi — would trigger tension_resolve/call_echo if a call were live
        **{**BASE, "call_phrase": [], "call_tension": 0.9},
    )
    assert "tension_resolve" not in b
    assert "direction_contrast" not in b
    assert "call_echo" not in b


def test_call_echo_fires_when_note_matches_the_calls_last_swara():
    r, b = compute_reward(
        note=11,  # Ni
        **{**BASE, "call_phrase": [0, 2, 4, 11], "call_echoed": False},
    )
    assert b["call_echo"] == 0.4


def test_call_echo_does_not_fire_on_a_mismatched_swara():
    r, b = compute_reward(
        note=4,  # Ga != call's last note (Ni)
        **{**BASE, "call_phrase": [0, 2, 4, 11], "call_echoed": False},
    )
    assert "call_echo" not in b


def test_call_echo_fires_at_most_once_per_call():
    r, b = compute_reward(
        note=11,
        **{**BASE, "call_phrase": [0, 2, 4, 11], "call_echoed": True},
    )
    assert "call_echo" not in b


# ── F11 fix (Phase 5.1): drought penalties are floored, never unbounded ────

def test_vadi_drought_penalty_unfloored_below_the_cap():
    # drought=20: raw = -0.05 * (20-8) = -0.6, well above the -1.0 floor
    r, b = compute_reward(note=0, **{**BASE, "vadi_drought": 20})
    assert b["vadi_drought_penalty"] == pytest.approx(-0.6)


def test_vadi_drought_penalty_never_exceeds_its_floor():
    # drought=1000: raw would be -0.05 * 992 = -49.6 without the floor
    r, b = compute_reward(note=0, **{**BASE, "vadi_drought": 1000})
    assert b["vadi_drought_penalty"] == VADI_DROUGHT_FLOOR


def test_pakad_drought_penalty_unfloored_below_the_cap():
    # drought=20: raw = -0.03 * (20-12) = -0.24, well above the -0.5 floor
    r, b = compute_reward(note=0, **{**BASE, "pakad_drought": 20})
    assert b["pakad_drought_penalty"] == pytest.approx(-0.24)


def test_pakad_drought_penalty_never_exceeds_its_floor():
    # drought=1000: raw would be -0.03 * 988 = -29.64 without the floor
    r, b = compute_reward(note=0, **{**BASE, "pakad_drought": 1000})
    assert b["pakad_drought_penalty"] == PAKAD_DROUGHT_FLOOR


def test_drought_floors_never_exceed_the_mildest_hard_rule_penalty():
    """The whole point of F11: neglecting good practice must never cost
    more than an actual rule violation. aaroha_violation (-1.0) is the
    mildest hard-rule penalty."""
    assert VADI_DROUGHT_FLOOR >= -1.0
    assert PAKAD_DROUGHT_FLOOR >= -1.0


def test_both_drought_penalties_floored_simultaneously_still_bounded():
    r, b = compute_reward(note=0, **{**BASE, "vadi_drought": 1000, "pakad_drought": 1000})
    assert b["vadi_drought_penalty"] == VADI_DROUGHT_FLOOR
    assert b["pakad_drought_penalty"] == PAKAD_DROUGHT_FLOOR
    # valid_note (+0.2) + both floors: still far better than a hard-rule violation.
    assert r == pytest.approx(0.2 + VADI_DROUGHT_FLOOR + PAKAD_DROUGHT_FLOOR)


# ── Audit 2026-10-07: rule direction is the move into the note ─────────────

def test_move_direction():
    assert move_direction(19, []) == 0
    assert move_direction(19, [19]) == 0
    assert move_direction(19, [16]) == 1
    assert move_direction(19, [21]) == 2


def test_ascending_into_pa_is_an_aaroha_violation_in_yaman():
    r, b = compute_reward(note=19, **{**BASE, "note_history": [12, 14, 16]})  # Ga -> Pa, rising
    assert "aaroha_violation" in b


def test_descending_into_pa_after_a_rise_is_legal_in_yaman():
    """Dha -> Pa is standard Yaman avaroha. The old contour-based direction
    called this ascending (the 3 notes before it rose) and penalised it."""
    r, b = compute_reward(note=19, **{**BASE, "note_history": [16, 18, 21]})
    assert "aaroha_violation" not in b
    assert r > 0


def test_ascending_into_pa_after_a_mixed_contour_is_still_a_violation():
    """Ni -> Sa -> Ga -> Pa: the old 3-note window was mixed (one fall, one
    rise) and let this rising Ga -> Pa through unpenalised."""
    r, b = compute_reward(note=19, **{**BASE, "note_history": [23, 12, 16]})
    assert "aaroha_violation" in b


# ── Audit 2026-10-07: leap penalty is floored like the drought penalties ───

def test_leap_penalty_never_exceeds_its_floor():
    r, b = compute_reward(note=23, **{**BASE, "note_history": [0]})  # 23-semitone leap
    assert b["large_leap_penalty"] == LEAP_PENALTY_FLOOR


def test_leap_penalty_unfloored_below_the_cap():
    r, b = compute_reward(note=11, **{**BASE, "note_history": [0]})  # 11 semitones: -0.8
    assert b["large_leap_penalty"] == pytest.approx(-0.8)


def test_leap_floor_never_exceeds_the_mildest_hard_rule_penalty():
    assert LEAP_PENALTY_FLOOR >= -1.0

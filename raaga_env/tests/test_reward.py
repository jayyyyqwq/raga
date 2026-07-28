import pytest
from raaga_env.ragas import RAGAS, TALAS
from raaga_env.reward import compute_reward

YAMAN = RAGAS["yaman"]
TEENTAAL = TALAS["teentaal"]

BASE = dict(
    duration=0,
    note_history=[0],
    dur_history=[2],
    tala_position=4,
    direction=0,
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
    # call goes up (2→9), agent going down (direction=2) should get contrast reward
    r, b = compute_reward(
        note=2,
        **{**BASE, "direction": 2, "call_phrase": [2, 4, 6, 9], "call_tension": 0.3}
    )
    assert "direction_contrast" in b

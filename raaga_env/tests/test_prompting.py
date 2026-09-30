# Phase 2.2/2.4 acceptance tests for raaga_env/prompting.py.

import pytest

from raaga_env.jugalbandi_env import JugalbandiEnv
from raaga_env.prompting import SYSTEM, Arm, StepFeedback, decode, parse_action, render_prompt

FORBIDDEN_BY_ARM = {
    Arm.DIAL: ("yaman", "bhairav", "grace", "switch"),
    Arm.HIDDEN: ("yaman", "bhairav", "grace", "dial", "switch"),
}


def _mid_drift_obs():
    """An obs vector where a leak is most likely: mid-episode, just switched
    raga, inside the grace period (F2's exact scenario)."""
    env = JugalbandiEnv(initial_dial=0.0)
    env.reset(seed=0)
    env.set_dial(0.75)  # switch to bhairav, enter grace period
    return env._get_obs().tolist(), env


# ── 2.2: arm leakage — the test that would have caught F2 ──────────────────

def test_dial_and_hidden_arms_leak_no_raga_identifying_text():
    obs, env = _mid_drift_obs()
    for arm, forbidden_words in FORBIDDEN_BY_ARM.items():
        prompt = render_prompt(obs, arm=arm, tala_pos=env.tala_position).lower()
        for word in forbidden_words:
            assert word not in prompt, f"{arm} prompt leaked {word!r}:\n{prompt}"


def test_oracle_arm_does_name_the_raga():
    """Positive control: a leak test alone doesn't prove ORACLE differs
    correctly from DIAL/HIDDEN — this confirms the arms actually diverge."""
    obs, env = _mid_drift_obs()
    prompt = render_prompt(obs, arm=Arm.ORACLE, tala_pos=env.tala_position, raga="bhairav")
    assert "bhairav" in prompt.lower()
    assert "grace" in prompt.lower()


def test_oracle_arm_requires_raga_argument():
    obs, env = _mid_drift_obs()
    with pytest.raises(ValueError):
        render_prompt(obs, arm=Arm.ORACLE, tala_pos=env.tala_position)


def test_shared_fields_present_in_every_arm():
    """Note history, tala, call-phrase tension, and both droughts are
    rule-agnostic and must survive in all three arms."""
    obs, env = _mid_drift_obs()
    for arm in Arm:
        raga = "bhairav" if arm is Arm.ORACLE else None
        prompt = render_prompt(obs, arm=arm, tala_pos=env.tala_position, raga=raga)
        assert "Tala position" in prompt
        assert "Last 4 notes" in prompt
        assert "Pakad drought" in prompt
        assert "Vadi drought" in prompt


def test_feedback_rendered_when_present_regardless_of_arm():
    obs, env = _mid_drift_obs()
    feedback = StepFeedback(note_name="Sa(sixteenth)", reward=-2.0)
    for arm in Arm:
        raga = "bhairav" if arm is Arm.ORACLE else None
        prompt = render_prompt(obs, arm=arm, tala_pos=env.tala_position, raga=raga, feedback=feedback)
        assert "penalised" in prompt
        assert "-2.00" in prompt


def test_feedback_present_does_not_reintroduce_a_raga_leak():
    """StepFeedback is rule-agnostic by construction, but check the
    combination anyway — this is the exact configuration Claim B training
    actually uses for DIAL/HIDDEN."""
    obs, env = _mid_drift_obs()
    feedback = StepFeedback(note_name="Sa(sixteenth)", reward=-2.0)
    for arm, forbidden_words in FORBIDDEN_BY_ARM.items():
        prompt = render_prompt(obs, arm=arm, tala_pos=env.tala_position, feedback=feedback).lower()
        for word in forbidden_words:
            assert word not in prompt, f"{arm} + feedback leaked {word!r}:\n{prompt}"


# ── 2.1: parse_action never clamps ──────────────────────────────────────────

def test_parse_action_accepts_valid_range():
    assert parse_action("0") == 0
    assert parse_action("95") == 95
    assert parse_action("  42 \n") == 42
    assert parse_action("42 trailing junk") == 42


@pytest.mark.parametrize("text", ["", "not a number", "96", "-1", "137"])
def test_parse_action_rejects_unparseable_and_out_of_range(text):
    assert parse_action(text) is None


# ── 2.4: decode() round-trips against the documented encoding, all 96 actions ──

def test_decode_matches_system_prompt_text():
    assert "note (0-23, absolute pitch across two registers)" in SYSTEM


def test_decode_matches_env_decode_for_every_action():
    env = JugalbandiEnv()
    for action in range(96):
        assert decode(action) == env._decode(action), f"mismatch at action {action}"

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
        assert "Recent notes" in prompt
        assert "Partner's call phrase" in prompt
        assert "Pakad drought" in prompt
        assert "Vadi drought" in prompt


# ── ML retrain fix (2026-10, docs/RETRAIN_PLAN.md): the call phrase itself
# must reach the prompt text, in every arm — this used to never render at
# all, regardless of arm, which is why the model was deaf to any call ──────

def test_call_phrase_renders_as_none_yet_before_any_call():
    obs, env = _mid_drift_obs()
    for arm in Arm:
        raga = "bhairav" if arm is Arm.ORACLE else None
        prompt = render_prompt(obs, arm=arm, tala_pos=env.tala_position, raga=raga)
        assert "Partner's call phrase: none yet" in prompt


def test_call_phrase_notes_render_once_a_call_is_submitted():
    env = JugalbandiEnv(initial_dial=0.0)
    env.reset(seed=0)
    env.set_call([0, 4, 7, 11])  # Sa Ga Pa Ni
    obs = env._get_obs().tolist()
    for arm in Arm:
        raga = "yaman" if arm is Arm.ORACLE else None
        prompt = render_prompt(obs, arm=arm, tala_pos=env.tala_position, raga=raga)
        assert "Partner's call phrase: Sa, Ga, Pa, Ni" in prompt
        assert "none yet" not in prompt


def test_call_phrase_present_does_not_reintroduce_a_raga_leak():
    """Mirrors test_feedback_present_does_not_reintroduce_a_raga_leak — a
    live call phrase is new text in the prompt and must not accidentally
    carry a raga name or dial/grace/switch word into DIAL/HIDDEN."""
    env = JugalbandiEnv(initial_dial=0.0)
    env.reset(seed=0)
    env.set_call([0, 4, 7, 11])
    obs = env._get_obs().tolist()
    for arm, forbidden_words in FORBIDDEN_BY_ARM.items():
        prompt = render_prompt(obs, arm=arm, tala_pos=env.tala_position).lower()
        for word in forbidden_words:
            assert word not in prompt, f"{arm} + call leaked {word!r}:\n{prompt}"


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


# ── Audit (2026-10-07): note decoding from the obs vector ──────────────────

@pytest.mark.parametrize("note", range(24))
def test_every_played_note_renders_as_itself(note):
    """obs stores pitch as float32 note/23; int()-truncating that back
    rendered 12 of 24 pitches a semitone flat (Sa as ṉNi, Re as Re♭, Pa as
    Ma#) in every prompt. Every pitch must round-trip exactly."""
    from raaga_env.prompting import note_name
    env = JugalbandiEnv(initial_dial=0.0)
    env.reset(seed=0)
    env.step(note + 2 * 24)  # quarter note
    prompt = render_prompt(env._get_obs().tolist(), arm=Arm.HIDDEN, tala_pos=env.tala_position)
    last_notes_line = next(l for l in prompt.splitlines() if l.startswith("Recent notes (oldest first):"))
    assert last_notes_line.endswith(note_name(note, 2)), last_notes_line


def test_most_recent_note_is_listed_last_even_early_in_an_episode():
    """With fewer than 4 notes of history, the real notes used to fill the
    leading slots and empty slots trailed — so the prompt's *last* listed
    note was a phantom ṉSa, not the note actually just played."""
    env = JugalbandiEnv(initial_dial=0.0)
    env.reset(seed=0)
    prompt = render_prompt(env._get_obs().tolist(), arm=Arm.HIDDEN, tala_pos=0)
    assert "Recent notes (oldest first):" in prompt
    assert prompt.split("Recent notes (oldest first): ")[1].splitlines()[0].endswith("Sa(quarter)")



def test_empty_history_slots_are_not_rendered_as_notes():
    env = JugalbandiEnv(initial_dial=0.0)
    env.reset(seed=0)
    prompt = render_prompt(env._get_obs().tolist(), arm=Arm.HIDDEN, tala_pos=0)
    line = next(l for l in prompt.splitlines() if l.startswith("Recent notes"))
    assert line == "Recent notes (oldest first): Sa(quarter)"


def test_a_real_all_sa_call_is_not_rendered_as_no_call():
    env = JugalbandiEnv(initial_dial=0.0)
    env.reset(seed=0)
    env.set_call([0, 0, 0, 0])
    prompt = render_prompt(env._get_obs().tolist(), arm=Arm.HIDDEN, tala_pos=0)
    assert "Partner's call phrase: Sa, Sa, Sa, Sa" in prompt

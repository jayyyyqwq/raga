# Contract tests (updatedplan.md Phase 0.2).
#
# These assert properties the codebase is *supposed* to have. Several of them
# currently FAIL on this tree — that is the acceptance criterion for this
# file, not a bug in the tests. Each failing test names the finding in
# updatedplan.md §2 it documents. Do not "fix" a test to make it pass; fix
# the code it is testing, in the phase updatedplan.md assigns that fix to.

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from raaga_env.jugalbandi_env import JugalbandiEnv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OPENENV_YAML = REPO_ROOT / "openenv.yaml"
SERVER_SRC = (REPO_ROOT / "openenv_server" / "server.py").read_text(encoding="utf-8")
TRAIN_SRC = (REPO_ROOT / "training" / "train_grpo.py").read_text(encoding="utf-8")


# ── F5: observations must stay inside the declared observation_space ───────

def test_jugalbandi_obs_always_in_bounds_over_10000_steps():
    env = JugalbandiEnv()
    env.action_space.seed(0)
    env.reset(seed=0)
    for _ in range(10_000):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        assert env.observation_space.contains(obs), f"Obs out of bounds: {obs}"
        if terminated or truncated:
            env.reset()


# ── F3: action_space.n must agree across env, manifest, and server contract ─

def test_action_space_size_matches_openenv_yaml():
    env = JugalbandiEnv()
    manifest = yaml.safe_load(OPENENV_YAML.read_text(encoding="utf-8"))
    assert env.action_space.n == manifest["action_space"]["n"]


def test_action_space_size_matches_step_request_bound():
    match = re.search(r"action:\s*int\s*=\s*Field\(\.\.\.,\s*ge=0,\s*le=(\d+)\)", SERVER_SRC)
    assert match, "Could not find StepRequest's action field bound in server.py"
    declared_max = int(match.group(1))

    env = JugalbandiEnv()
    assert declared_max == env.action_space.n - 1


# ── F4/F7: exactly one prompt builder must exist (raaga_env/prompting.py),
# and every call site must import it rather than re-deriving its own
# encoding text. Full decode-vs-SYSTEM-text correctness (all 96 actions) is
# covered in raaga_env/tests/test_prompting.py — this just guards against
# the three-builders regression F7 found.

def test_server_and_training_script_import_the_shared_prompt_module():
    assert "from raaga_env.prompting import" in SERVER_SRC
    assert "from raaga_env.prompting import" in TRAIN_SRC


def test_no_call_site_redefines_the_action_encoding_text():
    assert "note (0-23, absolute pitch across two registers)" not in SERVER_SRC
    assert "note (0-23, absolute pitch across two registers)" not in TRAIN_SRC


# ── F11: reward must stay inside the range openenv.yaml declares ───────────

def test_reward_always_within_declared_bounds():
    manifest = yaml.safe_load(OPENENV_YAML.read_text(encoding="utf-8"))
    lo, hi = manifest["reward"]["min"], manifest["reward"]["max"]

    env = JugalbandiEnv()
    env.reset(seed=0)
    # Sa (12) and Ni (23) alternating: both valid and aaroha-safe in Yaman, so
    # the hard-rule layer never fires and drought penalties accumulate
    # uncapped for the whole episode (F11), instead of being masked by an
    # early hard-rule return.
    safe_notes = [12, 23]
    violations = []
    for i in range(64):
        action = safe_notes[i % len(safe_notes)]
        _, reward, terminated, truncated, _ = env.step(action)
        if not (lo <= reward <= hi):
            violations.append((i, reward))
        if terminated or truncated:
            break
    assert not violations, (
        f"Reward left declared bounds [{lo}, {hi}] at steps (step, reward): {violations}"
    )


# ── F10: reward_breakdown must be float-valued and consistent with reward ──

def test_reward_breakdown_values_are_all_floats():
    env = JugalbandiEnv()
    env.reset(seed=0)
    bad_entries = []
    for _ in range(16):  # CALL_EVERY=8, so this crosses a call_requested step
        _, _, terminated, truncated, info = env.step(4)  # Ga = vadi, always valid
        for key, value in info["reward_breakdown"].items():
            if not isinstance(value, float):
                bad_entries.append((key, value, type(value).__name__))
        if terminated or truncated:
            break
    assert not bad_entries, f"Non-float breakdown entries: {bad_entries}"


def test_adaptation_bonus_fires_when_new_raga_pakad_is_played_within_window():
    """The drift-specific adaptation bonus must award ADAPTATION_BONUS when
    the agent plays the new raga's pakad within ADAPTATION_WINDOW steps of a
    switch — this is what Phase 4's drift_adaptation_speed metric measures.
    It currently never fires: DriftManager.step's inline pakad matcher
    iterates `(phrase, multiplier)` tuples without unpacking them, so
    `len(pakad)` is always 2 and `note_history[-n:] == pakad` compares a
    list of ints to a (list, float) tuple, which is never True. This is a
    stricter break than F7 ('re-implements pakad matching inline instead of
    calling ragas.match_pakad') documents — the reimplementation isn't just
    a drift risk, it is dead on arrival."""
    env = JugalbandiEnv(initial_dial=0.0)
    env.reset(seed=0)
    env.set_dial(0.75)  # switch to bhairav; opens the adaptation-bonus window
    pakad = [12, 13, 16, 17]  # Sa re Ga Ma — bhairav's shortest opening pakad
    breakdown = {}
    for note in pakad:
        _, _, _, _, info = env.step(note)
        breakdown = info["reward_breakdown"]
    assert "adaptation_bonus" in breakdown, "Expected the adaptation bonus to fire"


def test_reward_breakdown_total_matches_returned_reward_when_drift_bonus_applies(monkeypatch):
    """breakdown['total'] is set inside compute_reward() (reward.py:127)
    before jugalbandi_env.step() adds the drift/adaptation bonus on top
    (jugalbandi_env.py:93-96), so the two go out of sync whenever the bonus
    is nonzero (F10). Isolated from the adaptation-bonus-never-fires bug
    above via a monkeypatched DriftManager.step so this test exercises only
    the ordering bug."""
    env = JugalbandiEnv(initial_dial=0.0)
    env.reset(seed=0)
    monkeypatch.setattr(env.drift, "step", lambda note, history: 3.0)

    _, reward, _, _, info = env.step(4)  # any valid note; the patched bonus always fires
    breakdown = info["reward_breakdown"]
    assert breakdown["total"] == pytest.approx(reward), (
        f"breakdown['total']={breakdown['total']} != returned reward={reward}"
    )


# ── F2: the HIDDEN arm must leak no raga-identifying text into the prompt ──
# raaga_env/prompting.py now exists (Phase 2) — see raaga_env/tests/
# test_prompting.py for the full arm-leakage and encoding test suite this
# placeholder was standing in for.

def test_hidden_arm_prompt_leaks_no_raga_identifying_text():
    from raaga_env.prompting import Arm, render_prompt

    env = JugalbandiEnv(initial_dial=0.0)
    obs, _ = env.reset(seed=0)
    env.set_dial(0.75)  # switch + grace period, the exact leak F2 found
    obs = env._get_obs()

    prompt = render_prompt(obs, arm=Arm.HIDDEN, tala_pos=env.tala_position, feedback=None)

    lowered = prompt.lower()
    for substring in ("yaman", "bhairav", "grace", "dial", "switch"):
        assert substring not in lowered, f"HIDDEN-arm prompt leaked {substring!r}"

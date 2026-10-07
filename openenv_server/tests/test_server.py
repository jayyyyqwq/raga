# Phase 2.3 acceptance tests for the server's StepFeedback (Claim B) wiring.
#
# /infer itself needs torch (a GPU-only dep, deliberately not installed
# locally — see requirements-train.txt). What's tested here is everything up
# to that boundary: that /reset and /step correctly maintain the module's
# _last_step_feedback, and that it's shaped the way render_prompt expects —
# render_prompt's own rendering of a StepFeedback is already covered by
# raaga_env/tests/test_prompting.py.

import pytest
from fastapi.testclient import TestClient

import openenv_server.server as server_module
from raaga_env.prompting import Arm, StepFeedback, render_prompt

client = TestClient(server_module.app)


@pytest.fixture(autouse=True)
def _reset_env():
    client.post("/reset", json={"dial": 0.0})
    yield


def test_reset_clears_last_step_feedback():
    client.post("/step", json={"action": 4})  # Ga, vadi of Yaman — always valid
    assert server_module._last_step_feedback is not None

    client.post("/reset", json={"dial": 0.0})
    assert server_module._last_step_feedback is None


def test_step_sets_last_step_feedback_from_the_actual_outcome():
    resp = client.post("/step", json={"action": 4})  # Ga = vadi, positive reward
    reward = resp.json()["reward"]

    feedback = server_module._last_step_feedback
    assert isinstance(feedback, StepFeedback)
    assert feedback.reward == pytest.approx(reward)
    assert "Ga" in feedback.note_name


def test_step_sets_last_step_feedback_for_a_penalised_action():
    resp = client.post("/step", json={"action": 5})  # natural Ma, forbidden in Yaman
    reward = resp.json()["reward"]
    assert reward < 0

    feedback = server_module._last_step_feedback
    assert feedback.reward == pytest.approx(reward)
    assert "penalised" in feedback.render()


def test_last_step_feedback_reflects_only_the_most_recent_step():
    client.post("/step", json={"action": 4})   # Ga: rewarded
    client.post("/step", json={"action": 5})   # natural Ma: penalised
    feedback = server_module._last_step_feedback
    assert feedback.reward < 0
    assert "penalised" in feedback.render()


def test_last_step_feedback_is_rule_agnostic():
    """The whole point of Claim B's channel: it says outcome, never why."""
    client.post("/step", json={"action": 5})  # forbidden in yaman
    feedback = server_module._last_step_feedback
    rendered = feedback.render().lower()
    for forbidden_word in ("yaman", "bhairav", "forbidden", "raga"):
        assert forbidden_word not in rendered


def test_last_step_feedback_integrates_with_render_prompt_for_every_arm():
    """Confirms _last_step_feedback is exactly the shape /infer feeds into
    render_prompt — the rendering itself is covered by test_prompting.py."""
    client.post("/step", json={"action": 5})  # forbidden, produces a penalty
    feedback = server_module._last_step_feedback
    obs = server_module.env._get_obs().tolist()

    for arm in Arm:
        raga = "yaman" if arm is Arm.ORACLE else None
        prompt = render_prompt(
            obs, arm=arm, tala_pos=server_module.env.tala_position, feedback=feedback, raga=raga
        )
        assert "Last action" in prompt
        assert "penalised" in prompt


# ── UI static mount (docs/imrpovedui.md "one process, one port") ───────────

def test_root_serves_the_ui_index_page():
    res = client.get("/")
    assert res.status_code == 200
    assert "Jugalbandi" in res.text


def test_static_mount_serves_an_app_module():
    res = client.get("/app.js")
    assert res.status_code == 200
    assert "EnvClient" in res.text


def test_api_routes_still_take_priority_over_the_static_mount():
    """The mount is at "/" — this is what guarantees /reset, /step etc.
    never get shadowed by StaticFiles looking for files named "reset"."""
    res = client.post("/reset", json={"dial": 0.0})
    assert res.status_code == 200
    assert "active_raga" in res.json()


@pytest.mark.parametrize("notes", [[12], [-1], [0, 2, 4, 99]])
def test_call_rejects_out_of_range_swaras(notes):
    """Calls are swaras 0-11; anything else used to reach set_call() and
    then crash prompt rendering (IndexError) or push obs outside [0, 1]."""
    resp = client.post("/call", json={"notes": notes})
    assert resp.status_code == 422

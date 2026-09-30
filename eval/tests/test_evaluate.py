import json

import pytest

from eval.evaluate import SCRIPTED_POLICIES, run_and_save, run_policy_over_eval_set, save_result
from eval.episodes import EVAL_EPISODES


def test_run_policy_over_eval_set_covers_every_episode():
    trajectories = run_policy_over_eval_set(SCRIPTED_POLICIES["random-valid"])
    assert len(trajectories) == len(EVAL_EPISODES)


def test_run_policy_over_eval_set_replays_each_episodes_own_schedule():
    trajectories = run_policy_over_eval_set(SCRIPTED_POLICIES["scripted-oracle"])
    for traj, ep in zip(trajectories, EVAL_EPISODES):
        assert traj.switch_steps == (ep.switch_step,)


def test_save_result_writes_a_fingerprinted_json_file(tmp_path, monkeypatch):
    import eval.evaluate as evaluate_module

    monkeypatch.setattr(evaluate_module, "RESULTS_DIR", tmp_path)
    trajectories = run_policy_over_eval_set(SCRIPTED_POLICIES["safe-set-cycle"])
    path = save_result("safe-set-cycle", trajectories)

    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["policy"] == "safe-set-cycle"
    assert payload["n_episodes"] == len(EVAL_EPISODES)
    assert "fingerprint" in payload and payload["fingerprint"]["git_sha"]
    assert payload["metrics"]["safe_set_occupancy"] == 1.0


@pytest.mark.parametrize("name", sorted(SCRIPTED_POLICIES))
def test_run_and_save_works_for_every_scripted_policy(tmp_path, monkeypatch, name):
    import eval.evaluate as evaluate_module

    monkeypatch.setattr(evaluate_module, "RESULTS_DIR", tmp_path)
    path = run_and_save(name)
    payload = json.loads(path.read_text())
    assert payload["n_episodes"] == len(EVAL_EPISODES)


def test_run_and_save_rejects_unknown_policy():
    with pytest.raises(ValueError):
        run_and_save("grpo-hidden")  # needs a real model, not runnable here

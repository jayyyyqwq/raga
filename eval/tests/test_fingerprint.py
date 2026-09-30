from eval.fingerprint import compute_fingerprint


def test_fingerprint_has_expected_keys():
    fp = compute_fingerprint()
    assert set(fp.keys()) == {
        "git_sha",
        "requirements_train_hash",
        "ragas_hash",
        "reward_hash",
        "eval_episodes_hash",
        "metrics_hash",
        "policies_hash",
    }


def test_fingerprint_is_deterministic_for_the_same_tree():
    assert compute_fingerprint() == compute_fingerprint()


def test_fingerprint_changes_when_reward_file_changes(tmp_path, monkeypatch):
    import eval.fingerprint as fp_module

    fake_root = tmp_path
    (fake_root / "raaga_env").mkdir()
    reward_file = fake_root / "raaga_env" / "reward.py"
    reward_file.write_text("VERSION = 1\n")
    monkeypatch.setattr(fp_module, "REPO_ROOT", fake_root)

    first = fp_module.compute_fingerprint()
    reward_file.write_text("VERSION = 2\n")
    second = fp_module.compute_fingerprint()

    assert first["reward_hash"] != second["reward_hash"]

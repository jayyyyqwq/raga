from eval.episodes import DRIFT_WINDOW, N_EVAL_EPISODES, EVAL_EPISODES, generate_eval_episodes


def test_eval_episodes_has_the_documented_count():
    assert len(EVAL_EPISODES) == N_EVAL_EPISODES == 200


def test_eval_episodes_is_deterministic():
    a = generate_eval_episodes()
    b = generate_eval_episodes()
    assert a == b


def test_eval_episodes_every_episode_has_exactly_one_switch_in_the_drift_window():
    for ep in EVAL_EPISODES:
        assert DRIFT_WINDOW[0] <= ep.switch_step <= DRIFT_WINDOW[1]
        assert ep.switch_dial != ep.initial_dial


def test_eval_episodes_split_evenly_between_starting_ragas():
    starts_yaman = sum(1 for ep in EVAL_EPISODES if ep.initial_dial < 0.5)
    starts_bhairav = sum(1 for ep in EVAL_EPISODES if ep.initial_dial >= 0.5)
    assert starts_yaman == starts_bhairav == 100


def test_eval_episodes_have_unique_episode_ids():
    ids = [ep.episode_id for ep in EVAL_EPISODES]
    assert len(set(ids)) == len(ids)


def test_eval_episodes_different_seed_gives_a_different_set():
    a = generate_eval_episodes(master_seed=1)
    b = generate_eval_episodes(master_seed=2)
    assert a != b

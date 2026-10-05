# Tests for eval/llm_policy.py's _build_policy — the pure-Python half (no
# torch/transformers/peft needed, see that module's header comment on why
# the split exists). make_llm_policy() itself (the GPU-dependent half that
# actually loads a model) is exercised for real on Colab, same as every
# other GPU-only code path in this project.

from raaga_env.prompting import Arm

from eval.llm_policy import LLMPolicyStats, _build_policy


def _fake_generate(fixed_output: str):
    calls = []

    def generate_fn(prompt: str) -> str:
        calls.append(prompt)
        return fixed_output

    return generate_fn, calls


def _obs_with_call(swaras):
    """A 22-dim obs with call_phrase slots (8-11) set to `swaras`, and dial
    (15) at 0.0 (Yaman) unless overridden — matches JugalbandiEnv._get_obs's
    encoding (n/11.0 for calls, see that method's docstring)."""
    obs = [0.0] * 22
    for i, s in enumerate(swaras[:4]):
        obs[8 + i] = s / 11.0
    return obs


def test_valid_completion_returns_the_parsed_action():
    generate_fn, _ = _fake_generate("42")
    stats = LLMPolicyStats()
    policy = _build_policy(generate_fn, arm=Arm.HIDDEN, fallback_action=4, stats=stats)

    action = policy(_obs_with_call([0, 0, 0, 0]), {})
    assert action == 42
    assert stats.n_total == 1
    assert stats.n_valid == 1


def test_unparseable_completion_falls_back_and_is_counted_invalid():
    generate_fn, _ = _fake_generate("not a number")
    stats = LLMPolicyStats()
    policy = _build_policy(generate_fn, arm=Arm.HIDDEN, fallback_action=4, stats=stats)

    action = policy(_obs_with_call([0, 0, 0, 0]), {})
    assert action == 4  # fallback
    assert stats.n_total == 1
    assert stats.n_valid == 0
    assert stats.validity_rate == 0.0


def test_out_of_range_completion_also_falls_back():
    generate_fn, _ = _fake_generate("137")  # parse_action rejects >= 96
    stats = LLMPolicyStats()
    policy = _build_policy(generate_fn, arm=Arm.HIDDEN, fallback_action=4, stats=stats)

    assert policy(_obs_with_call([0, 0, 0, 0]), {}) == 4
    assert stats.n_valid == 0


def test_validity_rate_accumulates_across_calls():
    outputs = iter(["4", "garbage", "11"])

    def generate_fn(prompt):
        return next(outputs)

    stats = LLMPolicyStats()
    policy = _build_policy(generate_fn, arm=Arm.HIDDEN, fallback_action=0, stats=stats)

    for _ in range(3):
        policy(_obs_with_call([0, 0, 0, 0]), {})
    assert stats.n_total == 3
    assert stats.n_valid == 2
    assert stats.validity_rate == 2 / 3


def test_first_call_with_empty_info_has_no_feedback_line():
    generate_fn, calls = _fake_generate("4")
    stats = LLMPolicyStats()
    policy = _build_policy(generate_fn, arm=Arm.HIDDEN, fallback_action=0, stats=stats)

    policy(_obs_with_call([0, 0, 0, 0]), {})
    assert "Last action" not in calls[0]


def test_subsequent_call_with_info_renders_feedback():
    generate_fn, calls = _fake_generate("4")
    stats = LLMPolicyStats()
    policy = _build_policy(generate_fn, arm=Arm.HIDDEN, fallback_action=0, stats=stats)

    info = {"note": 4, "duration": 0, "reward_breakdown": {"total": -2.0}, "tala_position": 3}
    policy(_obs_with_call([0, 0, 0, 0]), info)
    assert "Last action" in calls[0]
    assert "penalised" in calls[0]
    assert "-2.00" in calls[0]


def test_oracle_arm_names_the_raga_dial_hidden_does_not():
    generate_fn, calls = _fake_generate("4")
    stats = LLMPolicyStats()

    hidden_policy = _build_policy(generate_fn, arm=Arm.HIDDEN, fallback_action=0, stats=stats)
    hidden_policy(_obs_with_call([0, 0, 0, 0]), {})
    assert "yaman" not in calls[-1].lower()
    assert "bhairav" not in calls[-1].lower()

    oracle_policy = _build_policy(generate_fn, arm=Arm.ORACLE, fallback_action=0, stats=stats)
    oracle_policy(_obs_with_call([0, 0, 0, 0]), {})
    assert "yaman" in calls[-1].lower()

# Wraps a trained LoRA adapter as an eval.rollout Policy — the piece
# eval/evaluate.py's own module docstring said would "live with the
# training code once a checkpoint exists to evaluate." One now does.
#
# WHY this doesn't live in openenv_server/server.py: that module serves one
# decision at a time over HTTP for the live demo. This runs many episodes
# in-process through eval.rollout.rollout() — the project's single source
# of truth for every reported metric (see that module's own docstring).
# Driving the model a second, HTTP-shaped way here would be exactly the
# divergent-implementation problem F7 already named once (three prompt
# builders that were supposed to agree and didn't).
#
# Testability: policy construction is split in two. _build_policy() takes
# a plain generate_fn(prompt: str) -> str and contains all the actual
# policy logic (prompt building, StepFeedback reconstruction, parsing,
# stats) — no torch import, fully unit-testable with a fake generate_fn.
# make_llm_policy() is the thin GPU-dependent half: it loads the real
# model/tokenizer (torch/transformers/peft, lazily imported so this module
# stays importable without the GPU stack, same pattern as
# openenv_server/server.py's _load_inference_model()) and hands
# _build_policy() a real generate_fn closing over them.

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from raaga_env.prompting import Arm, StepFeedback, note_name, parse_action, render_prompt
from raaga_env.ragas import raga_from_dial


@dataclass
class LLMPolicyStats:
    """What eval.metrics.action_validity_rate needs. A Trajectory only
    contains steps whose completion DID parse (see that function's own
    docstring: "a Trajectory only exists for actions that did parse") —
    the denominator lives upstream of that, here, not in the Trajectory."""

    n_total: int = 0
    n_valid: int = 0

    @property
    def validity_rate(self) -> float:
        return self.n_valid / self.n_total if self.n_total else 1.0


def _build_policy(
    generate_fn: Callable[[str], str],
    *,
    arm: Arm,
    fallback_action: int,
    stats: LLMPolicyStats,
) -> Callable:
    """Returns an eval.rollout.Policy-shaped callable. `fallback_action` is
    substituted on a parse failure (default: Ga, valid in both Yaman and
    Bhairav) so a 200-episode unattended eval run doesn't abort on one bad
    completion — unlike openenv_server/server.py's /infer, which raises,
    because that's one live decision a human is watching, not one row of a
    long batch job. The substitution is never silent: `stats` counts every
    attempt and every success, so action_validity_rate reports exactly how
    often this had to happen."""

    def policy(observation, info) -> int:
        feedback = None
        if info:
            # info comes from the *previous* env.step() (see
            # eval/rollout.py's _run_steps) — {} only at episode start.
            # reward_breakdown["total"] is kept equal to the step's
            # returned reward by construction (jugalbandi_env.py, F10),
            # so this needs nothing eval.rollout doesn't already give.
            feedback = StepFeedback(
                note_name=note_name(info["note"], info["duration"]),
                reward=float(info["reward_breakdown"]["total"]),
            )
        raga = raga_from_dial(observation[15]) if arm is Arm.ORACLE else None
        prompt = render_prompt(
            list(observation),
            arm=arm,
            tala_pos=info.get("tala_position", 0),
            feedback=feedback,
            raga=raga,
        )

        decoded = generate_fn(prompt)
        stats.n_total += 1
        action = parse_action(decoded)
        if action is None:
            return fallback_action
        stats.n_valid += 1
        return action

    return policy


def make_llm_policy(
    adapter_repo: str,
    *,
    base_model: str = "unsloth/Qwen2.5-0.5B-Instruct",
    arm: Arm = Arm.HIDDEN,
    max_new_tokens: int = 4,
    temperature: float = 0.1,
    fallback_action: int = 4,
) -> tuple[Callable, LLMPolicyStats]:
    """Loads `adapter_repo` (a local path or HF Hub repo id — anything
    `PeftModel.from_pretrained` accepts) over `base_model` and returns
    (policy, stats). GPU/CUDA stack required (requirements-train.txt);
    this is what actually needs torch, imported lazily here so the rest of
    this module — and every test of _build_policy — never does."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Real bug, not just inherent per-call overhead: neither
    # AutoModelForCausalLM.from_pretrained nor PeftModel.from_pretrained
    # moves the model onto a GPU by default — with no explicit device_map
    # or .to(device), this ran entirely on CPU even on a Colab T4 instance,
    # which is easily 10-30x slower than it needed to be over a 200-episode
    # (up to 12,800-call) run. Found the hard way, mid-run, 2026-10.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if device == "cuda":
        # 4-bit NF4, matching training (unsloth load_in_4bit=True): the
        # adapter was learned against the quantised base, and evaluating it
        # on a 16-bit base is a train/eval mismatch (audit 2026-10-07).
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
        )
        base = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=quant, device_map={"": 0})
    else:
        base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, adapter_repo)
    model.eval()
    # The base model's generation_config.json ships a default max_length
    # (32768) alongside the max_new_tokens we pass per call below — with
    # both set, transformers warns "Both max_new_tokens and max_length seem
    # to have been set" on every single .generate() call. Harmless (the
    # warning itself says max_new_tokens wins, which is what we want), but
    # a 200-episode x up to-64-step run means up to ~12,800 near-identical
    # warning lines — noisy enough to make the real output hard to find and
    # slow to scroll through. Clearing it once here, rather than at each
    # call site, removes the conflict for the rest of this policy's life.
    model.generation_config.max_length = None

    def generate_fn(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0.0,
            )
        return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    stats = LLMPolicyStats()
    policy = _build_policy(generate_fn, arm=arm, fallback_action=fallback_action, stats=stats)
    return policy, stats

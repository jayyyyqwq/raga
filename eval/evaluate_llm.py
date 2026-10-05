# Evaluates a trained LoRA adapter over the fixed, shared eval set — the
# runner eval/evaluate.py's own docstring said would exist "once a
# checkpoint exists to evaluate." Needs the GPU training stack
# (requirements-train.txt); run this on Colab, same place training ran.
#
# Usage (Colab, after training/train_grpo.ipynb has saved an adapter):
#   python -m eval.evaluate_llm \
#       --adapter-repo /content/drive/MyDrive/jugalbandi/jugalbandi-grpo-hidden-v2/final \
#       --arm hidden --name grpo-hidden-v2
#
# Quick check on a subset first (minutes, not the full 200-episode run):
#   python -m eval.evaluate_llm --adapter-repo ... --arm hidden --name grpo-hidden-v2-quick --limit 20
#
# `run_llm_over_eval_set` takes an already-built policy (not an
# adapter_repo string) so it's unit-testable with a fake policy — only
# `main()` below actually calls make_llm_policy() and needs a GPU.

from __future__ import annotations

import argparse

from raaga_env.prompting import Arm

from .episodes import EVAL_EPISODES
from .evaluate import save_result
from .policies import sample_call_phrase
from .rollout import DriftSchedule, Trajectory, rollout


def run_llm_over_eval_set(policy, *, arm_value: str, limit: int | None = None) -> list[Trajectory]:
    """Runs `policy` over the fixed eval set (or its first `limit`
    episodes), each with its own recorded drift schedule replayed exactly
    — same contract as eval.evaluate.run_policy_over_eval_set (including
    call_phrase_fn=sample_call_phrase, so jugalbandi_coherence/
    call_echo_rate are measurable here too), so results from this and the
    scripted baselines are directly comparable."""
    episodes = EVAL_EPISODES[:limit] if limit else EVAL_EPISODES
    trajectories = []
    for ep in episodes:
        schedule = DriftSchedule(switches=((ep.switch_step, ep.switch_dial),))
        traj = rollout(
            policy,
            seed=ep.seed,
            initial_dial=ep.initial_dial,
            drift_schedule=schedule,
            arm=arm_value,
            episode_length=64,
            call_phrase_fn=sample_call_phrase,
        )
        trajectories.append(traj)
    return trajectories


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-repo", required=True, help="local path or HF Hub repo id")
    parser.add_argument("--base-model", default="unsloth/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--arm", choices=[a.value for a in Arm], default=Arm.HIDDEN.value)
    parser.add_argument("--name", required=True, help="result file name, e.g. grpo-hidden-v2")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="run only the first N of the 200 eval episodes (for a quick check before the full run)",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    arm = Arm(args.arm)

    from .llm_policy import make_llm_policy  # GPU-dependent — deferred, see that module's header

    print(f"Loading {args.adapter_repo} (base: {args.base_model}, arm: {arm.value}) ...")
    policy, stats = make_llm_policy(
        args.adapter_repo,
        base_model=args.base_model,
        arm=arm,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    n_episodes = args.limit or len(EVAL_EPISODES)
    print(f"Running {n_episodes} episodes ...")
    trajectories = run_llm_over_eval_set(policy, arm_value=arm.value, limit=args.limit)

    print(f"Action validity: {stats.n_valid}/{stats.n_total} ({stats.validity_rate:.1%})")
    path = save_result(args.name, trajectories, action_validity_rate=stats.validity_rate)
    print(f"{args.name}: wrote {path}")
    if args.limit:
        print(
            f"This was a {args.limit}-episode subset, not the full 200 — re-run without "
            "--limit for a result worth comparing against the baselines in eval/results/."
        )


if __name__ == "__main__":
    main()

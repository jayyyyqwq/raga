# Runs one policy over the fixed eval set and writes a fingerprinted result
# file (updatedplan.md Phase 4.1/4.2).
#
# Usage:
#   venv/Scripts/python.exe -m eval.evaluate random-uniform
#   venv/Scripts/python.exe -m eval.evaluate --all
#
# Only the four scripted baselines (random-uniform, random-valid,
# safe-set-cycle, scripted-oracle) can run here — base-zeroshot and every
# grpo-* arm need a real model (torch/transformers, not installed locally;
# requirements-train.txt is Colab/GPU-only). Their runner lives with the
# training code once a checkpoint exists to evaluate.

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .episodes import EVAL_EPISODES, EVAL_MASTER_SEED
from .fingerprint import compute_fingerprint
from .metrics import compute_all_metrics
from .policies import (
    random_uniform_policy,
    random_valid_policy,
    safe_set_cycle_policy,
    sample_call_phrase,
    scripted_oracle_policy,
)
from .rollout import DriftSchedule, Trajectory, rollout

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# name -> policy_factory(episode: EvalEpisode) -> Policy. Called once per
# episode so a stateful or rng-seeded policy (scripted-oracle; every
# random-* policy) gets a fresh, per-episode-deterministic instance instead
# of sharing state or an RNG stream across the whole 200-episode set — that
# would make episode k's result depend on the order episodes were run in.
SCRIPTED_POLICIES = {
    "random-uniform": lambda ep: random_uniform_policy(random.Random(ep.seed)),
    "random-valid": lambda ep: random_valid_policy(random.Random(ep.seed)),
    "safe-set-cycle": lambda ep: safe_set_cycle_policy(),
    "scripted-oracle": lambda ep: scripted_oracle_policy(
        DriftSchedule(switches=((ep.switch_step, ep.switch_dial),)), random.Random(ep.seed)
    ),
}


def run_policy_over_eval_set(
    policy_factory, *, arm: str | None = None
) -> list[Trajectory]:
    """Runs policy_factory(episode) over every episode in the fixed,
    shared eval set (Phase 4.2), each with its own recorded drift schedule
    replayed exactly. Also submits a call every CALL_EVERY steps
    (call_phrase_fn=sample_call_phrase — eval.rollout.rollout()'s
    docstring explains why this must be explicit) so jugalbandi_coherence
    and call_echo_rate are measurable for these baselines too, not just
    whatever trained model gets compared against them."""
    trajectories = []
    for ep in EVAL_EPISODES:
        policy = policy_factory(ep)
        schedule = DriftSchedule(switches=((ep.switch_step, ep.switch_dial),))
        traj = rollout(
            policy,
            seed=ep.seed,
            initial_dial=ep.initial_dial,
            drift_schedule=schedule,
            arm=arm,
            episode_length=64,
            call_phrase_fn=sample_call_phrase,
        )
        trajectories.append(traj)
    return trajectories


def save_result(
    name: str, trajectories: list[Trajectory], action_validity_rate: float = 1.0
) -> Path:
    report = compute_all_metrics(trajectories, action_validity_rate_override=action_validity_rate)
    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {
        "policy": name,
        "n_episodes": len(trajectories),
        "eval_master_seed": EVAL_MASTER_SEED,
        "fingerprint": compute_fingerprint(),
        "metrics": report.to_dict(),
    }
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def run_and_save(name: str) -> Path:
    if name not in SCRIPTED_POLICIES:
        raise ValueError(
            f"Unknown scripted policy {name!r}. Available: {sorted(SCRIPTED_POLICIES)}. "
            "base-zeroshot and grpo-* need a real model and are not run here."
        )
    trajectories = run_policy_over_eval_set(SCRIPTED_POLICIES[name])
    return save_result(name, trajectories)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", nargs="?", choices=sorted(SCRIPTED_POLICIES))
    parser.add_argument("--all", action="store_true", help="run every scripted baseline")
    args = parser.parse_args()

    if not args.all and not args.policy:
        parser.error("pass a policy name or --all")

    names = sorted(SCRIPTED_POLICIES) if args.all else [args.policy]
    for name in names:
        path = run_and_save(name)
        print(f"{name}: wrote {path}")


if __name__ == "__main__":
    main()

"""Paper statistics for the four scripted baselines (EXPERIMENT_PLAN.md §11).

Re-runs each baseline over the fixed 200-episode eval set through the same
harness eval/evaluate.py uses, then reports per-policy mean/SD episode return,
paired bootstrap 95% CIs on per-episode return differences, Wilcoxon
signed-rank tests, the post-switch hard-rule violation curve, and a hard-rule
violation breakdown. Writes paper/data/baseline_stats.json.

    venv/Scripts/python -m paper.scripts.baseline_stats
"""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path

from eval.evaluate import SCRIPTED_POLICIES, run_policy_over_eval_set
from eval.fingerprint import compute_fingerprint
from eval.metrics import HARD_RULE_KEYS
from eval.rollout import Trajectory

OUT = Path(__file__).resolve().parent.parent / "data" / "baseline_stats.json"
N_BOOT = 10_000
BOOT_SEED = 0
DECAY_HORIZON = 16
PAIRS = (
    ("scripted-oracle", "safe-set-cycle"),
    ("scripted-oracle", "random-valid"),
    ("safe-set-cycle", "random-valid"),
    ("random-valid", "random-uniform"),
)


def wilcoxon_signed_rank(diffs: list[float]) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test, normal approximation with tie
    correction; zero differences dropped (Wilcoxon's original convention).
    Implemented here rather than adding scipy as a dependency — with n=200
    paired episodes the normal approximation is the standard choice anyway.
    Returns (W+, p)."""
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n == 0:
        return 0.0, 1.0
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    tie_term = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        t = j - i + 1
        tie_term += t ** 3 - t
        i = j + 1
    w_plus = sum(r for r, d in zip(ranks, nz) if d > 0)
    mean = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24 - tie_term / 48
    z = (w_plus - mean) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2))
    return w_plus, p


def paired_bootstrap_ci(diffs: list[float], rng: random.Random) -> tuple[float, float]:
    n = len(diffs)
    means = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT)
    )
    return means[int(0.025 * N_BOOT)], means[int(0.975 * N_BOOT) - 1]


def hard_violation_decay(trajs: list[Trajectory]) -> dict[int, float]:
    counts = {k: [0, 0] for k in range(DECAY_HORIZON)}
    for t in trajs:
        switch = t.switch_steps[0]
        for s in t.steps:
            off = s.step - switch
            if 0 <= off < DECAY_HORIZON:
                counts[off][1] += 1
                counts[off][0] += any(k in s.reward_breakdown for k in HARD_RULE_KEYS)
    return {k: v[0] / v[1] for k, v in counts.items()}


def main() -> None:
    trajs = {name: run_policy_over_eval_set(f) for name, f in SCRIPTED_POLICIES.items()}
    returns = {name: [t.total_reward for t in ts] for name, ts in trajs.items()}

    per_policy = {
        name: {
            "mean_return": statistics.fmean(r),
            "sd_return": statistics.stdev(r),
            "hard_violation_counts": dict(Counter(
                k for t in trajs[name] for s in t.steps for k in s.reward_breakdown if k in HARD_RULE_KEYS
            )),
            "hard_violation_decay": hard_violation_decay(trajs[name]),
        }
        for name, r in returns.items()
    }

    rng = random.Random(BOOT_SEED)
    pairwise = []
    for a, b in PAIRS:
        diffs = [x - y for x, y in zip(returns[a], returns[b])]
        lo, hi = paired_bootstrap_ci(diffs, rng)
        w_plus, p_value = wilcoxon_signed_rank(diffs)
        sd = statistics.stdev(diffs)
        pairwise.append({
            "a": a, "b": b,
            "mean_diff": statistics.fmean(diffs),
            "ci95": [lo, hi],
            "wilcoxon_W_plus": w_plus,
            "wilcoxon_p": p_value,
            "cohens_dz": statistics.fmean(diffs) / sd if sd else None,
            "a_wins": sum(d > 0 for d in diffs),
            "n": len(diffs),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "fingerprint": compute_fingerprint(),
        "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
        "per_policy": per_policy, "pairwise": pairwise,
    }, indent=2))
    for name, p in per_policy.items():
        print(f"{name:16s} return {p['mean_return']:7.2f} ± {p['sd_return']:5.2f}  {p['hard_violation_counts']}")
        print("   hard-decay", [round(p["hard_violation_decay"][k], 3) for k in range(DECAY_HORIZON)])
    for c in pairwise:
        print(f"{c['a']} - {c['b']}: {c['mean_diff']:.2f} CI[{c['ci95'][0]:.2f},{c['ci95'][1]:.2f}] "
              f"W+={c['wilcoxon_W_plus']:.0f} p={c['wilcoxon_p']:.2e} dz={c['cohens_dz']:.2f} wins={c['a_wins']}/{c['n']}")


if __name__ == "__main__":
    main()

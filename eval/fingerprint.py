# Result fingerprinting (updatedplan.md Phase 0.5).
#
# Every result file (baseline run, trained-arm eval, ablation, figure) must
# embed compute_fingerprint()'s output. A result whose fingerprint does not
# match the current tree is invalid and must be regenerated, not trusted —
# this is what stops a stale number from being reused after ragas.py or
# reward.py changes underneath it.
#
# WHY file hashes and not a hand-maintained dict of "the reward constants":
# a dict would itself be a second copy of every magic number in reward.py,
# and could silently drift out of sync with the file it's supposed to be
# fingerprinting. Hashing the source file directly cannot drift.

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _hash_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT, text=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return bool(out.strip())


def compute_fingerprint() -> dict:
    """Git SHA plus content hashes of everything a reported number depends
    on: the pinned training stack, the raga rule definitions, the reward
    function, and (from Phase 4 on) the fixed eval-episode set, the metric
    formulas, and the reference policies a baseline result was produced
    with. Call this once per result and store the output alongside it."""
    # git_sha can never equal the commit that *contains* a result file
    # (committing changes the SHA), so a result committed alongside the code
    # that produced it records the parent commit with git_dirty=True. The
    # content hashes below — not git_sha — are the staleness check.
    return {
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "requirements_train_hash": _hash_file(REPO_ROOT / "requirements-train.txt"),
        "ragas_hash": _hash_file(REPO_ROOT / "raaga_env" / "ragas.py"),
        "reward_hash": _hash_file(REPO_ROOT / "raaga_env" / "reward.py"),
        "eval_episodes_hash": _hash_file(REPO_ROOT / "eval" / "episodes.py"),
        "metrics_hash": _hash_file(REPO_ROOT / "eval" / "metrics.py"),
        "policies_hash": _hash_file(REPO_ROOT / "eval" / "policies.py"),
        # Added 2026-10 when rollout.py gained call-phrase injection
        # (eval.rollout.rollout()'s call_phrase_fn) — a behaviour change to
        # the harness itself that none of the fields above would have
        # caught, since rollout.py/evaluate.py weren't hashed. A silent gap
        # in exactly the mechanism this function exists to provide.
        "rollout_hash": _hash_file(REPO_ROOT / "eval" / "rollout.py"),
        "evaluate_hash": _hash_file(REPO_ROOT / "eval" / "evaluate.py"),
        # Added 2026-10-07 (audit): env dynamics (CALL_EVERY, obs layout,
        # grace/adaptation constants) and prompt rendering change every
        # number too, and none of the fields above covered them — the
        # prompt note-decoding fix would have left old LLM results looking
        # valid.
        "env_hash": _hash_file(REPO_ROOT / "raaga_env" / "env.py"),
        "jugalbandi_env_hash": _hash_file(REPO_ROOT / "raaga_env" / "jugalbandi_env.py"),
        "drift_hash": _hash_file(REPO_ROOT / "raaga_env" / "drift.py"),
        "prompting_hash": _hash_file(REPO_ROOT / "raaga_env" / "prompting.py"),
        "llm_policy_hash": _hash_file(REPO_ROOT / "eval" / "llm_policy.py"),
    }

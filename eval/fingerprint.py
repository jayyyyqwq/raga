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


def compute_fingerprint() -> dict:
    """Git SHA plus content hashes of everything a reported number depends
    on: the pinned training stack, the raga rule definitions, the reward
    function, and (from Phase 4 on) the fixed eval-episode set, the metric
    formulas, and the reference policies a baseline result was produced
    with. Call this once per result and store the output alongside it."""
    return {
        "git_sha": _git_sha(),
        "requirements_train_hash": _hash_file(REPO_ROOT / "requirements-train.txt"),
        "ragas_hash": _hash_file(REPO_ROOT / "raaga_env" / "ragas.py"),
        "reward_hash": _hash_file(REPO_ROOT / "raaga_env" / "reward.py"),
        "eval_episodes_hash": _hash_file(REPO_ROOT / "eval" / "episodes.py"),
        "metrics_hash": _hash_file(REPO_ROOT / "eval" / "metrics.py"),
        "policies_hash": _hash_file(REPO_ROOT / "eval" / "policies.py"),
    }

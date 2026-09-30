# Single prompt-rendering interface (updatedplan.md Phase 2.1).
#
# WHY one module: there used to be three independent prompt builders
# (openenv_server/server.py, training/train_grpo.py, and the notebook) that
# were supposed to agree and didn't (F7) — and none of them had any concept
# of an "arm": the raga name and the grace-period flag were rendered into
# every prompt unconditionally, in plain English (F2), which is the finding
# that kills the "the model infers the rule set" claim regardless of what
# the observation vector contains. Every prompt-producing call site must
# import this module rather than building its own string, so a field can't
# silently leak into an arm that isn't supposed to see it.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .ragas import NOTE_NAMES

ACTION_SPACE_N = 96

DURATION_NAMES = {0: "sixteenth", 1: "eighth", 2: "quarter", 3: "half"}

SYSTEM = (
    "You are an expert Indian classical musician composing in a raga. "
    f"Given the current musical context, choose the next note action (0-{ACTION_SPACE_N - 1}). "
    "Action encodes: note (0-23, absolute pitch across two registers) "
    "+ duration_index (0-3) * 24. "
    "Output ONLY the integer, nothing else."
)


class Arm(Enum):
    """Observation arms (updatedplan.md §3 / Phase 2.2). The obs vector and
    env dynamics are identical across arms — only what render_prompt puts
    into text differs."""

    ORACLE = "oracle"  # raga named, dial shown, grace flag, steps since switch
    DIAL = "dial"       # dial shown only
    HIDDEN = "hidden"   # no raga-identifying signal at all


@dataclass(frozen=True)
class StepFeedback:
    """Rule-agnostic outcome of the previous step, available to every arm
    (Phase 2.3) — it never says *why* a step was rewarded or penalised, so
    it can't leak the rule set. This is Claim B's information channel
    (updatedplan.md §9 decision 2, resolved in favour of Claim B: in-context
    regime inference from reward feedback alone): without it, "infer the
    rule change from reward correlation" cannot happen in-context, because
    reward was never in the context."""

    note_name: str
    reward: float

    def render(self) -> str:
        outcome = "rewarded" if self.reward >= 0 else "penalised"
        return f"Last action: {self.note_name} -> outcome: {outcome} ({self.reward:+.2f})"


def note_name(note: int, duration: int) -> str:
    """Renders a raw (note, duration) pair — as returned by
    JugalbandiEnv._decode / info["note"]/info["duration"] — into the same
    note-name text the prompt uses elsewhere. Used directly by callers that
    just stepped the env (e.g. server.py building the next StepFeedback),
    and by _note_name_from_obs below for decoding a prompt's own obs slots."""
    return f"{NOTE_NAMES[note]}({DURATION_NAMES[duration]})"


def _note_name_from_obs(obs: list[float], slot: int) -> str:
    note_idx = min(int(obs[slot * 2] * 23), 23)  # obs pitch dims are note/23.0
    dur_idx = min(int(obs[slot * 2 + 1] * 3), 3)
    return note_name(note_idx, dur_idx)


def _context_lines(
    obs: list[float],
    *,
    arm: Arm,
    tala_pos: int,
    feedback: StepFeedback | None = None,
    raga: str | None = None,
) -> list[str]:
    """Everything in the prompt except the system header and the final
    instruction line. Split out from render_prompt so arm-gating logic lives
    in exactly one place, the way F7 (three divergent prompt builders)
    should have worked from the start."""
    if arm is Arm.ORACLE and raga is None:
        raise ValueError("ORACLE arm requires the active raga name")

    dial = round(obs[15], 2)
    pakad_drought = round(obs[16], 2)
    vadi_drought = round(obs[20], 2)
    tension = round(obs[14], 2)
    in_grace = obs[21] > 0.5
    steps_since_switch = round(obs[17], 2)

    lines: list[str] = []

    if arm is Arm.ORACLE:
        grace_text = "  GRACE PERIOD - rules just changed" if in_grace else ""
        lines.append(f"Active raga: {raga} (dial={dial}{grace_text})")
        lines.append(f"Steps since last rule change: {steps_since_switch} (normalised)")
    elif arm is Arm.DIAL:
        lines.append(f"Raga dial: {dial}")
    # HIDDEN: no raga name, no dial, no grace flag, no steps-since-switch.

    last_notes = [_note_name_from_obs(obs, i) for i in range(4)]
    lines.append(f"Tala position: beat {tala_pos}/16")
    lines.append(f"Last 4 notes: {', '.join(last_notes)}")
    lines.append(f"Human call tension: {tension:.2f} (how unresolved their phrase was)")
    lines.append(f"Pakad drought: {pakad_drought:.2f} (0=just played a phrase, 1=very long since last phrase)")
    lines.append(f"Vadi drought: {vadi_drought:.2f} (0=vadi just played, 1=long since)")
    if feedback is not None:
        lines.append(feedback.render())

    return lines


def render_prompt(
    obs: list[float],
    *,
    arm: Arm,
    tala_pos: int,
    feedback: StepFeedback | None = None,
    raga: str | None = None,
) -> str:
    """Render the user-turn prompt for one observation arm, asking for a
    single next action.

    `obs` is JugalbandiEnv's 22-dim observation (see JugalbandiEnv's
    docstring for the dim layout). `raga` is the active raga's name and is
    used only by the ORACLE arm — DIAL and HIDDEN must never receive or
    render it (F2); pass None for those arms.
    """
    lines = ["<|system|>", SYSTEM, "<|user|>"]
    lines += _context_lines(obs, arm=arm, tala_pos=tala_pos, feedback=feedback, raga=raga)
    lines.append(f"Choose action (0-{ACTION_SPACE_N - 1}):")
    lines.append("<|assistant|>")
    return "\n".join(lines) + "\n"


def parse_action(text: str) -> int | None:
    """Returns None on a parse failure or an out-of-range value. Never
    clamps via modulo (Phase 1.2): an out-of-range action is an
    instruction-following failure, not a different legal action."""
    try:
        action = int(text.strip().split()[0])
    except (ValueError, IndexError):
        return None
    if not (0 <= action < ACTION_SPACE_N):
        return None
    return action


def decode(action: int) -> tuple[int, int]:
    """The encoding SYSTEM describes in English, and the one
    JugalbandiEnv._decode implements: note = action % 24 (absolute pitch),
    duration = action // 24."""
    return action % 24, action // 24

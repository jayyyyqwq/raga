# Drift mechanic: manages raga switching, grace period, and adaptation bonus.
#
# WHY a separate module: drift logic is stateful and complex enough that
# embedding it in env.py would push that file past 300 lines. Clean separation.

from .ragas import raga_from_dial, RAGAS, match_pakad

GRACE_PERIOD_STEPS = 3
GRACE_PENALTY_FACTOR = 0.20   # only 20% of hard penalties during grace (80% reduction)
ADAPTATION_WINDOW = 5          # steps to play new-raga pakad for bonus
ADAPTATION_BONUS = 3.0


class DriftManager:
    def __init__(self, initial_dial: float = 0.0):
        self.dial = initial_dial
        self.active_raga_name = raga_from_dial(initial_dial)
        self.steps_since_switch = 999  # large = no recent switch
        self.grace_steps_remaining = 0
        self.adaptation_window_remaining = 0
        self.adaptation_bonus_awarded = False

    @property
    def active_raga(self) -> dict:
        return RAGAS[self.active_raga_name]

    @property
    def in_grace_period(self) -> bool:
        return self.grace_steps_remaining > 0

    def set_dial(self, new_dial: float) -> bool:
        """Update dial; return True if raga switched."""
        new_raga = raga_from_dial(new_dial)
        switched = new_raga != self.active_raga_name
        self.dial = new_dial
        if switched:
            self.active_raga_name = new_raga
            self.steps_since_switch = 0
            self.grace_steps_remaining = GRACE_PERIOD_STEPS
            self.adaptation_window_remaining = ADAPTATION_WINDOW
            self.adaptation_bonus_awarded = False
        return switched

    def step(self, note: int, note_history: list[int]) -> float:
        """
        Advance drift state by one step. `note_history` is the history
        *before* `note` was played (same convention as ragas.match_pakad).
        Returns extra reward from drift-specific mechanics (grace + adaptation bonus).
        """
        extra_reward = 0.0
        self.steps_since_switch += 1

        if self.grace_steps_remaining > 0:
            self.grace_steps_remaining -= 1

        if self.adaptation_window_remaining > 0:
            self.adaptation_window_remaining -= 1
            if not self.adaptation_bonus_awarded:
                # Single source of truth for pakad matching (see ragas.py's
                # match_pakad docstring) — this used to hand-roll the same
                # check by iterating `(phrase, multiplier)` tuples without
                # unpacking them, which made the match condition unsatisfiable
                # and the adaptation bonus permanently dead.
                if match_pakad(note_history, note, self.active_raga) is not None:
                    extra_reward += ADAPTATION_BONUS
                    self.adaptation_bonus_awarded = True

        return extra_reward

    def get_state(self) -> dict:
        return {
            "dial": self.dial,
            "active_raga_name": self.active_raga_name,
            "steps_since_switch": self.steps_since_switch,
            "grace_steps_remaining": self.grace_steps_remaining,
            "adaptation_window_remaining": self.adaptation_window_remaining,
            "adaptation_bonus_awarded": self.adaptation_bonus_awarded,
        }

    def set_state(self, state: dict) -> None:
        self.dial = state["dial"]
        self.active_raga_name = state["active_raga_name"]
        self.steps_since_switch = state["steps_since_switch"]
        self.grace_steps_remaining = state["grace_steps_remaining"]
        self.adaptation_window_remaining = state["adaptation_window_remaining"]
        self.adaptation_bonus_awarded = state["adaptation_bonus_awarded"]

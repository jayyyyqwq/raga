# JugalbandiEnv — extends RaagaEnv with drift + call-response mechanics.
# This is what gets deployed to HF Spaces and used in training.
# The base RaagaEnv stays clean; all extensions live here.

import numpy as np
from gymnasium import spaces
from .env import RaagaEnv
from .drift import DriftManager
from .reward import compute_reward

CALL_EVERY = 8   # human injects a 4-note call every N steps


class JugalbandiEnv(RaagaEnv):
    """
    22-dim observation space.

    Extra dims vs base (14):
      [14-21] human call phrase (last 4 notes, note + duration each = 8 dims)
      BUT we collapse to: [14-17] call notes, [18] tension, [19] dial, [20] pakad_drought, [21] steps_since_switch
    So base obs is 12 dims (last 4 notes only) + 8 jugalbandi dims = 22 total.

    Obs layout:
      0-7  : last 4 played notes (note/11, duration/3 interleaved)
      8-11 : human call phrase (4 note values / 11)
      12   : melodic direction
      13   : tala position
      14   : tension metric (how unresolved human left their phrase)
      15   : raga dial value (0-1, the implicit schema signal)
      16   : pakad drought (normalised)
      17   : steps since last schema switch (normalised)
      18   : vadi distance
      19   : samvadi distance
      20   : vadi drought
      21   : in grace period (0 or 1)
    """

    def __init__(self, initial_dial: float = 0.0, **kwargs):
        kwargs.setdefault("episode_length", 64)
        super().__init__(**kwargs)

        self.drift = DriftManager(initial_dial)
        # Override raga to follow the dial
        self.raga_name = self.drift.active_raga_name
        self.raga = self.drift.active_raga

        # Override obs space
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(22,), dtype=np.float32
        )

        self.call_phrase: list[int] = [0, 0, 0, 0]
        self.call_tension: float = 0.0
        self.steps_until_call = CALL_EVERY

    # ------------------------------------------------------------------
    # Gym API overrides
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.drift = DriftManager(self.drift.dial)
        self.raga_name = self.drift.active_raga_name
        self.raga = self.drift.active_raga
        self.call_phrase = [0, 0, 0, 0]
        self.call_tension = 0.0
        self.steps_until_call = CALL_EVERY
        return self._get_obs(), info

    def step(self, action: int):
        note, duration = self._decode(action)

        # Sync raga to drift dial before computing reward
        self.raga = self.drift.active_raga

        grace = self.drift.apply_grace
        reward, breakdown = compute_reward(
            note=note,
            duration=duration,
            note_history=list(self.note_history),
            dur_history=list(self.dur_history),
            tala_position=self.tala_position,
            direction=self._direction(),
            raga=self.raga,
            tala=self.tala,
            pakad_drought=self.pakad_drought,
            vadi_drought=self.vadi_drought,
            grace_factor=0.2 if self.drift.in_grace_period else 1.0,
            call_phrase=self.call_phrase,
            call_tension=self.call_tension,
        )

        drift_bonus = self.drift.step(note, list(self.note_history) + [note])
        reward += drift_bonus
        if drift_bonus > 0:
            breakdown["adaptation_bonus"] = drift_bonus

        self._update_state(note, duration)
        self.episode_reward += reward

        self.steps_until_call -= 1
        if self.steps_until_call <= 0:
            self.steps_until_call = CALL_EVERY
            # Signal to the server/UI that it's time for a human call phrase
            breakdown["call_requested"] = True

        terminated = self.step_count >= self.episode_length
        info = {
            "reward_breakdown": breakdown,
            "forbidden_note_count": self.forbidden_count,
            "pakad_completions": self.pakad_completions,
            "vadi_emphasis_count": self.vadi_count,
            "episode_reward": self.episode_reward,
            "note": note,
            "duration": duration,
            "tala_position": self.tala_position,
            "active_raga": self.drift.active_raga_name,
            "dial": self.drift.dial,
            "in_grace_period": self.drift.in_grace_period,
            "call_requested": breakdown.get("call_requested", False),
        }
        return self._get_obs(), reward, terminated, False, info

    # ------------------------------------------------------------------
    # Human interaction API (called by the server, not the agent)
    # ------------------------------------------------------------------

    def set_dial(self, dial: float) -> bool:
        """Move the raga dial. Returns True if raga switched."""
        switched = self.drift.set_dial(dial)
        if switched:
            self.raga_name = self.drift.active_raga_name
            self.raga = self.drift.active_raga
        return switched

    def set_call(self, notes: list[int]) -> None:
        """Human submits a 4-note call phrase."""
        self.call_phrase = notes[:4]
        # Tension: how far from vadi the last call note lands
        last = notes[-1] if notes else 0
        self.call_tension = abs(last - self.raga["vadi"]) / 11.0

    # ------------------------------------------------------------------
    # Obs override
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(22, dtype=np.float32)
        hist = list(self.note_history)
        dur = list(self.dur_history)

        # 0-7: last 4 played notes
        for i, idx in enumerate(range(max(0, len(hist) - 4), len(hist))):
            obs[i * 2] = hist[idx] / 11.0
            obs[i * 2 + 1] = dur[idx] / 3.0 if idx < len(dur) else 0.0

        # 8-11: human call phrase (4 note values)
        for i, n in enumerate(self.call_phrase[:4]):
            obs[8 + i] = n / 11.0

        obs[12] = self._direction() / 2.0
        obs[13] = self.tala_position / (self.tala["beats"] - 1)
        obs[14] = self.call_tension
        obs[15] = self.drift.dial
        obs[16] = min(self.pakad_drought / 20.0, 1.0)
        obs[17] = min(self.drift.steps_since_switch / 20.0, 1.0)
        last = hist[-1] if hist else 0
        obs[18] = abs(last - self.raga["vadi"]) / 11.0
        obs[19] = abs(last - self.raga["samvadi"]) / 11.0
        obs[20] = min(self.vadi_drought / 16.0, 1.0)
        obs[21] = 1.0 if self.drift.in_grace_period else 0.0
        return obs

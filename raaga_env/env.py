# Base Gymnasium environment — single-raga, no drift, no call-response.
# Keep this clean. JugalbandiEnv extends it with drift + human-in-loop.

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
from .ragas import RAGAS, TALAS, DURATIONS, NOTE_NAMES


class RaagaEnv(gym.Env):
    """
    RL environment for Indian classical raga composition.

    Observation: 14-dim continuous vector (see _get_obs)
    Action:      Discrete 48 = 12 semitones × 4 durations
    Reward:      Rule-based raga grammar (see reward.py)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        raga: str = "yaman",
        tala: str = "teentaal",
        episode_length: int = 32,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.raga_name = raga
        self.raga = RAGAS[raga]
        self.tala = TALAS[tala]
        self.episode_length = episode_length
        self.render_mode = render_mode
        self._history_len = 6

        # 24 absolute pitches (mandra 0-11, madhya 12-23) × 4 durations
        self.action_space = spaces.Discrete(96)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(14,), dtype=np.float32
        )

        # initialise state so _get_obs never fails before first reset
        self._init_state()

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._init_state()
        self.note_history.append(12)  # start on madhya Sa — home register
        self.dur_history.append(2)
        return self._get_obs(), {}

    def step(self, action: int):
        note, duration = self._decode(action)

        from .reward import compute_reward
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
        )

        self._update_state(note, duration)
        self.episode_reward += reward
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
        }
        return self._get_obs(), reward, terminated, False, info

    def render(self):
        if self.render_mode != "human":
            return
        recent = [NOTE_NAMES[n % len(NOTE_NAMES)] for n in list(self.note_history)[-4:]]
        print(
            f"Step {self.step_count:3d} | {recent} | "
            f"Tala {self.tala_position:2d} | R {self.episode_reward:.2f} | "
            f"Pakads {self.pakad_completions} | Forbidden {self.forbidden_count}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_state(self):
        self.note_history: deque[int] = deque(maxlen=self._history_len)
        self.dur_history: deque[int] = deque(maxlen=self._history_len)
        self.step_count = 0
        self.tala_position = 0
        self.pakad_drought = 0
        self.vadi_drought = 0
        self.episode_reward = 0.0
        self.forbidden_count = 0
        self.pakad_completions = 0
        self.vadi_count = 0

    def _decode(self, action: int) -> tuple[int, int]:
        return action % 24, action // 24

    def _direction(self) -> int:
        """0=neutral, 1=ascending, 2=descending based on last 3 notes."""
        hist = list(self.note_history)
        if len(hist) < 2:
            return 0
        recent = hist[-3:]
        asc = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i - 1])
        dsc = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i - 1])
        if asc > dsc:
            return 1
        if dsc > asc:
            return 2
        return 0

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(14, dtype=np.float32)
        hist = list(self.note_history)
        dur = list(self.dur_history)
        for i, idx in enumerate(range(max(0, len(hist) - 4), len(hist))):
            slot = i
            obs[slot * 2] = hist[idx] / 23.0          # normalise over 24-note range
            obs[slot * 2 + 1] = dur[idx] / 3.0 if idx < len(dur) else 0.0
        obs[8] = self._direction() / 2.0
        obs[9] = self.tala_position / (self.tala["beats"] - 1)
        # Swara-level distance (circular on 12-note wheel) so octave doesn't distort distance.
        last_swara = (hist[-1] if hist else 12) % 12
        vadi, samvadi = self.raga["vadi"], self.raga["samvadi"]
        obs[10] = min(abs(last_swara - vadi),    12 - abs(last_swara - vadi))    / 6.0
        obs[11] = min(abs(last_swara - samvadi), 12 - abs(last_swara - samvadi)) / 6.0
        obs[12] = min(self.pakad_drought / 20.0, 1.0)
        obs[13] = min(self.vadi_drought / 16.0, 1.0)
        return obs

    def _update_state(self, note: int, duration: int):
        self.note_history.append(note)
        self.dur_history.append(duration)
        self.step_count += 1
        self.tala_position = (
            self.tala_position + DURATIONS[duration]
        ) % self.tala["beats"]

        if note % 12 == self.raga["vadi"]:
            self.vadi_drought = 0
            self.vadi_count += 1
        else:
            self.vadi_drought += 1

        recent = list(self.note_history)
        pakad_hit = False
        for phrase, _reward in self.raga["pakads"]:
            n = len(phrase)
            if len(recent) >= n and recent[-n:] == phrase:
                self.pakad_drought = 0
                self.pakad_completions += 1
                pakad_hit = True
                break
        if not pakad_hit:
            self.pakad_drought += 1

        if note % 12 in self.raga["forbidden_notes"]:
            self.forbidden_count += 1

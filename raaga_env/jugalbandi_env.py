# JugalbandiEnv — extends RaagaEnv with drift + call-response mechanics.
# This is what the server, training, and eval harness all use.
# The base RaagaEnv stays clean; all extensions live here.

import numpy as np
from gymnasium import spaces
from .env import RaagaEnv
from .drift import DriftManager, GRACE_PENALTY_FACTOR
from .reward import compute_reward

CALL_EVERY = 8   # human injects a 4-note call every N steps


class JugalbandiEnv(RaagaEnv):
    """
    22-dim observation space.

    Obs layout:
      0-7  : last 4 played notes (note/23, duration/3 interleaved, right-
             aligned: slots 6-7 are the most recent note) — reused verbatim
             from RaagaEnv._get_obs() so the two can't diverge again
      8-11 : human call phrase (4 note values / 11)
      12   : melodic direction        (from RaagaEnv._get_obs())
      13   : tala position            (from RaagaEnv._get_obs())
      14   : tension metric (how unresolved human left their phrase)
      15   : raga dial value (0-1, the implicit schema signal)
      16   : pakad drought (normalised)   (from RaagaEnv._get_obs())
      17   : steps since last schema switch (normalised)
      18   : vadi distance, circular     (from RaagaEnv._get_obs())
      19   : samvadi distance, circular  (from RaagaEnv._get_obs())
      20   : vadi drought                (from RaagaEnv._get_obs())
      21   : in grace period (0 or 1)
    """

    def __init__(self, initial_dial: float = 0.0, **kwargs):
        kwargs.setdefault("episode_length", 64)
        super().__init__(**kwargs)

        self.initial_dial = initial_dial
        self.drift = DriftManager(initial_dial)
        # Override raga to follow the dial
        self.raga_name = self.drift.active_raga_name
        self.raga = self.drift.active_raga

        # Inherited from RaagaEnv, but declared again here so the 96-action
        # contract is local and directly testable on this class rather than
        # only implied by inheritance (updatedplan.md Phase 1.1).
        self.action_space = spaces.Discrete(96)

        # Override obs space
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(22,), dtype=np.float32
        )

        # Empty, not [0,0,0,0]: a call of all-Sa is a real, legal call and
        # must be distinguishable from "no call has been submitted yet".
        # [0,0,0,0] used to be the default, which is truthy in Python, so
        # reward.py's `if call_phrase:` guard on the whole jugalbandi layer
        # fired on every single step of every training episode — training
        # never called set_call() before the 2026-10 fix, so the v1 model was
        # scored the entire time against a phantom call it never received
        # (ML retrain finding, 2026-10; see docs/RETRAIN_PLAN.md).
        self.call_phrase: list[int] = []
        self.call_tension: float = 0.0
        self.call_echoed: bool = False
        self.steps_until_call = CALL_EVERY

    # ------------------------------------------------------------------
    # Gym API overrides
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # Start from options["dial"] if given, else the constructor's
        # initial_dial — not whatever dial the *previous* episode ended on
        # (audit 2026-10-07: a reset after a mid-episode switch used to start
        # the next episode in the switched-to raga).
        dial = (options or {}).get("dial", self.initial_dial)
        self.drift = DriftManager(dial)
        self.raga_name = self.drift.active_raga_name
        self.raga = self.drift.active_raga
        self.call_phrase = []
        self.call_tension = 0.0
        self.call_echoed = False
        self.steps_until_call = CALL_EVERY
        return self._get_obs(), info

    def step(self, action: int):
        note, duration = self._decode(action)

        # Sync raga to drift dial before computing reward
        self.raga = self.drift.active_raga

        reward, breakdown = compute_reward(
            note=note,
            duration=duration,
            note_history=list(self.note_history),
            dur_history=list(self.dur_history),
            tala_position=self.tala_position,
            raga=self.raga,
            tala=self.tala,
            pakad_drought=self.pakad_drought,
            vadi_drought=self.vadi_drought,
            grace_factor=GRACE_PENALTY_FACTOR if self.drift.in_grace_period else 1.0,
            call_phrase=self.call_phrase,
            call_tension=self.call_tension,
            call_echoed=self.call_echoed,
        )
        if "call_echo" in breakdown:
            self.call_echoed = True

        # note_history is passed *before* this note is appended, matching
        # ragas.match_pakad's contract (see DriftManager.step).
        drift_bonus = self.drift.step(note, list(self.note_history))
        reward += drift_bonus
        if drift_bonus > 0:
            breakdown["adaptation_bonus"] = drift_bonus
        # compute_reward() set breakdown["total"] before the drift bonus
        # existed; keep it in sync with the reward actually returned (F10).
        breakdown["total"] = reward

        self._update_state(note, duration)
        self.episode_reward += reward

        terminated = self.step_count >= self.episode_length

        # Never on the terminal step: a call requested there can't be
        # submitted or answered, but eval.metrics still counted it as a
        # call-response pair — 8 per 64-step episode instead of the 7 that
        # actually happen, capping call_echo_rate at 0.875 (audit 2026-10-07).
        self.steps_until_call -= 1
        call_requested = False
        if self.steps_until_call <= 0:
            self.steps_until_call = CALL_EVERY
            call_requested = not terminated
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
            # Not a reward component — belongs in info, not reward_breakdown
            # (F10: it was previously stuffed into the breakdown dict, which
            # broke any code summing breakdown values as float rewards).
            "call_requested": call_requested,
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
        """Human submits a 4-note call phrase (swara values 0-11, matching
        obs[8:12]'s n/11.0 encoding — no register, calls carry no octave)."""
        self.call_phrase = notes[:4]
        # Tension: how far from vadi the last call note lands. Read from the
        # stored (truncated) phrase so tension and call_echo agree on which
        # note "ended" the call.
        last = self.call_phrase[-1] if self.call_phrase else 0
        # Circular swara distance (max 6 semitones), matching the vadi-
        # distance obs dims — linear |last - vadi| / 11 treated Ni and Sa,
        # one semitone apart, as maximally distant (audit 2026-10-07).
        dist = abs(last - self.raga["vadi"])
        self.call_tension = min(dist, 12 - dist) / 6.0
        # A new call is a fresh chance to answer it.
        self.call_echoed = False

    # ------------------------------------------------------------------
    # State serialisation (extends RaagaEnv.get_state/set_state with drift
    # and call-response state — updatedplan.md Phase 0.4)
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        state = super().get_state()
        state["drift"] = self.drift.get_state()
        state["call_phrase"] = list(self.call_phrase)
        state["call_tension"] = self.call_tension
        state["call_echoed"] = self.call_echoed
        state["steps_until_call"] = self.steps_until_call
        return state

    def set_state(self, state: dict) -> None:
        super().set_state(state)
        self.drift.set_state(state["drift"])
        self.raga_name = self.drift.active_raga_name
        self.raga = self.drift.active_raga
        self.call_phrase = list(state["call_phrase"])
        self.call_tension = state["call_tension"]
        # .get(): tolerates state_json saved before call_echoed existed
        # (Phase 0.4 snapshots predating the ML retrain fix).
        self.call_echoed = state.get("call_echoed", False)
        self.steps_until_call = state["steps_until_call"]

    # ------------------------------------------------------------------
    # Obs override
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        # base layout (14-dim): notes[0:8], direction[8], tala[9],
        # vadi_dist[10], samvadi_dist[11], pakad_drought[12], vadi_drought[13]
        base = super()._get_obs()
        obs = np.zeros(22, dtype=np.float32)

        obs[0:8] = base[0:8]  # last 4 played notes (note/23, duration/3)

        # 8-11: human call phrase (4 note values)
        for i, n in enumerate(self.call_phrase[:4]):
            obs[8 + i] = n / 11.0

        obs[12] = base[8]   # melodic direction
        obs[13] = base[9]   # tala position
        obs[14] = self.call_tension
        obs[15] = self.drift.dial
        obs[16] = base[12]  # pakad drought (normalised)
        obs[17] = min(self.drift.steps_since_switch / 20.0, 1.0)
        obs[18] = base[10]  # vadi distance, circular
        obs[19] = base[11]  # samvadi distance, circular
        obs[20] = base[13]  # vadi drought
        obs[21] = 1.0 if self.drift.in_grace_period else 0.0
        return obs

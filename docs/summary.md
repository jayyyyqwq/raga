# Jugalbandi — Project Summary

> An OpenEnv-compatible reinforcement learning environment that trains an LLM to
> compose Indian classical music (ragas) while the rule system ("grammar") drifts
> mid-episode. Built for the Meta PyTorch OpenEnv Hackathon (Theme 5 — Wild Card).
> The underlying research claim: this is a testbed for **schema adherence under
> constraint drift** — the same shape of problem as an LLM having to keep
> following an API contract, a compliance rule set, or a JSON schema after it
> changes mid-task, but rendered in a domain (music) where a human listener can
> instantly hear whether the model got it right, and specifically designed to
> separate two capabilities usually conflated in the drift literature:
> *conditioned* policy learning over an observed context variable, versus
> *in-context regime inference* from reward feedback alone (see
> `updatedplan.md` §3 and `docs/EXPERIMENT_PLAN.md` §4 for the full framing).

**No model has been trained as of this writing.** Everything in this document
describing the environment, the training pipeline, and the evaluation harness
is accurate against the current code and backed by a 120-test suite. Anything
about a *trained model's behavior* is either explicitly marked pending, or
comes from the four scripted baselines in `eval/results/` — which are real
measurements, not targets.

---

## 1. Repository Layout

```
Raaga_trial_1/
├── raaga_env/                  # Core RL environment (pure Python, no server/UI deps)
│   ├── ragas.py                 # Raga rule DSL: scales, pakads, vadi/samvadi, talas
│   ├── env.py                   # RaagaEnv — base Gymnasium env (single raga, no drift)
│   ├── jugalbandi_env.py        # JugalbandiEnv — extends RaagaEnv with drift + call-response
│   ├── reward.py                # 5-layer reward function
│   ├── drift.py                 # DriftManager — raga switching / grace period / adaptation bonus
│   ├── prompting.py             # THE single prompt-rendering interface (all arms, all callers)
│   └── tests/                   # 40+ tests: contracts, prompting, state, env, reward
│
├── openenv_server/              # HTTP wrapper exposing the env, OpenEnv-style
│   ├── server.py                 # FastAPI app: /reset /step /set_dial /call /state /infer /health
│   ├── tests/test_server.py      # StepFeedback wiring tests
│   ├── requirements.txt
│   └── Dockerfile
│
├── training/
│   ├── train_grpo.py             # GRPO training script — in-process, per-step, feedback-conditioned
│   ├── train_grpo.ipynb          # Thin-driver Colab notebook over train_grpo.py
│   └── tests/test_train_grpo.py  # Data-pipeline tests (everything short of GRPOTrainer.train())
│
├── eval/                        # In-process evaluation harness (Phase 0.3–4)
│   ├── rollout.py                # rollout() / rollout_from_state() — the one place a step gets scored
│   ├── policies.py               # Reference/baseline policies (random-uniform/valid, safe-set-cycle, scripted-oracle)
│   ├── episodes.py               # The fixed, shared, deterministic 200-episode eval set
│   ├── metrics.py                # Every precisely-defined Phase 4.3 metric
│   ├── evaluate.py               # Runner: policy × eval set → fingerprinted result file
│   ├── fingerprint.py            # Git SHA + content hashes of everything a result depends on
│   ├── results/                  # Real, fingerprinted baseline results (regenerate if stale)
│   └── tests/
│
├── ui/                           # Browser demo client (vanilla JS, ES modules, no build step)
│   ├── index.html / app.js / env_client.js / sitar.js / tabla.js / styles.css
│
├── docs/
│   ├── Jugalbandi.md              # The pitch — problem framing, mechanics, demo script, Q&A prep
│   ├── EXPERIMENT_PLAN.md         # Phase 4 pre-registration — read this first if you're new here
│   ├── CLASSICAL_MUSIC_AUDIT.md
│   ├── PHASE_PLAN.md / RaagaRL - Build Guide.md / UIChoice.md / Mid_life_crisis.md
│   ├── Train.md                   # Training walkthrough, kept in sync with the current pipeline
│   └── report-midway.md
│
├── openenv.yaml                  # OpenEnv manifest: env class, obs/action spaces, reward, rubric
├── requirements.txt               # Root deps (env + server + test)
├── requirements-train.txt         # Pinned GPU-only training stack (Colab only)
├── updatedplan.md                 # The remediation plan this whole codebase is executing against
└── CLAUDE.md
```

**Design principle used throughout:** each layer only imports the layer below
it. UI → `env_client.js` → HTTP → `server.py` → `raaga_env/*`. Training and
evaluation both import `raaga_env` and `eval` directly, in-process — neither
goes through HTTP (`updatedplan.md` Phase 3.1). The RL environment itself has
zero knowledge of FastAPI, Tone.js, or training.

---

## 2. What Each Component Does

### 2.1 `raaga_env/ragas.py` — the music theory DSL

The data layer encoding Hindustani classical music rules as plain Python
dicts, cross-checked against Bhatkhande's Kramik Pustak Malika (KPM) Vol. I
and the Parrikar raga archive (`docs/CLASSICAL_MUSIC_AUDIT.md`). Two ragas:

- **Yaman** (evening, shringar mood): sharpened Ma (Ma#), natural Ma
  forbidden, Pa skipped on ascent only (vakra rule), vadi = Ga, samvadi = Ni.
- **Bhairav** (morning, devotional mood): komal Re and komal Dha, natural Re
  and natural Dha forbidden, vadi = komal Dha, samvadi = komal Re.

Notes are absolute pitch integers 0–23 across two octaves (mandra 0–11,
madhya 12–23): `note % 12` gives swara identity register-agnostically, while
raw pitch differences capture melodic leaps and octave jumps correctly.
`match_pakad(note_history, note, raga)` is the single source of truth for
phrase detection — every call site (env bookkeeping, reward scoring, the
drift adaptation bonus, evaluation metrics) uses this one function.

### 2.2 `raaga_env/env.py` — `RaagaEnv` (base Gymnasium environment)

The undrifted single-raga case.

- **Action space:** `Discrete(96)` — `note = action % 24`, `duration =
  action // 24` (24 absolute pitches × 4 durations, two registers). This is
  the one consistent action-space contract across the entire codebase — env,
  server, training, and `openenv.yaml` all agree on 96
  (`updatedplan.md` Phase 1.1; cross-register pakads like Yaman's mandra-Ni
  opening or Bhairav's mandra-Ni close are only reachable at 96).
- **Observation space:** `Box(shape=(14,), low=0, high=1)` — last 4
  (note/23, duration/3) pairs, melodic direction, tala position, circular
  swara-distance from vadi/samvadi, pakad drought, vadi drought.
- **`get_state()` / `set_state()`** — full episode state serialization
  (`updatedplan.md` Phase 0.4), used for deterministic replay in the
  evaluation harness and for per-step training's state-restoring reward
  scoring.
- **`step(action)`**: decodes the action, calls `compute_reward(...)`,
  updates counters, terminates after `episode_length` steps.

### 2.3 `raaga_env/jugalbandi_env.py` — `JugalbandiEnv` (the actual training/demo env)

Subclasses `RaagaEnv`, adds:

1. **Drift** — wraps a `DriftManager` owning which raga is active based on a
   continuous "dial" value, plus grace-period and adaptation-bonus state.
2. **Call-response ("jugalbandi")** — a human submits a 4-note "call" phrase
   via `set_call()`; the env tracks `call_tension` and folds it into reward.
3. **22-dim observation space** — reuses `RaagaEnv._get_obs()` for every
   shared dimension (note history, direction, tala position, vadi/samvadi
   distance, both droughts) rather than recomputing them, so the two can't
   drift out of sync (`updatedplan.md` Phase 1.3 fixed a real bug here: the
   observation used to divide pitch by 11 instead of 23, putting values
   outside the declared `Box(0,1)` for most of the madhya register). Dim 15
   is the raw, unlabeled dial value — the implicit schema signal.

Episode length defaults to 64 steps. Every 8 steps (`CALL_EVERY`), `info`
flags `call_requested=True`.

### 2.4 `raaga_env/reward.py` — the 5-layer reward function

`compute_reward(...)` returns `(total_reward: float, breakdown: dict[str,
float])`.

**Layer 1 — Hard rules (early return):** forbidden note → `-2.0 *
grace_factor`; direction-sensitive aaroha violation → `-1.0 * grace_factor`;
any other out-of-scale note → `-1.5 * grace_factor`.

**Layer 2 — Soft rules (cumulative):** valid note `+0.2`; vadi `+0.4`;
samvadi `+0.3`; vadi-drought penalty (unplayed >8 steps) `-0.05` per extra
step, **floored at -1.0**; large-leap penalty `-0.2` per excess semitone
(currently the only *unbounded* per-step penalty — see §4); repetition
penalty `-0.3`; holding the vadi `+0.15`.

**Layer 3 — Sequence-level:** pakad completion `+1.0 × multiplier` (0.5–1.2);
pakad-drought penalty (none completed in >12 steps) `-0.03` per extra step,
**floored at -0.5**; vadi-on-sam `+0.8`; weak-note-on-sam `-0.4`;
vadi/samvadi on a strong beat `+0.2`.

**Layer 4 — Jugalbandi (call-response):** resolving a high-tension human call
by landing on the vadi, up to `+0.5`; contrasting melodic direction with the
human's call, `+0.3`.

**Layer 5 — Drift-specific (`drift.py`, folded in by the caller):** grace
period scales hard-rule penalties by `GRACE_PENALTY_FACTOR = 0.2` for 3
steps after a switch; adaptation bonus `+3.0` for completing a new-raga
pakad within 5 steps of a switch.

**The drought-penalty floor is `updatedplan.md` Phase 5.1's fix for finding
F11**: both drought penalties used to grow without limit, so a long enough
drought eventually cost more than an actual forbidden-note violation — the
environment was punishing *neglect* harder than *rule-breaking*. Both are now
floored at a value that can never exceed the mildest hard-rule penalty
(`aaroha_violation`, -1.0). The real, measured effect is visible in
`eval/results/`: before the fix, `safe-set-cycle` (a policy that never
adapts to drift at all) beat `random-valid` on reward; after the fix the gap
narrowed sharply, but `scripted-oracle` (genuine, fast adaptation) still only
barely beats `safe-set-cycle` on mean reward (-22.24 vs -22.34 over the fixed
200-episode set) — meaning the reward function still doesn't reward real
adaptation as strongly as it probably should. That gap is real, measured, and
open; it is not yet fixed, and Phase 5.2/5.3's ablations are the next honest
step, not this paragraph asserting it's solved.

Declared reward range in `openenv.yaml`: **min -6.0, max 7.0** per step — a
conservative analytical envelope, re-derived after the F11 fix (see
`openenv.yaml`'s own comments for the exact derivation), not a measured
range.

### 2.5 `raaga_env/drift.py` — `DriftManager`

Owns dial/raga-switch state, grace period, and the adaptation bonus.
`step(note, note_history)` calls `ragas.match_pakad()` — the same shared
pakad-matching function everything else uses. (An earlier version of this
method re-implemented pakad matching inline, iterating `(phrase,
multiplier)` tuples without unpacking them; the match condition was
structurally unsatisfiable, so the adaptation bonus could never fire. Fixed
by routing through `match_pakad` like everywhere else.)

### 2.6 `raaga_env/prompting.py` — the single prompt-rendering interface

**Every** prompt shown to a model, anywhere in this codebase, is built by
this one module (`updatedplan.md` Phase 2.1) — there used to be three
independent, hand-duplicated prompt builders (server, training script,
notebook) that were supposed to agree and didn't; one of them named the raga
and announced grace periods in plain English regardless of which "arm" was
supposedly being trained (finding F2, the one that would have invalidated
the whole research claim if it had shipped unnoticed).

- **`Arm`** — `ORACLE` (raga named, dial shown, grace flag, steps-since-switch),
  `DIAL` (dial number only), `HIDDEN` (none of the above). Enforced by test:
  the DIAL and HIDDEN renderers are asserted to never contain the words
  "yaman", "bhairav", "grace", "dial" (DIAL excepted for its own field), or
  "switch".
- **`StepFeedback`** — the rule-agnostic "Last action: X -> outcome:
  rewarded/penalised (±R)" line shown to every arm (Phase 2.3, Claim B).
  Never says *why* — this is what makes in-context inference possible at all
  for DIAL/HIDDEN, since without it reward is never in the model's context.
- **`render_prompt(...)`** — one action per call, used by both live
  inference (`server.py`) and per-step training.
- **`parse_action(text)`** — returns `None` on any parse failure or
  out-of-range value; never clamps via modulo. An out-of-range model output
  is recorded as an instruction-following failure, not silently turned into
  a different, legal action (`updatedplan.md` Phase 1.2).

### 2.7 `openenv_server/server.py` — FastAPI HTTP wrapper

Wraps a process-global `JugalbandiEnv` (`initial_dial=0.0, episode_length=64`).

| Method & path | Purpose |
|---|---|
| `POST /reset` | `{seed?, dial}` → resets episode, clears the feedback channel |
| `POST /step` | `{action: 0-95}` → advances one step, records the outcome as the next `StepFeedback` |
| `POST /set_dial` | `{value: 0-1}` → moves the dial mid-episode |
| `POST /call` | `{notes: [..]}` → human submits a call phrase |
| `GET /state` | snapshot of current raga/dial/step/reward/counters |
| `POST /infer` | runs the trained policy on the current obs (rendered via `raaga_env.prompting`, arm set by `JUGALBANDI_INFER_ARM`), returns a chosen action — does not step the env |
| `GET /health` | liveness probe |

`/infer` lazily imports `torch`/`transformers`/`peft` (zero GPU dependency
for normal env use), loads a base model plus a LoRA adapter from
`JUGALBANDI_ADAPTER_REPO`, and parses the output via `prompting.parse_action`
— an out-of-range or unparseable output is a clean 502, never silently
clamped.

### 2.8 `training/train_grpo.py` — GRPO fine-tuning script

Runs on a Colab GPU, not locally — GPU-only imports (`unsloth`, `trl`,
`torch`, `wandb`) are lazily deferred into `main()` so the data pipeline
itself stays importable and unit-testable without a training stack
installed.

**Training is in-process and per-step**, not the original per-episode HTTP
bandit design (`updatedplan.md` Phase 3, revised after Phase 2.3 settled on
Claim B — see that section for why a full blind 64-action completion per
sample is structurally incompatible with using per-step feedback):

1. `sample_drift_schedule()` — one drift event per reference episode,
   uniform over step 16–48, flipped to the opposite raga; 20% of episodes
   get no switch at all (a control condition, so the model can't just learn
   "a switch always happens near the middle").
2. `build_dataset(n, arm)` — walks a fresh reference episode forward under
   `random_valid_policy` (with the sampled schedule), stops at a uniformly
   chosen step, and snapshots: the prompt at that point (with the previous
   step's `StepFeedback`), and the exact env state (`get_state()`, as JSON).
3. `make_step_reward_fn(arm)` — GRPO's reward function. Restores the env to
   the snapshot's exact state, applies the sampled action, then continues
   for a short horizon under `random_valid_policy` (a Monte-Carlo return) via
   `eval.rollout.rollout_from_state` — the same function every baseline and
   evaluation number goes through, so training reward and eval reward can't
   silently diverge.
4. Trains with TRL's `GRPOTrainer`, one action per completion
   (`max_completion_length=8`), logs to Weights & Biases, saves the LoRA
   adapter for upload to the HF Hub.

`training/train_grpo.ipynb` is a thin driver over this script — it imports
`build_dataset`/`make_step_reward_fn`/etc. rather than redefining them, so it
can't drift out of sync with the script the way the old notebook did.

### 2.9 `eval/` — the in-process evaluation harness

Built in `updatedplan.md` Phase 0.3 and expanded through Phase 4. No HTTP —
every downstream number (baselines, trained arms, ablations) goes through
this package, so a metric definition change applies everywhere at once.

- **`rollout.py`** — `rollout()` (fresh episode) and `rollout_from_state()`
  (continue from a saved snapshot). `Trajectory` carries its own
  `switch_steps`, so every timing-based metric is self-contained.
- **`policies.py`** — `random_uniform_policy` (absolute floor),
  `random_valid_policy` (honest floor — used by training too),
  `safe_set_cycle_policy` (names the F13 exploit explicitly),
  `scripted_oracle_policy` (the ceiling — "cheats" by being told the switch
  step directly, to calibrate what `drift_adaptation_speed` can physically be).
- **`episodes.py`** — the fixed, shared, deterministic 200-episode
  evaluation set (100 starting each raga, one switch each, fixed master
  seed) every policy is compared on.
- **`metrics.py`** — every `updatedplan.md` Phase 4.3 metric, precisely
  defined once: 3-way `valid_raga_adherence` split (pre-switch / grace /
  post-grace), censored `drift_adaptation_speed`, `adaptation_success_rate@K`,
  `post_switch_violation_decay`, `pakad_rate`, `jugalbandi_coherence`,
  `safe_set_occupancy`, `action_validity_rate`.
- **`evaluate.py`** — runs a policy over the fixed eval set and writes a
  fingerprinted result file. `eval/results/*.json` holds real output from
  the four scripted baselines.
- **`fingerprint.py`** — git SHA + content hashes of everything a result
  depends on (reward constants, raga rules, eval-episode definitions, metric
  formulas, the training-stack pin). A result whose fingerprint doesn't
  match the current tree is stale and must be regenerated, not trusted.

### 2.10 `ui/` — the browser demo

Framework-free ES-module JS (no build step). `env_client.js` is the only
place that calls `fetch()`. `app.js` drives the demo state machine
(`idle → human_call → ai_response → loop/ended`), calling `/infer` then
`/step` for each AI turn, falling back to a fixed safe note if no adapter is
deployed yet. `sitar.js` / `tabla.js` render SVG strings and drive Tone.js
synths (`PluckSynth`, `MembraneSynth`) — no audio sample files shipped.

### 2.11 `openenv.yaml` — the OpenEnv manifest

Declares `JugalbandiEnv` (96-action `Discrete`, 22-dim `Box` observation,
reward range `[-6.0, 7.0]`, both re-derived analytically and documented
inline with the derivation), episode config (`max_steps: 64`, human turn
every 8 steps), and a 4-criterion self-scoring rubric mapping directly onto
`eval/metrics.py`'s implementations.

### 2.12 Tests

120 tests across `raaga_env/tests/`, `openenv_server/tests/`, `eval/tests/`,
and `training/tests/`: environment contracts (obs-space bounds, action-space
agreement, reward-bounds, breakdown integrity — `raaga_env/tests/
test_contracts.py`), prompting/arm-leakage enforcement, state
serialization round-trips, the full reward function including the F11
drought-floor fix, the entire per-step training data pipeline (everything
short of an actual GPU `trainer.train()` call), and the evaluation harness
including every Phase 4.3 metric.

---

## 3. What The Project Produces / The Point Of It

**Core deliverable:** an OpenEnv-compliant RL environment (`JugalbandiEnv`)
and an evaluation harness capable of answering, with real statistics, whether
a small LLM can learn — from reward signal alone — to:

1. Stay inside a raga's valid-note grammar, including direction-sensitive
   rules a plain scale-membership check would miss.
2. Emphasize the raga's structurally important notes and complete
   recognizable characteristic phrases, rather than degenerating into a
   trivial safe policy (`safe_set_occupancy` exists specifically to detect
   this).
3. Respond coherently to a 4-note human "call" phrase.
4. **Detect and adapt to a rule change it is never explicitly told about** —
   this is Claim B (`updatedplan.md` §9 decision 2, resolved), and it is
   specifically the DIAL and HIDDEN arms' job, tested against the ORACLE
   arm as an upper bound and the scripted baselines as floor/ceiling.

**Status:** the environment, the training pipeline, and the evaluation
harness are built and tested (120 tests). The four scripted baselines have
been run for real (`eval/results/`). No model has been trained — that step
needs a GPU this development environment doesn't have; `docs/EXPERIMENT_PLAN.md`
has the exact commands and the full pre-registered protocol for when one is
available.

---

## 4. Notable Implementation Details (for anyone extending this)

- **Absolute-pitch vs swara-identity encoding is the key trick** that lets
  smoothness/leap penalties work naturally across octave boundaries while
  raga-rule checks stay register-agnostic — always compute rule membership
  on `note % 12`, melodic distance on the raw absolute pitch.
- **`match_pakad` is the single source of truth** for phrase detection,
  shared by env bookkeeping, reward scoring, and the drift adaptation bonus.
- **The dial's implicit-learning property is load-bearing for the research
  claim** — do not add an explicit "raga changed" observation dimension for
  the DIAL/HIDDEN arms; the whole point is inferring it from reward
  correlation (ORACLE) or reward correlation plus `StepFeedback`
  (DIAL/HIDDEN).
- **The large-melodic-leap penalty is still unbounded per step** — F11's fix
  (Phase 5.1) floored both drought penalties, not this one. It's currently
  the dominant term in the worst-case single-step reward bound
  (`openenv.yaml`'s declared `min`). Worth the same treatment F11 got, if a
  similar distortion shows up in the Phase 5.2/5.3 ablations.
- **Every prompt-producing call site imports `raaga_env.prompting`** — do
  not add a fourth hand-written prompt builder; F7 (three divergent
  builders) and F2 (one of them leaking the raga name to every arm) are both
  what happens when that discipline slips.
- **Training and evaluation are both in-process, never HTTP** — the server
  stays for the UI and OpenEnv discoverability, but a training run that
  round-trips over HTTP for every sample is both slow and (via one shared
  mutable env instance) a correctness hazard.
- **The server holds one global env instance per process** — fine for a
  single-user demo; a real multi-user deployment needs one process/worker
  per session.
- **Heavy ML dependencies are optional and lazily imported** — the base
  environment, tests, evaluation harness, and the server's non-`/infer`
  endpoints run with just `gymnasium`, `numpy`, `fastapi`, `uvicorn`,
  `pydantic`, `pyyaml`. `requirements-train.txt` (pinned, GPU-only) is a
  separate file specifically so the base stack never accidentally depends
  on it.

---

## 5. How To Rebuild This From Scratch (high-level steps)

1. Define the raga rule DSL as plain dict data (`ragas.py`): scale,
   forbidden notes, aaroha/avaroha contour, vadi/samvadi, weighted pakad
   phrases, tala beat structure. Absolute-pitch-with-modulo-12 encoding from
   the start.
2. Write a minimal Gymnasium `Env` (`env.py`): discrete action = note ×
   duration, a small continuous observation vector, plus `get_state()`/
   `set_state()` from day one — it's needed later for both training and
   evaluation, and is much more annoying to retrofit.
3. Extract reward logic into a pure function (`reward.py`) returning
   `(total, breakdown_dict)`: hard rules with early return, additive soft
   rules, sequence-level bonuses. **Floor every penalty that scales with an
   unbounded counter** (a drought, a streak) from the start — an unbounded
   penalty term will eventually dominate a bounded one, and it's a much
   worse bug once a training run has already used it.
4. Add a second raga, a `dial → raga` mapping, a stateful `DriftManager`
   (grace period + adaptation bonus routed through the *same* phrase-
   matching function as everything else), a call-phrase concept, and an
   expanded observation vector — reusing the base env's `_get_obs()` for
   every dimension that isn't new, rather than recomputing it.
5. Build **one** prompt-rendering module before writing a second call site
   that needs a prompt. Define your "arms"/conditions as an enum from the
   start if the project has any comparative claim at all, and write the
   leakage-enforcement test in the same commit as the enum.
6. Build an in-process evaluation harness (fresh-reset rollout + restore-
   from-state rollout) *before* writing a training script — training's
   reward function should call the harness, not the other way around, so
   there is only ever one place a number can come from.
7. Wrap the env in FastAPI with Pydantic-validated bodies; keep GPU/LLM
   inference behind a lazy import and a feature-flag env var.
8. Write the training script against the harness from step 6, in-process,
   never over HTTP. Pin the GPU-only dependency stack in its own
   requirements file from the start.
9. Build a static-file, no-build-step browser client talking only to your
   server.
10. Write an OpenEnv manifest with re-derivable (not just asserted) space
    bounds and reward bounds, and a rubric that maps onto real metric
    implementations, not just prose descriptions.
11. Write contract tests for the properties that actually break in practice
    — exact penalty/bound values, exact space agreement across every
    declaration site, arm-leakage, state round-trips — before writing the
    training or evaluation code that depends on them holding.

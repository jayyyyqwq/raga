# Jugalbandi — Updated Remediation & Research Plan

**Status:** working plan, supersedes the priority list in `docs/report-midway.md` §8.
**Written:** 2026-09-10
**Basis:** direct code inspection plus verification scripts run against the repo at commit
`06656a5` using the project's own `venv` (gymnasium 1.3.0, numpy 2.4.4). Every claim in §2 was
executed, not inferred. The commands are reproduced in §10 so they can be re-run.

**How to read this:** §1 corrects the incoming review. §2 is the verified defect list — this is the
part that matters, and it is worse than the review suggested. §3 states the single diagnosis. §4 is
the phased plan. §5–§10 are the supporting apparatus (tests, compute, risk, sequencing, decisions).

---

## 1. Corrections to the incoming review

The review is directionally right about the research claim and wrong about the project's state.
Both matter, because the second changes what work is actually needed.

| Review says | Reality | Consequence for the plan |
|---|---|---|
| "You have one trained policy and a rubric that self-scores it" | **No model has ever been trained.** `assets/reward_curves/` is empty (verified). `docs/report-midway.md` §5.1 states this plainly. No checkpoint, no wandb run, no adapter exists. | There is nothing to re-measure. Every number in the pitch docs is a target written before any run. This is *better* news: no results need retracting, they need producing. |
| "Fix the action-space inconsistency — pick 48" | The inconsistency is real, but 48 is the wrong choice. See F3 and Phase 1.1. | Choose 96, not 48. Reasoning below. |
| "Prompt template duplication is a silent-drift risk" | Not a risk. It has already happened, and there are **three** builders, not two. | Phase 2 is a deletion job, not a precaution. |
| "The dial being unlabeled makes no difference to the MDP formalism" | Correct, and understated. **The prompt names the raga in plain English and announces the rule change.** See F2. | The dial argument is moot. The leak is in the text channel, which is the only channel the model reads. |
| "Reward weights need justification or a sensitivity analysis" | Correct, and there is a specific inversion: an unbounded drought penalty grows larger than the forbidden-note penalty. See F11. | Phase 5 is not optional polish. One reward term currently contradicts the environment's own headline metric. |
| "Report the failure mode honestly, e.g. the agent thrashes" | Right instinct. There is a specific degenerate policy to test for: the four swaras legal in both ragas. See F13. | Add it as a named baseline, not an anecdote. |

The review's central recommendation — **make the observation arm the independent variable and run
Arm A against Arm B** — is correct and is adopted as Phase 4. One addition the review could not have
known: the pitch document already asserts that result as though it were measured.
`docs/Jugalbandi.md:161` reads *"We also ablate by withholding the dial value entirely from the obs
-- that agent never adapts below 18 steps."* That experiment has never been run. This plan converts
a fabricated sentence into a real result.

---

## 2. Verified ground truth

Ordered by severity. Each finding lists the evidence that produced it.

### F1 — CRITICAL. Training is a contextual bandit with a mismatched context, and contains no drift event.

`training/train_grpo.py:69-91`, notebook Cell 14. `env_reward()` calls `POST /reset {"dial": 0.0}`
before scoring **each** completion, then applies one action and reads the reward.

The prompt the model saw was built in `build_dataset()` from some random rollout state. The reward
is computed in a **freshly reset episode** — Yaman, note history `[12]`, tala position 0. The prompt
and the reward therefore come from different environment states. There is no trajectory and no
credit assignment. `max_completion_length=4` confirms the design intent: one action per sample.

Verified:

```
calls /set_dial in training: False
dial values sent to /reset:  {'0.0'}
```

No drift event ever occurs during training, in either the script or the notebook. The rubric metric
`drift_adaptation_speed` (`openenv.yaml`, weight 0.25) has **zero** training signal.

It is worse than absent signal. Notebook Cell 16 resets with `random.uniform(0, 1)` at episode end,
so a prompt can read `Active raga: bhairav` while `env_reward` scores that completion in **Yaman**.
The gradient actively teaches the model that the raga label does not predict reward.

### F2 — CRITICAL. The prompt names the raga and announces the rule change in English.

`openenv_server/server.py:96` and notebook Cell 12:

```python
f"Active raga: {raga} (dial={dial}{'  GRACE PERIOD - rules just changed' if in_grace else ''})\n"
```

`training/train_grpo.py:57`: `f"Active raga: {raga}\n"`.

The observation vector is never given to the model. It is rendered into a prompt string, and that
string states the active raga by name and flags the grace period. `summary.md` §4 records the design
rule *"do not add an explicit 'raga changed' observation dimension"* — that rule is already
violated, in the only channel the model can read.

This is the finding that kills the headline claim, and it is independent of the dial argument.
Removing `obs[15]` while leaving `Active raga: bhairav` in the prompt changes nothing at all.

### F3 — HIGH. The action-space contract is inconsistent three ways, and the deployed range makes half the durations unreachable.

| Source | Claims |
|---|---|
| `raaga_env/env.py:38` | `Discrete(96)` |
| `raaga_env/env.py:117` | `_decode = (action % 24, action // 24)` |
| `raaga_env/jugalbandi_env.py:16-36` docstring | 48, and overrides neither `action_space` nor `_decode` |
| `openenv.yaml` | `action_space.n: 48` |
| `openenv_server/server.py:46` | `Field(..., ge=0, le=47)` |

Verified:

```
durations reachable via server (0..47): [0, 1]
durations in full space (0..95):        [0, 1, 2, 3]
actions with duration>=2 in 0..47:      []
```

Consequence: `reward.py:85` (`vadi_held`, requires `duration >= 2`) is **dead code in every
server-mediated rollout** — all training, all UI play, all evaluation. A declared reward layer can
never fire.

### F4 — HIGH. The prompt's stated action encoding does not match the environment's decoder.

Prompt (server.py:70, notebook Cell 12, train_grpo.py:48): *"Action encodes: note_semitone (0-11) +
duration_index (0-3) * 12."*
Environment (`env.py:117`): `action % 24, action // 24`.

Verified:

```
action 16: prompt says (note=4, dur=1)  |  env does (note=16, dur=0)
action 28: prompt says (note=4, dur=2)  |  env does (note=4,  dur=1)
action 40: prompt says (note=4, dur=3)  |  env does (note=16, dur=1)
```

Any instruction-following the model does is systematically mis-decoded. It cannot express a musical
intention even in principle. The `INFER_SYSTEM` string and the training `SYSTEM` string are both
wrong, identically.

### F5 — HIGH. `JugalbandiEnv` emits observations outside its own declared `Box(0, 1)`.

`jugalbandi_env.py:154` divides absolute pitch (range 0-23) by `11.0`. Lines 168-169 compute
vadi/samvadi distance on absolute pitch instead of the circular swara distance.

Verified:

```
reset obs: [1.091 0.667 0. ... 1. 0.727 0.091 0. 0.]
observation_space.contains(obs) at reset: False
out-of-bounds observation steps over one random episode: 58/64
```

The base class does this correctly (`env.py:139` uses `/23.0`; `env.py:146-147` uses circular
distance). `JugalbandiEnv` is a regression from its own parent.

The test suite does not catch it. `test_obs_always_in_bounds` (`test_env.py:41`) covers only
`RaagaEnv`; `test_jugalbandi_obs_shape` (`test_env.py:67`) asserts shape and nothing else. This is
why "20/20 passing" is not evidence of correctness.

Downstream, `_make_infer_prompt` decodes note names with `int(obs[i*2] * 23)`, which assumes the
base env's `/23.0`. Verified against the actual `/11.0` values:

```
played Ga  (16) -> prompt renders "Ni" (23)   WRONG
played Ma# (18) -> prompt renders "Ni" (23)   WRONG
```

Every madhya-register note above Ga saturates to "Ni" in the prompt the model reads.

### F6 — HIGH. The notebook will not construct a trainer, and the training stack is unpinned.

Cell 18 sets `per_device_train_batch_size=2` with `num_generations=4`. TRL's `GRPOTrainer` requires
the global train batch size (`num_processes * per_device_train_batch_size`) to be divisible by
`num_generations`; `2 % 4 != 0`. Also `tokenizer=` is deprecated in favour of `processing_class=`
in TRL 0.12 and later.

Not empirically confirmed here — TRL is not installed locally, because `requirements.txt` leaves
every training dependency commented out with **no pinned versions at all**. That is itself the
finding: the training stack is a version lottery on first run. Phase 0.1 fixes it, and the exact
TRL constraint gets verified against the pinned version at that point.

### F7 — MEDIUM. Duplicated logic in exactly the places the code warns against duplicating.

- `drift.py:59-64` re-implements pakad matching inline instead of calling `ragas.match_pakad`,
  despite `ragas.py:120-133` carrying an explicit comment that both call sites must use the shared
  helper "so the two code paths can't silently drift out of sync again". A third path was then added
  that does exactly that.
- `jugalbandi_env.py:76` assigns `grace = self.drift.apply_grace` and never uses it.
  `DriftManager.apply_grace` and `DriftManager.obs_dims` are both dead code.
- Three prompt builders exist: `server.py:_make_infer_prompt`, notebook Cell 12 `make_prompt`, and
  `train_grpo.py:52 make_prompt`. The first two agree. The third does not — it dumps raw floats
  (`Last 4 notes (note/11, dur/3): [1.09, 0.67, ...]`) instead of note names. A model trained via the
  script and served by the server sees two different input distributions.

### F8 — MEDIUM. `/reset` destroys drift state, so no episode can begin mid-drift.

`server.py:148-149` calls `env.set_dial(req.dial)` and then `env.reset()`. `jugalbandi_env.py:62`
constructs a fresh `DriftManager`, wiping `grace_steps_remaining` and `adaptation_window_remaining`.
Drift can therefore only be induced by a separate mid-episode `POST /set_dial`, which nothing in the
training pipeline calls (see F1).

### F9 — MEDIUM. One global mutable env instance behind async endpoints.

`server.py:40`. Every request mutates shared state. Concurrent clients, or any multi-worker
deployment, silently interleave steps into one another's episodes. Acceptable for a single-user
demo; disqualifying for generating paper results.

### F10 — MEDIUM. Reward-breakdown logging is unsound, and the declared reward range is wrong.

- `jugalbandi_env.py:106` writes `breakdown["call_requested"] = True` into the same dict that holds
  float reward components. Any code that sums the breakdown to produce a per-component reward plot —
  which is exactly the planned figure — adds `1.0` per call event and attributes it to a reward
  component that does not exist. Confirmed: a decomposition script picked up a spurious `+8.00`.
- `reward.py:127` sets `breakdown["total"]` before the drift bonus is added at
  `jugalbandi_env.py:94`, so `breakdown["total"] != reward` on exactly the steps the paper cares about.
- `openenv.yaml` declares `reward: min -2.0, max 3.0`. Both are violated. The adaptation bonus stacks
  on top of the per-step reward (`0.2 + 0.4 + 0.15 + 1.2 + 0.8 + 0.2 + 0.5 + 0.3 + 3.0 = 6.75`), and
  the vadi-drought penalty alone reaches `-2.80` per step by step 64.

### F11 — MEDIUM/HIGH. Unbounded drought penalties dominate the reward and invert its priorities.

`reward.py:67-70` and `reward.py:99-102` apply penalties linear in drought length with no floor.

Verified over one 64-step episode of a legal-but-unmusical policy:

| Component | Episode total |
|---|---|
| `pakad_drought_penalty` | -39.78 |
| `vadi_drought_penalty` | -16.25 |
| `large_leap_penalty` | -12.00 |
| all positive components combined | +27.50 |

Per-step magnitudes late in an episode:

| Steps of drought | `pakad_drought` per step | `vadi_drought` per step |
|---|---|---|
| 20 | -0.24 | -0.60 |
| 40 | -0.84 | -1.60 |
| 64 | -1.56 | -2.80 |

The forbidden-note penalty is a flat **-2.00**. So by step 40 the environment penalises *not having
played your vadi recently* more harshly than *playing a note forbidden in the raga*. The reward
function's dominant gradient points away from the metric the project reports as its primary rubric
item (`valid_raga_adherence`, weight 0.35). No amount of arm-splitting fixes a reward that is
optimising for something else.

### F12 — LOW/MEDIUM. Claim hygiene: measured-sounding numbers that were never measured.

`docs/Jugalbandi.md` lines 107, 123, 125, 161 and 223 assert: reward climbing "-8 to +4";
forbidden-note rate "25% to under 5%"; "3.4 steps" average adaptation with "72% success rate" and
"standard deviation under 1.5" across "100 evaluation episodes"; "response coherence 89%"; and the
dial-ablation result quoted in §1. None were measured. `summary.md` §4 states that the repetition and
drought penalties "close a known reward-hacking exploit ... observed in early training runs" — there
are no training runs. These sentences are the project's largest reputational liability and cost
nothing to fix.

### F13 — NEW FINDING. A raga-agnostic safe set exists and is a live degenerate-policy risk.

Verified:

```
Yaman valid   : [0, 2, 4, 6, 7, 9, 11]   forbidden: [5]
Bhairav valid : [0, 1, 4, 5, 7, 8, 11]   forbidden: [2, 9]
RAGA-AGNOSTIC SAFE SET (penalised in neither): [0, 4, 7, 11]   = Sa, Ga, Pa, Ni
```

A policy confined to `{Sa, Ga, Pa, Ni}` never triggers a forbidden-note penalty in **either** raga,
so it scores identically before and after every drift event. It would produce a flat,
perfect-looking `valid_raga_adherence` curve across drift while demonstrating exactly zero
adaptation.

Two things currently limit it, and both are accidents rather than design: Yaman's `aaroha_swaras`
excludes Pa, so ascending Pa costs -1.0; and the drought penalties (F11) punish the resulting lack
of pakads. Note also that Yaman's vadi (Ga) and samvadi (Ni) are both *inside* the safe set while
Bhairav's (komal Dha, komal Re) are both outside — so the exploit is actively rewarded in Yaman and
merely unpunished in Bhairav.

This must be an explicitly reported baseline (`safe-set-cycle`) and a monitored failure mode, not a
footnote. If a trained policy converges here, every headline metric except pakad rate looks good.

---

## 3. Diagnosis

One sentence: **the project's research claim is contradicted in three independent places, and no
experiment has been run that could have detected it.**

- The claim requires the agent to infer the active rule set. The prompt states it in English (F2).
- The claim requires the agent to experience rule changes. Training contains none (F1).
- The claim requires reward to track musical correctness. It is dominated by an unbounded drought
  term (F11).

The review framed this as "a defensible experiment is available if you split the arms". That is
correct but not sufficient. Arm-splitting on top of F1, F2 and F11 would produce three arms that all
train on the same broken bandit and all read the raga name from their own prompt. The arms would be
indistinguishable, and the null result would be an artifact of the harness rather than a finding.

**Fix the harness first, then run the experiment.** Phases 0–3 are not cleanup. They are the
precondition for Phase 4 meaning anything.

There is also a claim to be made precise, which the review touched but did not separate:

- **Claim A (in-weights conditioning).** Over training, the policy learns the mapping
  dial-value to rule set. Requires the dial in the observation. This is a Contextual MDP
  (Hallak et al., 2015) and is, as the review says, ordinary conditioned policy learning.
- **Claim B (in-context regime inference).** Within a single episode, the agent detects that the
  rule set changed from feedback alone and switches behaviour. This is a Hidden-Parameter MDP or
  POMDP result, and it is the only version that is novel.

**Claim B is currently impossible in this codebase, for a reason nobody has noticed: the reward is
never shown to the model.** The prompt contains note history, tala position, dial, tension and
drought — and no reward, no penalty signal, no indication that the last action was punished.
"Infer the rule change from reward correlation alone" cannot happen in-context when reward is not in
the context. Phase 2 must add a feedback channel, or the paper must commit to Claim A and drop the
detection language entirely.

---

## 4. Phase plan

Phases 0–3 are sequential and blocking. Phase 4 depends on all of them. Phases 5–7 run in parallel
once Phase 4 has started. Phase 8 runs continuously.

---

### Phase 0 — Reproducibility harness (blocking, about 1 day)

Nothing can be measured until measurement is deterministic. Do this before touching any logic.

**0.1 Pin the training stack.** Create `requirements-train.txt` with exact versions for `torch`,
`transformers`, `trl`, `peft`, `accelerate`, `datasets`, `bitsandbytes`, `unsloth` and `wandb`. The
current file comments them all out unversioned. Record the resolved versions in every result file.
Verify the TRL batch-size constraint from F6 against the pinned version and fix the config.

**0.2 Write the failing contract tests first (RED).** New file `raaga_env/tests/test_contracts.py`
asserting the properties currently violated. These must fail on the current tree — that is the
acceptance criterion for this step.

- `JugalbandiEnv` observations lie inside `observation_space` for 10,000 random steps.
- `action_space.n` agrees with `openenv.yaml`, with `StepRequest`'s bound, and with the prompt text.
- `_decode` round-trips against the encoding described in the system prompt.
- Every reward returned lies inside the range declared in `openenv.yaml`.
- Every value in `reward_breakdown` is a float, and `breakdown["total"]` equals the returned reward.
- `render_prompt(arm=HIDDEN)` contains none of: `yaman`, `bhairav`, `grace`, `dial`, `switch`.

**0.3 Build the in-process evaluation harness.** New `eval/rollout.py`:

```python
def rollout(policy, *, seed: int, drift_schedule: DriftSchedule, arm: Arm) -> Trajectory
def metrics(trajectories: list[Trajectory]) -> MetricSet
```

No HTTP. Every downstream number — baselines, arms, ablations, listening-study stimuli — comes
through this one function, so a change to a metric definition cannot silently apply to some results
and not others.

**0.4 Add environment state serialisation.** `JugalbandiEnv.get_state() -> dict` and
`set_state(dict)`, covering note and duration history, tala position, droughts, counters, and the
full `DriftManager`. Needed for correct GRPO group scoring (Phase 3.2) and for deterministic
re-scoring of saved trajectories.

**0.5 Fingerprint every result.** Each result file records the git SHA, a hash of
`requirements-train.txt`, a hash of the `RAGAS` dict, and a hash of the reward constants. A figure
whose fingerprint does not match the current tree is invalid and gets regenerated, not trusted.

**Acceptance:** the new contract tests exist and fail for the documented reasons; `rollout()` runs a
scripted policy deterministically and returns identical metrics across two invocations at the same
seed.

---

### Phase 1 — Contract correctness (blocking, about 1 day)

**1.1 Resolve the action space to 96, not 48.**

The review recommends 48. Disagree, for a reason specific to the domain modelling this project has
already done. The pakads in `ragas.py` are written in absolute pitch and several span both
registers: Yaman `[11, 14, 16]` (mandra Ni, Re, Ga) and Bhairav `[13, 12, 11, 12]` (re, Sa, mandra
Ni, Sa). The source annotates these as the most distinctive phrases, and
`docs/CLASSICAL_MUSIC_AUDIT.md` cites them as the reason the action space was widened to two octaves
in the first place. Collapsing to one octave makes them unreachable and reverts a documented audit
fix. Collapsing to 48 while keeping the `% 24` decoder discards two of four durations (F3).

So: make everything 96. Change `openenv.yaml` `action_space.n` to 96 and rewrite its description;
change `StepRequest` to `le=95`; declare `action_space` explicitly on `JugalbandiEnv` even though it
inherits, so the contract is local and testable; update all prompt strings; update `build_dataset`'s
`random.randint(0, 47)`.

*Caveat requiring a decision:* if `openenv.yaml` is frozen by a submission already filed against that
manifest, this flips. See §9.

**1.2 Replace silent modulo clamping with explicit rejection.** `server.py:244`, `train_grpo.py:81`
and notebook Cell 14 all parse with `int(...) % 48`. A model that emits `137` currently becomes a
legal action. That destroys any measurement of instruction-following. Parse failures and
out-of-range outputs must be recorded and penalised as such, and `action_validity_rate` becomes a
reported metric (§4.3).

**1.3 Fix the observation normalisation.** In `jugalbandi_env._get_obs`: divide pitch by `23.0`, and
replace `obs[18]`/`obs[19]` with the circular swara distance already implemented correctly at
`env.py:146-147`. Then remove the duplication — have `JugalbandiEnv._get_obs` call
`super()._get_obs()` for the shared dimensions rather than recomputing them, so the two cannot
diverge again.

**1.4 Fix reward-range and breakdown integrity.** Move `call_requested` out of `reward_breakdown`
and into `info`. Compute `breakdown["total"]` after the drift bonus is applied. Re-derive the true
reward bounds analytically and update `openenv.yaml` — or bound the reward, which is Phase 5's job.

**Acceptance:** every Phase 0 contract test passes; the existing 20 tests still pass; a scripted
policy that plays each cross-register pakad receives the pakad bonus.

---

### Phase 2 — One prompt interface, arms enforced at the text layer (blocking, 1–2 days)

**2.1 Create `raaga_env/prompting.py`** as the single source of truth:

```python
class Arm(Enum): ORACLE = "oracle"; DIAL = "dial"; HIDDEN = "hidden"

def render_prompt(obs, *, arm: Arm, tala_pos: int, feedback: StepFeedback | None) -> str
def parse_action(text: str) -> int | None      # None means invalid; never clamps
```

Delete `_make_infer_prompt` from `server.py`, `make_prompt` from `train_grpo.py`, and Cell 12's copy
from the notebook. All call sites import this module, and the notebook imports it from the repo
rather than redefining it. That import is the specific mechanism preventing F7 from recurring.

**2.2 Define the arms as data, and enforce them by test.**

| Field in prompt | ORACLE | DIAL | HIDDEN |
|---|---|---|---|
| Raga named in English | yes | **no** | **no** |
| `dial` float | yes | yes | **no** |
| Grace-period flag | yes | **no** | **no** |
| Steps since switch | yes | **no** | **no** |
| Note history, tala, call phrase, droughts | yes | yes | yes |
| Last-step feedback (see 2.3) | yes | yes | yes |

`steps_since_switch` must be removed from DIAL as well as HIDDEN — it leaks the switch event on a
delay, which is the same information the dial carries. The review flagged this and it is correct.

The enforcement test asserts the rendered string for each arm contains no forbidden substring. It is
three lines long, and it is the test that would have caught F2.

**2.3 Add a feedback channel — a design decision, not a bug fix.** Per §3, in-context inference is
impossible while reward is invisible. Add to the prompt, for all arms:

```
Last action: <note name>  ->  outcome: rewarded (+0.6) | penalised (-2.0)
```

Coarse and rule-agnostic. It must not say *why* — never "forbidden in bhairav". This gives Claim B
an information channel without leaking the rule set, and it is the honest reading of "infer from
reward correlation".

If Claim A is chosen instead (§9), skip 2.3 and delete the detection language from the paper. Do not
ship both.

**2.4 Fix the action-encoding text (F4)** in the one remaining prompt builder, and add a test
asserting the documented encoding matches `_decode` for all 96 actions.

**Acceptance:** one prompt builder in the repo, verified by grep; arm-leakage tests pass; a
round-trip test confirms that actions chosen from a rendered prompt decode to the notes that prompt
described.

---

### Phase 3 — Make training a real MDP containing drift (blocking, 3–4 days)

**3.1 Move training off HTTP.** Import `JugalbandiEnv` in-process. The HTTP server stays for the UI
and for OpenEnv discoverability, but it is not a training surface: a thousand round trips per dataset
build against one global mutable env (F9) is both slow and a correctness hazard. This also removes
the F8 reset-order problem from the training path entirely.

**3.2 Replace the bandit with episode rollouts.** Recommended design: each GRPO sample is a **full
episode**. The prompt is the episode's opening context plus the drift-schedule seed, the completion
is a sequence of actions, and the reward is the episode return. This gives credit assignment across
the switch, and it makes `drift_adaptation_speed` computable per sample — which is the metric the
paper needs.

The alternative is to keep per-step GRPO but restore environment state before scoring each member of
a group (using Phase 0.4) and use a Monte-Carlo return from a policy rollout. That is more
principled and much higher variance at 0.5B. Start with episode rollouts; if the return signal is
too coarse to learn from, fall back, and report which was used and why.

**3.3 Every training episode contains a drift event.** Sample `drift_at_step` uniformly from
`[16, 48]` and flip the dial to the opposite side. Hold out a control condition of episodes with no
switch (recommend 20%) so the policy does not simply learn "a switch always happens near the middle".

**3.4 Fix the notebook config** per F6, and regenerate the notebook from the script rather than
maintaining both by hand. The notebook should be a thin driver that imports the repo.

**Acceptance:** a 20-step training run completes without crashing; logged episodes show drift events
occurring; a scripted oracle policy run through the same loop achieves the return the reward
function predicts analytically.

---

### Phase 4 — The experiment (this is the paper; 1–2 weeks including compute)

Pre-register before running. Write `docs/EXPERIMENT_PLAN.md` with the metric definitions, statistical
tests and exclusion rules, and commit it with a timestamp **before** the first production run. It
costs an hour and it is the strongest available answer to "did you p-hack this".

**4.1 Policies to evaluate.**

| ID | Description | Role |
|---|---|---|
| `random-uniform` | Uniform over the 96 actions | Absolute floor |
| `random-valid` | Uniform over notes valid in the *currently active* raga | Honest floor. Without it, a policy that learned only Yaman's scale looks impressive |
| `safe-set-cycle` | Cycles `{Sa, Ga, Pa, Ni}` (F13) | Names the degenerate strategy so it is detected, not discovered by a reviewer |
| `scripted-oracle` | Plays the shortest new-raga pakad immediately on switch | Ceiling. Calibrates what `drift_adaptation_speed` can physically be |
| `base-zeroshot` | Qwen2.5-0.5B-Instruct, untrained, same prompt | Shows what GRPO actually bought |
| `grpo-ORACLE` | Trained, raga named in prompt | Upper bound / oracle arm |
| `grpo-DIAL` | Trained, dial only | What the project claims to do today |
| `grpo-HIDDEN` | Trained, no context signal | The research claim |

The three scripted policies cost minutes to run and make the trained results interpretable. Skipping
them is how a mediocre result gets mistaken for a good one.

**4.2 Seeds and evaluation protocol.** Five training seeds per trained arm. Evaluation on a **fixed,
shared set of 200 episodes** with pre-generated drift timings, identical across every policy, so all
comparisons are paired. Pairing is worth roughly a factor of three in statistical power here and
costs nothing.

**4.3 Metrics, defined precisely, because they currently are not.**

- `valid_raga_adherence` — fraction of steps whose swara is in the active raga's valid set and not
  forbidden. **Reported split three ways:** pre-switch, inside the grace window, post-grace. A single
  average hides the entire effect.
- `drift_adaptation_speed` — steps from switch to first new-raga pakad. **Right-censored** when no
  pakad occurs before the episode ends. Report a Kaplan-Meier survival curve, or a median with the
  censoring rate stated. Dropping non-adapting episodes is the easiest way to manufacture "3.4 steps",
  and it is what the current pitch number would have to be doing.
- `adaptation_success_rate@K` for K in {5, 10, 20} — report the curve, not a single number.
- `post_switch_violation_decay` — forbidden-note rate as a function of steps since switch, averaged
  across episodes. This is the real adaptation figure and is more informative than any scalar.
- `pakad_rate` and `jugalbandi_coherence` — as in `openenv.yaml`, with formulas written down for the
  first time.
- `action_validity_rate` — fraction of completions parsing to a legal action with no clamping. New,
  enabled by 1.2.
- `safe_set_occupancy` — fraction of steps inside `{0, 4, 7, 11}` (F13). Flags the degenerate policy.

**4.4 Statistics.** Paired bootstrap over the shared episode set; report effect sizes with confidence
intervals. With five seeds, use Wilcoxon signed-rank on paired episodes. Do not run t-tests on five
points, and do not report a mean of five seeds as though it were a distribution.

**4.5 Expected result, and how to report it.** Likely ordering:
`scripted-oracle > grpo-ORACLE > grpo-DIAL >= grpo-HIDDEN >= random-valid`.

`grpo-HIDDEN` may fail to beat `random-valid` at 0.5B. **That is a publishable negative result** —
but only because Phases 0–3 make the harness trustworthy. The contribution then becomes: a clean
environment separating conditioned policy learning from regime inference, plus evidence that the
latter does not emerge at this scale under this reward density. Characterise the failure mode
concretely using `safe_set_occupancy` and `post_switch_violation_decay` rather than describing it
qualitatively.

---

### Phase 5 — Reward ablations and sensitivity (parallel with Phase 4, about 3 days)

**5.1 Fix F11 first — a correction, not an ablation.** The unbounded drought penalties must be
floored (for example, clamped at -0.5 per step) or made saturating. Leaving a penalty that grows past
the forbidden-note penalty is not a tuning preference, it is a reward function optimising for the
wrong thing. Re-derive the reward bounds and update `openenv.yaml`.

**5.2 Leave-one-out ablations** from the corrected full model, three seeds each, on the best arm:
adaptation bonus (3.0 / 1.0 / 0.0), grace period (3 steps / 0), repetition penalty (on/off), drought
penalties (on/off, post-fix). Eight runs. Justify leave-one-out over a full factorial in the methods
section — it is a legitimate choice, it just has to be stated.

**5.3 Sensitivity sweep.** Perturb each continuous weight by plus or minus 50% and confirm the
*ordering* of arms is preserved. That is the defensible answer to "you tuned these by hand".
Robustness of ordering is a weaker and far more honest claim than robustness of magnitudes.

**5.4 Delete the false provenance.** `summary.md` §4 claims the repetition and drought penalties were
added in response to exploits "observed in early training runs". No such runs exist. Either re-label
them as anticipated failure modes, or better, actually observe them now — Phase 4 gives you the
ablation runs that would demonstrate it, at which point the sentence becomes true and citable.

---

### Phase 6 — Listening study (start recruiting during Phase 3; 2–3 weeks calendar)

This is the only evidence for the paper's stated advantage over a JSON-schema environment. Nothing
currently tests it.

**6.1 Stimuli.** Thirty clips of roughly ten seconds, rendered from Phase 4 trajectories via the
existing Tone.js path or offline MIDI. Conditions: `random-valid`, `base-zeroshot`, the best trained
arm, and a reference human phrase. Include matched pre-switch and post-switch segments from the same
episodes.

**6.2 Task design.** Two-alternative forced choice — *"which of these two continues in the same raga
as the opening?"* — plus a five-point raga-correctness rating. 2AFC is far more reliable than
absolute ratings with untrained listeners and gives a clean 50% chance baseline.

**6.3 Participants, two tiers.** (a) 15–20 non-musicians, testing the claim that violations are
audible to a lay audience, which is the claim `summary.md` §3 actually makes. (b) 3–5 trained
Hindustani musicians, for the correctness claim. Indore has a real classical scene and IPS Academy
has faculty routes; a gharana teacher or a music college is a realistic ask, and the same musicians
can validate any Phase 7 raga additions.

**6.4 Analysis.** Inter-rater agreement (Krippendorff's alpha) first — if agreement sits at chance,
the stimuli are the problem, not the policy. Then per-condition accuracy against the 50% baseline.

**6.5 Consent.** A one-paragraph consent form, no personal data beyond self-reported musical
training. Any venue will ask for it.

**Sequencing note:** this is mostly calendar time, not work time. Start recruiting in Phase 3.

---

### Phase 7 — Beyond two ragas (only after Phase 4 produces a result, about 1 week)

Do not simply add "a handful more" — that turns a case study into a slightly larger case study.
Add ragas that vary along a **controlled axis**, so drift magnitude becomes a second independent
variable:

- **Near pairs** differing by one swara: Yaman and Yaman-Kalyan; Bhairav and Ahir Bhairav.
- **Far pairs** differing by several: the existing Yaman and Bhairav.

The paper can then ask whether adaptation speed scales with the size of the rule change, which is a
genuine second finding rather than more instances of the first.

Budget roughly three hours per raga to meet the sourcing standard already set in
`docs/CLASSICAL_MUSIC_AUDIT.md` (Bhatkhande KPM plus cross-reference), and have the Phase 6 musicians
check the additions.

---

### Phase 8 — Claim hygiene and paper assembly (continuous)

**8.1 Purge or label every unmeasured number.** Exact locations: `docs/Jugalbandi.md` lines 107, 123,
125, 161, 223; `summary.md` §4 (training-run provenance); `docs/Train.md` (past-tense results voice
throughout). Either delete them or mark them inline as pre-registration targets. Do this **now**, not
at submission. A target sitting in a repo for months reads as a result.

**8.2 Rewrite the contribution claim.** Proposed:

> We present Jugalbandi, an RL environment for schema adherence under mid-episode constraint drift,
> instantiated on Hindustani raga grammar so that violations are audible to human listeners. We use
> it to separate two capabilities usually conflated in the drift literature: conditioned policy
> learning over an observed context variable, and regime inference from reward feedback alone. We
> report results for a 0.5B policy trained with GRPO across three observation arms, together with a
> listening study establishing that the environment's rule violations are perceptible.

**8.3 Related work — currently cited: none.** Minimum set:

- Contextual MDPs — Hallak, Di Castro and Mannor (2015). The formalism the review invoked; the paper
  must engage with it directly.
- Hidden-Parameter MDPs — Doshi-Velez and Konidaris (2016).
- Meta-RL and in-context adaptation — Duan et al. (2016), RL^2; Wang et al. (2016).
- In-context RL from trajectories — Laskin et al. (2022), Algorithm Distillation; Lee et al. (2023),
  Decision-Pretrained Transformer.
- Non-stationary MDPs and change-point detection — Padakandla et al.; Da Silva et al.
- Reward hacking — Skalse et al. (2022), for the ablation framing and F13.
- GRPO — Shao et al. (2024), DeepSeekMath.
- Music — Bhatkhande KPM (already used); computational raga work such as Chordia and Rae on raga
  recognition, and Ross and Rao on motif spotting.

**8.4 Venue realism.** This is a workshop paper, not a main-track submission. Target an ICML or
NeurIPS workshop on agents or evaluation, ISMIR for the music-and-ML angle, or an ICLR
Tiny-Papers-style track. Scoping to a workshop is precisely what makes a two-raga case study
acceptable, and saying so up front is better than having a reviewer say it.

---

## 5. Test plan

Included because new work ships with tests, and because F5 shows the current suite passes while the
environment is broken.

**Unit.** The contract assertions from Phase 0.2. Decode round-trip across all 96 actions.
Prompt-leakage per arm. `parse_action` rejects out-of-range and non-numeric input without clamping.
`DriftManager` state transitions: a switch sets grace and the adaptation window; grace expires after
exactly three steps; the adaptation bonus fires at most once per switch. Each reward layer in
isolation, including the currently dead `vadi_held` branch once F3 is fixed.

**Property-based.** Over 10,000 random (state, action) pairs: the observation lies inside
`observation_space`; the reward lies inside the declared bounds; the sum of float values in the
breakdown equals the returned reward.

**Regression / golden trajectory.** A fixed seed plus a fixed action sequence produces a stored hash
of observations, rewards and breakdown keys. Any reward change breaks it loudly and deliberately.
This is the guard that makes Phase 5's ablations trustworthy.

**Integration.** Server endpoints exercised against the in-process env. `get_state`/`set_state`
round-trip produces byte-identical subsequent trajectories.

**Harness self-test — do not skip this one.** Run `scripted-oracle` through `eval/rollout.py` and
assert `drift_adaptation_speed` equals the length of the shortest new-raga pakad. If the harness
cannot recover the known-correct answer from a known-correct policy, no number it produces about a
trained policy means anything.

---

## 6. Compute budget

Order of magnitude, to be refined once Phase 3 measures step time.

| Item | Runs | Estimated GPU-hours (T4) |
|---|---|---|
| Phase 4 trained arms (3 arms x 5 seeds x ~300 steps) | 15 | 25–45 |
| Phase 5 ablations (8 configs x 3 seeds) | 24 | 30–60 |
| Evaluation rollouts (all policies x 200 episodes) | — | 3–6 |

The Colab free tier will not carry this; session limits alone make fifteen sequential runs
impractical. The GCP $300 credit referenced in `docs/report-midway.md` §9.2 is the realistic route,
with Kaggle's 30 weekly GPU-hours as a supplement. Decide before Phase 3 finishes, because it
determines whether Phase 5 runs at three seeds or two.

---

## 7. Risk register

| Risk | Likelihood | Response |
|---|---|---|
| `grpo-HIDDEN` fails to learn anything | High | Planned for. It is the finding, provided the harness is trustworthy (Phases 0–3) and `random-valid` and `scripted-oracle` bracket the result |
| All three arms are indistinguishable | Medium | Indicates the reward signal is too coarse. Diagnose with `scripted-oracle`: if the oracle's return is not clearly separated from `random-valid`, the reward is the problem, not the policy |
| Trained policy converges to the safe set (F13) | Medium | `safe_set_occupancy` detects it. Response is a reward correction in Phase 5, reported as an ablation |
| Episode-level GRPO too high-variance at 0.5B | Medium | Fall back to state-restoring per-step GRPO (Phase 3.2 alternative); report which was used and why |
| Listening study under-recruits | Medium | Tier (a) is the load-bearing claim and is easy to fill. Tier (b) can be three musicians; report n honestly |
| Unpinned training stack breaks on the first cloud run | Was high, now mitigated | Phase 0.1 |
| Scope creep into Phase 7 before Phase 4 lands | High, and the most expensive failure mode | Phase 7 is explicitly gated. More ragas do not fix a broken harness |

---

## 8. Sequencing

```
Week 1   Phase 0 (harness, pins, failing tests)  ->  Phase 1 (contracts)
Week 2   Phase 2 (prompting, arms)               ->  Phase 3 (real MDP training)
         [start listening-study recruitment in parallel]
Week 3   Phase 4 pre-registration + scripted baselines + first trained arm
Week 4   Phase 4 remaining arms and seeds; Phase 5 ablations in parallel
Week 5   Phase 6 listening study runs; Phase 8 claim hygiene and related work
Week 6   Analysis, figures, write-up. Phase 7 only if Weeks 1-5 held.
```

The hard gate: **do not start Phase 4 until every Phase 0 contract test passes.** The entire reason
this plan exists is that conclusions were drawn on an unvalidated harness. Repeating that with more
arms produces more invalid numbers, faster.

---

## 9. Decisions needed before work starts

These change downstream work and cannot be defaulted.

1. **Action space: 96 or 48?** This plan recommends 96 (Phase 1.1), because cross-register pakads are
   otherwise unreachable and a documented audit fix gets reverted. Choosing 48 means deleting those
   pakads from `ragas.py` and redoing that section of `CLASSICAL_MUSIC_AUDIT.md`. If `openenv.yaml`
   is already frozen under a filed submission, that inverts the recommendation.

2. **Claim A or Claim B?** (§3.) Claim A (in-weights conditioning over an observed dial) is safe,
   modest and already achievable. Claim B (in-context regime inference from reward feedback) is the
   interesting one and requires the feedback channel in Phase 2.3. Choosing B commits you to
   defending a harder result; choosing A commits you to deleting the detection language from
   `Jugalbandi.md` and `summary.md`. Do not ship both.

3. **Compute source and budget ceiling.** (§6.) Determines seeds per arm, and whether Phase 5 runs at
   three seeds or two.

4. **Venue and deadline.** (§8.4.) Determines whether Phases 6 and 7 are in scope at all.

5. **Musician access for Phase 6.** If tier (b) is unreachable, the paper drops the expert-correctness
   claim and keeps only the lay-audibility claim. Better to know in Week 1 than Week 5.

---

## 10. Appendix — verification commands

Every finding in §2 came from one of these. All were run against `06656a5` with the project venv.

```bash
# F3, F4: action-space and prompt-encoding mismatch
venv/Scripts/python.exe -c "from raaga_env.jugalbandi_env import JugalbandiEnv as E; e=E(); \
print(sorted({e._decode(a)[1] for a in range(48)}), sorted({e._decode(a)[1] for a in range(96)})); \
print([(a, a%12, a//12, e._decode(a)) for a in (16,28,40)])"

# F5: observations outside the declared Box(0,1)
venv/Scripts/python.exe -c "from raaga_env.jugalbandi_env import JugalbandiEnv as E; e=E(); \
o,_=e.reset(); print(o.max(), e.observation_space.contains(o))"

# F1: no drift during training
grep -c "set_dial" training/train_grpo.py          # -> 0
grep -o '"dial": [0-9.]*' training/train_grpo.py   # -> only 0.0

# F2: raga named in the prompt
grep -n "Active raga" openenv_server/server.py training/train_grpo.py

# F13: raga-agnostic safe set
venv/Scripts/python.exe -c "from raaga_env.ragas import RAGAS as R; Y,B=R['yaman'],R['bhairav']; \
print(sorted((Y['valid_notes']&B['valid_notes'])-Y['forbidden_notes']-B['forbidden_notes']))"

# baseline: the current test suite passes while all of the above is true
venv/Scripts/python.exe -m pytest raaga_env/tests/ -q   # -> 20 passed
```

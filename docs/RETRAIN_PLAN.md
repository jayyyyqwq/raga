# ML Retrain — call-response fix (2026-10)

Status: **code done, tests passing locally. GPU retrain not run yet — needs Colab.**

Scope: Option A from [`imrpovedui.md`](imrpovedui.md) §7, pulled forward and done first, on its
own, before any UI work. The UI plan is unchanged and picks up again once this lands.

## What was wrong

The trained adapter (`jugalbandi-grpo-hidden-v1`, see [`FIRST_TRAINING_RUN.md`](FIRST_TRAINING_RUN.md))
is a solo raga improviser with drift adaptation. It was never trained to respond to a human call
phrase, because:

1. **The call never reached the prompt.** [`prompting.py`](../raaga_env/prompting.py)'s
   `_context_lines` rendered `Human call tension: 0.xx` — a single derived float — but never the
   call's actual notes. The model had no way to know what was played, in any arm.
2. **Training never submitted a call.** [`train_grpo.py`](../training/train_grpo.py)'s
   `build_dataset` never called `env.set_call(...)`. Every training episode had a call phrase of
   `[0, 0, 0, 0]` (`JugalbandiEnv`'s old default) for its entire length.
3. **That default was a bug, not a neutral value.** `[0, 0, 0, 0]` is truthy in Python, and
   [`reward.py`](../raaga_env/reward.py)'s call-response layer gates on `if call_phrase:`. So that
   layer fired on *every single training step*, scored against a call that was never actually
   played — most visibly, `direction_contrast` rewarded any ascending/flat move on effectively
   every step, since `call_phrase[-1] > call_phrase[0]` is always `False` for `[0,0,0,0]`.

Net effect: the call-response reward layer was active throughout the whole first training run, but
scoring against a phantom call — not a bug that silently did nothing, a bug that silently taught
something unintended.

## What changed

| File | Change |
|---|---|
| [`raaga_env/jugalbandi_env.py`](../raaga_env/jugalbandi_env.py) | `call_phrase` defaults to `[]` (falsy), not `[0,0,0,0]` (truthy) — "no call yet" is now distinguishable from a real call. Added `call_echoed: bool`, reset on every `set_call()`, persisted through `get_state()`/`set_state()` |
| [`raaga_env/reward.py`](../raaga_env/reward.py) | New `call_echo` term (+0.4, once per call): rewards landing on the swara the human's call ended on — the simplest, most legible form of "the AI answered you". Existing `tension_resolve`/`direction_contrast` unchanged, now correctly gated |
| [`raaga_env/prompting.py`](../raaga_env/prompting.py) | The call phrase's actual notes now render as text — `Partner's call phrase: Sa, Ga, Pa, Ni` or `Partner's call phrase: none yet` — in **every arm**, including HIDDEN. It's musical input, not a rule signal |
| [`training/train_grpo.py`](../training/train_grpo.py) | New `sample_call_phrase()`: a raga-valid 4-swara phrase. `build_dataset()` now calls `env.set_call(...)` on the env's own `CALL_EVERY` cadence (`info["call_requested"]`), so training data actually contains calls |
| [`training/train_grpo.ipynb`](../training/train_grpo.ipynb) Cell 8 | The post-training inference sanity check now also submits calls on the same cadence, and reports a call-echo count, so it exercises the fixed path instead of only a solo episode |
| Tests | `raaga_env/tests/{test_env,test_reward,test_prompting}.py`, `training/tests/test_train_grpo.py` — 22 new tests, all passing (134/134 total) |

## A documented trade-off, not an oversight

`sample_call_phrase()` draws from the active raga's valid notes, not uniformly over 0–11. That's
deliberate — a real human partner's phrase is musically coherent too, and training on raga-invalid
noise would teach the model to ignore the call as meaningless. The cost: the call's note choices
are now a soft, implicit signal about the active raga, reaching HIDDEN's prompt along with
everything else.

HIDDEN's actual guarantee — *no raga name or dial value is ever written in words* — is unchanged
and still enforced by `test_prompting.py`'s and `test_train_grpo.py`'s leak tests (both extended to
check this specifically with a live call present). What's no longer true is that *zero* information
about the raga reaches HIDDEN's prompt. If `FIRST_TRAINING_RUN.md`-style write-ups get a sequel for
this run, say this plainly rather than letting "HIDDEN" imply more than it now does.

## Still scoped out (left for the UI phase)

- Call length stays 4 notes (`imrpovedui.md`'s 8-note lines are a UI-phase decision)
- No echo-arc/motif-matching reward term beyond the single `call_echo` bonus
- No change to `openenv_server/server.py` — not needed: `/infer` already passes the live `obs`
  (which includes the call dims) through to `render_prompt`, so the fix applies automatically once
  a new adapter is loaded

## Running the retrain

Same process as the first run (see [`FIRST_TRAINING_RUN.md`](FIRST_TRAINING_RUN.md) and the
README's Colab section) — no new steps, just re-run `training/train_grpo.ipynb` top to bottom
against this code. Cannot be run here: `train_grpo.py`'s `main()` needs `unsloth`/`trl`/`torch` on a
CUDA GPU (`requirements-train.txt`), which this machine doesn't have. What *was* verified locally,
without the GPU stack:

```bash
python -m venv venv && venv/Scripts/activate
pip install -r requirements.txt
pytest raaga_env/tests training/tests eval/tests openenv_server/tests -q
# 134 passed
```

This exercises everything up to the GPU boundary: `build_dataset()` actually produces rows with a
live call in the prompt, `make_step_reward_fn()` scores them correctly, and the leak tests confirm
HIDDEN still never writes a raga name. The one thing that cannot be checked without Colab is whether
GRPO actually learns to use the call — that's what the retrain itself, plus a re-run of
`eval/evaluate.py` comparing against `random-valid`, would show.

**Before re-running:** pull latest, confirm `pytest` is green locally first (catches anything Colab
would otherwise fail on 45 minutes into a T4 run), then run the notebook. Save the new adapter as a
new run name (e.g. `jugalbandi-grpo-hidden-v2`) rather than overwriting `v1` on Drive — `v1` stays
as the documented baseline in `FIRST_TRAINING_RUN.md`.

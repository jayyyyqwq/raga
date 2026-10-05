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

Cannot be run here: `train_grpo.py`'s `main()` needs `unsloth`/`trl`/`torch` on a CUDA GPU
(`requirements-train.txt`), which this machine doesn't have. What *was* verified locally, without
the GPU stack:

```bash
python -m venv venv && venv/Scripts/activate
pip install -r requirements.txt
pytest raaga_env/tests training/tests eval/tests openenv_server/tests -q
# 137 passed
```

This exercises everything up to the GPU boundary: `build_dataset()` actually produces rows with a
live call in the prompt, `make_step_reward_fn()` scores them correctly, and the leak tests confirm
HIDDEN still never writes a raga name. The one thing that cannot be checked without Colab is whether
GRPO actually learns to use the call — that's what the retrain itself, plus a re-run of
`eval/evaluate.py` comparing against `random-valid`, would show.

### 2026-10 script fixes (do before any run, not just the first one)

Found on a pre-flight review, not from a failed run — fixed now rather than waiting to hit them:

- `train_grpo.py`'s `main()` called `wandb.init()` unconditionally — would hang on an interactive
  login prompt or crash outright if run exactly as its own usage comment says to (plain
  `python train_grpo.py ...`, no notebook secret-handling in front of it) without `WANDB_API_KEY`
  set. Now conditional, matching the notebook's Cell 2 pattern.
- Both `train_grpo.py` and the notebook's Cell 6 set `num_train_epochs=1` on `GRPOConfig` — dead
  config. HF's Trainer always lets a positive `max_steps` override `num_train_epochs`, so this
  implied something false. What actually happens: every row `build_dataset()` generates gets
  visited `gradient_accumulation_steps` (4) times over a full run — the same "K epochs per rollout
  batch" pattern PPO uses, not a bug, just previously undocumented. Removed the misleading line,
  added a comment explaining the real relationship in both places.
- The notebook's `RUN_NAME` was still hardcoded to `-v1` — running it as-is would have silently
  overwritten the first run's checkpoint on Drive, the exact thing this doc already warned against
  below. Bumped to `-v2` in the notebook itself, not just this doc.

### Pilot run first, then decide on the final run — this is the actual recommendation

Run a cheap pilot before any long run: in the notebook, set `STEPS = 50` (a few minutes on a T4)
instead of 300, run top to bottom, and look at three things before going further:

1. **Did it finish without crashing?** `requirements-train.txt` now pins only `unsloth` and lets it
   resolve its own compatible `trl`/`transformers`/`peft` — a real version jump from the `trl==0.12.2`
   this script was originally written against. `GRPOTrainer`'s constructor has changed shape between
   trl releases before; a pilot catches that in minutes instead of an hour into a "final" run.
2. **Does `train/reward_mean` trend up** (wandb, if configured) or at least not look broken in the
   per-`logging_steps` stdout output?
3. **Does Cell 8's inference check run end to end**, and does its new `Call-echo bonuses` count show
   anything nonzero at least some of the time? Zero every time across a few pilot runs would be a
   sign the call-response fix isn't actually landing in the trained behaviour, worth investigating
   before spending a long run on it.

Only once the pilot looks sane, raise `STEPS` for the real run. **How high:** the first run (v1) used
300 steps and only had to learn drift adaptation; this run also has to learn to use a call phrase it
never saw before, which is strictly more to learn from the same model size. Recommend starting the
final run at **500–800 steps** rather than repeating 300 — but let the pilot's actual reward/loss
trend inform that number rather than picking it blind; if reward is still climbing steadily at 300 in
the pilot's own curve, that's a stronger signal than any number in this doc.

**Before re-running:** pull latest, confirm `pytest` is green locally first (catches anything Colab
would otherwise fail on 45 minutes into a T4 run), then run the notebook — pilot first, per above.
`RUN_NAME` is `jugalbandi-grpo-hidden-v2` now (fixed above); `v1` stays on Drive untouched as the
documented baseline in `FIRST_TRAINING_RUN.md`. A pilot run also saves under `-v2` and gets
overwritten by the final run under the same name — that's fine, nothing worth keeping from a 50-step
pilot once it's told you what you needed to know. Only bump to `-v3` if you want to keep both a pilot
and a final run's checkpoints side by side on purpose.

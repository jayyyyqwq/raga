# Jugalbandi

An OpenEnv-compatible RL environment that trains a small LLM to compose Indian classical music
(ragas) while the rule system drifts mid-episode — testing whether a model can notice its rules
changed from reward feedback alone, without being told in words.

Full technical writeup: [`docs/summary.md`](docs/summary.md). Experiment protocol (pre-registered,
locked before any training run): [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md). Research
framing and related-work notes: [`docs/research.md`](docs/research.md).

## Status

- Environment, reward function, drift mechanic, prompting layer, HTTP server, and evaluation harness:
  **built and tested** (151 tests, `pytest -q`).
- Four scripted baselines (`random-uniform`, `random-valid`, `safe-set-cycle`, `scripted-oracle`):
  **run for real** — see `eval/results/`.
- Trained model: **`grpo-HIDDEN` v1 run completed on Colab** (300 steps, Qwen2.5-0.5B QLoRA) — see
  [`docs/FIRST_TRAINING_RUN.md`](docs/FIRST_TRAINING_RUN.md). That run predates the call-response fix
  ([`docs/RETRAIN_PLAN.md`](docs/RETRAIN_PLAN.md)) — it's a solo improviser with drift adaptation,
  not a call-and-response result. A **v2 pilot (50 steps) with the fix has run** and the pipeline
  survives on the current dependency stack; a real full-length v2 run and the proper 200-episode
  comparison (`eval/evaluate_llm.py`, new — see `docs/RETRAIN_PLAN.md`) haven't happened yet.
  `DIAL`/`ORACLE` arms: **not run yet**.

## Training on Colab — 3 steps

This repo is public on GitHub, so Colab opens and clones it directly — no download, no zip
upload, no Hugging Face account, no sign-in beyond your own Google account.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jayyyyqwq/raga/blob/main/training/train_grpo.ipynb)

1. **Click the badge above.** It opens `training/train_grpo.ipynb` directly in Colab.
2. `Runtime → Change runtime type → T4 GPU` (free tier) → `Save`.
3. `Runtime → Run all`.

That's it — 3 clicks, then wait. Takes roughly 45–60 minutes; the trained model saves itself to
your Google Drive automatically, no further action needed. Everything below is optional extra
context, not required steps.

**If Cell 1 raises `RuntimeError: pip install did not actually install: [...]`** — that's the
install-verification check doing its job: `requirements-train.txt`'s `unsloth` pin has gone stale
against whatever PyPI/Colab ships now. Scroll up in that cell's output for pip's actual error, then
see `requirements-train.txt`'s header comment for how to re-pin it (this has happened twice before;
the fix is always the same shape).

If the badge or GitHub's clone is ever flaky (offline mirror, you're testing local uncommitted
changes), there's a fallback: double-click **`make_training_zip.bat`** in this folder to build a
`raga.zip`, then drag it into Colab's Files panel (folder icon, left sidebar) before running the
first code cell — the notebook auto-detects the zip and uses that instead of cloning.

### What about Hugging Face?

Hugging Face is a free website that hosts trained AI models, the same way GitHub hosts code. This
project can optionally upload the trained model there at the end — **but it's entirely optional and
off by default.** If you never set up an `HF_TOKEN`, that step is silently skipped and nothing
breaks; the model is already safe in your Google Drive regardless. A past version of this notebook
required a GitHub token too, which caused problems — that requirement has since been removed
entirely (see git history). If you don't touch the "secrets" step in Colab, you will never interact
with Hugging Face at all.

The same applies to `WANDB_API_KEY` (Weights & Biases, a live training-chart dashboard) — also
optional, also safe to ignore.

### Advanced / manual version

If you'd rather run the zip step yourself instead of the `.bat` file (e.g. on Mac/Linux), the
`.bat` file just runs this one command, which uses git's own tracked-file list so it automatically
excludes `venv/`, `.git/`, and everything else in `.gitignore`:

```bash
git archive HEAD -o raga.zip --prefix=raga/
```

See `docs/EXPERIMENT_PLAN.md` for what the resulting model is evaluated against.

## Running the interactive demo

A turn-based jugalbandi stage — you play a recorded intro (real sitar samples, tanpura drone), the
AI answers (real flute/bansuri samples), you reply live on the keyboard, it answers again, each
turn a step up an escalation ladder. Design doc: [`docs/imrpovedui.md`](docs/imrpovedui.md). Audio
sources and licences: [`docs/AUDIO_SOURCES.md`](docs/AUDIO_SOURCES.md) /
[`ATTRIBUTION.md`](ATTRIBUTION.md).

It runs **with or without a trained adapter** — without one, `/infer` falls back to a safe default
note (logged on screen) so the whole loop still plays end to end; the point of this step is the UI,
not the model (see [`docs/RETRAIN_PLAN.md`](docs/RETRAIN_PLAN.md) for where the model itself stands).

1. One-time setup, in a terminal in this folder:

   ```bash
   python -m venv venv
   venv/Scripts/activate          # Windows; source venv/bin/activate on Linux/Mac
   pip install -r requirements.txt -r requirements-infer.txt
   ```

2. **(Optional, for the real trained model)** In Google Drive, find `jugalbandi/<run-name>/final`
   (the folder Colab's Cell 7 saved to). Right-click → **Download**, unzip it so its files
   (`adapter_config.json`, `adapter_model.safetensors`, tokenizer files, …) land directly inside a
   `checkpoints/final` folder here — i.e. `checkpoints/final/adapter_config.json` should exist.
   (Gitignored — downloaded weights never get committed.) Skip this to run on the fallback notes.

3. Double-click **`run_demo.bat`**. It starts one server (API + the demo page, same origin — no
   second window, no CORS) and opens your browser to it. If a trained adapter is configured, the
   first time the AI plays it has to load the model into memory — a one-time pause (up to ~1-2
   minutes on CPU) is normal, not frozen.

4. Click **▶ Play intro** — your opening line plays on sitar, the tanpura drone fades in, tabla
   keeps teentaal. The AI answers on flute. Your turn: play up to 4 notes on the on-screen strings
   or the keyboard (`S R G M P D N`, `Shift` for the octave up, `Z X C V B N M` for the octave
   down) within the turn window. It keeps alternating, climbing the escalation ladder shown above
   the river, until the episode's step budget runs out. The raga dial above the stage still
   controls the drift mechanic live, same as before.

If something's missing (no venv set up), `run_demo.bat` tells you exactly what to fix instead of
just failing silently.

## Dev quickstart (code changes, not training)

```bash
python -m venv venv
venv/Scripts/activate          # Windows; source venv/bin/activate on Linux/Mac
pip install -r requirements.txt
pytest -q                      # 151 tests, no GPU needed
python -m eval.evaluate --all  # re-run the four scripted baselines
```

## Repo layout

```
raaga_env/        core RL environment (ragas.py, env.py, jugalbandi_env.py, reward.py, drift.py, prompting.py)
openenv_server/   FastAPI HTTP wrapper (server.py) — also serves ui/ itself
training/         GRPO training script + Colab notebook
eval/              in-process evaluation harness (rollout, policies, metrics, fixed 200-episode set)
ui/                jugalbandi stage (vanilla JS + PixiJS + Tone.js, no build step)
  stage/            river, tala mandala, orbs, theme, tween — the Pixi visuals
  audio/            sampled sitar/flute voices, tanpura drone, synth tabla
  analysis/         echo.js — pure functions behind the echo-arc visualisation
  samples/           sitar/flute note recordings + tanpura drone (see /ATTRIBUTION.md)
  app.js, input.js, presets.js, env_client.js — orchestrator, live input, intros, API client
scripts/           fetch_samples.py — re-downloads ui/samples/ if needed
docs/              full writeups — start with docs/summary.md
```

# Jugalbandi

An OpenEnv-compatible RL environment that trains a small LLM to compose Indian classical music
(ragas) while the rule system drifts mid-episode — testing whether a model can notice its rules
changed from reward feedback alone, without being told in words.

Full technical writeup: [`docs/summary.md`](docs/summary.md). Experiment protocol (pre-registered,
locked before any training run): [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md). Research
framing and related-work notes: [`docs/research.md`](docs/research.md).

## Status

- Environment, reward function, drift mechanic, prompting layer, HTTP server, and evaluation harness:
  **built and tested** (134 tests, `pytest -q`).
- Four scripted baselines (`random-uniform`, `random-valid`, `safe-set-cycle`, `scripted-oracle`):
  **run for real** — see `eval/results/`.
- Trained model: **one `grpo-HIDDEN` run completed on Colab** (300 steps, Qwen2.5-0.5B QLoRA) — see
  [`docs/FIRST_TRAINING_RUN.md`](docs/FIRST_TRAINING_RUN.md) for what the numbers mean. That run
  predates a call-response bug fix ([`docs/RETRAIN_PLAN.md`](docs/RETRAIN_PLAN.md)) — it's a solo
  improviser with drift adaptation, not a call-and-response result. A retrain with the fix hasn't
  been run yet. `DIAL`/`ORACLE` arms and a full baseline comparison: **not run yet**.

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

## Running the interactive demo (after training)

Once you have a trained adapter, you can play against it in the browser — the sitar/tabla UI
talking to a local server that runs the real model. No GPU needed for this part; it runs fine on
CPU, just slower per move (a second or two) than it would on a GPU.

1. In Google Drive, find `jugalbandi/<run-name>/final` (the folder Colab's Cell 7 saved to —
   e.g. `jugalbandi/jugalbandi-grpo-hidden-v1/final`). Right-click it → **Download**. Drive zips it
   for you.
2. Unzip it so its files (`adapter_config.json`, `adapter_model.safetensors`, tokenizer files, …)
   land directly inside a `checkpoints/final` folder in this repo — i.e.
   `checkpoints/final/adapter_config.json` should exist. Create the `checkpoints` folder if it's
   not there yet. (This folder is gitignored — your downloaded weights never get committed.)
3. One-time setup, in a terminal in this folder:

   ```bash
   python -m venv venv
   venv/Scripts/activate          # Windows; source venv/bin/activate on Linux/Mac
   pip install -r requirements.txt -r requirements-infer.txt
   ```

4. Double-click **`run_demo.bat`**. It starts the server, starts a page server for `ui/`, and opens
   your browser to the demo. The first time you click Start and the AI takes its turn, it has to
   load the model into memory — that one-time pause (up to ~1-2 minutes on CPU) is normal, not frozen.
5. Click **Start**, pluck 4 sitar strings for your "call" phrase, and watch the trained model
   respond. The raga dial slider controls the drift mechanic live.

If something's missing (no adapter downloaded yet, no venv set up), `run_demo.bat` tells you
exactly what to fix instead of just failing silently.

> **UI redesign planned.** The current UI is being replaced by a turn-based jugalbandi stage
> (recorded intro → AI answers → you reply → AI answers, with sampled sitar/bansuri/tanpura).
> Plan: [`docs/imrpovedui.md`](docs/imrpovedui.md). Audio sources and licences:
> [`docs/AUDIO_SOURCES.md`](docs/AUDIO_SOURCES.md).

## Dev quickstart (code changes, not training)

```bash
python -m venv venv
venv/Scripts/activate          # Windows; source venv/bin/activate on Linux/Mac
pip install -r requirements.txt
pytest -q                      # 134 tests, no GPU needed
python -m eval.evaluate --all  # re-run the four scripted baselines
```

## Repo layout

```
raaga_env/        core RL environment (ragas.py, env.py, jugalbandi_env.py, reward.py, drift.py, prompting.py)
openenv_server/   FastAPI HTTP wrapper (server.py)
training/         GRPO training script + Colab notebook
eval/              in-process evaluation harness (rollout, policies, metrics, fixed 200-episode set)
ui/                browser demo client (vanilla JS)
docs/              full writeups — start with docs/summary.md
```

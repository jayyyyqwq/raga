# Jugalbandi

An OpenEnv-compatible RL environment that trains a small LLM to compose Indian classical music
(ragas) while the rule system drifts mid-episode — testing whether a model can notice its rules
changed from reward feedback alone, without being told in words.

Full technical writeup: [`docs/summary.md`](docs/summary.md). Experiment protocol (pre-registered,
locked before any training run): [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md). Research
framing and related-work notes: [`docs/research.md`](docs/research.md).

## Status

- Environment, reward function, drift mechanic, prompting layer, HTTP server, and evaluation harness:
  **built and tested** (120 tests, `pytest -q`).
- Four scripted baselines (`random-uniform`, `random-valid`, `safe-set-cycle`, `scripted-oracle`):
  **run for real** — see `eval/results/`.
- Trained model (`grpo-HIDDEN`/`DIAL`/`ORACLE`): **pending a GPU run** — see
  [`training/train_grpo.ipynb`](training/train_grpo.ipynb), meant to run on Colab.

## Training on Colab — zero-setup version

This repo is public on GitHub, so Colab can pull it directly. No download, no zip, no upload, no
Hugging Face account required. Four steps:

1. Go to [colab.research.google.com](https://colab.research.google.com) → `File → Open notebook`
   → the **GitHub** tab → paste `https://github.com/jayyyyqwq/raga` → open `training/train_grpo.ipynb`.
2. `Runtime → Change runtime type → T4 GPU` (free tier) → `Save`.
3. `Runtime → Run all`.
4. Wait roughly 45–60 minutes. The trained model saves itself to your Google Drive automatically —
   no further action needed.

That's it. Everything below is optional extra context, not required steps.

If GitHub's own Colab link is ever flaky, there's a fallback: double-click
**`make_training_zip.bat`** in this folder to build a `raga.zip`, then drag it into Colab's Files
panel (folder icon, left sidebar) before running the first code cell — the notebook auto-detects
the zip and uses that instead of cloning.

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

## Dev quickstart (code changes, not training)

```bash
python -m venv venv
venv/Scripts/activate          # Windows; source venv/bin/activate on Linux/Mac
pip install -r requirements.txt
pytest -q                      # 120 tests, no GPU needed
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

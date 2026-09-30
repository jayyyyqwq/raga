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

## Quickstart

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

## Training on Colab

Open [`training/train_grpo.ipynb`](training/train_grpo.ipynb) in Colab. It clones this repo (private
— needs a `GH_TOKEN` Colab secret, see Cell 1 for how to create one) and needs `WANDB_API_KEY` and
`HF_TOKEN` Colab secrets too. See the notebook's own cells for details; see
`docs/EXPERIMENT_PLAN.md` for what the resulting model is evaluated against.

This repo is private. Don't make it public or share the link.

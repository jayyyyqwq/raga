# First training run — results and what they mean

Plain-language companion to the first completed GRPO run. For the full research design behind
*why* any of this is set up this way, see `docs/EXPERIMENT_PLAN.md` and `docs/summary.md`.

## What was trained

- **Arm:** `HIDDEN` — the model gets no raga name, no dial value, and no explicit signal that a
  rule change happened. The only thing it sees each step is the outcome (reward) of its *previous*
  move. This is the actual research question of the project: can a model notice the rules changed,
  purely from reward feedback, without being told in words?
- **Base model:** `unsloth/Qwen2.5-0.5B-Instruct`, fine-tuned with QLoRA (4-bit) + GRPO.
- **Steps:** 300 training steps, batch size 4, 4 generations per group (1,200 decision points
  sampled total).
- **Where it ran:** Google Colab, free-tier T4 GPU, via `training/train_grpo.ipynb`.
- **wandb / Hugging Face:** both skipped (optional, off by default) — no live charts were recorded
  for this run, only the end-of-notebook inference check below.

## The inference check (Cell 8)

This is a single improvised episode run right after training, as a sanity check that the trained
adapter actually produces valid, playable output — not a real evaluation of how *good* the model
is (see caveat below). Result from this run:

| Metric | Value | Meaning |
|---|---|---|
| Drift schedule | initial=0.0, switch at step 37 | Episode started in Yaman (dial 0.0); the raga's rules silently flipped to Bhairav at step 37 of 64. |
| Steps completed | 64 / 64 | The model never produced an unparseable or out-of-range move — it stayed "in the game" the whole episode. This is actually the baseline bar to clear: a model outputting garbage would end the episode early. |
| Forbidden-note steps | 0 / 64 | The model never played a note outside the currently-active raga's allowed note set, across the entire episode — including after the silent rule switch at step 37. |
| Episode return | −46.12 | Total reward summed over all 64 steps. Negative. |

## How to read the negative return

A negative total reward on one episode is **not** evidence the model failed. Reasons not to
over-read it:

1. **It's one episode.** GRPO reward in this project stacks several soft penalties (repetition,
   "pakad drought", ignoring vadi/samvadi emphasis, etc. — see `raaga_env/reward.py`) on top of the
   hard forbidden-note rule. Even a well-behaved policy can rack up small negative penalties across
   64 steps without ever breaking a hard rule. One sample is noisy; it could easily have been +20
   or −80 on a different random episode.
2. **300 steps is a short run.** It's enough to prove the pipeline works end-to-end (which it did —
   see the 64/64 and 0/64 lines above), but likely not enough for GRPO to fully converge on a
   0.5B-parameter model learning a fairly subtle in-context-inference task.
3. **The project's own risk register already names this as a possible, still-valid outcome** (see
   `training/train_grpo.ipynb`, Cell 6 markdown, and `docs/updatedplan.md`'s risk section):
   `grpo-HIDDEN` failing to clearly beat a random-valid baseline at 0.5B scale is a legitimate,
   reportable result — not necessarily a bug to chase.

## What would actually answer "did it work?"

A single episode's return is not a verdict. To get a real answer, this project already has a
proper evaluation harness (`eval/`) built for exactly this — it runs many episodes and compares
against the four scripted baselines (`random-uniform`, `random-valid`, `safe-set-cycle`,
`scripted-oracle`) under the same conditions:

```bash
python -m eval.evaluate --all
```

Comparing the trained `grpo-HIDDEN` adapter's average return (over, say, 50-200 episodes) against
`random-valid` is what would actually show whether training produced a measurable improvement, as
opposed to one noisy episode's score. That comparison hasn't been run yet — it's the natural next
step once you want more than "the pipeline works," which this run already confirms.

## Where the model lives

The trained adapter is saved to Google Drive (not committed to this repo — model weights are
gitignored): `jugalbandi/jugalbandi-grpo-hidden-v1/final/`. See the README's "Running the
interactive demo" section for downloading it and playing against it in the browser UI.

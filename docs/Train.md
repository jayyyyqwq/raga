# Train.md — How to Actually Train Jugalbandi

> Written for Jay. Assumes you understand Python and can read code, but this is your
> first time doing LLM fine-tuning with GRPO + Unsloth. This walks every decision
> from "why this model" to "how to push the checkpoint to HF."

**This file was rewritten** (`updatedplan.md` Phase 8.1) — the previous version
described a training design that no longer exists: a 0-47 action space, TRL
0.8.6, and a training loop that scored actions over HTTP against a live
server. All three are gone. `training/train_grpo.py` is now the source of
truth; this document walks through what it actually does. If the two ever
disagree, trust the code and consider this file stale until it's fixed.

---

## The Big Picture First

You are not training a model from scratch. You are taking a small model that
already knows English, already knows how to generate tokens — and pushing its
behavior in one specific direction: **when given a musical context, output an
integer 0–95 that follows raga grammar rules, and — this is the interesting
part — notice when the rules changed mid-performance from feedback alone.**

The way you push that behavior is **GRPO** — Group Relative Policy
Optimization. Instead of "here is the correct answer" (supervised learning),
you say "here are 4 things you tried, here is how well each one scored, now do
more of the good stuff." The reward comes entirely from the environment
(`raaga_env/reward.py`), replayed in-process — no server involved in training.

**Read `docs/EXPERIMENT_PLAN.md` first if you haven't** — it explains the
*why* (the ORACLE/DIAL/HIDDEN arms, the feedback channel, what's actually
being tested) at more length than this document will. This document is the
*how*.

---

## Part 1: Why This Model — Qwen2.5-0.5B-Instruct

**Size:** 0.5 billion parameters — small enough that a free Colab T4 (15GB
VRAM) handles it in 4-bit quantization comfortably.

**The Instruct variant:** already fine-tuned to follow a chat format
(`<|system|>`, `<|user|>`, `<|assistant|>`), which is exactly the format
`raaga_env/prompting.py` renders.

**Unsloth support:** optimized kernels specifically for Qwen2.5.

---

## Part 2: Why Unsloth (QLoRA)

Training on a T4 without help is slow and OOMs easily. Unsloth handles two things:

**QLoRA** — the model's weights are stored in 4-bit, but you can't fine-tune
in 4-bit directly, so small trainable "adapter" matrices (LoRA) sit alongside
the frozen weights. Only the adapters (`r=16`, ~4M params) get gradient
updates — roughly 1% of the full model's parameter count.

**Triton kernels** — Unsloth rewrites attention and RoPE using Triton for a
2-3x speedup over vanilla HuggingFace.

```python
r=16                    # adapter rank — 16 is a standard default
target_modules=[...]    # q,k,v,o attention projections get adapters
lora_alpha=16           # alpha/r = 1.0, no extra scaling
lora_dropout=0          # Unsloth recommends 0 for speed
bias="none"
use_gradient_checkpointing="unsloth"  # recompute activations instead of storing them
```

---

## Part 3: Why GRPO, and Why Per-Step Instead of Per-Episode

**GRPO vs PPO:** PPO needs a value network roughly as large as the policy,
and struggles when reward is sparse and delayed (pakad completions are rare
early in training). GRPO samples G completions per prompt (`G=4` here),
scores all of them, and pushes the model toward whichever scored above the
group average — no value network needed.

**Why one action per GRPO sample, not a whole episode.** An earlier version
of this pipeline had GRPO generate a full 64-action episode in one blind
completion. That's a real design GRPO supports, but it has a fatal problem
for *this* project specifically: a model committing to all 64 actions before
any of them are scored cannot react to feedback about steps it hasn't taken
yet. Once `updatedplan.md` §9 decision 2 settled on **Claim B** — testing
whether the model can infer a rule change *in-context*, from reward feedback
alone, within a single episode — that design became unusable, because
in-context reaction requires the model to actually see feedback before
choosing its next action. So training went back to one action per GRPO
sample, the way GRPO is normally used, with the twist described below.

---

## Part 4: The Training Flow, Step by Step

Here's what `training/train_grpo.py` actually does, top to bottom:

```
1. Parse --arm (oracle/dial/hidden), --seed, --steps, --batch, --num-generations.
   --batch must be divisible by --num-generations (TRL's GRPOTrainer constraint).

2. Load Qwen2.5-0.5B-Instruct in 4-bit via Unsloth, attach LoRA adapters.

3. build_dataset(n, arm) — build n training rows, EACH ONE A SNAPSHOT, not a
   full episode:
     a. Sample a drift schedule (sample_drift_schedule()): a switch step
        uniform in [16, 48], flipped to the opposite raga — or, 20% of the
        time, no switch at all (a control condition, so the model can't
        just learn "a switch always happens near the middle").
     b. Walk a FRESH episode forward, in-process, under a random-valid
        reference policy (uniformly samples among notes legal in whichever
        raga is *currently* active — see eval/policies.py).
     c. Stop at a uniformly chosen step. Render the prompt at that point —
        including StepFeedback from the immediately preceding step (or none,
        if this is step 0) — and save the env's exact state
        (JugalbandiEnv.get_state(), as a JSON string).
   Each row is: {prompt, state_json, mc_seed}.

4. make_step_reward_fn(arm) — the GRPO reward function. For each sampled
   completion (one action):
     a. Parse it (raaga_env.prompting.parse_action — never clamps; an
        out-of-range or unparseable output is a fixed penalty, not silently
        wrapped into a legal action).
     b. Restore a fresh JugalbandiEnv to the row's exact saved state
        (set_state()).
     c. Apply the sampled action, record its immediate reward.
     d. If the episode isn't over, continue for 8 more steps under the same
        random-valid reference policy (a Monte-Carlo continuation), via
        eval.rollout.rollout_from_state — the SAME function every baseline
        and evaluation result goes through, so training reward and eval
        reward can never silently diverge.
     e. Return immediate + continuation reward as this completion's score.

5. GRPOTrainer.train() — num_generations=4 completions per prompt,
   max_completion_length=8 (an action is 1-2 tokens), logs to wandb.

6. Save the LoRA adapter locally; upload to HF Hub with huggingface-cli.
```

**Why a Monte-Carlo continuation instead of just the immediate reward:**
scoring only the one step is the "purest" bandit-style GRPO, but it's very
myopic — an action's value often depends on what happens a few steps later
(did playing that note set up a pakad completion?). Continuing under a
reference policy for a short, fixed horizon gives partial credit assignment
without needing the model itself in the scoring loop, which would be far
more expensive per training step.

**Why the reference policy is `random-valid`, not something fancier:** it's
the same policy used as an evaluation baseline (`docs/EXPERIMENT_PLAN.md`
§5), so the number that comes out of training reward and the number that
comes out of evaluating `random-valid` are produced by literally the same
code. Using a different, bespoke "training-only" policy would reintroduce
exactly the kind of silent-drift risk `eval/rollout.py`'s whole design
exists to prevent.

---

## Part 5: Which Arm to Train

```bash
python training/train_grpo.py --arm hidden --seed 1 --steps 300 --run grpo-hidden-seed1
```

- `--arm oracle` — sees the raga name, the dial, and the grace flag. Upper bound.
- `--arm dial` — sees only the raw 0.0–1.0 dial number. "What the project
  currently claims to do."
- `--arm hidden` — sees none of that, only `StepFeedback`. **The actual
  research question.**

Per `docs/EXPERIMENT_PLAN.md` §10, the full run is 3 arms × 5 seeds = 15
training runs, each evaluated on the same fixed 200-episode set every
baseline was already evaluated on.

---

## Part 6: The Notebook

`training/train_grpo.ipynb` is a **thin driver** over `train_grpo.py` — it
imports `build_dataset`, `make_step_reward_fn`, `sample_drift_schedule`,
`schedule_from_row`, `EPISODE_LENGTH` rather than redefining any of them, so
it cannot drift out of sync with the script the way the pre-rewrite notebook
did (three independent, hand-duplicated prompt/reward implementations, one of
which leaked the raga name to every arm regardless of the leakage tests
passing elsewhere — see `updatedplan.md` finding F2/F7 for the full story of
why that's the specific failure mode this thin-driver structure exists to
prevent).

Cell structure, in order: install + clone the repo (needed — the notebook
imports `raaga_env`/`eval`/`training` from the repo, not just from pip
packages), mount Drive + secrets, load the model, import the training
pipeline and pick `ARM`, build `reward_fn`, build the dataset, configure and
run `GRPOTrainer`, save + push to Hub, and a genuine multi-turn inference
check (see Part 7).

Run cells top to bottom. There's no separate "verify server" smoke-test cell
any more — training doesn't touch the HTTP server at all (`updatedplan.md`
Phase 3.1).

---

## Part 7: The Inference Check (After Training)

The model was trained on **one action per call, with live feedback** — so
the honest sanity check is a genuine multi-turn loop, not a single
`/infer`-style call. The notebook's last cell does exactly this:

```python
# One env, reset once. For each of 64 steps:
#   1. render_prompt(obs, arm=ARM, feedback=<previous step's StepFeedback>, ...)
#   2. generate ONE action from the model
#   3. parse_action() it — stop early and report if it's unparseable
#   4. env.step(action), record the new StepFeedback for the next prompt
```

This is the same interaction pattern `openenv_server/server.py`'s `/infer` +
`/step` loop uses in production — if the notebook's check works, the
deployed server loop will behave the same way. Report `episode return`,
`forbidden-note steps`, and how many of the 64 steps actually completed
(a model that goes off-format stops the loop early and that's reported
honestly, not silently papered over).

**Do not evaluate the model informally and call it done.** The real
evaluation protocol — the fixed 200-episode set, the precise metrics, the
statistical comparison against the baselines — is `docs/EXPERIMENT_PLAN.md`,
not this cell. This cell answers "did something obviously break," not "is
the result real."

---

## Part 8: Reading the WandB Dashboard

- `train/reward_mean` should climb over the course of training. Because this
  is a Monte-Carlo return (immediate + 8-step continuation under a random
  reference policy), expect it to be noisier than a pure single-step bandit
  signal — that's expected, not a sign something's broken.
- `train/reward_std` collapsing to 0 means the model converged, possibly to
  a degenerate policy — check `safe_set_occupancy` (`docs/EXPERIMENT_PLAN.md`
  §7.8) on the trained model once you can evaluate it; a value near 1.0 is
  the specific named failure mode (F13) that metric exists to catch.
- **`grpo-HIDDEN` may simply fail to separate from `random-valid`.** This is
  explicitly named in `updatedplan.md`'s risk register as a real possible
  outcome, not a bug — see `docs/EXPERIMENT_PLAN.md` §12 for how to report
  it honestly if it happens.

---

## Part 9: Compute and Timing

| Hardware | Notes |
|---|---|
| Colab T4 (free) | May time out after ~90 min; save checkpoints to Drive, not `/content` |
| Colab A100 / L4 (paid) | Faster, recommended for the full 15-run sweep |
| Local GPU | Fine if you have one — same `requirements-train.txt` |

Per `updatedplan.md` §6: roughly 25–45 GPU-hours for the full 15 training
runs (3 arms × 5 seeds), plus 3–6 more for evaluating everything. That does
not fit in one free Colab session — plan for multiple sessions, or a paid
tier. The exact seed count is still an open decision
(`docs/EXPERIMENT_PLAN.md` §9) — fewer seeds if compute is constrained, but
that's a real tradeoff to make explicitly, not discover by running out of
GPU-hours partway through.

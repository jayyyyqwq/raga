# GRPO training script — run on Colab A100 with HF credits.
# WHY GRPO over PPO: no value model needed; reward comes directly from env.
# WHY Qwen2.5-0.5B: smallest Unsloth-supported model that fits T4 (15GB) with QLoRA.
#
# Usage (Colab):
#   !pip install -r requirements-train.txt
#   !python train_grpo.py --arm hidden --steps 500
#
# Run a cheap pilot (e.g. --steps 50, a few minutes on a T4) before a long
# "final" run — this is the actual, current trl/unsloth/transformers stack
# (requirements-train.txt pins only unsloth and lets it resolve the rest;
# see that file's header), not the exact versions this script was last
# verified against, and GRPOTrainer's constructor has changed shape between
# trl releases before. A pilot run is what catches an API mismatch, a
# reward/loss that isn't moving, or a training-loop crash in minutes
# instead of an hour. See docs/RETRAIN_PLAN.md.
#
# Training is in-process (updatedplan.md Phase 3.1) — no HTTP server, no
# shared mutable env instance across samples (F9).
#
# Each GRPO sample is ONE decision point within an episode, not a whole
# blind episode (Phase 3.2's "alternative design", adopted deliberately: the
# episode-blind-completion design where the model commits to all 64 actions
# up front cannot react to feedback about steps it hasn't taken yet, which
# is incompatible with Claim B — in-context regime inference from reward
# feedback alone, updatedplan.md §9 decision 2). The prompt shows
# raaga_env.prompting.StepFeedback for the *previous* step (Phase 2.3);
# scoring restores the env to the exact state the prompt was drawn from
# (Phase 0.4's get_state/set_state), applies the sampled action, and
# continues for a short horizon under a reference policy for a Monte-Carlo
# return (eval.rollout.rollout_from_state) — giving the sampled action more
# than one step's worth of credit assignment without needing the model
# itself in the scoring loop.
#
# GPU-only imports (unsloth/trl/torch/wandb) are deferred into main() so
# this module — and its dataset/reward-scoring logic — stays importable and
# testable without a training stack installed, the same lazy-import pattern
# openenv_server/server.py uses for its inference deps.

import argparse
import json
import os
import random

from raaga_env.jugalbandi_env import JugalbandiEnv
from raaga_env.prompting import Arm, StepFeedback, note_name, parse_action, render_prompt
from eval.policies import random_valid_policy, sample_call_phrase
from eval.rollout import DriftSchedule, rollout_from_state

EPISODE_LENGTH = 64          # matches JugalbandiEnv's default / openenv.yaml episode.max_steps
DRIFT_WINDOW = (16, 48)      # updatedplan.md Phase 3.3
NO_SWITCH_CONTROL_PROB = 0.20
MC_HORIZON = 8                # extra steps of reference-policy continuation after the sampled action
# A parse failure gets a fixed penalty scaled to (1 + MC_HORIZON) "virtual
# steps" — comparably harsh to the old single-step parse-failure penalty of
# -1.0 (Phase 1.2), scaled to the size of the return it's being compared
# against in the same GRPO group.
STEP_PARSE_FAILURE_PENALTY = -1.0 * (MC_HORIZON + 1)

# sample_call_phrase used to be defined here; moved to eval.policies
# (2026-10) once eval.rollout.rollout() needed the exact same generator to
# make jugalbandi_coherence/call_echo_rate measurable at all — see that
# function's docstring for the full rationale and the HIDDEN-arm caveat.


def sample_drift_schedule(rng: random.Random = random) -> tuple[float, DriftSchedule]:
    """One drift event per training episode, sampled uniformly over
    DRIFT_WINDOW, flipping the dial to the opposite raga. A
    NO_SWITCH_CONTROL_PROB fraction of episodes get no switch at all — a
    control condition so the policy can't just learn "a switch always
    happens near the middle" (updatedplan.md Phase 3.3).
    """
    initial_dial = rng.choice([0.0, 1.0])
    if rng.random() < NO_SWITCH_CONTROL_PROB:
        return initial_dial, DriftSchedule()
    switch_step = rng.randint(*DRIFT_WINDOW)
    new_dial = 1.0 if initial_dial < 0.5 else 0.0
    return initial_dial, DriftSchedule(switches=((switch_step, new_dial),))


def build_dataset(n: int, arm: Arm, rng: random.Random = random) -> list[dict]:
    """Each row is one decision point sampled from a reference episode: walk
    a fresh episode forward under random_valid_policy (with a sampled drift
    schedule), stop at a uniformly chosen step, and snapshot the prompt (with
    StepFeedback from the immediately preceding step, or none at step 0) plus
    the exact env state at that point. `state_json` carries JugalbandiEnv's
    full get_state() (Phase 0.4) as a JSON string — a single plain-string
    column, so nothing about it depends on how TRL's dataset conversion
    handles nested/struct columns. `mc_seed` makes the Monte-Carlo
    continuation used when scoring this row reproducible on repeat (used by
    tests); during training it still varies row to row, avoiding a single
    fixed continuation-policy draw across the whole dataset."""
    rows = []
    while len(rows) < n:
        initial_dial, schedule = sample_drift_schedule(rng)
        schedule_by_step = schedule.as_dict()
        snapshot_step = rng.randint(0, EPISODE_LENGTH - 1)

        env = JugalbandiEnv(initial_dial=initial_dial, episode_length=EPISODE_LENGTH)
        obs, info = env.reset()
        policy = random_valid_policy(rng)
        feedback: StepFeedback | None = None

        for step_idx in range(EPISODE_LENGTH):
            if step_idx in schedule_by_step:
                env.set_dial(schedule_by_step[step_idx])
                obs = env._get_obs()

            if step_idx == snapshot_step:
                raga = env.drift.active_raga_name if arm is Arm.ORACLE else None
                prompt = render_prompt(
                    list(obs), arm=arm, tala_pos=env.tala_position,
                    feedback=feedback, raga=raga,
                )
                rows.append({
                    "prompt": prompt,
                    "state_json": json.dumps(env.get_state()),
                    "mc_seed": rng.randint(0, 2**31 - 1),
                })
                break

            action = policy(obs, info)
            note, duration = env._decode(action)
            obs, reward, terminated, truncated, info = env.step(action)
            feedback = StepFeedback(note_name=note_name(note, duration), reward=float(reward))
            if terminated or truncated:
                break  # episode ended before reaching snapshot_step; retry

            # JugalbandiEnv's own CALL_EVERY cadence (info["call_requested"])
            # says a fresh call is due now. Training used to never call
            # set_call() at all — see jugalbandi_env.py's call_phrase
            # comment — so this is what actually exercises the call-response
            # reward layer and puts a call phrase in front of the model for
            # the first time (ML retrain, 2026-10; docs/RETRAIN_PLAN.md).
            if info.get("call_requested"):
                env.set_call(sample_call_phrase(env.raga, rng))
                obs = env._get_obs()
    return rows[:n]


def make_step_reward_fn(arm: Arm):
    """Returns a TRL reward_funcs-compatible callable bound to `arm`. GRPO
    scores one sampled action per completion: restore the env to the row's
    exact snapshot state, apply the action, then continue with
    random_valid_policy for MC_HORIZON steps (eval.rollout.rollout_from_state
    — Phase 0.3's single source of truth, same as every other result) to get
    a Monte-Carlo return. Never a shared/mutated env instance across samples
    (Phase 3.1 / F9), never a freshly reset, unrelated episode (F1)."""

    def step_reward(
        prompts: list[str],
        completions: list[str],
        state_json: list[str],
        mc_seed: list[int],
        **kwargs,
    ) -> list[float]:
        rewards = []
        for completion, state_str, seed in zip(completions, state_json, mc_seed):
            action = parse_action(completion)
            if action is None:
                rewards.append(STEP_PARSE_FAILURE_PENALTY)
                continue

            state = json.loads(state_str)
            env = JugalbandiEnv(episode_length=EPISODE_LENGTH)
            env.set_state(state)
            _, immediate_reward, terminated, truncated, _ = env.step(action)

            total = immediate_reward
            if not (terminated or truncated):
                continuation = rollout_from_state(
                    random_valid_policy(random.Random(seed)),
                    state=env.get_state(),
                    episode_length=EPISODE_LENGTH,
                    max_extra_steps=MC_HORIZON,
                    arm=arm.value,
                )
                total += continuation.total_reward
            rewards.append(total)
        return rewards

    return step_reward


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--run", default="jugalbandi-grpo-v1")
    parser.add_argument("--num-generations", type=int, default=4)
    # Which observation arm to train against (updatedplan.md Phase 4.1: this
    # script gets invoked once per arm — grpo-ORACLE, grpo-DIAL, grpo-HIDDEN).
    parser.add_argument("--arm", choices=[a.value for a in Arm], default=Arm.HIDDEN.value)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.arm = Arm(args.arm)

    # TRL's GRPOTrainer requires per_device_train_batch_size to be divisible
    # by num_generations (it must fit whole groups of completions per device).
    if args.batch % args.num_generations != 0:
        parser.error(
            f"--batch ({args.batch}) must be divisible by --num-generations ({args.num_generations})"
        )

    random.seed(args.seed)

    # Optional, same as the notebook's Cell 2 (docs/RETRAIN_PLAN.md-adjacent
    # fix, 2026-10): this used to call wandb.init() unconditionally, which
    # either hangs on an interactive login prompt or raises outright when
    # run exactly as this file's own usage comment above says to (plain
    # `python train_grpo.py ...`, no notebook secret-handling in front of
    # it) and WANDB_API_KEY isn't set. A training run must not depend on
    # wandb to start.
    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        import wandb
        wandb.init(project="jugalbandi", name=args.run, config={**vars(args), "arm": args.arm.value})
    else:
        print("No WANDB_API_KEY set — training without wandb logging. "
              "Reward/loss still print to stdout every `logging_steps`.")

    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=512,
        load_in_4bit=True,       # QLoRA — fits T4 with 15GB VRAM
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    config = GRPOConfig(
        output_dir=f"./checkpoints/{args.run}",
        # No num_train_epochs: HF's Trainer always lets a positive max_steps
        # override num_train_epochs, so setting it here would be dead config
        # implying something false. What actually happens: each optimizer
        # step consumes per_device_train_batch_size * gradient_accumulation_
        # steps rows, so with gradient_accumulation_steps=4 below, every row
        # build_dataset() generates gets visited ~4 times over a full
        # --steps run — the same "K epochs per rollout batch" pattern PPO
        # uses, not a bug. mc_seed (see build_dataset's docstring) exists
        # partly because of this: a row scored more than once needs a
        # varied Monte-Carlo continuation each time, not a frozen one.
        max_steps=args.steps,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=50,
        report_to="wandb" if use_wandb else "none",
        # GRPO-specific
        num_generations=args.num_generations,
        max_completion_length=8,  # one action index; 1-2 tokens
    )

    print("Building dataset …")
    dataset = build_dataset(args.steps * args.batch, args.arm)

    trainer = GRPOTrainer(
        model=model,
        args=config,
        reward_funcs=make_step_reward_fn(args.arm),
        train_dataset=dataset,
        processing_class=tokenizer,  # `tokenizer=` is deprecated in TRL 0.12+
    )

    print(f"Training {args.steps} steps, arm={args.arm.value} …")
    trainer.train()

    model.save_pretrained(f"./checkpoints/{args.run}/final")
    tokenizer.save_pretrained(f"./checkpoints/{args.run}/final")
    print("Done. Upload checkpoint to HF Hub with `huggingface-cli upload`.")


if __name__ == "__main__":
    main()

# GRPO training script — run on Colab A100 with HF credits.
# WHY GRPO over PPO: no value model needed; reward comes directly from env.
# WHY Qwen2.5-0.5B: smallest Unsloth-supported model that fits T4 (15GB) with QLoRA.
#
# Usage (Colab):
#   !pip install unsloth trl wandb httpx
#   !python train_grpo.py --server http://<your-hf-space>.hf.space --steps 500

import argparse
import httpx
import wandb
import torch
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

# ── Args ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--server",  default="http://localhost:7860")
parser.add_argument("--model",   default="unsloth/Qwen2.5-0.5B-Instruct")
parser.add_argument("--steps",   type=int, default=200)
parser.add_argument("--batch",   type=int, default=4)
parser.add_argument("--run",     default="jugalbandi-grpo-v1")
args = parser.parse_args()

# ── Model ──────────────────────────────────────────────────────────────
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

# ── Prompt template ────────────────────────────────────────────────────
# Intern note: each "turn" is one note pick. The LLM reads musical context,
# outputs a number 0-47 (action index). GRPO optimises this against env reward.

SYSTEM = (
    "You are an expert Indian classical musician composing in a raga. "
    "Given the current musical context, choose the next note action (0-47). "
    "Action = note_semitone + duration_index * 12. "
    "Output ONLY the integer, nothing else."
)

def make_prompt(obs: list[float], raga: str, tala_pos: int) -> str:
    note_hist = obs[0:8]
    return (
        f"<|system|>\n{SYSTEM}\n"
        f"<|user|>\n"
        f"Active raga: {raga}\n"
        f"Tala position: {tala_pos}/16\n"
        f"Last 4 notes (note/11, dur/3): {[round(x, 2) for x in note_hist]}\n"
        f"Raga dial: {round(obs[15], 2)}\n"
        f"Pakad drought: {round(obs[16], 2)}\n"
        f"Choose action (0-47):\n"
        f"<|assistant|>\n"
    )

# ── Reward function (calls env server) ────────────────────────────────
client = httpx.Client(base_url=args.server, timeout=10.0)

def env_reward(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    """
    GRPO calls this with a batch of (prompt, completion) pairs.
    We parse the completion as an action int, step the env, return the reward.
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        try:
            action = int(completion.strip().split()[0]) % 48
        except (ValueError, IndexError):
            rewards.append(-1.0)  # parse failure → penalty
            continue
        try:
            res = client.post("/step", json={"action": action})
            reward = res.json()["reward"]
            # Reset if episode ended
            if res.json()["terminated"]:
                client.post("/reset", json={"dial": 0.0})
        except Exception:
            reward = -0.5
        rewards.append(float(reward))
    return rewards

# ── Training ───────────────────────────────────────────────────────────

def build_dataset(n: int = 500) -> list[dict]:
    """
    Seed dataset: reset env, collect n prompts from random rollouts.
    GRPO will then improve from these starting points.
    """
    client.post("/reset", json={"dial": 0.0})
    dataset = []
    for _ in range(n):
        state = client.get("/state").json()
        obs_dummy = [0.0] * 22  # placeholder; real obs comes from /state
        prompt = make_prompt(obs_dummy, state["raga"], state["tala_position"])
        dataset.append({"prompt": prompt})
        # random step to vary starting states
        import random
        res = client.post("/step", json={"action": random.randint(0, 47)})
        if res.json()["terminated"]:
            client.post("/reset", json={"dial": 0.0})
    return dataset

wandb.init(project="jugalbandi", name=args.run, config=vars(args))

config = GRPOConfig(
    output_dir=f"./checkpoints/{args.run}",
    num_train_epochs=1,
    max_steps=args.steps,
    per_device_train_batch_size=args.batch,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=50,
    report_to="wandb",
    # GRPO-specific
    num_generations=4,        # how many completions to sample per prompt
    max_completion_length=8,  # action index fits in 1-2 tokens
)

print("Building dataset …")
dataset = build_dataset(args.steps * args.batch)

trainer = GRPOTrainer(
    model=model,
    args=config,
    reward_funcs=env_reward,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

print(f"Training {args.steps} steps on {args.server} …")
trainer.train()

model.save_pretrained(f"./checkpoints/{args.run}/final")
tokenizer.save_pretrained(f"./checkpoints/{args.run}/final")
print("Done. Upload checkpoint to HF Hub with `huggingface-cli upload`.")

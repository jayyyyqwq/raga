# Train.md — How to Actually Train Jugalbandi

> Written for Jay. Assumes you understand Python and can read code, but this is your first time doing LLM fine-tuning with GRPO + Unsloth. This walks every decision from "why this model" to "how to push the checkpoint to HF."

---

## The Big Picture First

You are not training a model from scratch. You are taking a small model that already knows English, already knows how to generate tokens, already has some vague sense of what numbers mean — and you are pushing its behavior in one specific direction: **when given a musical context, output an integer 0–47 that follows raga grammar rules.**

The way you push that behavior is called **GRPO** — Group Relative Policy Optimization. Instead of telling the model "here is the correct answer" (supervised learning), you say "here are 4 things you tried, here is how well each one scored, now learn to do more of the good stuff." The reward signal comes entirely from the OpenEnv environment server — specifically, from what `reward.py` returns after each note action.

---

## Part 1: Why This Model — Qwen2.5-0.5B-Instruct

There are hundreds of open models. Here is why this one:

**Size:** 0.5 billion parameters. That is tiny by modern standards. GPT-4 is ~1.7 trillion. The reason we use tiny is hardware — specifically, a free Google Colab T4 GPU has 15GB of VRAM. A 0.5B model in 4-bit quantization (explained below) uses ~1.2GB for weights alone, leaving room for activations, gradients, and batch data.

**The Instruct variant:** The `-Instruct` suffix means this model was already fine-tuned by Qwen's team to follow instructions in a chat format (`<|system|>`, `<|user|>`, `<|assistant|>` tags). This matters because our prompt uses exactly that format. A base model would need additional prompt engineering to follow the "output ONLY the integer" instruction reliably.

**Unsloth support:** Unsloth (explained below) has optimized kernels specifically for Qwen2.5 architecture. If you pick a random model that Unsloth hasn't explicitly optimized, you lose the speed benefits.

**Why not Llama-3.2-1B or Mistral-0.5B?**
- Llama-3.2-1B is 2x the size, needs more VRAM, slower training
- Mistral-0.5B doesn't exist
- Phi-3.5-mini is 3.8B — too big for T4 with this batch size
- Qwen2.5-0.5B hits the sweet spot for this hardware and task

---

## Part 2: Why Unsloth

Plain PyTorch + HuggingFace Transformers training on a T4 is slow and often OOMs (out-of-memory errors). Unsloth solves two things:

**1. QLoRA (4-bit quantization + LoRA adapters)**

The model's weights are stored in 4-bit integers instead of 32-bit floats. This reduces weight memory by ~8x. The model cannot be fine-tuned in 4-bit directly (gradients need higher precision), so Unsloth inserts small trainable "adapter" matrices (LoRA — Low-Rank Adaptation) alongside the frozen 4-bit weights. Only the adapters get gradient updates — they are ~1-5% of total parameters. You are training ~5-25M parameters instead of 500M.

```
Frozen 4-bit weights (Qwen2.5-0.5B)
    +
Trainable LoRA adapters (r=16, ~4M params)
    =
Fine-tuned model that fits on T4
```

**2. Triton kernel optimizations**

Unsloth rewrites certain compute-heavy ops (attention, RoPE embedding) using Triton (a GPU programming language). This gives 2-3x faster training vs vanilla HuggingFace with identical results. On a T4, this is the difference between "500 steps in 45 minutes" and "500 steps in 2.5 hours."

**LoRA config explained (from train_grpo.py):**
```python
r=16                    # rank — size of the adapter matrices. Higher = more expressive, more params. 16 is standard.
target_modules=[...]    # which attention projections get adapters. q,k,v,o = all attention heads.
lora_alpha=16           # scaling factor. alpha/r = 1.0 means no extra scaling. Standard default.
lora_dropout=0          # Unsloth recommends 0 for speed; dropout here rarely helps anyway.
bias="none"             # don't train bias terms; they add params without proportional benefit
use_gradient_checkpointing="unsloth"  # recompute activations instead of storing them → 30% less VRAM
```

---

## Part 3: Why GRPO (Not PPO)

Both are RL algorithms for fine-tuning language models. The difference matters for your reward structure.

**PPO (Proximal Policy Optimization):**
Requires two models at all times — a policy model (the one you're training) and a value model (predicts expected future reward from current state). The value model needs to be almost as large as the policy. On T4, that means you can only fit a ~0.25B model effectively. Also, value networks struggle when rewards are sparse and delayed — which ours are. Pakad completions are rare early in training. The value network would have bad estimates for many steps before seeing any reward signal.

**GRPO (Group Relative Policy Optimization):**
No value network. Instead, for each prompt, you sample G completions (we use G=4). You compute rewards for all G. You normalize them relative to each other within the group. The model is pushed toward completions that scored above the group average. This is structurally cleaner for sparse rewards because the comparison is always relative — "this action was better than these 3 alternatives" — rather than "this action should produce X reward in 3 more steps."

In plain terms: GRPO is simpler, fits on smaller hardware, and handles our reward structure better.

---

## Part 4: The Training Flow, Step by Step

Here is what actually happens when training runs, in order:

```
1. model loaded from HF Hub (Qwen2.5-0.5B-Instruct, 4-bit)
2. LoRA adapters attached to attention layers
3. build_dataset() called → seeds 2000 prompts by running random actions on the env server
4. GRPOTrainer initialized with reward_funcs=env_reward
5. For each training step:
   a. Take a batch of prompts from the dataset
   b. For each prompt, sample 4 completions (num_generations=4)
   c. Pass all completions to env_reward() → each one posts /step to the env server
   d. Normalize rewards within each group of 4
   e. Compute GRPO loss (push policy toward above-average completions)
   f. Backward pass → update LoRA adapter weights only
   g. Log to wandb every 10 steps
   h. Save checkpoint every 50 steps
6. Final adapter saved locally
7. (Manual step) Push to HF Hub
```

---

## Part 5: Known Issues in the Current train_grpo.py

Be honest about these before running. Three real problems:

### Problem 1: Environment State Drift During Batch Reward

`GRPOTrainer` generates all 4 completions for a prompt before calling `env_reward`. Inside `env_reward`, each call posts `/step` to the env server. This steps the environment 4 times sequentially. That means completion #1 is evaluated at env state T, completion #2 at state T+1, completion #3 at T+2, completion #4 at T+3. The rewards are not comparable because they're from different states.

**Fix (minimal):** In `env_reward`, call `/reset` before evaluating each completion. This resets the env to a fresh episode for each evaluation. You lose episode continuity but the reward comparison within the group becomes valid. Add this:

```python
def env_reward(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    rewards = []
    for prompt, completion in zip(prompts, completions):
        client.post("/reset", json={"dial": 0.0})   # add this line
        try:
            action = int(completion.strip().split()[0]) % 48
        ...
```

### Problem 2: obs_to_prompt Uses Dummy Obs

In `build_dataset()`, the actual observation from the env (`/state`) is not decoded into the prompt. A dummy `[0.0] * 22` array is used instead:

```python
obs_dummy = [0.0] * 22  # placeholder; real obs comes from /state — THIS IS THE BUG
prompt = make_prompt(obs_dummy, state["raga"], state["tala_position"])
```

The `/state` endpoint returns the state dict, but not the `obs` array directly. You need to call `/reset` to get a fresh obs, then use those values. Fix:

```python
reset_resp = client.post("/reset", json={"dial": 0.0}).json()
obs = reset_resp.get("obs", [0.0] * 22)  # server needs to return obs in reset response
prompt = make_prompt(obs, state["raga"], state["tala_position"])
```

Check `server.py` to confirm the reset endpoint returns `obs`. If it doesn't, add it.

### Problem 3: Notebook Format Required for Submission

The hackathon requires a `.ipynb` Colab notebook, not a `.py` script. Judges will click "Open in Colab" and run cells. The `.py` script approach is fine for local dev but is not the submission artifact.

You need to convert `train_grpo.py` into a notebook with clearly labeled cells. See Part 6 below.

---

## Part 6: The Actual Colab Notebook Setup

This is the cell-by-cell structure the notebook should have.

### Cell 1 — Install Dependencies
```python
# Install all required packages. Takes ~3 minutes on Colab.
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install trl==0.8.6 httpx wandb gymnasium numpy
```

Why `trl==0.8.6`: pin the version. GRPO API changed between versions. Pin prevents breakage when Colab updates packages.

### Cell 2 — Mount Drive + Set Secrets (Optional but Recommended)
```python
from google.colab import drive, userdata
drive.mount('/content/drive')

import os
os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")   # set in Colab Secrets
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")             # for pushing model at end
```

### Cell 3 — Start the OpenEnv Server

The environment server needs to be running before training starts. Two options:

**Option A (simplest for hackathon):** Deploy your `openenv_server/server.py` to a HF Space before the notebook runs, get the public URL, hardcode it.

```python
SERVER_URL = "https://your-username-jugalbandi-env.hf.space"  # replace with your HF Space URL
```

**Option B (self-contained in Colab):** Run the server in a background thread inside Colab.
```python
import threading
import subprocess

server_proc = subprocess.Popen(
    ["python", "openenv_server/server.py"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)
import time; time.sleep(5)  # wait for server to start
SERVER_URL = "http://localhost:7860"
```

Option A is more reliable for a demo. Option B works for a self-contained notebook but requires the server code to be present in Colab (clone your repo first).

### Cell 4 — Load Model
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-0.5B-Instruct",
    max_seq_length=512,
    load_in_4bit=True,
    token=os.environ["HF_TOKEN"],
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
print(model.print_trainable_parameters())  # should say ~4M trainable params
```

### Cell 5 — Verify Server is Reachable
```python
import httpx
client = httpx.Client(base_url=SERVER_URL, timeout=10.0)
resp = client.post("/reset", json={"dial": 0.0})
print(resp.json())  # should show {"obs": [...22 floats...], "raga": "yaman", ...}
```

If this cell fails, the server is not running. Fix the server first. Training will not work without it.

### Cell 6 — Prompt Template
```python
SYSTEM = (
    "You are an expert Indian classical musician composing in a raga. "
    "Given the current musical context, choose the next note action (0-47). "
    "Action encodes: note_semitone (0-11) + duration_index (0-3) * 12. "
    "Output ONLY the integer, nothing else."
)

NOTE_NAMES = [
    "ṉSa","ṉRe♭","ṉRe","ṉGa♭","ṉGa","ṉMa","ṉMa#","ṉPa","ṉDha♭","ṉDha","ṉNi♭","ṉNi",
    "Sa","Re♭","Re","Ga♭","Ga","Ma","Ma#","Pa","Dha♭","Dha","Ni♭","Ni",
]
DURATION_NAMES = {0: "sixteenth", 1: "eighth", 2: "quarter", 3: "half"}

def make_prompt(obs: list[float], raga: str, tala_pos: int) -> str:
    # Decode last 4 notes from obs[0:8]
    last_notes = []
    for i in range(4):
        note_idx = min(int(obs[i*2] * 11), 23)
        dur_idx  = min(int(obs[i*2+1] * 3), 3)
        last_notes.append(f"{NOTE_NAMES[note_idx]}({DURATION_NAMES[dur_idx]})")

    dial      = round(obs[15], 2)
    drought   = round(obs[16], 2)
    tension   = round(obs[14], 2)
    in_grace  = obs[21] > 0.5

    return (
        f"<|system|>\n{SYSTEM}\n"
        f"<|user|>\n"
        f"Active raga: {raga} (dial={dial}{'  ⚠ GRACE PERIOD — rules just changed' if in_grace else ''})\n"
        f"Tala position: beat {tala_pos}/16\n"
        f"Last 4 notes: {', '.join(last_notes)}\n"
        f"Human call tension: {tension:.2f} (how unresolved their phrase was)\n"
        f"Pakad drought: {drought:.2f} (0=just played a phrase, 1=very long since last phrase)\n"
        f"Choose action (0-47):\n"
        f"<|assistant|>\n"
    )
```

This is significantly better than the current dummy prompt. It gives the model human-readable note names, tells it about grace periods, and decodes all observation dimensions.

### Cell 7 — Reward Function
```python
import httpx

client = httpx.Client(base_url=SERVER_URL, timeout=10.0)

def env_reward(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    rewards = []
    for prompt, completion in zip(prompts, completions):
        # Reset env before each evaluation so states are comparable
        client.post("/reset", json={"dial": 0.0})
        try:
            action = int(completion.strip().split()[0]) % 48
        except (ValueError, IndexError):
            rewards.append(-1.0)
            continue
        try:
            res = client.post("/step", json={"action": action})
            data = res.json()
            reward = data["reward"]
        except Exception:
            reward = -0.5
        rewards.append(float(reward))
    return rewards
```

### Cell 8 — Build Dataset
```python
import random

def build_dataset(n: int = 1000) -> list[dict]:
    dataset = []
    resp = client.post("/reset", json={"dial": 0.0}).json()
    obs = resp.get("obs", [0.0] * 22)

    for i in range(n):
        state = client.get("/state").json()
        prompt = make_prompt(obs, state["raga"], state["tala_position"])
        dataset.append({"prompt": prompt})

        action = random.randint(0, 47)
        step_resp = client.post("/step", json={"action": action}).json()
        obs = step_resp.get("obs", [0.0] * 22)

        if step_resp.get("terminated"):
            resp = client.post("/reset", json={"dial": random.uniform(0, 1)}).json()
            obs = resp.get("obs", [0.0] * 22)

        if (i+1) % 100 == 0:
            print(f"  {i+1}/{n} prompts collected")

    return dataset

print("Building training dataset...")
train_dataset = build_dataset(1000)
print(f"Dataset size: {len(train_dataset)} prompts")
print(f"Sample prompt:\n{train_dataset[0]['prompt'][:400]}")
```

### Cell 9 — Training Config + Run
```python
import wandb
from trl import GRPOConfig, GRPOTrainer

RUN_NAME = "jugalbandi-grpo-v1"
STEPS = 300  # on T4: ~45-60 min. Increase if you have A100 credits.

wandb.init(project="jugalbandi", name=RUN_NAME)

config = GRPOConfig(
    output_dir=f"/content/drive/MyDrive/jugalbandi/{RUN_NAME}",
    num_train_epochs=1,
    max_steps=STEPS,
    per_device_train_batch_size=2,     # lower than default — env calls are slow
    gradient_accumulation_steps=8,    # effective batch = 16
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=50,
    report_to="wandb",
    num_generations=4,
    max_completion_length=4,           # action index is 1-2 tokens; 4 is generous
    temperature=0.9,                   # some exploration
    seed=42,
)

trainer = GRPOTrainer(
    model=model,
    args=config,
    reward_funcs=env_reward,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

print(f"Starting training: {STEPS} steps")
trainer.train()
print("Training complete.")
```

**What to watch on wandb:**
- `train/reward_mean` — should climb from negative to positive over 200+ steps
- `train/reward_std` — if this collapses to 0, the model has converged (possibly degenerate)
- If reward stays flat after 100 steps, check if the server is responding correctly

### Cell 10 — Save and Push to Hub
```python
from huggingface_hub import HfApi

SAVE_DIR = f"/content/drive/MyDrive/jugalbandi/{RUN_NAME}/final"

model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Saved to {SAVE_DIR}")

# Push to HF Hub
api = HfApi()
api.upload_folder(
    folder_path=SAVE_DIR,
    repo_id="your-username/jugalbandi-qwen-grpo",  # replace with your HF username
    repo_type="model",
    token=os.environ["HF_TOKEN"],
)
print("Pushed to HuggingFace Hub.")
```

---

## Part 7: Reading the WandB Dashboard

After Cell 9 starts, go to wandb.ai → your project → jugalbandi-grpo-v1.

**Reward curve:** Should look like a noisy climb from ~-1.0 at step 0 toward +2.0 to +5.0 by step 300. It won't be smooth — GRPO reward curves are noisy. Look at the 50-step moving average.

**What good looks like at step 300:**
- Average episode reward: +2.0 to +4.0
- Reward still trending up (not plateaued)

**What bad looks like:**
- Reward stuck at -1.0 forever → server not responding, action parse failing, all actions hitting forbidden notes
- Reward at +0.2 flatline → degenerate policy (model always outputs same number)

**Quick debug if reward stays at -1.0:**
In a separate cell during training, manually call env_reward with a known good action:
```python
test = env_reward(["dummy prompt"], ["7"])  # Sa = action 7
print(test)  # should be ~0.2 (valid_note base reward)
```

---

## Part 8: The Inference Check (After Training)

After training, verify the model actually learned something before claiming results:

```python
from unsloth import FastLanguageModel

# Load fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(SAVE_DIR, load_in_4bit=True)
FastLanguageModel.for_inference(model)

# Test on a fresh env state
client.post("/reset", json={"dial": 0.0})
state = client.get("/state").json()
obs = [0.0] * 22  # fresh episode, all zeros is fine for a quick test
prompt = make_prompt(obs, "yaman", 0)

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=4, temperature=0.1)
decoded = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"Model output: {decoded}")
action = int(decoded.strip().split()[0]) % 48
note = action % 12
print(f"Action {action} → note semitone {note}")
# For Yaman: valid notes are {0,2,4,6,7,9,11}. Invalid: {1,3,5,8,10}
print(f"Valid in Yaman: {note in {0,2,4,6,7,9,11}}")
```

If the model consistently outputs valid Yaman notes, training worked. If it outputs random numbers, something went wrong.

---

## Part 9: Approximate Timings on Different Hardware

| Hardware | Steps | Time | Notes |
|---|---|---|---|
| Colab T4 (free) | 300 | ~60 min | May timeout after 90 min; save frequently |
| Colab A100 (paid) | 500 | ~30 min | Recommended for the actual hackathon run |
| Colab L4 (paid) | 500 | ~45 min | Cheaper than A100, still fast |
| Local RTX 3090 | 500 | ~25 min | If you have it |

Save checkpoints to Google Drive (not `/content`), which is wiped when Colab disconnects.

---

## What to Actually Do Right Now

Priority order:

1. Verify the OpenEnv server `/reset` response includes the `obs` array — if it doesn't, add it to `server.py`
2. Convert `train_grpo.py` into a `.ipynb` using the cell structure above
3. Run Cells 1-5 only to confirm server connectivity before doing a full training run
4. Do a short 50-step test run to confirm wandb is logging and rewards are non-degenerate
5. Do the full 300-500 step run with A100 credits the day before submission

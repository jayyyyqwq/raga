# FastAPI server wrapping JugalbandiEnv.
# This is the OpenEnv-compatible HTTP interface — clients hit these endpoints.
# The UI's env_client.js talks to this. The GRPO training loop talks to this.
#
# WHY FastAPI: async, auto-generates OpenAPI docs at /docs, pydantic validation free.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from raaga_env.jugalbandi_env import JugalbandiEnv
from raaga_env.prompting import Arm, StepFeedback, note_name, parse_action, render_prompt

# torch/transformers/peft are only needed for /infer, and only once a trained
# adapter exists (see docs/report-midway.md §9.1) — importing them eagerly
# would make the whole server (and every other endpoint) depend on a GPU-era
# dependency stack that most of this project's lifetime doesn't have. Loaded
# lazily inside _load_inference_model() instead.

app = FastAPI(
    title="Jugalbandi OpenEnv Server",
    description="RL environment: raga grammar adherence under dynamic constraint drift.",
    version="1.0.0",
)

# Allow the UI (different port in dev, same origin in prod) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One env instance per process.
# In prod (HF Spaces), each user session should get its own process via workers.
env = JugalbandiEnv(initial_dial=0.0, episode_length=64)

# Claim B's feedback channel (updatedplan.md Phase 2.3): the outcome of the
# most recent /step call, shown to the next /infer prompt regardless of arm.
# None at episode start — there is no "last action" yet. Rule-agnostic by
# construction (StepFeedback never says *why*), so this is safe to show
# ORACLE, DIAL, and HIDDEN alike.
_last_step_feedback: StepFeedback | None = None


# ── Request / Response models ──────────────────────────────────────────

class StepRequest(BaseModel):
    action: int = Field(..., ge=0, le=95)

class ResetRequest(BaseModel):
    seed: int | None = None
    dial: float = Field(0.0, ge=0.0, le=1.0)

class DialRequest(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0)

class CallRequest(BaseModel):
    notes: list[int] = Field(..., min_length=1, max_length=4)

class InferRequest(BaseModel):
    max_new_tokens: int = Field(4, ge=1, le=16)
    temperature: float = Field(0.1, ge=0.0, le=2.0)


# ── Inference (trained policy) ───────────────────────────────────────────
# Prompt format comes from raaga_env.prompting — the single source of truth
# for every prompt-producing call site (updatedplan.md Phase 2.1). The
# policy only learned to respond to whatever arm it was trained under, so
# this must match training/train_grpo.py's --arm for whatever adapter
# JUGALBANDI_ADAPTER_REPO points at.

INFER_ARM = Arm(os.environ.get("JUGALBANDI_INFER_ARM", "oracle"))

_inference_cache: dict = {}


def _load_inference_model():
    """Lazily load base model + LoRA adapter. Cached after first call."""
    if _inference_cache:
        return _inference_cache["model"], _inference_cache["tokenizer"]

    adapter_repo = os.environ.get("JUGALBANDI_ADAPTER_REPO")
    if not adapter_repo:
        raise HTTPException(
            503,
            "No trained adapter configured. Set JUGALBANDI_ADAPTER_REPO to the "
            "HF Hub repo id produced by training/train_grpo.ipynb (see docs/report-midway.md §9).",
        )

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as exc:
        raise HTTPException(
            503,
            f"Inference dependencies not installed ({exc}). "
            "pip install torch transformers peft.",
        )

    base_repo = os.environ.get("JUGALBANDI_BASE_MODEL", "unsloth/Qwen2.5-0.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(base_repo)
    base_model = AutoModelForCausalLM.from_pretrained(base_repo, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base_model, adapter_repo)
    model.eval()

    _inference_cache["model"] = model
    _inference_cache["tokenizer"] = tokenizer
    return model, tokenizer


# ── Endpoints ──────────────────────────────────────────────────────────

@app.post("/reset")
async def reset(req: ResetRequest):
    global _last_step_feedback
    env.set_dial(req.dial)
    obs, info = env.reset(seed=req.seed)
    _last_step_feedback = None  # new episode: no "last action" yet
    return {
        "observation": obs.tolist(),
        "info": info,
        "active_raga": env.drift.active_raga_name,
    }


@app.post("/step")
async def step(req: StepRequest):
    global _last_step_feedback
    if env.step_count >= env.episode_length:
        raise HTTPException(400, "Episode ended. Call /reset.")
    obs, reward, terminated, truncated, info = env.step(req.action)
    _last_step_feedback = StepFeedback(
        note_name=note_name(info["note"], info["duration"]), reward=float(reward)
    )
    info_clean = {
        k: (v.tolist() if hasattr(v, "tolist") else v)
        for k, v in info.items()
        if k != "reward_breakdown"
    }
    return {
        "observation": obs.tolist(),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": info_clean,
        "reward_breakdown": info.get("reward_breakdown", {}),
    }


@app.post("/set_dial")
async def set_dial(req: DialRequest):
    switched = env.set_dial(req.value)
    return {
        "dial": env.drift.dial,
        "active_raga": env.drift.active_raga_name,
        "switched": switched,
        "in_grace_period": env.drift.in_grace_period,
    }


@app.post("/call")
async def submit_call(req: CallRequest):
    """Human submits a 4-note call phrase. Agent will respond on next steps."""
    env.set_call(req.notes)
    return {
        "call_phrase": env.call_phrase,
        "call_tension": env.call_tension,
        "active_raga": env.drift.active_raga_name,
    }


@app.get("/state")
async def state():
    return {
        "raga": env.drift.active_raga_name,
        "dial": env.drift.dial,
        "step_count": env.step_count,
        "episode_length": env.episode_length,
        "episode_reward": env.episode_reward,
        "pakad_completions": env.pakad_completions,
        "forbidden_notes": env.forbidden_count,
        "tala_position": env.tala_position,
        "in_grace_period": env.drift.in_grace_period,
        "call_phrase": env.call_phrase,
    }


@app.post("/infer")
async def infer(req: InferRequest):
    """
    Runs the trained policy on the environment's *current* observation and
    returns the action it picks — this does not step the environment.
    Callers apply the action with a separate POST /step.
    """
    model, tokenizer = _load_inference_model()  # raises a clean 503 if deps/adapter are missing
    import torch  # safe now — _load_inference_model() already proved this import works

    obs = env._get_obs().tolist()
    prompt = render_prompt(
        obs,
        arm=INFER_ARM,
        tala_pos=env.tala_position,
        feedback=_last_step_feedback,
        raga=env.drift.active_raga_name if INFER_ARM is Arm.ORACLE else None,
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            do_sample=req.temperature > 0.0,
        )
    decoded = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )

    # No silent modulo-clamping into range (Phase 1.2): a model emitting an
    # out-of-range or unparseable action must be recorded as invalid, not
    # quietly turned into a different, legal action.
    action = parse_action(decoded)
    if action is None:
        raise HTTPException(
            502, f"Model returned unparseable or out-of-range output: {decoded!r}"
        )

    return {"action": action, "raw_output": decoded}


@app.get("/health")
async def health():
    return {"status": "ok", "env": "JugalbandiEnv"}


# ── Serve the UI itself ───────────────────────────────────────────────
# One process, one port (docs/imrpovedui.md's "serving" decision) — replaces
# the old setup of this API server plus a separate `python -m http.server`
# for ui/, which needed the CORS "*" above and a second terminal window.
# Mounted last and at "/" so every @app.* route above still matches first;
# StaticFiles only sees requests nothing else claimed, and html=True serves
# ui/index.html for "/" itself.
UI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui")
if os.path.isdir(UI_DIR):
    app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)

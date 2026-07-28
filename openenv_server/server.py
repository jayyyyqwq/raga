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
from pydantic import BaseModel, Field
import uvicorn

from raaga_env.jugalbandi_env import JugalbandiEnv

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


# ── Request / Response models ──────────────────────────────────────────

class StepRequest(BaseModel):
    action: int = Field(..., ge=0, le=47)

class ResetRequest(BaseModel):
    seed: int | None = None
    dial: float = Field(0.0, ge=0.0, le=1.0)

class DialRequest(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0)

class CallRequest(BaseModel):
    notes: list[int] = Field(..., min_length=1, max_length=4)


# ── Endpoints ──────────────────────────────────────────────────────────

@app.post("/reset")
async def reset(req: ResetRequest):
    env.set_dial(req.dial)
    obs, info = env.reset(seed=req.seed)
    return {
        "observation": obs.tolist(),
        "info": info,
        "active_raga": env.drift.active_raga_name,
    }


@app.post("/step")
async def step(req: StepRequest):
    if env.step_count >= env.episode_length:
        raise HTTPException(400, "Episode ended. Call /reset.")
    obs, reward, terminated, truncated, info = env.step(req.action)
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


@app.get("/health")
async def health():
    return {"status": "ok", "env": "JugalbandiEnv"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)

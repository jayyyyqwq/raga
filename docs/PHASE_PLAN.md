# Jugalbandi — 36h Pre-Hackathon Phase Plan
> Hackathon starts 2026-04-25. Today: 2026-04-24.
> Goal: env working end-to-end + baseline training run BEFORE we arrive.

---

## Hard Gate (Hour 20)
> Track A must run end-to-end with a random policy.
> If it doesn't, ALL remaining time goes to Track A. Track B freezes wherever it is.

---

## Phase 0 — Setup (Hour 0–1)
- [ ] `python -m venv venv && source venv/Scripts/activate`
- [ ] `pip install gymnasium numpy fastapi uvicorn pydantic httpx pytest`
- [ ] `pytest raaga_env/tests/ -v` — all tests must pass before touching anything else
- [ ] Verify server starts: `python openenv_server/server.py` → `curl http://localhost:7860/health`

## Phase 1 — Env Validation (Hour 1–4)
Owner: Jay
- [ ] Fix any test failures from Phase 0
- [ ] Run a 100-step random rollout, print reward breakdown every 10 steps
- [ ] Verify: forbidden notes give -2.0, pakads give +1.0, grace period reduces penalty
- [ ] Run `JugalbandiEnv` specifically: set_dial(0.75) mid-episode, verify raga switches
- [ ] Script: `python -c "from raaga_env import JugalbandiEnv; env = JugalbandiEnv(); obs,_ = env.reset(); print(obs)"`

## Phase 2 — Server Integration (Hour 4–6)
Owner: Jay
- [ ] POST /reset, /step, /set_dial, /call all return valid JSON
- [ ] Run `ui/env_client.js` stubs against the live server (open index.html, check console)
- [ ] Verify CORS headers allow browser requests
- [ ] Verify Dockerfile builds: `docker build -t jugalbandi .`

## Phase 3 — Track B UI (Hour 6–14)
Owner: Teammate
- [ ] `sitar.js`: 7 strings render on mobile, touch triggers pluck sound + callback
- [ ] `tabla.js`: beat clock runs at 80bpm, 16 dots animate in sync
- [ ] `app.js`: Start → human plays 4 notes → AI responds (placeholder action=4) → loop
- [ ] `raga-dial`: drag updates active raga label, calls /set_dial
- [ ] Test on actual phone (Chrome mobile, not just DevTools)
- [ ] Reward log shows per-component breakdown after each AI step

## Phase 4 — Baseline Training Run (Hour 14–28)
Owner: Jay
- [ ] Open `training/train_grpo.py` in Colab (free T4)
- [ ] Run 50 steps with `--batch 2` to verify no crashes
- [ ] Run 200 steps, save reward curve screenshot → `assets/reward_curves/baseline_200.png`
- [ ] Compare: random policy average reward vs trained policy average reward
- [ ] This is your 20% judging criterion evidence — DO NOT skip

## Phase 5 — Wire UI to Real Policy (Hour 28–34)
Owner: Both
- [ ] Replace placeholder `action=4` in `app.js:runAIResponse()` with call to trained model
- [ ] Two options:
  - (a) Serve trained model from HF Inference API → UI calls that endpoint
  - (b) Add `/agent_step` endpoint to FastAPI server that loads the checkpoint locally
- [ ] Verify: played sitar string → AI responds with trained-policy note → tabla in sync

## Phase 6 — Polish + Submission Prep (Hour 34–36)
Owner: Both
- [ ] `README.md`: motivation, env description, results, links
- [ ] Reward curve plots embedded in README
- [ ] openenv.yaml valid: `python -c "import yaml; yaml.safe_load(open('openenv.yaml'))"`
- [ ] Record <2min screen capture showing the jugalbandi demo (this is your video)
- [ ] List of remaining tasks for onsite (HF Space deploy, HF blog post, big training run)

---

## Onsite Day (2026-04-25/26) — With $200 HF Credits
1. Deploy env server to HF Spaces (Docker space, 7860)
2. Point UI's `window.ENV_SERVER_URL` to the Space URL
3. Run full training: 500+ steps on A100, log to wandb
4. Screenshot wandb reward curves → assets/reward_curves/
5. Write HF blog post (use the storytelling angle: "Jugalbandi as schema-drift training")
6. Submit Space URL

---

## Risk Register
| Risk | Probability | Mitigation |
|---|---|---|
| Tests fail in Phase 0 | Low | Fix reward.py first; it's pure functions, easy to debug |
| GRPO trainer crashes on Colab | Medium | Pin `trl==0.8.6`; use `--batch 2` for first run |
| UI audio doesn't work on iOS | Medium | iOS requires user gesture before AudioContext; start-btn already handles this |
| Trained model too slow for live inference | Low | Cache last 4 AI responses, stream them |
| HF Space cold start too slow | Low | Keep env server warm with a /health ping |

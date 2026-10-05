# Improved UI — Jugalbandi Stage

Status: **plan only, no code.** Personal project, runs locally only (`run_demo.bat`). Audio sourcing details: [`AUDIO_SOURCES.md`](AUDIO_SOURCES.md).

**Goal:** an interactive jugalbandi where you and an RL-trained AI trade lines of Hindustani classical music. You start with a recorded intro, the AI answers with a stronger version, you reply to the AI, it replies to you. It should be obvious that classical music is the medium and that RL taught the AI to play inside the raga.

---

## 1. Why the current UI is weak

| Problem | Where | Effect |
|---|---|---|
| Looks like a debug panel, not a performance | `ui/index.html`, `ui/styles.css` | Nothing says "two musicians are trading phrases" |
| No sense of turns | `ui/app.js` state machine is invisible | Can't tell who is playing, or when |
| No visual memory | notes vanish after a 300 ms flash | Can't see the AI's answer relate to the call |
| Sound is a toy | `Tone.PluckSynth` + `MembraneSynth` (`ui/sitar.js`, `ui/tabla.js`) | Ringtone, not sitar. No tanpura drone |
| Cold start | `onUserPluck` requires 4 clicks on thin lines | Nothing happens until you work out the UI |
| Notes play as HTTP replies arrive | `runAIResponse` awaits `/infer` → `/step` per note | Rhythm set by network latency, not tala |
| Reward is a single number | `rewardDisplay` | Says nothing about why a line was good |

---

## 2. Hard truths from the code

### 2.1 The model can't hear your line — CRITICAL, and now on the critical path
- `raaga_env/prompting.py::_context_lines` renders only `Human call tension: 0.xx`. The call notes (`obs[8:12]`) never reach the prompt. The LLM only sees the prompt.
- `training/train_grpo.py` never calls `env.set_call(...)`; every training step had `call_phrase = [0,0,0,0]`, `call_tension = 0.0`.
- The jugalbandi reward (`reward.py:126-134`) therefore never trained a call-response skill. Worse, `[0,0,0,0]` is truthy, so the "contrary motion" term fired against a phantom all-Sa call.
- The current adapter (`jugalbandi-grpo-hidden-v1`) is a **solo raga improviser with drift adaptation**. It does not answer anyone.

**Why this now matters more:** the new core loop (§4) is literally "AI answers *your* line". With the current adapter, line 2 would be a solo that ignores line 1. So **Option A (§7) moves from "later" to the critical path.** The current adapter is fine to build and test the UI against, nothing more.

### 2.2 Encoding mismatches
- UI sends calls as swara `0–11`; env pitch is absolute `0–23` (madhya = `12–23`). Calls land an octave low.
- `set_call` tension = `abs(last - vadi)` mixes absolute pitch and swara.
- `set_call` keeps only 4 notes; new lines are 8 notes (§4.3).
- UI does 4 AI notes per turn; env `CALL_EVERY = 8`.

### 2.3 Model quality
- First run: 300 GRPO steps, return **−46.12**, 0/64 forbidden notes. Legal, not yet shown to be musical. The UI will expose that honestly; the drone, meend and good samples (§5) help it sound like music without changing a single note.

---

## 3. The vision

A stage. Two performers: **you** (sitar) and the **AI** (bansuri). A tanpura drones underneath, tabla keeps teentaal. You click once: your recorded intro plays and draws itself across the stage. The AI visibly listens, then answers: same raga, related shape, pushed a step higher. You answer the AI. It answers you. Each exchange climbs, and every line shows what the reward function thought of it.

---

## 4. The turn loop (core interaction)

### 4.1 Script

| Line | Who | What happens | Source of notes |
|---|---|---|---|
| 0 | — | Stage idle, tanpura drone fades in. One big button: **▶ Play intro** | — |
| 1 | You (recorded) | Intro plays on sitar samples, saffron notes draw across the swara river | Pre-composed note list (§4.2), rendered with sampled sitar |
| 2 | AI | "Listening…" (line 1 flies into the AI orb), then AI's **improved** line plays on bansuri, violet | Trained model via `/respond`, conditioned on line 1 |
| 3 | You (live) | Your turn: play a reply on keyboard / on-screen strings, tala keeps running, notes snap to the beat. Or pick one of 3 suggested reply lines | Live input, quantised (§4.4) |
| 4 | AI | Answers line 3, one step higher on the escalation ladder | Model, conditioned on line 3 |
| 5… | alternate | Continue until the episode budget runs out (§4.3) | — |
| End | — | Finale: whole river as one picture, per-line reward chips, total | — |

### 4.2 The recorded intro

"Recorded" = a **pre-composed phrase stored as note data** (pitch + duration + beat), played through real instrument samples. Not an audio clip.

Why note data and not an audio recording: the model needs the notes as input. A sitar audio clip from the internet would sound nice but the AI couldn't read it without a pitch-tracker, and any pitch-tracker error would show up as the AI "mishearing".

Proposed intros (absolute pitches, Sa = 12 = C4):

| Raga | Notes | Swaras | Why |
|---|---|---|---|
| Yaman | `11 14 16 18 16 14 12 12` | ṉNi Re Ga Ma# Ga Re Sa Sa | Contains two Yaman pakads (`ṉNi Re Ga`, `Ga Ma# Ga Re Sa`) — instantly identifies the raga |
| Bhairav | `16 17 20 19 16 17 13 12` | Ga Ma dha Pa Ga Ma re Sa | Ends on the `Ga Ma re Sa` pakad, hits komal dha (vadi) |

Durations/beats get set so the intro lands on sam (beat 1) at the end. Stored in `ui/presets.js`.

### 4.3 Line lengths and the episode budget

- Env episode = 64 steps; only AI notes consume steps (your lines go in via `/call`, not `/step`).
- Every line is 8 notes, matching `CALL_EVERY = 8`. That gives **8 AI lines → 16 lines total** (you 8, AI 8) per performance.
- `set_call` must accept the full 8-note line (currently truncates to 4).

### 4.4 Your live reply (line 3, 5, …)

- **Keys:** `S R G M P D N` play the active raga's swaras; hold `Shift` for the upper octave, `Z X C V B N M` for mandra (lower) octave. Labels change with the raga (Ma# in Yaman, komal re/dha in Bhairav). Forbidden swaras simply aren't on the keyboard.
- **On-screen:** the same 7 swaras as wide plucked strings at the bottom of the stage, for mouse/touch.
- **Quantising:** keypress timestamps snap to the nearest 16th note at the tabla's tempo, so your line is in time even if you aren't.
- **Turn window:** your turn lasts 2 tala cycles max (32 beats) or until you've played 8 notes; a countdown ring on the tala wheel shows time left. Fewer than 8 notes is fine — send what you played.
- **"Suggest" fallback:** 3 ghost reply lines generated from the AI's last line (rule-based: echo its ending, invert its contour, resolve to Sa). Click one to play it as your reply. For when you just want to watch the conversation.

### 4.5 Escalation ladder (what "takes it up" means each AI line)

| AI line | Transform | Label on stage | Real or presentation? |
|---|---|---|---|
| 1 | Base answer | `UTTAR` (answer) | Real |
| 2 | Echo + extend — AI line built off your last 2–3 notes | `BADHAT` (development) | Real once Option A lands |
| 3 | **Dugun** — played at 2× laya | `LAYA ×2` | Presentation (same notes, faster) |
| 4 | **Taar saptak** — voiced an octave up | `TAAR SAPTAK` | Presentation (same swaras, +12) |
| 5–8 | Combined: dugun + taar + longer density | `TIHAI ↑` | Mixed |

The stage always shows the label so the climb is visible, and presentation transforms are marked with a small "voicing" tag so there's no confusion about what the model chose vs what the UI did.

### 4.6 "The AI understood" — how it's shown

1. **Listen animation:** your notes lift off the river and stream into the AI orb.
2. **Echo arcs:** after the AI plays, gold arcs link your notes to AI notes where a computed relation exists (same motif, transposed motif, mirrored contour, shared ending). Computed from the actual notes in `ui/analysis/echo.js` — no relation, no arc.
3. **Caption:** one line from the same analysis, e.g. *"picked up your Ga–Ma#, answered higher, resolved to Sa on sam."*
4. **Reward chips** for the AI line from `reward_breakdown`: `pakad ✓`, `vadi on sam ✓`, `answered your tension ✓`, `forbidden ✕`. This is the RL made visible — "the reward function liked this, here's why".

---

## 5. Audio — sourced, free

Full source list, licences and attribution in [`AUDIO_SOURCES.md`](AUDIO_SOURCES.md). Summary:

| Role | Instrument | Source | Licence |
|---|---|---|---|
| You | **Sitar** | FluidR3_GM `sitar-mp3` (per-note mp3, C3–C6 verified) | CC-BY 3.0 |
| AI | **Bansuri** (GM `flute`) — alt: **Shehnai** (`shanai`) | FluidR3_GM | CC-BY 3.0 |
| Drone | **Tanpura**, tuned to C | Freesound #35476 `tanpura2.wav` (marvman), 5.8 s, loop with crossfade | CC0 |
| Rhythm | **Tabla** | Option 1: Freesound `mmiron` tabla-bols pack (na, tun, te, re, ke, tas) — licence to verify on download. Option 2: keep current synth tabla | TBD / none |

**Combo plan:** sitar (plucked) vs bansuri (sustained) is the strongest contrast; your lines and AI lines are distinguishable by ear with eyes closed. **Single-instrument fallback** if the combo sounds bad: sitar for both, AI line panned right and voiced with more reverb.

Playback details:
- Sa = C4 (261.63 Hz) to match `ui/sitar.js` and the tanpura's C tuning.
- Samples cover C3–C6: env range 0–23 is C3–B4, taar saptak escalation reaches B5. Covered.
- **Meend:** for intervals ≤ 2 semitones, glide pitch into the next note (Tone.js `Sampler` + `detune` ramp). Presentation only.
- **Scheduling:** the AI line is fully generated *before* playback, then scheduled on `Tone.Transport` from the next beat. Latency hides behind the listen animation.
- Samples get downloaded once into `ui/samples/` (dirs already exist, empty) — no CDN at runtime.

---

## 6. Visual design

Look: **traditional modern** — deep indigo ground, gold and saffron line work, miniature-painting palette, Devanagari swara labels beside Latin (सा रे ग म प ध नि). Yaman = indigo dusk, Bhairav = dawn amber.

```
┌──────────────────────────────────────────────────────────────┐
│  JUGALBANDI          Raag Yaman · evening · teentaal 80bpm   │
│                                                              │
│   ◉ YOU · sitar          ╭─ tala mandala ─╮    AI · bansuri ◉ │
│   (saffron)              │ 16 beats, sam ✕ │       (violet)   │
│                          ╰────────────────╯                  │
│   escalation: UTTAR › BADHAT › LAYA×2 › TAAR SAPTAK › TIHAI  │
│  ═════════════════ SWARA RIVER (scrolling) ════════════════  │
│  नि Ni ─                       ●━━●                          │
│  ध Dha ─        ●━━━●        ╱      ╲●                       │
│  ग Ga ─   ●━━●╱       ╲● ⌒⌒●     (gold echo arcs)            │
│  सा Sa ─ ●                                                   │
│  ═════════════════════════════════════════════════════════  │
│  Line 4 · AI   "picked up your Ga–Ma#, answered higher"     │
│  pakad ✓  vadi on sam ✓  answered tension ✓  forbidden 0    │
│                                                              │
│      [ S ][ R ][ G ][ M# ][ P ][ D ][ N ]   ← your strings   │
└──────────────────────────────────────────────────────────────┘
```

Pieces:
1. **Swara river** — scrolling pitch ribbon, notes joined by glowing meend curves. Persists for the whole performance.
2. **Tala mandala** — 16-beat wheel, sweeping playhead, sam ✕, khali ○, turn-countdown ring on your turn.
3. **Performer orbs** — breathe with the drone; whoever plays glows. AI orb "inhales" your notes during the listen phase.
4. **Escalation strip** — shows the ladder from §4.5 with the current step lit.
5. **Echo arcs + caption + reward chips** — §4.6.
6. **Drift** — raga dial on the side; crossing 0.5 morphs lane labels, dims forbidden lanes, shifts the palette dusk ↔ dawn, 3-beat amber shimmer for the grace period.

---

## 7. Making "the AI answers you" true

> **2026-10 update:** the code for this section's Option A is done — see
> [`RETRAIN_PLAN.md`](RETRAIN_PLAN.md) for exactly what changed. The GPU retrain itself hasn't run
> yet (needs Colab). Nothing below is stale, this just means "critical path" now has a tracked plan
> instead of being an open problem.

| Option | What changes | Status |
|---|---|---|
| **A. Line in the prompt + retrain** | `prompting.py` renders `Your partner just played: Ni Re Ga Ma# Ga Re Sa Sa`; `train_grpo.py` injects a raga-valid 8-note call every `CALL_EVERY` steps (sampled from pakads + random valid walks); fix tension calc and the phantom-call bug; `set_call` takes 8 notes | **Critical path** — required for line 2+ to be a real answer |
| **B. Current adapter, honest captions** | Nothing on the model side; captions say "improvises in the raga after your line"; echo arcs only where they genuinely occur | Development stand-in only |
| C. Fake it (transpose your line, call it the AI's) | — | No. Defeats the whole point |

Note on HIDDEN: a call built only from raga-valid notes hints at the active raga. That's what a real accompanist uses and is fine, but it changes what "HIDDEN" means — state it in `FIRST_TRAINING_RUN.md`'s successor write-up.

Reward change worth adding with A: a **call-relatedness** term (e.g. shares a 2–3 note motif or contour with the call, without copying it verbatim) so the model is actually paid for answering, not just for being in the raga. Must be capped and anti-copy, otherwise it learns to parrot.

---

## 8. Tech

- **Rendering:** PixiJS **v7** (pixi.js@7.4.3, pinned, jsdelivr CDN) — not v8. v8.x (0.0 through at
  least 8.22.0) has a confirmed WebGL-renderer crash (`pixijs/pixijs#12048`) triggered by mutating a
  Container's transform every tick with a Graphics anywhere beneath it — exactly the "breathing
  orb" pattern this stage needs — reproduced independent of this app down to a single Graphics
  circle, and not fixed by any workaround tried (extra wrapper Container, mutating the Graphics'
  own transform directly, different GL backends). v7 doesn't have v8's RenderGroup system and
  doesn't hit this at all; same-frame visual quality for this stage's needs. three.js unnecessary
  for a 2D stage.
- **Audio:** Tone.js, pinned on jsdelivr (replace unpinned skypack). `Tone.Sampler` for sitar/bansuri, `Tone.Player` loop for tanpura.
- **Serving:** FastAPI serves `ui/` itself → one process, one port, drop CORS `*` and the separate `http.server` in `run_demo.bat`.
- **New endpoint `/respond`:** body = `{call: [8 notes], n: 8}` → sets call, runs n × (infer + step) server-side, returns `{notes, durations, rewards, breakdowns, raga, in_grace}`. One round trip per AI line instead of 16.
- **Files (proposed):**
  - `ui/stage/river.js`, `ui/stage/talaMandala.js`, `ui/stage/orbs.js`, `ui/stage/escalation.js`
  - `ui/audio/drone.js`, `ui/audio/voices.js` (sitar/bansuri samplers + meend), `ui/audio/clock.js`
  - `ui/input/keyboard.js` (keys + on-screen strings + quantiser)
  - `ui/analysis/echo.js` (pure functions: motif/contour/ending matching), `ui/analysis/suggest.js` (3 reply suggestions)
  - `ui/presets.js` (intros per raga), `ui/turns.js` (turn-loop state machine, replaces `app.js`)

---

## 9. Build phases (next steps)

| # | Deliverable | Depends on | Status |
|---|---|---|---|
| 0 | Download + licence-check samples per `AUDIO_SOURCES.md`; write `ATTRIBUTION.md` | — | **Done** (2026-10-05) |
| 1 | Audio engine: drone, sitar + bansuri samplers, transport clock, meend. Play the Yaman intro end-to-end | 0 | **Done** — meend shipped as a visual curve only, not a synthesised pitch glide (see §9.1) |
| 2 | Env/server fixes: absolute-pitch calls, 8-note `set_call`, tension calc, phantom-call bug, `/respond`, FastAPI serves `ui/` | — | **Partly done, partly descoped** — phantom-call bug and tension calc were already fixed in the ML retrain pass (`RETRAIN_PLAN.md`); FastAPI now serves `ui/`; call stayed 4 notes and there's no `/respond` endpoint — see §9.1 |
| 3 | Turn loop with current adapter (Option B): intro → AI line → your live line → AI line | 1, 2 | **Done** |
| 4 | **Option A:** call in prompt + call injection in training + relatedness reward; retrain on Colab; drop new adapter into `checkpoints/final` | 2 | **Code done** (`RETRAIN_PLAN.md`), **GPU retrain not run** — needs Colab |
| 5 | Stage visuals: river, tala mandala, orbs, escalation strip, palettes | 3 | **Done** |
| 6 | Echo arcs, captions, reward chips, suggest-reply | 3, 5 | **Echo arcs/captions/chips done. Suggest-reply (ghost lines) descoped** — not built, see §9.1 |
| 7 | Drift visuals + finale screen | 5 | **Partly done** — dial/raga switch repaints the palette and river lane labels live; no dedicated finale screen or grace-period shimmer animation yet |
| 8 | Polish: full-screen laptop sizing, CPU latency, before/after toggle (random-valid vs trained) | all | **Not done** — no before/after toggle mode; basic responsive CSS only |

Phases 1, 2 and 4 ran in parallel, as planned.

### 9.1 What shipped differently than planned, and why

| Planned | Shipped | Why |
|---|---|---|
| Call phrase extended to 8 notes | **Stayed at 4** (`JugalbandiEnv.set_call`'s existing contract) | Extending it means growing the 22-dim observation space (`obs[8:12]` only has room for 4 call-note slots) — that's an env/model contract change, not a UI change, and would invalidate the ML retrain work that just landed. Deferred; line *length* escalation is simulated differently (next row) |
| Lines grow 4→8→16 notes as part of the escalation ladder | AI line is a **fixed 8 notes per turn** (matches `CALL_EVERY`); the "growth" step (`BADHAT`) is relabelled as an echo/development emphasis rather than literal length growth | Keeps the ladder's visible 5-step climb without an env change. `BADHAT`'s distinctiveness from `UTTAR` is weaker than planned until the retrained model (Phase 4) actually conditions on the call — tracked, not silently dropped |
| New `/respond` server endpoint (one round trip per AI line) | **No server change** — client prefetches all 8 notes via sequential `/infer` + `/step` calls *before* playing any of them, then schedules playback on the next beat | Same latency-hiding effect (§4.3's goal) without touching `server.py`'s contract; simpler, no new endpoint to version |
| Meend = synthesised pitch glide between samples | **Visual-only** — a curved glow line on the river; audio plays the two notes back to back, no pitch-bend | Tone.Sampler has no clean per-note portamento API; a real glide would need custom playback-rate automation. Visual meend delivers most of the "feels like a glide" effect for a fraction of the risk |
| Suggest-reply (3 ghost lines) if you don't want to play | **Not built** — if your turn times out with 0 notes played, your previous call is resubmitted unchanged | Scope cut to ship the core loop; this is the first thing to add next if "watch only" mode matters |
| Input quantised to the nearest 16th note | **Not built** — notes play the instant you press a key/button | Scope cut; doesn't block the core experience, worth adding if live input feels loose against the tabla |
| Sitar / bansuri **or shehnai** | **Sitar (you) + flute/bansuri (AI)** shipped; shehnai not downloaded | See §11 — shehnai stays a one-line swap in `scripts/fetch_samples.py`'s `INSTRUMENTS` dict if you want to A/B it |

### What to test
- `echo.js`: detects exact repeat, transposition, inversion, shared ending; no arcs on random phrases.
- Quantiser: jittered keypress timings snap to the right 16th; turn ends at 32 beats or 8 notes.
- Encoding: UI swara ↔ env absolute pitch round-trip; tension calc after fix; 8-note `set_call`.
- `/respond`: returns exactly n notes, errors after episode end, matches sequential `/step` for the same seed.
- Prompting: HIDDEN arm with call notes still has no raga name / dial (`test_prompting.py`).
- Training: call injected every `CALL_EVERY` steps; relatedness reward is capped and pays 0 for verbatim copy.
- Scheduler: AI line starts on a beat boundary regardless of response latency.

---

## 10. Decisions log

| Date | Topic | Decision |
|---|---|---|
| 2026-10-04 | AI listening | Start on current adapter (B) for UI dev; **A is now critical path** because the loop is call-and-answer |
| 2026-10-04 | Escalation | All three (longer, dugun, taar saptak), shown on an escalation strip |
| 2026-10-04 | Look | Traditional modern |
| 2026-10-04 | Deploy | Local only, personal project. No HF Space/upload. Adapter from `checkpoints\final`; base model downloads once anonymously |
| 2026-10-04 | Deps | PixiJS (pinned) allowed; Tone.js pinned |
| 2026-10-04 | Audio | All free pre-recorded samples, sourced by Claude. Combo: sitar (you) + bansuri (AI); single-instrument fallback = sitar |
| 2026-10-04 | Flow | Click → recorded intro → AI improved line → your live reply → AI reply → … |
| 2026-10-05 | Build | Shipped the full stage per §9's table — river, tala mandala, orbs, escalation ladder, echo arcs, reward chips, sampled sitar+flute+tanpura, single-process serving. Deviations logged in §9.1 |
| 2026-10-05 | AI voice | Flute (bansuri stand-in) — no longer open, see §11 |
| 2026-10-05 | Tabla | Synth (`Tone.MembraneSynth`), per `AUDIO_SOURCES.md` §3's decision — no cleared-licence real tabla pack found |

## 11. Still open

1. **Visual references** for the traditional-modern look, if you have any — current palette/layout is my own interpretation, not checked against a reference.
2. **Timeline for the Option A retrain** (`RETRAIN_PLAN.md`): target date or open-ended? The stage runs fine on the current (un-retrained) adapter in the meantime — captions say "improvises after your line" rather than claiming it's answering you, honestly, until the retrain lands.
3. **Shehnai A/B** — only flute (bansuri) shipped for the AI voice; want it compared against shehnai? One line in `scripts/fetch_samples.py`'s `INSTRUMENTS` dict plus a constructor arg change in `app.js`.
4. **Which §9.1 deviation to close next** — quantised input, suggest-reply, a before/after toggle, or a finale screen, if any of them turn out to matter once you've actually played with it.

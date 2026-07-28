// State machine for the Jugalbandi demo UI.
// Owns: env session, UI sync, call-response flow, raga dial.
// Talks to:  env_client.js (server), sitar.js, tabla.js

import { EnvClient } from "./env_client.js";
import { Sitar, RAGA_NOTES } from "./sitar.js";
import { Tabla } from "./tabla.js";

// ── State ──────────────────────────────────────────────────────────────

const state = {
  phase: "idle",        // idle | human_call | ai_response | ended
  callBuffer: [],       // notes user played during human_call phase
  callExpected: 4,
  aiResponseNotes: [],
  stepCount: 0,
  totalReward: 0,
  lastBreakdown: {},
};

// ── DOM refs ───────────────────────────────────────────────────────────

const $ = (id) => document.getElementById(id);
const ragaLabel     = $("raga-label");
const phaseLabel    = $("phase-label");
const rewardDisplay = $("reward-display");
const talaDisplay   = $("tala-display");
const beatDots      = Array.from(document.querySelectorAll(".beat-dot"));
const dialInput     = $("raga-dial");
const dialValue     = $("dial-value");
const startBtn      = $("start-btn");
const resetBtn      = $("reset-btn");
const logEl         = $("reward-log");

// ── Instruments ────────────────────────────────────────────────────────

const sitar = new Sitar("sitar-container", onUserPluck);
const tabla = new Tabla(onTableBeat);

// ── Lifecycle ──────────────────────────────────────────────────────────

startBtn.addEventListener("click", async () => {
  await EnvClient.health();
  const dial = parseFloat(dialInput.value);
  const data = await EnvClient.reset(dial);
  state.phase = "human_call";
  state.callBuffer = [];
  state.stepCount = 0;
  state.totalReward = 0;
  sitar.setRaga(data.active_raga);
  tabla.start();
  updateUI(data.active_raga, data.observation);
  log("Episode started. Play your call phrase (4 notes).");
  startBtn.disabled = true;
  resetBtn.disabled = false;
});

resetBtn.addEventListener("click", async () => {
  tabla.stop();
  state.phase = "idle";
  startBtn.disabled = false;
  resetBtn.disabled = true;
  phaseLabel.textContent = "—";
  log("Reset.");
});

dialInput.addEventListener("input", async () => {
  const v = parseFloat(dialInput.value);
  dialValue.textContent = v.toFixed(2);
  if (state.phase === "idle") return;
  const res = await EnvClient.setDial(v);
  sitar.setRaga(res.active_raga);
  ragaLabel.textContent = res.active_raga;
  if (res.switched) {
    log(`⚡ Raga switched → ${res.active_raga}${res.in_grace_period ? " (grace period)" : ""}`);
  }
});

// ── User pluck ─────────────────────────────────────────────────────────

async function onUserPluck(noteIdx) {
  if (state.phase !== "human_call") return;

  const semitone = RAGA_NOTES[sitar.raga][noteIdx];
  state.callBuffer.push(semitone);
  log(`Call note ${state.callBuffer.length}/4: semitone ${semitone}`);

  if (state.callBuffer.length >= state.callExpected) {
    await EnvClient.submitCall(state.callBuffer);
    state.callBuffer = [];
    state.phase = "ai_response";
    phaseLabel.textContent = "AI responding…";
    log("Call sent — AI responding…");
    await runAIResponse();
  }
}

// ── AI turn ────────────────────────────────────────────────────────────

async function runAIResponse() {
  // AI plays 4 steps; server returns the notes via step responses.
  // In training, the trained LLM picks actions. In the demo UI we send
  // action=null to let the server use a simple greedy policy fallback
  // (or the trained model if loaded). For now we send action 4 (Ga, safe vadi).
  // TODO: replace with trained model inference endpoint when ready.

  for (let i = 0; i < 4; i++) {
    const res = await EnvClient.step(4);  // placeholder: always play Ga (vadi of Yaman)
    state.totalReward += res.reward;
    state.lastBreakdown = res.reward_breakdown;
    state.stepCount++;

    // animate the string that corresponds to the note the AI played
    const noteIdx = mapSemitoneToStringIndex(res.info.note);
    if (noteIdx >= 0) sitar.animatePluck(noteIdx);
    tabla.syncBeat(res.info.tala_position);
    updateReward();
    logBreakdown(res.reward_breakdown, res.reward);

    await delay(400);

    if (res.terminated) {
      state.phase = "ended";
      phaseLabel.textContent = "Episode ended";
      tabla.stop();
      log(`Episode complete. Total reward: ${state.totalReward.toFixed(2)}`);
      startBtn.disabled = false;
      return;
    }
  }

  // Hand back to user for next call phrase
  state.phase = "human_call";
  phaseLabel.textContent = "Your turn — play 4 notes";
  log("Your turn. Play your call phrase.");
}

// ── Tabla beat sync ────────────────────────────────────────────────────

function onTableBeat(beat) {
  beatDots.forEach((dot, i) => dot.classList.toggle("active", i === beat));
  talaDisplay.textContent = `Beat ${beat + 1}/16`;
}

// ── UI helpers ─────────────────────────────────────────────────────────

function updateUI(ragaName, obs) {
  ragaLabel.textContent = ragaName;
  phaseLabel.textContent = "Your turn — play 4 notes";
  rewardDisplay.textContent = "0.00";
}

function updateReward() {
  rewardDisplay.textContent = state.totalReward.toFixed(2);
}

function log(msg) {
  const el = document.createElement("div");
  el.className = "log-line";
  el.textContent = msg;
  logEl.prepend(el);
  if (logEl.children.length > 30) logEl.lastChild.remove();
}

function logBreakdown(bd, total) {
  const parts = Object.entries(bd)
    .filter(([k]) => k !== "total")
    .map(([k, v]) => `${k}: ${(+v).toFixed(2)}`)
    .join(" | ");
  log(`Reward ${total >= 0 ? "+" : ""}${total.toFixed(2)} → ${parts || "—"}`);
}

function mapSemitoneToStringIndex(semitone) {
  const notes = RAGA_NOTES[sitar.raga];
  return notes.indexOf(semitone);
}

function delay(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

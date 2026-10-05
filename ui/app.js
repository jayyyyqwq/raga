// Jugalbandi stage — turn-based call-and-response (docs/imrpovedui.md).
// Owns: episode lifecycle, the turn loop, and wiring the audio/visual/input
// modules together. Those modules know nothing about each other or about
// the env — this file is the only place that does.

import { Application } from "https://cdn.jsdelivr.net/npm/pixi.js@8.22.0/+esm";
import * as Tone from "https://cdn.jsdelivr.net/npm/tone@15.1.22/+esm";

import { EnvClient } from "./env_client.js";
import { Voice } from "./audio/voices.js";
import { Drone } from "./audio/drone.js";
import { Tabla } from "./audio/tabla.js";
import { River } from "./stage/river.js";
import { TalaMandala } from "./stage/talaMandala.js";
import { Orbs } from "./stage/orbs.js";
import { InputStrip } from "./input.js";
import { INTROS, callPhraseFromIntro } from "./presets.js";
import { analyzeResponse } from "./analysis/echo.js";
import { RAGA_THEME, ragaFromDial, escalationForTurn, ESCALATION, SWARA_LATIN } from "./stage/theme.js";

const STAGE_WIDTH = 880;
const STAGE_HEIGHT = 620;
const AI_NOTES_PER_TURN = 8;     // matches JugalbandiEnv.CALL_EVERY
const YOUR_TURN_MS = 22000;
const NOTE_PACE_MS = 260;        // intro/fixed pacing between visible notes

const $ = (id) => document.getElementById(id);
const dom = {
  startBtn: $("start-btn"),
  resetBtn: $("reset-btn"),
  dialInput: $("raga-dial"),
  dialValue: $("dial-value"),
  ragaLabel: $("raga-label"),
  phaseLabel: $("phase-label"),
  rewardDisplay: $("reward-display"),
  log: $("log"),
  caption: $("caption"),
  chips: $("chips"),
  escalation: $("escalation"),
  stageHost: $("stage-host"),
  inputHost: $("input-strip"),
};

const state = {
  phase: "idle",          // idle | intro | listening | ai_playing | your_turn | ended
  raga: "yaman",
  turnIndex: 0,            // 0-based AI-turn counter, drives the escalation ladder
  totalReward: 0,
  lastCall: [],            // swara 0-11, your most recent submitted call
  yourTurnBuffer: [],
  yourTurnDeadline: 0,
  yourTurnTimer: null,
  callPositions: [],       // river {x,y} per note of lastCall, for echo arcs
};

let app, river, mandala, orbs, input, tabla, sitarVoice, fluteVoice, drone;

// ── Bootstrap ──────────────────────────────────────────────────────────

async function init() {
  app = new Application();
  await app.init({ width: STAGE_WIDTH, height: STAGE_HEIGHT, backgroundAlpha: 0, antialias: true, autoDensity: true, resolution: window.devicePixelRatio || 1 });
  dom.stageHost.appendChild(app.canvas);

  river = new River(app, { x: 0, y: 0, width: STAGE_WIDTH, height: 400 });
  mandala = new TalaMandala(app, { x: 440, y: 515, radius: 78 });
  orbs = new Orbs(app, {
    you: { x: 150, y: 515, radius: 28, color: RAGA_THEME.yaman.you },
    ai: { x: 730, y: 515, radius: 28, color: RAGA_THEME.yaman.ai },
  });
  river.setRaga("yaman");

  input = new InputStrip("input-strip", onYourNote);
  input.setEnabled(false);

  sitarVoice = new Voice("sitar");
  fluteVoice = new Voice("flute", { volume: -3 });
  drone = new Drone();
  tabla = new Tabla(onBeat);

  dom.startBtn.disabled = true;
  dom.startBtn.textContent = "Loading instruments…";
  await Promise.all([sitarVoice.loaded(), fluteVoice.loaded(), drone.loaded()]);
  dom.startBtn.disabled = false;
  dom.startBtn.textContent = "▶ Play intro";

  buildEscalationStrip();

  dom.startBtn.addEventListener("click", startPerformance);
  dom.resetBtn.addEventListener("click", resetPerformance);
  dom.dialInput.addEventListener("input", onDialInput);
}

function buildEscalationStrip() {
  dom.escalation.innerHTML = "";
  ESCALATION.forEach((step, i) => {
    const el = document.createElement("span");
    el.className = "escalation-step";
    el.dataset.index = i;
    el.textContent = step.label;
    el.title = step.sub;
    dom.escalation.appendChild(el);
    if (i < ESCALATION.length - 1) {
      const sep = document.createElement("span");
      sep.className = "escalation-sep";
      sep.textContent = "›";
      dom.escalation.appendChild(sep);
    }
  });
  highlightEscalation(0);
}

function highlightEscalation(turnIndex) {
  const idx = Math.min(turnIndex, ESCALATION.length - 1);
  [...dom.escalation.children].forEach((el) => {
    if (!el.dataset.index) return;
    el.classList.toggle("active", Number(el.dataset.index) === idx);
  });
}

// ── Lifecycle ──────────────────────────────────────────────────────────

async function startPerformance() {
  await Tone.start();
  await EnvClient.health();

  const dial = parseFloat(dom.dialInput.value);
  state.raga = ragaFromDial(dial);
  state.turnIndex = 0;
  state.totalReward = 0;
  state.yourTurnBuffer = [];
  river.reset();
  river.setRaga(state.raga);
  input.setRaga(state.raga);
  applyTheme(state.raga);
  highlightEscalation(0);
  updateReward();

  const data = await EnvClient.reset(dial);
  state.raga = data.active_raga;
  river.setRaga(state.raga);
  input.setRaga(state.raga);
  applyTheme(state.raga);
  dom.ragaLabel.textContent = `${RAGA_THEME[state.raga].label} · ${RAGA_THEME[state.raga].time}`;

  tabla.start();
  drone.start();

  dom.startBtn.disabled = true;
  dom.resetBtn.disabled = false;
  log("Episode started.");

  await playIntro(state.raga);
}

function resetPerformance() {
  clearTimeout(state.yourTurnTimer);
  tabla.stop();
  drone.stop();
  input.setEnabled(false);
  state.phase = "idle";
  dom.startBtn.disabled = false;
  dom.resetBtn.disabled = true;
  dom.phaseLabel.textContent = "—";
  dom.caption.textContent = "";
  dom.chips.innerHTML = "";
  log("Reset.");
}

async function onDialInput() {
  const v = parseFloat(dom.dialInput.value);
  dom.dialValue.textContent = v.toFixed(2);
  if (state.phase === "idle") return;
  const res = await EnvClient.setDial(v);
  state.raga = res.active_raga;
  river.setRaga(state.raga);
  input.setRaga(state.raga);
  applyTheme(state.raga);
  dom.ragaLabel.textContent = `${RAGA_THEME[state.raga].label} · ${RAGA_THEME[state.raga].time}`;
  if (res.switched) {
    log(`⚡ Raga switched → ${state.raga}${res.in_grace_period ? " (grace period)" : ""}`);
  }
}

function applyTheme(raga) {
  const t = RAGA_THEME[raga];
  document.documentElement.style.setProperty("--bg0", t.bg0);
  document.documentElement.style.setProperty("--bg1", t.bg1);
  document.documentElement.style.setProperty("--accent", t.accent);
  document.documentElement.style.setProperty("--accent2", t.accent2);
  document.documentElement.style.setProperty("--you", t.you);
  document.documentElement.style.setProperty("--ai", t.ai);
  orbs.you.color = t.you;
  orbs.ai.color = t.ai;
}

// ── Intro (your recorded opening line) ───────────────────────────────────

async function playIntro(raga) {
  state.phase = "intro";
  dom.phaseLabel.textContent = "Your intro";
  orbs.setTurn("you");

  const { pitches, durations } = INTROS[raga];
  state.callPositions = [];
  for (let i = 0; i < pitches.length; i++) {
    const pos = river.addNote(pitches[i] % 12, "you", { octaveUp: pitches[i] >= 24 });
    state.callPositions.push(pos);
    sitarVoice.play(pitches[i], durations[i]);
    orbs.you.pulse();
    await delay(NOTE_PACE_MS);
  }

  const callSwaras = callPhraseFromIntro(raga);
  state.lastCall = callSwaras;
  state.callPositions = state.callPositions.slice(-4); // only the last 4 notes became the call
  await EnvClient.submitCall(callSwaras);
  log(`Call: ${callSwaras.map((s) => SWARA_LATIN[s]).join(", ")}`);

  await runAiTurn();
}

// ── AI turn ────────────────────────────────────────────────────────────

async function runAiTurn() {
  state.phase = "listening";
  dom.phaseLabel.textContent = "AI listening…";
  orbs.setTurn("ai");
  input.setEnabled(false);

  for (const pos of state.callPositions) {
    orbs.listenTravel(pos.stage);
    await delay(120);
  }
  await delay(300);

  const escalation = escalationForTurn(state.turnIndex);
  highlightEscalation(state.turnIndex);

  // Prefetch the whole line before playing any of it, so network latency
  // hides behind the listen phase instead of breaking the beat
  // (docs/imrpovedui.md §4.3 "scheduled, not streamed").
  const notes = [];
  let terminated = false;
  let inferenceUnavailable = false;

  for (let i = 0; i < AI_NOTES_PER_TURN && !terminated; i++) {
    let action = 4; // fallback: Ga, Yaman's vadi — keeps the demo running if /infer is down
    try {
      const inferRes = await EnvClient.infer();
      action = inferRes.action;
    } catch (err) {
      if (!inferenceUnavailable) {
        inferenceUnavailable = true;
        log(`⚠ Model inference unavailable (${err.message}) — using fallback notes.`);
      }
    }
    const res = await EnvClient.step(action);
    state.totalReward += res.reward;
    notes.push({ note: res.info.note, duration: res.info.duration, reward: res.reward, breakdown: res.reward_breakdown });
    terminated = res.terminated;
  }

  // ── Scheduled playback, on the beat, with this turn's escalation voicing ──
  state.phase = "ai_playing";
  dom.phaseLabel.textContent = "AI responding…";
  orbs.setTurn("ai");
  await delay(tabla.msUntilNextBeat());

  const responsePositions = [];
  const breakdownUnion = {};
  for (const n of notes) {
    const voicedPitch = n.note + escalation.octaveUp;
    const pos = river.addNote(voicedPitch % 12, "ai", { octaveUp: escalation.octaveUp > 0 });
    responsePositions.push(pos);
    const seconds = fluteVoice.play(voicedPitch, n.duration, { speed: escalation.speed });
    orbs.ai.pulse();
    Object.assign(breakdownUnion, n.breakdown);
    updateReward();
    // `seconds` is already speed-adjusted inside Voice.play — don't divide again.
    await delay(seconds * 1000 + 40);
  }

  const responseSwaras = notes.map((n) => n.note % 12);
  const { arcs, caption } = analyzeResponse(state.lastCall, responseSwaras);
  for (const arc of arcs) {
    const from = state.callPositions[Math.min(arc.fromIndex, state.callPositions.length - 1)];
    const to = responsePositions[Math.min(arc.toIndex, responsePositions.length - 1)];
    if (from && to) river.drawArc(from.local, to.local);
  }
  dom.caption.textContent = `${escalation.label} — ${caption}`;
  renderChips(breakdownUnion);
  log(`AI turn ${state.turnIndex + 1} (${escalation.label}): reward ${notes.reduce((s, n) => s + n.reward, 0).toFixed(2)}`);

  state.turnIndex += 1;

  if (terminated) {
    endPerformance();
    return;
  }

  beginYourTurn();
}

// ── Your live reply ───────────────────────────────────────────────────

function beginYourTurn() {
  state.phase = "your_turn";
  state.yourTurnBuffer = [];
  state.callPositions = [];
  dom.phaseLabel.textContent = "Your turn — play up to 4 notes";
  orbs.setTurn("you");
  input.setEnabled(true);

  state.yourTurnDeadline = performance.now() + YOUR_TURN_MS;
  tickYourTurnCountdown();
}

function tickYourTurnCountdown() {
  const remaining = state.yourTurnDeadline - performance.now();
  mandala.setCountdown(Math.max(0, remaining / YOUR_TURN_MS));
  if (remaining <= 0) {
    finishYourTurn();
    return;
  }
  state.yourTurnTimer = setTimeout(tickYourTurnCountdown, 120);
}

function onYourNote(swara, { register }) {
  if (state.phase !== "your_turn") return;
  const pitch = 12 + swara + register * 12; // madhya baseline, ± one register
  const pos = river.addNote(swara, "you", { octaveUp: register > 0 });
  state.callPositions.push(pos);
  sitarVoice.play(pitch, 1);
  orbs.you.pulse();

  state.yourTurnBuffer.push(swara);
  if (state.yourTurnBuffer.length >= 4) {
    finishYourTurn();
  }
}

async function finishYourTurn() {
  clearTimeout(state.yourTurnTimer);
  mandala.setCountdown(0);
  input.setEnabled(false);

  if (state.yourTurnBuffer.length === 0) {
    log("No reply played — repeating your last call.");
  } else {
    state.lastCall = state.yourTurnBuffer;
  }
  await EnvClient.submitCall(state.lastCall);
  log(`Call: ${state.lastCall.map((s) => SWARA_LATIN[s]).join(", ")}`);

  await runAiTurn();
}

// ── Tabla beat sync ────────────────────────────────────────────────────

function onBeat(beat) {
  mandala.setBeat(beat);
}

function endPerformance() {
  state.phase = "ended";
  dom.phaseLabel.textContent = "Performance complete";
  tabla.stop();
  input.setEnabled(false);
  log(`Performance complete. Total reward: ${state.totalReward.toFixed(2)}`);
  dom.startBtn.disabled = false;
  dom.startBtn.textContent = "▶ Play again";
}

// ── UI helpers ─────────────────────────────────────────────────────────

function updateReward() {
  dom.rewardDisplay.textContent = state.totalReward.toFixed(2);
}

function renderChips(breakdown) {
  dom.chips.innerHTML = "";
  const entries = Object.entries(breakdown).filter(([k]) => k !== "total");
  if (!entries.length) {
    dom.chips.innerHTML = '<span class="chip chip-empty">no reward events this turn</span>';
    return;
  }
  for (const [key, value] of entries) {
    const chip = document.createElement("span");
    const good = value >= 0;
    chip.className = `chip ${good ? "chip-good" : "chip-bad"}`;
    chip.textContent = `${key.replace(/_/g, " ")} ${good ? "✓" : "✕"} ${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
    dom.chips.appendChild(chip);
  }
}

function log(msg) {
  const el = document.createElement("div");
  el.className = "log-line";
  el.textContent = msg;
  dom.log.prepend(el);
  while (dom.log.children.length > 40) dom.log.lastChild.remove();
}

function delay(ms) {
  return new Promise((r) => setTimeout(r, Math.max(0, ms)));
}

init();

// Shared visual + musical constants for the Jugalbandi stage.
// Traditional-modern palette: deep indigo ground, gold/saffron line work,
// Devanagari swara labels beside Latin (docs/imrpovedui.md §6 decision).

// Swara index (0-11) -> Latin label, matches raaga_env.ragas.NOTE_NAMES[12:]
// and ui/env_client.js's call-phrase encoding (swara, not absolute pitch).
export const SWARA_LATIN = ["Sa", "Re♭", "Re", "Ga♭", "Ga", "Ma", "Ma#", "Pa", "Dha♭", "Dha", "Ni♭", "Ni"];
export const SWARA_DEVANAGARI = ["सा", "रे♭", "रे", "ग♭", "ग", "म", "म#", "प", "ध♭", "ध", "नि♭", "नि"];

// Which swaras are in each raga, aaroha (ascending) order — mirrors
// raaga_env.ragas.RAGAS[*]['valid_notes'] / sitar.js's old RAGA_NOTES.
export const RAGA_SWARAS = {
  yaman: [0, 2, 4, 6, 7, 9, 11],
  bhairav: [0, 1, 4, 5, 7, 8, 11],
};

export const RAGA_FORBIDDEN = {
  yaman: new Set([5]),
  bhairav: new Set([2, 9]),
};

export const RAGA_VADI = { yaman: 4, bhairav: 8 };
export const RAGA_SAMVADI = { yaman: 11, bhairav: 1 };

// Sa = C4 (261.63 Hz) — matches the old ui/sitar.js and the tanpura sample's
// own tuning (docs/AUDIO_SOURCES.md). Absolute pitch 12 = madhya Sa = C4.
export const SA_HZ = 261.63;

export const RAGA_THEME = {
  yaman: {
    label: "Raag Yaman",
    time: "evening",
    bg0: "#12101f",
    bg1: "#1c1933",
    panel: "#1a1730",
    accent: "#d9a441",      // gold
    accent2: "#f2c14e",     // bright gold glow
    you: "#e08a3c",         // saffron
    ai: "#8f7bd1",          // violet
    forbidden: "#7a2f3a",
  },
  bhairav: {
    label: "Raag Bhairav",
    time: "morning",
    bg0: "#211409",
    bg1: "#2f1d0e",
    panel: "#2a1a0f",
    accent: "#e8833a",      // dawn amber
    accent2: "#ffb15c",
    you: "#e0a23c",
    ai: "#b17bd1",
    forbidden: "#7a2f3a",
  },
};

export function ragaFromDial(dial) {
  return dial < 0.5 ? "yaman" : "bhairav";
}

// Escalation ladder (docs/imrpovedui.md §4.5), applied per AI turn, not per
// note — turn 1 is a plain answer, each later turn adds one presentation
// transform on top of the model's own notes. These are UI-side voicings,
// never a different action from the model; each is labelled honestly.
export const ESCALATION = [
  { id: "uttar", label: "UTTAR", sub: "answer", speed: 1, octaveUp: 0 },
  { id: "badhat", label: "BADHAT", sub: "development", speed: 1, octaveUp: 0 },
  { id: "laya2", label: "LAYA ×2", sub: "double speed", speed: 2, octaveUp: 0 },
  { id: "taar", label: "TAAR SAPTAK", sub: "octave up", speed: 1, octaveUp: 12 },
  { id: "tihai", label: "TIHAI ↑", sub: "fast + high", speed: 2, octaveUp: 12 },
];

export function escalationForTurn(turnIndex) {
  return ESCALATION[Math.min(turnIndex, ESCALATION.length - 1)];
}

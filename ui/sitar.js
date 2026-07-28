// Sitar — 7 SVG strings, one per raga note.
// Touch/mouse swipe triggers a pluck: plays Tone.js synth + returns semitone.
//
// WHY PluckSynth: it models a damped string (Karplus-Strong). Closest free
// approximation to sitar timbre without loading audio samples.

import * as Tone from "https://cdn.skypack.dev/tone";

// Semitone indices for the 7 sitar strings.
// These update when the raga changes (active notes differ between Yaman/Bhairav).
const RAGA_NOTES = {
  yaman:   [0, 2, 4, 6, 7, 9, 11],  // Sa Re Ga Ma# Pa Dha Ni
  bhairav: [0, 1, 4, 5, 7, 8, 11],  // Sa Reb Ga Ma Pa Dhab Ni
};

const NOTE_LABELS = {
  yaman:   ["Sa", "Re", "Ga", "Ma#", "Pa", "Dha", "Ni"],
  bhairav: ["Sa", "Re♭", "Ga", "Ma", "Pa", "Dha♭", "Ni"],
};

// Base frequency for Sa (C4 = 261.63 Hz)
const SA_HZ = 261.63;
const SEMITONE = 2 ** (1 / 12);

function noteToHz(semitone) {
  return SA_HZ * SEMITONE ** semitone;
}

class Sitar {
  constructor(containerId, onPluck) {
    this.container = document.getElementById(containerId);
    this.onPluck = onPluck;        // callback(semitoneIndex) called on pluck
    this.raga = "yaman";
    this.activeStrings = new Set();

    this.synth = new Tone.PluckSynth({
      attackNoise: 1.2,
      dampening: 4000,
      resonance: 0.96,
    }).toDestination();

    // Slight reverb — sitars resonate
    this.reverb = new Tone.Reverb({ decay: 1.2, wet: 0.25 }).toDestination();
    this.synth.connect(this.reverb);

    this._buildSVG();
    this._bindEvents();
  }

  setRaga(ragaName) {
    this.raga = ragaName;
    this._updateLabels();
  }

  // Flash a string visually (called when AI plays a note)
  animatePluck(noteIndex) {
    const stringEl = this.container.querySelector(`[data-idx="${noteIndex}"]`);
    if (!stringEl) return;
    stringEl.classList.add("plucking");
    setTimeout(() => stringEl.classList.remove("plucking"), 300);
    this._playNote(noteIndex);
  }

  _buildSVG() {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 280 500");
    svg.setAttribute("class", "sitar-svg");

    const notes = RAGA_NOTES[this.raga];
    const labels = NOTE_LABELS[this.raga];
    const count = notes.length;
    const spacing = 280 / (count + 1);

    notes.forEach((semitone, i) => {
      const x = spacing * (i + 1);

      // String line
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", x); line.setAttribute("y1", 40);
      line.setAttribute("x2", x); line.setAttribute("y2", 460);
      line.setAttribute("stroke", `hsl(${30 + i * 20}, 70%, 55%)`);
      line.setAttribute("stroke-width", "3");
      line.setAttribute("stroke-linecap", "round");
      line.setAttribute("class", "sitar-string");
      line.setAttribute("data-idx", i);
      line.setAttribute("data-semitone", semitone);
      svg.appendChild(line);

      // Label
      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("x", x);
      text.setAttribute("y", 478);
      text.setAttribute("text-anchor", "middle");
      text.setAttribute("class", "string-label");
      text.setAttribute("data-label-idx", i);
      text.textContent = labels[i];
      svg.appendChild(text);
    });

    this.svg = svg;
    this.container.appendChild(svg);
  }

  _updateLabels() {
    const notes = RAGA_NOTES[this.raga];
    const labels = NOTE_LABELS[this.raga];
    notes.forEach((semitone, i) => {
      const line = this.container.querySelector(`[data-idx="${i}"]`);
      if (line) line.setAttribute("data-semitone", semitone);
      const label = this.container.querySelector(`[data-label-idx="${i}"]`);
      if (label) label.textContent = labels[i];
    });
  }

  _bindEvents() {
    const handle = (e) => {
      e.preventDefault();
      const touches = e.changedTouches || [e];
      Array.from(touches).forEach((touch) => {
        const el = document.elementFromPoint(touch.clientX, touch.clientY);
        if (!el || !el.classList.contains("sitar-string")) return;
        const idx = parseInt(el.getAttribute("data-idx"), 10);
        if (this.activeStrings.has(idx)) return;  // debounce per gesture
        this.activeStrings.add(idx);
        this._pluck(idx);
      });
    };

    const clear = () => this.activeStrings.clear();

    this.svg.addEventListener("touchstart", handle, { passive: false });
    this.svg.addEventListener("touchmove", handle, { passive: false });
    this.svg.addEventListener("touchend", clear);
    this.svg.addEventListener("mousedown", handle);
    this.svg.addEventListener("mousemove", (e) => { if (e.buttons) handle(e); });
    this.svg.addEventListener("mouseup", clear);
  }

  _pluck(idx) {
    const line = this.container.querySelector(`[data-idx="${idx}"]`);
    if (line) {
      line.classList.add("plucking");
      setTimeout(() => line.classList.remove("plucking"), 300);
    }
    this._playNote(idx);
    this.onPluck(idx);
  }

  _playNote(idx) {
    const semitone = parseInt(
      this.container.querySelector(`[data-idx="${idx}"]`)?.getAttribute("data-semitone") ?? "0",
      10
    );
    Tone.start();
    this.synth.triggerAttack(noteToHz(semitone));
  }
}

export { Sitar, RAGA_NOTES };

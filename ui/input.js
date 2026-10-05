// Your live input: the 7 on-screen strings + a computer-keyboard shortcut
// for each, in 3 registers (docs/imrpovedui.md §4.4). Framework-agnostic —
// plain DOM buttons + a keydown listener, not Pixi, so it stays simple and
// accessible; the stage (river/orbs/mandala) is purely a display of what
// this module reports.
//
// Positional mapping, not a fixed swara: "the 3rd string" is Ga in Yaman
// but Ga in Bhairav too here (both ragas' 3rd valid-note happens to be Ga),
// yet the 2nd string is Re in Yaman and komal Re in Bhairav — the actual
// swara always comes from RAGA_SWARAS[raga][slot], matching the old
// ui/sitar.js's RAGA_NOTES approach, extended with register.

import { RAGA_SWARAS, SWARA_LATIN } from "./stage/theme.js";

const BASE_KEYS = ["s", "r", "g", "m", "p", "d", "n"];       // madhya (register 0)
const MANDRA_KEYS = ["z", "x", "c", "v", "b", "n", "m"];     // mandra (register -1)

export class InputStrip {
  constructor(containerId, onNote) {
    this.container = document.getElementById(containerId);
    this.onNote = onNote; // (swara, { register }) => void
    this.raga = "yaman";
    this.enabled = false;

    this.buttons = [];
    this._buildButtons();
    this._bindKeyboard();
  }

  _buildButtons() {
    this.container.innerHTML = "";
    RAGA_SWARAS[this.raga].forEach((swara, i) => {
      const btn = document.createElement("button");
      btn.className = "swara-key";
      btn.type = "button";
      btn.dataset.slot = i;
      btn.addEventListener("click", () => this._fire(i, 0, btn));
      this.container.appendChild(btn);
      this.buttons.push(btn);
    });
    this._relabel();
  }

  _relabel() {
    RAGA_SWARAS[this.raga].forEach((swara, i) => {
      if (this.buttons[i]) this.buttons[i].textContent = SWARA_LATIN[swara];
    });
  }

  setRaga(raga) {
    this.raga = raga;
    this._relabel();
  }

  setEnabled(enabled) {
    this.enabled = enabled;
    this.container.classList.toggle("disabled", !enabled);
  }

  _fire(slot, register, btnEl) {
    if (!this.enabled) return;
    const swara = RAGA_SWARAS[this.raga][slot];
    if (swara === undefined) return;
    if (btnEl) {
      btnEl.classList.add("pressed");
      setTimeout(() => btnEl.classList.remove("pressed"), 180);
    }
    this.onNote(swara, { register });
  }

  _bindKeyboard() {
    window.addEventListener("keydown", (e) => {
      if (!this.enabled || e.repeat) return;
      const key = e.key.toLowerCase();
      const baseIdx = BASE_KEYS.indexOf(key);
      if (baseIdx !== -1) {
        this._fire(baseIdx, e.shiftKey ? 1 : 0, this.buttons[baseIdx]);
        return;
      }
      const mandraIdx = MANDRA_KEYS.indexOf(key);
      if (mandraIdx !== -1 && !e.shiftKey) {
        this._fire(mandraIdx, -1, this.buttons[mandraIdx]);
      }
    });
  }
}

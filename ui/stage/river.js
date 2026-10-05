// Swara river — the stage's main "you can see the conversation" element.
// Scrolling pitch ribbon: your notes in saffron, the AI's in violet, joined
// by glowing curves. Persists for the whole performance (docs/imrpovedui.md §4.1).
//
// Fixed 12-row layout (one row per swara, Ni at top to Sa at bottom) rather
// than only the 7 currently-active ones — a raga switch just relabels which
// rows are dim/forbidden instead of reflowing already-played notes (which
// were legitimately played under the old raga's mapping).
//
// Pixi version note: pinned to v7 (pixi.js@7.4.3), not v8 — v8.x (0.0 through
// at least 8.22.0) has a confirmed renderer bug (pixijs/pixijs#12048) where
// mutating a Container's transform every tick, with a Graphics anywhere
// beneath it, crashes the WebGL renderer after a couple of frames
// ("Cannot read properties of undefined (reading 'updateRenderable')") —
// reproduced here independent of this app's own code, down to a single
// Graphics circle. v7 doesn't have v8's RenderGroup system and doesn't hit
// this at all. See stage/orbs.js, the module that actually animates every
// frame, for where this matters most.

import * as PIXI from "https://cdn.jsdelivr.net/npm/pixi.js@7.4.3/+esm";
import { SWARA_LATIN, SWARA_DEVANAGARI, RAGA_FORBIDDEN, RAGA_VADI, RAGA_SAMVADI, RAGA_THEME } from "./theme.js";

const ROW_HEIGHT = 30;
const TOP_MARGIN = 14;
const LABEL_WIDTH = 64;
const NOTE_SPACING = 34;
const NOTE_RADIUS = 6;

// Row order, top to bottom: Ni(11) down to Sa(0).
const ROW_SWARAS = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0];

export class River {
  constructor(app, { x = 0, y = 0, width, height } = {}) {
    this.app = app;
    this.width = width;
    this.height = height ?? ROW_HEIGHT * ROW_SWARAS.length + TOP_MARGIN * 2;

    this.root = new PIXI.Container();
    this.root.x = x;
    this.root.y = y;
    app.stage.addChild(this.root);

    this.labelsLayer = new PIXI.Container();
    this.scrollLayer = new PIXI.Container();
    this.scrollLayer.x = LABEL_WIDTH;
    this.root.addChild(this.labelsLayer, this.scrollLayer);

    this.rowLabels = [];
    this._buildRows();

    this.cursorX = 8;
    this.history = []; // { swara, x, y, performer }
    this.raga = "yaman";
  }

  _buildRows() {
    const bg = new PIXI.Graphics();
    bg.beginFill(0x000000, 0.18);
    bg.drawRect(0, 0, this.width, this.height);
    bg.endFill();
    this.labelsLayer.addChild(bg);

    ROW_SWARAS.forEach((swara, i) => {
      const y = TOP_MARGIN + i * ROW_HEIGHT;

      const gridLine = new PIXI.Graphics();
      gridLine.lineStyle(1, 0xffffff, 0.05);
      gridLine.moveTo(0, y).lineTo(this.width, y);
      this.labelsLayer.addChild(gridLine);

      const label = new PIXI.Text(
        `${SWARA_DEVANAGARI[swara]} ${SWARA_LATIN[swara]}`,
        new PIXI.TextStyle({ fontFamily: "'Noto Sans Devanagari', 'Segoe UI', sans-serif", fontSize: 13, fill: 0xd8d2e8 }),
      );
      label.x = 8;
      label.y = y - label.height / 2;
      this.labelsLayer.addChild(label);
      this.rowLabels.push({ swara, label, baseY: y });
    });
  }

  setRaga(raga) {
    this.raga = raga;
    const theme = RAGA_THEME[raga];
    const forbidden = RAGA_FORBIDDEN[raga];
    const vadi = RAGA_VADI[raga];
    const samvadi = RAGA_SAMVADI[raga];

    for (const row of this.rowLabels) {
      if (forbidden.has(row.swara)) {
        row.label.style.fill = 0x6b4a4f;
        row.label.alpha = 0.55;
      } else if (row.swara === vadi) {
        row.label.style.fill = theme.accent2;
        row.label.alpha = 1;
      } else if (row.swara === samvadi) {
        row.label.style.fill = theme.accent;
        row.label.alpha = 1;
      } else {
        row.label.style.fill = 0xd8d2e8;
        row.label.alpha = 0.9;
      }
    }
  }

  rowY(swara) {
    const idx = ROW_SWARAS.indexOf(((swara % 12) + 12) % 12);
    return TOP_MARGIN + idx * ROW_HEIGHT;
  }

  /** Appends one note. `swara` 0-11 (register-stripped, matches the row
   * layout — register is shown only as a slightly larger dot for taar
   * saptak, see `octaveUp`). `performer` is "you" | "ai". Returns the dot's
   * stage position (for orbs.js's listen-particle travel). */
  addNote(swara, performer, { forbidden = false, octaveUp = false } = {}) {
    const theme = RAGA_THEME[this.raga];
    const color = forbidden ? theme.forbidden : performer === "you" ? theme.you : theme.ai;
    const y = this.rowY(swara);
    const x = this.cursorX;

    const prev = this.history[this.history.length - 1];
    if (prev) {
      const interval = Math.abs(swara - prev.swara);
      const line = new PIXI.Graphics();
      if (interval <= 2 && interval !== 0) {
        // Meend: a glowing curve for a close interval (visual only — see
        // audio/voices.js's comment on why the glide itself isn't synthesised).
        const midX = (prev.x + x) / 2;
        line.lineStyle(2, color, 0.5);
        line.moveTo(prev.x, prev.y).quadraticCurveTo(midX, (prev.y + y) / 2, x, y);
      } else {
        line.lineStyle(1, 0xffffff, 0.12);
        line.moveTo(prev.x, prev.y).lineTo(x, y);
      }
      this.scrollLayer.addChildAt(line, 0);
    }

    const dot = new PIXI.Graphics();
    const r = octaveUp ? NOTE_RADIUS + 2 : NOTE_RADIUS;
    dot.beginFill(color, 0.18);
    dot.drawCircle(0, 0, r + 3); // soft glow halo
    dot.endFill();
    dot.beginFill(color, 1);
    dot.drawCircle(0, 0, r);
    dot.endFill();
    dot.x = x;
    dot.y = y;
    this.scrollLayer.addChild(dot);

    this.cursorX += NOTE_SPACING;
    if (this.cursorX > this.width - 40) {
      this.scrollLayer.x = LABEL_WIDTH - (this.cursorX - (this.width - 40));
    }

    this.history.push({ swara, x, y, performer });
    // Two coordinate spaces, for two different consumers: `local` is
    // scrollLayer-relative and pans with the river (drawArc draws inside
    // scrollLayer, so arcs must stay in this space to track their notes);
    // `stage` is app.stage-absolute, for orbs.js's particleLayer, which
    // sits outside scrollLayer and never pans.
    return {
      local: { x, y },
      stage: { x: this.root.x + this.scrollLayer.x + x, y: this.root.y + y },
    };
  }

  /** Draws a gold arc between two already-returned addNote() positions —
   * the visual proof of analysis/echo.js's finding, never staged (§4.6):
   * only called when a real relation was found. `from`/`to` are stage
   * coordinates as returned by addNote(), already absolute. */
  drawArc(from, to) {
    const arc = new PIXI.Graphics();
    const midX = (from.x + to.x) / 2;
    const liftY = Math.min(from.y, to.y) - 22;
    arc.lineStyle(1.5, 0xf2c14e, 0.55);
    arc.moveTo(from.x, from.y).quadraticCurveTo(midX, liftY, to.x, to.y);
    this.scrollLayer.addChild(arc);
  }

  reset() {
    this.scrollLayer.removeChildren();
    this.scrollLayer.x = LABEL_WIDTH;
    this.cursorX = 8;
    this.history = [];
  }
}

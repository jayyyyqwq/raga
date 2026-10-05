// Tala mandala — a circular 16-beat Teentaal wheel. Makes rhythm legible
// without needing to read a "beat 7/16" label (docs/imrpovedui.md §6).
//
// Pixi v7 (see stage/river.js's header comment for why, not v8). The
// playhead is repositioned (never redrawn) each beat, and the countdown
// ring is 32 pre-drawn dots toggled via .visible — reasonable practice
// regardless of Pixi version, and it was already the plan before the v8
// bug was found.

import * as PIXI from "https://cdn.jsdelivr.net/npm/pixi.js@7.4.3/+esm";

const COUNTDOWN_SEGMENTS = 32;

export class TalaMandala {
  constructor(app, { x, y, radius }) {
    this.cx = x;
    this.cy = y;
    this.r = radius;

    this.root = new PIXI.Container();
    app.stage.addChild(this.root);

    this.ring = new PIXI.Graphics();
    this.ticks = new PIXI.Graphics();
    this.root.addChild(this.ring, this.ticks);

    this.countdownLayer = new PIXI.Container();
    this.root.addChild(this.countdownLayer);
    this._buildCountdownDots();

    this.playhead = new PIXI.Container();
    this.playheadGlow = new PIXI.Graphics();
    this.playheadDot = new PIXI.Graphics();
    this.playhead.addChild(this.playheadGlow, this.playheadDot);
    this.root.addChild(this.playhead);
    this._paintPlayhead();

    this.currentBeat = 0;
    this._drawStatic();
    this.setBeat(0);
  }

  _paintPlayhead() {
    this.playheadGlow.beginFill(0xf2c14e, 0.22);
    this.playheadGlow.drawCircle(0, 0, 9);
    this.playheadGlow.endFill();
    this.playheadDot.beginFill(0xf2c14e, 1);
    this.playheadDot.drawCircle(0, 0, 5);
    this.playheadDot.endFill();
  }

  _buildCountdownDots() {
    this.countdownDots = [];
    for (let i = 0; i < COUNTDOWN_SEGMENTS; i++) {
      const angle = (i / COUNTDOWN_SEGMENTS) * Math.PI * 2 - Math.PI / 2;
      const dot = new PIXI.Graphics();
      dot.beginFill(0x8f7bd1, 1);
      dot.drawCircle(0, 0, 2.5);
      dot.endFill();
      dot.x = this.cx + Math.cos(angle) * (this.r + 8);
      dot.y = this.cy + Math.sin(angle) * (this.r + 8);
      dot.visible = false;
      this.countdownLayer.addChild(dot);
      this.countdownDots.push(dot);
    }
  }

  _drawStatic() {
    this.ring.lineStyle(2, 0xffffff, 0.15);
    this.ring.drawCircle(this.cx, this.cy, this.r);

    for (let i = 0; i < 16; i++) {
      const angle = (i / 16) * Math.PI * 2 - Math.PI / 2;
      const inner = this.r - 10;
      const outer = this.r;
      const x1 = this.cx + Math.cos(angle) * inner;
      const y1 = this.cy + Math.sin(angle) * inner;
      const x2 = this.cx + Math.cos(angle) * outer;
      const y2 = this.cy + Math.sin(angle) * outer;
      const isSam = i === 0;
      const isKhali = i === 8;

      this.ticks.lineStyle(isSam ? 3 : 1.5, isSam ? 0xf2c14e : 0xffffff, isSam ? 1 : isKhali ? 0.6 : 0.35);
      this.ticks.moveTo(x1, y1).lineTo(x2, y2);

      if (isSam) {
        this.ticks.lineStyle(0);
        this.ticks.beginFill(0xf2c14e, 1);
        this.ticks.drawCircle(x2, y2, 4); // sam — filled gold
        this.ticks.endFill();
      } else if (isKhali) {
        this.ticks.lineStyle(1.5, 0xffffff, 0.7);
        this.ticks.drawCircle(x2, y2, 3); // khali — hollow
      }
    }
  }

  setBeat(beat) {
    this.currentBeat = beat;
    const angle = (beat / 16) * Math.PI * 2 - Math.PI / 2;
    this.playhead.x = this.cx + Math.cos(angle) * (this.r - 10);
    this.playhead.y = this.cy + Math.sin(angle) * (this.r - 10);
  }

  /** fraction 1 = just started (full ring lit), 0 = turn over. Used during
   * the human's live-reply turn (docs/imrpovedui.md §4.4's turn window). */
  setCountdown(fraction) {
    const lit = Math.round(Math.max(0, Math.min(1, fraction)) * COUNTDOWN_SEGMENTS);
    for (let i = 0; i < COUNTDOWN_SEGMENTS; i++) {
      this.countdownDots[i].visible = i < lit;
    }
  }
}

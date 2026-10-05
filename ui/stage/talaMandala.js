// Tala mandala — a circular 16-beat Teentaal wheel. Makes rhythm legible
// without needing to read a "beat 7/16" label (docs/imrpovedui.md §6).

import { Container, Graphics } from "https://cdn.jsdelivr.net/npm/pixi.js@8.22.0/+esm";

export class TalaMandala {
  constructor(app, { x, y, radius }) {
    this.cx = x;
    this.cy = y;
    this.r = radius;

    this.root = new Container();
    app.stage.addChild(this.root);

    this.ring = new Graphics();
    this.ticks = new Graphics();
    this.countdownArc = new Graphics();
    this.playhead = new Graphics();
    this.root.addChild(this.ring, this.ticks, this.countdownArc, this.playhead);

    this.currentBeat = 0;
    this._drawStatic();
    this.setBeat(0);
  }

  _drawStatic() {
    this.ring.circle(this.cx, this.cy, this.r).stroke({ width: 2, color: 0xffffff, alpha: 0.15 });

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
      this.ticks
        .moveTo(x1, y1)
        .lineTo(x2, y2)
        .stroke({ width: isSam ? 3 : 1.5, color: isSam ? 0xf2c14e : 0xffffff, alpha: isSam ? 1 : isKhali ? 0.6 : 0.35 });
      if (isSam) {
        this.ticks.circle(x2, y2, 4).fill({ color: 0xf2c14e }); // sam — filled gold
      } else if (isKhali) {
        this.ticks.circle(x2, y2, 3).stroke({ width: 1.5, color: 0xffffff, alpha: 0.7 }); // khali — hollow
      }
    }
  }

  setBeat(beat) {
    this.currentBeat = beat;
    const angle = (beat / 16) * Math.PI * 2 - Math.PI / 2;
    const px = this.cx + Math.cos(angle) * (this.r - 10);
    const py = this.cy + Math.sin(angle) * (this.r - 10);
    this.playhead.clear();
    this.playhead.circle(px, py, 9).fill({ color: 0xf2c14e, alpha: 0.22 });
    this.playhead.circle(px, py, 5).fill({ color: 0xf2c14e });
  }

  /** fraction 1 = just started (full ring left), 0 = turn over. Used during
   * the human's live-reply turn (docs/imrpovedui.md §4.4's turn window). */
  setCountdown(fraction) {
    this.countdownArc.clear();
    if (fraction <= 0) return;
    const start = -Math.PI / 2;
    const end = start + Math.min(1, fraction) * Math.PI * 2;
    this.countdownArc.arc(this.cx, this.cy, this.r + 7, start, end).stroke({ width: 3, color: 0x8f7bd1, alpha: 0.7 });
  }
}

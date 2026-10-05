// Performer orbs — breathe with the drone; whoever is playing glows. During
// the "listening" phase, the human's notes visibly fly into the AI orb
// (docs/imrpovedui.md §4.6 point 1 — makes "the AI is taking in your line"
// a thing you can watch happen, not just a caption).

import { Container, Graphics, Text, TextStyle } from "https://cdn.jsdelivr.net/npm/pixi.js@8.22.0/+esm";
import { loopPulse, tween } from "./tween.js";

class Orb {
  constructor(app, { x, y, radius, color, label }) {
    this.app = app;
    this.x = x;
    this.y = y;
    this.radius = radius;
    this.color = color;
    this.active = false;

    this.root = new Container();
    this.root.x = x;
    this.root.y = y;
    app.stage.addChild(this.root);

    this.glow = new Graphics();
    this.body = new Graphics();
    this.root.addChild(this.glow, this.body);

    const text = new Text({
      text: label,
      style: new TextStyle({ fontFamily: "'Segoe UI', sans-serif", fontSize: 12, fill: 0xd8d2e8, letterSpacing: 2 }),
    });
    text.anchor.set(0.5);
    text.y = radius + 16;
    this.root.addChild(text);

    this._breathPhase = 0;
    loopPulse(3200, (v) => { this._breathPhase = v; this._redraw(); });
  }

  _redraw() {
    const breath = 1 + this._breathPhase * 0.06;
    const activeBoost = this.active ? 1.15 : 1;
    const scale = breath * activeBoost;

    this.glow.clear();
    this.glow.circle(0, 0, this.radius * scale * 1.6).fill({ color: this.color, alpha: this.active ? 0.22 : 0.1 });
    this.glow.circle(0, 0, this.radius * scale * 1.25).fill({ color: this.color, alpha: this.active ? 0.3 : 0.14 });

    this.body.clear();
    this.body.circle(0, 0, this.radius * scale).fill({ color: this.color, alpha: this.active ? 1 : 0.75 });
  }

  setActive(active) {
    this.active = active;
    this._redraw();
  }

  /** A quick outward pulse when this performer plays a note. */
  pulse() {
    tween(260, (t) => {
      this._extraScale = 1 + (1 - t) * 0.18;
      this._redrawWithExtra();
    }, { ease: "outCubic" });
  }

  _redrawWithExtra() {
    const breath = 1 + this._breathPhase * 0.06;
    const activeBoost = this.active ? 1.15 : 1;
    const scale = breath * activeBoost * (this._extraScale || 1);
    this.body.clear();
    this.body.circle(0, 0, this.radius * scale).fill({ color: this.color, alpha: this.active ? 1 : 0.75 });
  }

  get center() {
    return { x: this.root.x, y: this.root.y };
  }
}

export class Orbs {
  constructor(app, { you, ai }) {
    this.app = app;
    this.you = new Orb(app, { ...you, label: "YOU · sitar" });
    this.ai = new Orb(app, { ...ai, label: "AI · bansuri" });
    this.particleLayer = new Container();
    app.stage.addChild(this.particleLayer);
  }

  setTurn(who) {
    this.you.setActive(who === "you");
    this.ai.setActive(who === "ai");
  }

  /** Animates a small mote travelling from a river note's screen position
   * into the AI orb — the "listening" visual. */
  listenTravel(fromPoint, durationMs = 550) {
    const dot = new Graphics();
    dot.circle(0, 0, 4).fill({ color: this.ai.color, alpha: 0.9 });
    dot.x = fromPoint.x;
    dot.y = fromPoint.y;
    this.particleLayer.addChild(dot);

    const target = this.ai.center;
    const start = { x: fromPoint.x, y: fromPoint.y };
    tween(durationMs, (t) => {
      dot.x = start.x + (target.x - start.x) * t;
      dot.y = start.y + (target.y - start.y) * t;
      dot.alpha = 1 - t * 0.7;
    }, {
      ease: "inOutSine",
      onComplete: () => this.particleLayer.removeChild(dot),
    });
  }
}

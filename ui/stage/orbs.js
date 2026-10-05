// Performer orbs — breathe with the drone; whoever is playing glows. During
// the "listening" phase, the human's notes visibly fly into the AI orb
// (docs/imrpovedui.md §4.6 point 1 — makes "the AI is taking in your line"
// a thing you can watch happen, not just a caption).
//
// Pixi v7 (see stage/river.js's header comment — v8.x crashes when a
// Container's transform is mutated every tick with a Graphics beneath it,
// confirmed independent of this app via pixijs/pixijs#12048 and a minimal
// repro). Even on v7, animation here still only ever moves a transform
// (scale/alpha/position) rather than clearing and redrawing Graphics
// geometry every frame — not required for correctness on v7, but it's
// cheaper (no per-frame CPU geometry rebuild) and was already the better
// pattern regardless of which Pixi version avoids the bug.

import * as PIXI from "https://cdn.jsdelivr.net/npm/pixi.js@7.4.3/+esm";
import { loopPulse, tween } from "./tween.js";

const GLOW_OUTER_RATIO = 1.6;
const GLOW_INNER_RATIO = 1.25;

class Orb {
  constructor(app, { x, y, radius, color, label }) {
    this.app = app;
    this.radius = radius;
    this.color = color;
    this.active = false;
    this._breathScale = 1;
    this._pulseScale = 1;

    this.root = new PIXI.Container();
    this.root.x = x;
    this.root.y = y;
    app.stage.addChild(this.root);

    // Everything that breathes/pulses lives in here, scaled as a whole —
    // never redrawn per frame, only transformed.
    this.pulseLayer = new PIXI.Container();
    this.root.addChild(this.pulseLayer);

    this.glow = new PIXI.Graphics();
    this.body = new PIXI.Graphics();
    this.pulseLayer.addChild(this.glow, this.body);
    this._paint();

    const text = new PIXI.Text(
      label,
      new PIXI.TextStyle({ fontFamily: "'Segoe UI', sans-serif", fontSize: 12, fill: 0xd8d2e8, letterSpacing: 2 }),
    );
    text.anchor.set(0.5);
    text.y = radius + 16;
    this.root.addChild(text);

    loopPulse(3200, (v) => {
      this._breathScale = 1 + v * 0.06;
      this._applyScale();
    });
  }

  /** The one place Graphics content actually gets (re)drawn — on
   * construction and whenever `color` changes (setColor). Base geometry
   * only; activity and breathing are transforms applied on top. */
  _paint() {
    this.glow.clear();
    this.glow.beginFill(this.color, 1);
    this.glow.drawCircle(0, 0, this.radius * GLOW_OUTER_RATIO);
    this.glow.drawCircle(0, 0, this.radius * GLOW_INNER_RATIO);
    this.glow.endFill();

    this.body.clear();
    this.body.beginFill(this.color, 1);
    this.body.drawCircle(0, 0, this.radius);
    this.body.endFill();

    this._applyActiveAlpha();
  }

  setColor(color) {
    this.color = color;
    this._paint();
  }

  _applyActiveAlpha() {
    this.glow.alpha = this.active ? 0.9 : 0.55;
    this.body.alpha = this.active ? 1 : 0.75;
    this.pulseLayer.alpha = this.active ? 1 : 0.85;
  }

  setActive(active) {
    this.active = active;
    this._applyActiveAlpha();
    this._applyScale();
  }

  _applyScale() {
    const activeBoost = this.active ? 1.15 : 1;
    this.pulseLayer.scale.set(this._breathScale * activeBoost * this._pulseScale);
  }

  /** A quick outward pulse when this performer plays a note — a transform
   * tween, no redraw. */
  pulse() {
    tween(260, (t) => {
      this._pulseScale = 1 + (1 - t) * 0.18;
      this._applyScale();
    }, {
      ease: "outCubic",
      onComplete: () => { this._pulseScale = 1; this._applyScale(); },
    });
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
    this.particleLayer = new PIXI.Container();
    app.stage.addChild(this.particleLayer);
  }

  setTurn(who) {
    this.you.setActive(who === "you");
    this.ai.setActive(who === "ai");
  }

  /** Animates a small mote travelling from a river note's screen position
   * into the AI orb — the "listening" visual. Drawn once, then only moved
   * (position/alpha), never redrawn. */
  listenTravel(fromPoint, durationMs = 550) {
    const dot = new PIXI.Graphics();
    dot.beginFill(this.ai.color, 0.9);
    dot.drawCircle(0, 0, 4);
    dot.endFill();
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

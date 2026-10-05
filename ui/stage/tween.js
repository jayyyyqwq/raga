// Minimal requestAnimationFrame tween helper — shared by river/orbs/mandala
// so the stage doesn't need a full animation library (GSAP etc.) just for
// breathing orbs, beat pulses, and the listen-phase particle travel.

const EASE = {
  linear: (t) => t,
  outCubic: (t) => 1 - (1 - t) ** 3,
  inOutSine: (t) => -(Math.cos(Math.PI * t) - 1) / 2,
};

/**
 * Runs onUpdate(progress 0-1) every frame for durationMs, then calls
 * onComplete (if given). Returns a cancel() function.
 */
export function tween(durationMs, onUpdate, { ease = "outCubic", onComplete } = {}) {
  const start = performance.now();
  const easeFn = EASE[ease] || EASE.linear;
  let cancelled = false;

  function frame(now) {
    if (cancelled) return;
    const t = Math.min(1, (now - start) / durationMs);
    onUpdate(easeFn(t));
    if (t < 1) {
      requestAnimationFrame(frame);
    } else if (onComplete) {
      onComplete();
    }
  }
  requestAnimationFrame(frame);

  return () => { cancelled = true; };
}

/** A looping, phase-based oscillator for breathing/pulse effects — returns
 * cancel(). onUpdate receives a 0-1 value that rises and falls (sine). */
export function loopPulse(periodMs, onUpdate) {
  let cancelled = false;
  function frame(now) {
    if (cancelled) return;
    const phase = (now % periodMs) / periodMs;
    onUpdate((Math.sin(phase * Math.PI * 2) + 1) / 2);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  return () => { cancelled = true; };
}

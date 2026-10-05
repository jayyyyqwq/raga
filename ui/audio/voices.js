// Sampled instrument voices — real recordings, not synths (docs/AUDIO_SOURCES.md).
// WHY samplers: a Tone.PluckSynth (the old ui/sitar.js) is a Karplus-Strong
// approximation that sounds like a ringtone, not a sitar. These are actual
// FluidR3_GM note recordings (CC-BY 3.0 — see /ATTRIBUTION.md).

import * as Tone from "https://cdn.jsdelivr.net/npm/tone@15.1.22/+esm";

const NOTE_LETTERS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"];

// Absolute pitch (0-35: mandra 0-11, madhya 12-23, taar saptak 24-35 — the
// last octave only exists as an escalation voicing, see stage/theme.js's
// ESCALATION) -> sample note name, e.g. pitch 16 -> "Ga" -> "E4".
export function pitchToSampleName(pitch) {
  const octave = 3 + Math.floor(pitch / 12);
  const letter = NOTE_LETTERS[((pitch % 12) + 12) % 12];
  return `${letter}${octave}`;
}

const DURATION_SECONDS = [0.18, 0.35, 0.7, 1.4]; // index 0..3, at laya x1 (see Voice.play's speed param)

export class Voice {
  constructor(sampleDir, { volume = 0 } = {}) {
    this.ready = new Promise((resolve) => {
      const urls = {};
      for (let pitch = 0; pitch < 36; pitch++) {
        urls[pitchToSampleName(pitch)] = `${pitchToSampleName(pitch)}.mp3`;
      }
      this.sampler = new Tone.Sampler({
        urls,
        baseUrl: `samples/${sampleDir}/`,
        onload: resolve,
      }).toDestination();
      this.sampler.volume.value = volume;
    });
  }

  async loaded() {
    return this.ready;
  }

  /** Plays one note. `pitch` is absolute (0-35, see pitchToSampleName).
   * `durationIndex` is 0-3 (sixteenth..half). `speed` (default 1) is the
   * escalation ladder's laya multiplier — 2 for LAYA×2/TIHAI, shortening
   * the sample's held duration without changing its pitch (a tempo change,
   * not a pitch change — real meend/laya instruments don't speed up pitch
   * when they play faster). `time` is a Tone.js-scheduler time (seconds,
   * AudioContext-relative) or undefined for "now". */
  play(pitch, durationIndex, { time, speed = 1, velocity = 0.9 } = {}) {
    const name = pitchToSampleName(pitch);
    const seconds = DURATION_SECONDS[durationIndex] / speed;
    this.sampler.triggerAttackRelease(name, seconds, time, velocity);
    return seconds;
  }
}

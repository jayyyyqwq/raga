// Tabla — auto-plays on the Teentaal beat cycle (16 beats).
// Drives its own clock; syncs to tala_position from env state.
//
// WHY MembraneSynth: free, no samples needed, gives a percussive hit.
// In production you'd replace with actual tabla samples (.wav / .ogg).

import * as Tone from "https://cdn.skypack.dev/tone";

const BEAT_PATTERNS = {
  //  beat index → { gain, pitch } for each of 16 beats
  //  Teentaal: Dha Dhin Dhin Dha | Dha Dhin Dhin Dha | Dha Tin Tin Ta | Ta Dhin Dhin Dha
  0:  { gain: 1.0, pitch: "C1" },   // sam — heaviest
  4:  { gain: 0.8, pitch: "C1" },
  8:  { gain: 0.5, pitch: "F1" },   // khali — lighter
  12: { gain: 0.8, pitch: "C1" },
};

class Tabla {
  constructor(onBeat) {
    this.onBeat = onBeat;   // callback(beatIndex) — lets app.js sync tala display
    this.bpm = 80;
    this.beatDuration = 60 / this.bpm;  // seconds per beat
    this.intervalId = null;
    this.currentBeat = 0;

    this.synth = new Tone.MembraneSynth({
      pitchDecay: 0.08,
      octaves: 4,
      envelope: { attack: 0.001, decay: 0.3, sustain: 0, release: 0.1 },
    }).toDestination();

    this.hiSynth = new Tone.MembraneSynth({
      pitchDecay: 0.04,
      octaves: 3,
      envelope: { attack: 0.001, decay: 0.15, sustain: 0, release: 0.05 },
    }).toDestination();
    this.hiSynth.volume.value = -6;
  }

  start() {
    if (this.intervalId) return;
    Tone.start();
    this._tick();
    this.intervalId = setInterval(() => this._tick(), this.beatDuration * 1000);
  }

  stop() {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
  }

  syncBeat(envTalaPosition) {
    // Called after each env step to keep tabla in sync with env state
    this.currentBeat = envTalaPosition;
  }

  setBpm(bpm) {
    this.bpm = bpm;
    this.beatDuration = 60 / bpm;
    if (this.intervalId) {
      this.stop();
      this.start();
    }
  }

  _tick() {
    const beat = this.currentBeat % 16;
    const pattern = BEAT_PATTERNS[beat];

    if (pattern) {
      this.synth.triggerAttackRelease(pattern.pitch, "8n");
    } else {
      // Off-beats: quiet hi-tap
      this.hiSynth.triggerAttackRelease("G1", "16n");
    }

    this.onBeat(beat);
    this.currentBeat = (beat + 1) % 16;
  }
}

export { Tabla };

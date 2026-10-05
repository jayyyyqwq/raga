// Tanpura drone — loops under the whole performance. The single biggest
// "sounds like Hindustani music, not MIDI" multiplier (docs/imrpovedui.md §4.3).
// Tuned to C (Sa), matching Voice's Sa=C4 and the sample's own tuning
// (docs/AUDIO_SOURCES.md — Freesound #35476, marvman, CC0).

import * as Tone from "https://cdn.jsdelivr.net/npm/tone@15.1.22/+esm";

export class Drone {
  constructor() {
    this.gain = new Tone.Gain(0).toDestination();
    this.ready = new Promise((resolve) => {
      this.player = new Tone.Player({
        url: "samples/tanpura/tanpura_C.mp3",
        loop: true,
        fadeIn: 0.4,
        fadeOut: 0.4,
        onload: resolve,
      }).connect(this.gain);
    });
  }

  async loaded() {
    return this.ready;
  }

  start(targetVolumeDb = -18) {
    this.player.start();
    this.gain.gain.rampTo(Tone.dbToGain(targetVolumeDb), 2.5);
  }

  stop() {
    this.gain.gain.rampTo(0, 1.5);
    setTimeout(() => this.player.stop(), 1600);
  }
}

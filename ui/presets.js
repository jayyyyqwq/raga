// Recorded intros — stored as note data, not audio (docs/imrpovedui.md §4.2):
// the model needs the actual notes as input, and a real audio clip would
// need a pitch-tracker (and its errors) to get them back out. Pitches are
// absolute (0-23, matching raaga_env's encoding — mandra 0-11, madhya
// 12-23); durations are duration-index units (0=sixteenth..3=half).

export const INTROS = {
  // ṉNi Re Ga Ma# Ga Re Sa Sa — two Yaman pakads back to back
  // (ṉNi-Re-Ga and Ga-Ma#-Ga-Re-Sa), landing on madhya Sa.
  yaman: {
    pitches: [11, 14, 16, 18, 16, 14, 12, 12],
    durations: [1, 1, 1, 1, 1, 1, 2, 3],
  },
  // Ga Ma dha Pa Ga Ma re Sa — ends on the Ga-Ma-re-Sa pakad, touches
  // komal dha (vadi).
  bhairav: {
    pitches: [16, 17, 20, 19, 16, 17, 13, 12],
    durations: [1, 1, 1, 1, 1, 1, 2, 3],
  },
};

/** The call phrase (swara 0-11, matching JugalbandiEnv.set_call's contract)
 * extracted from an intro's last 4 notes — this is what actually gets
 * submitted to the env as your "call". */
export function callPhraseFromIntro(raga) {
  return INTROS[raga].pitches.slice(-4).map((p) => p % 12);
}

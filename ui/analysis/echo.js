// Pure functions: did the AI's line actually relate to the human's last
// line? Computed from the real notes, never staged — if nothing matches, no
// arc gets drawn and no claim gets made (docs/imrpovedui.md §4.6 / §2.1's
// "honest, not theatre" requirement).
//
// Both `callSwaras` and `responseSwaras` are swara values 0-11 (register
// stripped — a human call carries no octave, see JugalbandiEnv.set_call;
// response swaras are the AI's played notes reduced via note % 12).

/**
 * @typedef {{ fromIndex: number, toIndex: number, kind: string }} EchoArc
 */

/** Does the response touch the call's last swara within its first 2 notes? */
function findEcho(callSwaras, responseSwaras) {
  if (!callSwaras.length) return null;
  const target = callSwaras[callSwaras.length - 1];
  const window = responseSwaras.slice(0, 2);
  const hitIdx = window.indexOf(target);
  if (hitIdx === -1) return null;
  return {
    fromIndex: callSwaras.length - 1,
    toIndex: hitIdx,
    kind: "echo",
  };
}

function signsOf(seq) {
  const signs = [];
  for (let i = 1; i < seq.length; i++) {
    const d = seq[i] - seq[i - 1];
    signs.push(d > 0 ? 1 : d < 0 ? -1 : 0);
  }
  return signs;
}

/** Do the two phrases move in the same overall direction most of the time? */
function findContourMatch(callSwaras, responseSwaras) {
  const callSigns = signsOf(callSwaras);
  const respSigns = signsOf(responseSwaras);
  if (callSigns.length < 2 || respSigns.length < 2) return null;

  const n = Math.min(callSigns.length, respSigns.length);
  let agree = 0;
  for (let i = 0; i < n; i++) {
    if (callSigns[i] !== 0 && callSigns[i] === respSigns[i]) agree++;
  }
  if (agree < Math.ceil(n * 0.6)) return null;
  return {
    fromIndex: 0,
    toIndex: 0,
    kind: "contour",
  };
}

/** Does the response repeat the call's exact interval pattern, shifted to a
 * different starting pitch (a transposed answer — classic jugalbandi move)? */
function findTransposition(callSwaras, responseSwaras) {
  const callIntervals = [];
  for (let i = 1; i < callSwaras.length; i++) callIntervals.push(callSwaras[i] - callSwaras[i - 1]);
  const respIntervals = [];
  for (let i = 1; i < responseSwaras.length; i++) respIntervals.push(responseSwaras[i] - responseSwaras[i - 1]);

  const n = Math.min(callIntervals.length, respIntervals.length);
  if (n < 2) return null;
  for (let start = 0; start + n <= respIntervals.length; start++) {
    let match = true;
    for (let i = 0; i < n; i++) {
      if (callIntervals[i] !== respIntervals[start + i]) { match = false; break; }
    }
    if (match) {
      return { fromIndex: 0, toIndex: start, kind: "transposition" };
    }
  }
  return null;
}

/** Everything the UI needs to draw/caption one AI response against the call
 * that preceded it: a list of arcs to draw (possibly empty) and a one-line
 * caption (possibly the honest "no match found" line). */
export function analyzeResponse(callSwaras, responseSwaras) {
  const arcs = [];
  const echo = findEcho(callSwaras, responseSwaras);
  if (echo) arcs.push(echo);

  const transposition = findTransposition(callSwaras, responseSwaras);
  if (transposition) arcs.push(transposition);

  const contour = !transposition ? findContourMatch(callSwaras, responseSwaras) : null;
  if (contour) arcs.push(contour);

  let caption;
  if (transposition) {
    caption = "answered in the same shape, shifted";
  } else if (echo && contour) {
    caption = "picked up your last note, matched your direction";
  } else if (echo) {
    caption = "landed on your last note";
  } else if (contour) {
    caption = "matched your direction";
  } else {
    caption = "a new idea in the raga, not obviously tied to your line";
  }

  return { arcs, caption };
}

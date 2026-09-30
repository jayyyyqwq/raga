// Thin wrapper around the FastAPI server.
// Everything in app.js talks to the env through this — no raw fetch calls elsewhere.

const BASE = window.ENV_SERVER_URL || "http://localhost:7860";

async function _post(path, body = {}) {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path} → ${res.status} ${await res.text()}`);
  return res.json();
}

async function _get(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${path} → ${res.status}`);
  return res.json();
}

export const EnvClient = {
  reset: (dial = 0.0, seed = null) => _post("/reset", { dial, seed }),
  step:  (action)                  => _post("/step",  { action }),
  setDial: (value)                 => _post("/set_dial", { value }),
  submitCall: (notes)              => _post("/call",   { notes }),
  state: ()                        => _get("/state"),
  health: ()                       => _get("/health"),
  infer: ()                        => _post("/infer",  {}),
};

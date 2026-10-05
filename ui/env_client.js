// Thin wrapper around the FastAPI server.
// Everything in app.js talks to the env through this — no raw fetch calls elsewhere.
//
// Default is same-origin ("") — openenv_server/server.py now serves this ui/
// directory itself (docs/imrpovedui.md's "one process, one port" decision),
// so the page and the API share an origin and need no CORS. Set
// window.ENV_SERVER_URL before this module loads only if running the UI
// from somewhere else (e.g. a separate static file server) against a
// server.py on a different host/port.

const BASE = window.ENV_SERVER_URL || "";

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

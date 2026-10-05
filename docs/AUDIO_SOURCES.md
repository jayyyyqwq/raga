# Audio Sources

Free, pre-recorded samples for the Jugalbandi stage (see [`imrpovedui.md`](imrpovedui.md) §5). Verified 2026-10-04. Nothing is downloaded yet.

All files get downloaded once into `ui/samples/` and served locally; nothing loads from a CDN at runtime.

---

## 1. Melodic instruments — FluidR3_GM (pre-rendered by gleitz/midi-js-soundfonts)

- Repo: https://github.com/gleitz/midi-js-soundfonts
- Licence: **CC-BY 3.0** — attribution required (see §5).
- One mp3 per note. URL pattern:
  `https://gleitz.github.io/midi-js-soundfonts/FluidR3_GM/<instrument>-mp3/<Note><Octave>.mp3`
- Note names use flats: `C Db D Eb E F Gb G Ab A Bb B`.
- Checked live (HTTP 200): `C3 Db4 C4 C5 C6` for all three instruments below.

| Role | Instrument | Folder | Notes needed |
|---|---|---|---|
| You | Sitar | `sitar-mp3/` | C3–B5 (36 files) |
| AI (option 1) | Bansuri — GM `flute` | `flute-mp3/` | C3–B5 (36 files) |
| AI (option 2) | Shehnai — GM `shanai` | `shanai-mp3/` | C3–B5 (36 files) |

Why C3–B5: Sa = C4. Env pitches 0–23 = C3–B4 (mandra + madhya); taar saptak escalation adds +12 → up to B5.

Target layout:

```text
ui/samples/sitar/C3.mp3 … B5.mp3
ui/samples/flute/C3.mp3 … B5.mp3
ui/samples/shanai/C3.mp3 … B5.mp3
```

Caveat: GM sitar/flute/shanai are soundfont approximations, not concert recordings. Meend (pitch glides) + tanpura drone do most of the work of making them sound Hindustani.

---

## 2. Tanpura drone — Freesound

| File | Author | Link | Licence | Details |
|---|---|---|---|---|
| `tanpura2.wav` | marvman | https://freesound.org/people/marvman/sounds/35476/ | **CC0** | 5.8 s, 44.1 kHz/16-bit stereo, tuned to C (Sa–Pa) |

- Matches Sa = C4. Loop with a short crossfade.
- **Downloaded (2026-10).** Turned out not to need a login: the sound page's own public preview CDN URL (`cdn.freesound.org/previews/35/35476_111012-hq.mp3`) is fetchable anonymously — that's what actually plays the waveform preview for anyone visiting the page logged out. Saved as `ui/samples/tanpura/tanpura_C.mp3` (the preview is an mp3, not the original wav — fine for a looping background drone).
- Backup candidates (CC0, other keys, would need pitch-shifting): Freesound #148850 "Tanpura in E" (iluppai), #506312 "Tambura_Eb_fat" (Kaczinski).

---

## 3. Tabla — open

| Option | Source | Licence | Coverage |
|---|---|---|---|
| 1 | Freesound pack "tabla bols" by mmiron — https://freesound.org/people/mmiron/packs/8162/ | **Not shown on pack page — check each sound's page before using** | na, na-open, na-sharp, tun, te, re, ke, tas. **No dha / dhin / ge** → dha would be layered as na + a bass stroke |
| 2 | gleitz `Tabla` soundfont — https://gleitz.github.io/midi-js-soundfonts/Tabla/ | **Not stated** — don't use until licence is known | Mapped to C4–E6 |
| 3 | Keep current `Tone.MembraneSynth` tabla (`ui/audio/tabla.js`) | n/a | Works now, sounds synthetic |

**Decided (2026-10): option 3**, for the first build of the new stage. Melodic voices (sitar/flute) are now real samples and carry most of the "sounds like Hindustani music" effect; tabla stays synth until a licence on a real pack is confirmed.

---

## 4. Download checklist (phase 0) — done, 2026-10

- [x] `scripts/fetch_samples.py` pulls sitar + flute note samples (72 files, C3–B5) into
      `ui/samples/<instrument>/` (pathlib + `urllib` only — no new deps). Shehnai was planned as a
      third option but shipped as sitar + flute (bansuri) only — see decision in §1; shehnai stays
      an easy later swap (`INSTRUMENTS` dict in the script).
- [x] Tanpura #35476 downloaded without needing a login — see §2.
- [x] Tabla: decided option 3 (keep the synth) — see §3.
- [x] Wrote `/ATTRIBUTION.md` at repo root.
- [x] **Commit samples** — ~110 files, ≈1.8MB total. Small enough to just check in; no gitignore
      exception needed.

---

## 5. Attribution

Moved to [`/ATTRIBUTION.md`](../ATTRIBUTION.md) (repo root) — that's the one a repo visitor will
actually find; this file stays the sourcing worksheet.

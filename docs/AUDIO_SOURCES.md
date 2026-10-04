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
- Freesound downloads need a (free) account login — download manually, save as `ui/samples/tanpura/tanpura_C.wav`.
- Backup candidates (CC0, other keys, would need pitch-shifting): Freesound #148850 "Tanpura in E" (iluppai), #506312 "Tambura_Eb_fat" (Kaczinski).

---

## 3. Tabla — open

| Option | Source | Licence | Coverage |
|---|---|---|---|
| 1 | Freesound pack "tabla bols" by mmiron — https://freesound.org/people/mmiron/packs/8162/ | **Not shown on pack page — check each sound's page before using** | na, na-open, na-sharp, tun, te, re, ke, tas. **No dha / dhin / ge** → dha would be layered as na + a bass stroke |
| 2 | gleitz `Tabla` soundfont — https://gleitz.github.io/midi-js-soundfonts/Tabla/ | **Not stated** — don't use until licence is known | Mapped to C4–E6 |
| 3 | Keep current `Tone.MembraneSynth` tabla (`ui/tabla.js`) | n/a | Works now, sounds synthetic |

Default until resolved: option 3.

---

## 4. Download checklist (phase 0)

- [ ] Script `scripts/fetch_samples.py` pulls the 3 × 36 FluidR3 mp3s into `ui/samples/<instrument>/` (pathlib, `urllib` only — no new deps)
- [ ] Manually download tanpura #35476 (Freesound login)
- [ ] Decide tabla option; if 1, open each sound page and confirm licence
- [ ] Write `ATTRIBUTION.md` at repo root (§5)
- [ ] Commit samples? ~110 small mp3s (≈25 KB each, ≈3 MB) — fine to commit; or gitignore + fetch script. Decide in phase 0.

---

## 5. Attribution text (for `ATTRIBUTION.md`)

```text
Sitar, flute and shanai samples: FluidR3_GM soundfont by Frank Wen, pre-rendered
by Benjamin Gleitzman (github.com/gleitz/midi-js-soundfonts), CC BY 3.0
(creativecommons.org/licenses/by/3.0/).

Tanpura drone: "tanpura2.wav" by marvman (freesound.org/s/35476/), CC0.
```

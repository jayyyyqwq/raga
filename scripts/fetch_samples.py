"""Re-downloads the FluidR3_GM note samples into ui/samples/<instrument>/.

The samples are already committed to this repo (ui/samples/ — ~1.8MB of
mp3s, small enough to just check in; see docs/AUDIO_SOURCES.md). This script
exists for two cases: re-fetching if a file ever goes missing, or adding
another instrument (e.g. shehnai — docs/imrpovedui.md §11 names it as the
easy alternative AI voice; swap INSTRUMENTS below and re-run).

Usage:
    python scripts/fetch_samples.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "https://gleitz.github.io/midi-js-soundfonts/FluidR3_GM"

# GM soundfont folder name -> local samples subdirectory.
INSTRUMENTS = {
    "sitar": "sitar",
    "flute": "flute",   # the AI's voice (bansuri stand-in) — see docs/AUDIO_SOURCES.md §1
}

OCTAVES = (3, 4, 5)  # covers env pitch 0-23 plus the taar-saptak (+12) escalation voicing
NOTE_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def fetch_instrument(gm_name: str, local_dir: str) -> None:
    dest_dir = REPO_ROOT / "ui" / "samples" / local_dir
    dest_dir.mkdir(parents=True, exist_ok=True)

    ok = skipped = failed = 0
    for octave in OCTAVES:
        for name in NOTE_NAMES:
            filename = f"{name}{octave}.mp3"
            dest = dest_dir / filename
            if dest.exists() and dest.stat().st_size > 0:
                skipped += 1
                continue
            url = f"{BASE_URL}/{gm_name}-mp3/{filename}"
            try:
                urllib.request.urlretrieve(url, dest)
                ok += 1
            except Exception as exc:  # noqa: BLE001 - report and keep going
                failed += 1
                print(f"  FAILED {url}: {exc}")

    print(f"{local_dir}: {ok} downloaded, {skipped} already present, {failed} failed")


def main() -> None:
    for gm_name, local_dir in INSTRUMENTS.items():
        fetch_instrument(gm_name, local_dir)


if __name__ == "__main__":
    main()

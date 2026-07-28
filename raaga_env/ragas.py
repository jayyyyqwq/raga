# Raga rule definitions — the DSL at the heart of Jugalbandi.
#
# Two-octave absolute pitch encoding:
#   Mandra (lower) saptak : 0-11
#   Madhya (home)  saptak : 12-23
#
# Swara identity is always  note % 12  →  Sa=0 Re♭=1 Re=2 Ga♭=3 Ga=4 Ma=5 Ma#=6
#                                          Pa=7 Dha♭=8 Dha=9 Ni♭=10 Ni=11
#
# Fields that hold swara integers (0-11, checked via % 12):
#   valid_notes, forbidden_notes, vadi, samvadi, weak_notes, aaroha_swaras
#
# Fields that hold absolute pitch integers (register matters):
#   aaroha, avaroha, and pakad phrase lists inside pakads

RAGAS: dict[str, dict] = {
    "yaman": {
        # --- scale ---
        "valid_notes":    {0, 2, 4, 6, 7, 9, 11},
        "forbidden_notes": {5},          # natural Ma (shuddha Ma) — strictly forbidden; Yaman's defining vakra

        # --- melodic contour (absolute pitches) ---
        # Starts from mandra Ni (11) per Bhatkhande; Pa skipped on ascent (alpa/varjit in aaroha).
        "aaroha":  [11, 14, 16, 18, 21, 23],   # ṉNi Re Ga Ma# Dha Ni
        "avaroha": [23, 21, 19, 18, 16, 14, 12], # Ni Dha Pa Ma# Ga Re Sa  — Pa reappears descending

        # Swara-level ascending rule (Pa excluded). Used for direction-sensitive validation.
        "aaroha_swaras": {0, 2, 4, 6, 9, 11},

        # --- hierarchy ---
        "vadi":    4,   # Ga — king note; raga lives in uttarang
        "samvadi": 11,  # Ni — minister; Bhatkhande, KPM Vol. I

        # --- characteristic phrases (pakad) —---
        # Format: (phrase as absolute pitches, reward multiplier)
        # Shorter prefixes have lower reward → implicit curriculum: agent learns fragments
        # first, earns bigger reward once it chains the full canonical phrase.
        # Source: Bhatkhande KPM Vol. I; cross-checked Parrikar raga archive (parrikar.org).
        "pakads": [
            ([11, 14, 16],          0.5),   # ṉNi Re Ga — cross-register opening; most iconic entry
            ([16, 18, 21, 23],      0.7),   # Ga Ma# Dha Ni — uttarang ascending sweep
            ([16, 18, 16, 14, 12],  1.2),   # Ga Ma# Ga Re Sa — signature turn + resolution (non-scalar ✓)
            ([23, 21, 19, 18, 16],  1.0),   # Ni Dha Pa Ma# Ga — avaroha body with Pa (non-scalar ✓)
        ],

        # Notes to penalise on sam (beat 1). Pa is alpa-swara and varjit in aaroha.
        "weak_notes": {7},

        "max_smooth_interval": 7,
        "mood":        "shringar",
        "time_of_day": "evening",
    },

    "bhairav": {
        # --- scale ---
        "valid_notes":    {0, 1, 4, 5, 7, 8, 11},
        "forbidden_notes": {2, 9},       # natural Re and natural Dha — forbidden throughout

        # --- melodic contour (absolute pitches) ---
        "aaroha":  [12, 13, 16, 17, 19, 20, 23],   # Sa re Ga Ma Pa dha Ni
        "avaroha": [23, 20, 19, 17, 16, 13, 12],    # Ni dha Pa Ma Ga re Sa

        # All valid notes are allowed ascending in Bhairav (no vakra).
        "aaroha_swaras": {0, 1, 4, 5, 7, 8, 11},

        # --- hierarchy ---
        # Source: Bhatkhande KPM Vol. I; Deepak Raja "Hindustani Music: A Tradition in Transition" Ch. 2.
        # The andolan (oscillation) on komal Dha and komal Re is what defines Bhairav's rasa;
        # these notes are therefore vadi/samvadi, NOT Ma/Sa as commonly mislabelled.
        "vadi":    8,   # komal Dha — primary andolan note
        "samvadi": 1,   # komal Re  — partner andolan note

        # --- characteristic phrases (pakad) ---
        # Source: Bhatkhande KPM; Parrikar raga archive.
        # Note: canonical Bhairav pakad uses andolan on dha and re, which discrete notes can't encode.
        # Phrases below are the closest discrete approximations.
        "pakads": [
            ([12, 13, 16, 17],  0.5),   # Sa re Ga Ma — standard opening
            ([20, 19, 17, 16],  0.7),   # dha Pa Ma Ga — komal-Dha descent (vadi emphasis)
            ([16, 17, 13, 12],  1.0),   # Ga Ma re Sa — resolution through komal Re (samvadi emphasis)
            ([13, 12, 11, 12],  1.2),   # re Sa ṉNi Sa — cross-register close; most distinctive (non-scalar ✓)
        ],

        # Ma on sam is unusual in Bhairav (gravity is on dha/re, not Ma).
        "weak_notes": {5},

        "max_smooth_interval": 7,
        "mood":        "devotion",
        "time_of_day": "morning",
    },
}

TALAS: dict[str, dict] = {
    "teentaal": {
        "beats":        16,
        "sam":          0,
        "khali":        8,
        "strong_beats": {0, 4, 12},
        "weak_beats":   {8},
    }
}

# Duration index → 16th-note units
DURATIONS: dict[int, int] = {0: 1, 1: 2, 2: 4, 3: 8}

# Human-readable note names for both saptaks.
# ṉ prefix = mandra (lower octave).
NOTE_NAMES: list[str] = [
    "ṉSa", "ṉRe♭", "ṉRe", "ṉGa♭", "ṉGa", "ṉMa", "ṉMa#", "ṉPa", "ṉDha♭", "ṉDha", "ṉNi♭", "ṉNi",  # 0-11
    "Sa",  "Re♭",  "Re",  "Ga♭",  "Ga",  "Ma",  "Ma#",  "Pa",  "Dha♭",  "Dha",  "Ni♭",  "Ni",   # 12-23
]


def raga_from_dial(dial: float) -> str:
    """Map a 0.0–1.0 dial value to a raga name. Crossing 0.5 is the drift point."""
    return "yaman" if dial < 0.5 else "bhairav"


def is_valid_note(note: int, raga: dict, direction: int) -> bool:
    """Swara-level validation. note is absolute (0-23); rules use % 12."""
    swara = note % 12
    if swara in raga["forbidden_notes"]:
        return False
    if swara not in raga["valid_notes"]:
        return False
    if direction == 1:
        return swara in raga["aaroha_swaras"]
    return True

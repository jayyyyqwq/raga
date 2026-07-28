# Classical Music Audit — Steps 1 & 2

**Scope.** Audit of `raaga_env/ragas.py` (Step 1) and `raaga_env/env.py` (Step 2) against Hindustani classical music theory (primarily Bhatkhande's codification, which is what your DSL implicitly follows).

**Verdict headline.** Step 1 is **~70% accurate**. Step 2 faithfully implements what Step 1 provides but inherits its errors plus a reachability bug. Core grammar (scales, forbidden notes, tala structure) is correct. The identity-defining parts of each raga (vadi/samvadi, pakad, andolan) are where the errors live — and those are the parts an informed listener or judge will notice first.

---

## 1. What the project is actually modelling

Hindustani classical music. Two ragas from two different thaats:

| Raga | Thaat | Time | Rasa | Defining features |
|---|---|---|---|---|
| Yaman | Kalyan | Evening (2nd prahar of night) | Shringar / longing | Teevra Ma (Ma#), shuddha rest; aaroha often skips Pa; starts from lower Ni |
| Bhairav | Bhairav | Dawn (1st prahar of day) | Devotion / gravitas | Komal Re and komal Dha with **andolan** (slow oscillation); all other notes shuddha |

Everything else — tala (rhythm cycle), pakad (signature phrase), vadi/samvadi (king/minister notes), aaroha/avaroha (ascent/descent scales) — are per-raga modifiers of this framework.

---

## 2. What the code gets right

### Scales and forbidden notes — **100% correct**
- Yaman swaras `{0,2,4,6,7,9,11}` with Ma (5) forbidden ✓ matches Kalyan thaat with teevra Ma.
- Bhairav swaras `{0,1,4,5,7,8,11}` with natural Re (2) and natural Dha (9) forbidden ✓ matches komal-Re / komal-Dha signature.

### Yaman vadi/samvadi — correct
- vadi=Ga(4), samvadi=Ni(11). Matches Bhatkhande's *Hindustani Sangeet Paddhati*. These are in the uttarang (upper tetrachord), which is the raga's poorvang/uttarang identity (Yaman = uttarang-pradhan).

### Teentaal structure — correct
- 16 beats, sam=0, khali=8, talis at 0/4/12. Textbook correct.

### Direction-sensitive aaroha (the novel claim in Step 1) — defensible
- Pa being valid descending but avoided ascending in Yaman is a real Bhatkhande-era convention (`varjit in aaroha`). Modern performance is laxer, but you can defend it as a classical-strict policy.

### Time-of-day tags — correct
- Yaman evening, Bhairav morning. Matches prahar theory.

---

## 3. What the code gets wrong or oversimplifies

### 3.1 Bhairav vadi/samvadi — **wrong**

```python
"vadi": 5,      # Ma
"samvadi": 0,   # Sa
```

Traditional Bhairav: **vadi = komal Dha (8), samvadi = komal Re (1)**. This is not a preference — it is the raga's core identity. Bhairav's rasa lives in the oscillation (andolan) on komal Dha and komal Re. By rewarding Ma and Sa as king/minister, the agent will learn a melodic shape that is *valid* Bhairav but not *recognisable* Bhairav. Any musician-judge will catch this in the demo.

**Fix.** `"vadi": 8, "samvadi": 1`. Pakads and sam-landing rewards will then point at the right notes automatically.

### 3.2 Yaman weak_notes — questionable

```python
"weak_notes": {2, 6}   # Re and Ma#
```

Re on sam is not weak in Yaman — several canonical Yaman bandishes resolve on Re. Ma# is the raga's *jaan* (signature) note; landing it on sam is unconventional but not "weak". Calling these weak will push the agent away from idiomatic Yaman.

**Better candidates for weak_notes in Yaman:** Pa (7) because Pa is alpa-swara and varjit in aaroha. Sa is strongest, Ga is vadi-strong, Ni is samvadi-strong, Re and Dha are moderate.

### 3.3 Pakads — valid phrases, but not canonical

The pakads encoded are all *legal* Yaman/Bhairav movements, but none are the textbook signature phrases that make the raga instantly recognisable:

| Raga | Canonical pakad (Bhatkhande) | Encoded |
|---|---|---|
| Yaman | `Ni Re Ga, Re Ga (Ma#) Re Sa` or `Pa Ma# Ga Re Sa` | Generic ascending/descending sweeps |
| Bhairav | `Ga Ma dha~ Pa, Ga Ma Re~ Sa` (~ = andolan) | 4-note sweeps without andolan |

The bigger problem is that **pakad in Hindustani music is not a discrete note sequence**. It includes andolan, meend (glide), and kan-swar (grace notes). Discrete-note pakad matching is a necessary simplification for RL, but the choice of which phrases to encode determines whether the agent learns Yaman or learns "a scale that happens to skip Ma".

**Fix.** Replace with the most iconic 4–5 phrases from a standard text (Bhatkhande's *Kramik Pustak Malika* is the primary source). For Yaman: `[11,2,4]` (Ni Re Ga — from below-Sa Ni, but see 3.5), `[4,6,4,2,0]` (Ga Ma# Ga Re Sa), `[7,6,4,2,0]` (Pa Ma# Ga Re Sa).

### 3.4 Octave bug — **pakad is unreachable**

Action space is `Discrete(48) = 12 notes × 4 durations`, decoded as `note = action % 12`. The agent can only emit notes 0–11. But Yaman pakad `[11, 12, 11, 9]` contains **note 12 (Sa of the upper octave)**. The agent can never complete this pakad and will never collect its reward.

This is a code bug, not a music theory bug. Two fixes:
- Expand action space to cover ≥2 octaves (e.g., 24 notes × 4 durations = 96 actions), or
- Rewrite pakads in mod-12 form and treat octave as a separate state dim.

The first is better musically — raga phrasing genuinely needs at least the lower Ni (Ni below Sa, i.e., note −1 or 11 in a lower octave) and the upper Sa. Hindustani phrases routinely span 1.5–2 octaves.

### 3.5 Yaman aaroha starts from Sa — simplification

Code: `[0, 2, 4, 6, 9, 11, 12]` (Sa Re Ga Ma# Dha Ni Sa').
Classical Yaman aaroha: `n R G M̄ D N S'` — **starts from lower Ni (below Sa)**. This lower-Ni entry is Yaman's characteristic opening. Without lower octave in the action space (see 3.4), this can't be encoded anyway. Acceptable simplification — but worth flagging in your pitch.

### 3.6 Missing musical primitives

Step 1 declares `ORNAMENTS = {0: none, 1: meend, 2: kan}` but Step 2's action space is flat Discrete(48) with no ornament dimension. They are decorative in the code and unused.

More importantly, these are missing entirely:
- **Andolan** — the slow oscillation on komal Dha / komal Re that defines Bhairav. Not expressible in discrete-note output.
- **Shruti / microtones** — 12-TET semitones are a Western approximation. Komal Re in Bhairav is actually flatter than a Western minor 2nd; Ga in Yaman is slightly sharper. RL-wise, fine. Musicologically, it's a lossy representation and judges will ask.
- **Gamak, kan-swar** — grace notes.
- **Poorvang/uttarang balance** — Yaman is uttarang-pradhan (upper-tetrachord-dominant). No reward signal encodes this, so the agent will not naturally emphasise the upper half.

---

## 4. Confidence summary

| Item | Status | Notes |
|---|---|---|
| Yaman scale (valid/forbidden) | ✅ Correct | |
| Bhairav scale (valid/forbidden) | ✅ Correct | |
| Yaman vadi/samvadi | ✅ Correct | Ga/Ni per Bhatkhande |
| Bhairav vadi/samvadi | ❌ **Wrong** | Should be komal Dha / komal Re, not Ma / Sa |
| Aaroha/avaroha shape | ⚠️ Simplified | No lower-octave entry; direction rule defensible |
| Pakads | ⚠️ Valid but generic | Not the canonical signature phrases |
| Yaman weak_notes | ⚠️ Questionable | Re shouldn't be weak |
| Teentaal structure | ✅ Correct | |
| Time-of-day tags | ✅ Correct | |
| Ornaments | ⚠️ Declared but unused | No ornament dim in action space |
| Octave range | ❌ **Bug** | Pakad uses note 12; action space tops at 11 |
| Andolan / shruti / gamak | ❌ Absent | Unavoidable for discrete RL, but limits authenticity |

**Numeric estimate.** Step 1 is ~70% faithful to classical Hindustani theory. The correct bits are the broad-grammar bits (scales, thaat, tala) — the parts a Wikipedia reader could verify. The wrong/weak bits are the identity-and-affect bits (vadi for Bhairav, pakad choices, andolan) — the parts a trained ear notices.

Step 2 does not introduce new musical errors, but it inherits Step 1's errors and adds the octave-reachability bug. Environment mechanics (observation vector, direction detection, drought counters, reward plumbing) are sound.

---

## 5. Recommended fixes before the hackathon demo

In descending order of impact-per-effort:

1. **Fix Bhairav vadi/samvadi** → `vadi=8, samvadi=1`. One-line change, biggest credibility recovery.
2. **Fix octave bug** → expand action space to 2 octaves (24 notes) or rewrite pakads in mod-12 and track octave separately. Otherwise one of your Yaman pakads is dead code.
3. **Replace pakads with canonical phrases from Bhatkhande.** ~1 hour of lookup per raga.
4. **Remove `weak_notes={2,6}` in Yaman** or replace with `{7}` (Pa as alpa-swara).
5. **Add a one-line "simplifications" disclaimer** to the README: "Discrete 12-TET notes; no andolan/meend/microtones. This is an RL grammar layer, not a performance synthesiser." Defuses the Q&A.
6. *(Stretch.)* Add an `andolan_flag` per-note action for Bhairav's komal Dha/Re, wired to a reward bonus when held on strong beats. Even a crude implementation shows you know what's missing.

---

## 6. Sources the report leans on

- V.N. Bhatkhande, *Hindustani Sangeet Paddhati* (standard-text vadi/samvadi, aaroha/avaroha, pakad).
- Standard thaat theory (Kalyan for Yaman, Bhairav for Bhairav).
- Prahar time-theory (Yaman = 2nd prahar of night; Bhairav = 1st prahar of day).

No proprietary or disputed sources used — all the above is in any introductory Hindustani theory textbook or the ITC Sangeet Research Academy reference corpus.

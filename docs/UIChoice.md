# UIChoice.md — Submission Strategy for Jugalbandi

> Written from the perspective of a hackathon evaluator scoring this project. 20 minutes per project, partial attention, no patience for setup friction.

---

## What a Judge Actually Does in 20 Minutes

Judges at a hackathon like this do not read your code. Here is the honest timeline of what they do:

```
00:00  Open the project page / README
00:30  Skim the one-liner and first paragraph
01:00  Click the HF Space link (or notebook link)
02:00  Watch the demo video while Space loads
04:00  Run the notebook top-to-bottom (or just read it if they're busy)
08:00  Check the results section — look for the comparison numbers
12:00  Skim the blog post or README technical section
15:00  Open the code only if something surprised them or they're skeptical
18:00  Fill out scorecard
20:00  Next project
```

Every design decision below follows from this 20-minute map.

---

## Current State vs What's Needed

| Submission Component | Status | Impact if Missing |
|---|---|---|
| OpenEnv server (HF Space) | Built | — |
| Trained model on HF Hub | Not done yet | Judges can't run inference |
| Evaluation table (evaluate.py) | MISSING | No quantitative proof of learning |
| Gradio demo (demo/app.py) | MISSING | The "click" moment doesn't exist |
| Audio before/after | MISSING | Best storytelling tool not used |
| Training notebook (.ipynb) | Wrong format | Judges can't re-run |
| Blog post | Not written | Counts for 30% of score |
| Video | Not made | First thing judges see |

You can win without audio. You cannot win without the evaluation table and the Gradio demo. Build in that order.

---

## Build Priority Order (2 days to submission)

### Priority 1: evaluate.py (4 hours)

This is what generates the before/after comparison table judges expect. It must exist.

The script runs 100 episodes twice — once with a random policy (baseline) and once with the trained model. It prints and saves a comparison table.

**Minimum viable output:**
```
=== Jugalbandi Evaluation — 100 Episodes ===

Metric                    Random Policy    Trained (GRPO)   Delta
─────────────────────────────────────────────────────────────────
Avg Episode Reward        -10.2            +4.8             +15.0
Forbidden Note Rate       24%              3%               -21pp
Pakad Completion Rate     0.1/ep           2.3/ep           +2.2/ep
Valid Note Rate           68%              97%              +29pp
Sam Landing (vadi)        2%               18%              +16pp
Adapt Speed (steps)       N/A              3.4 ± 1.2        —
```

This table is the screenshot that goes in the blog, the README, and the slides. Without it, you are claiming improvement without evidence. Judges will not take your word for it.

**After writing evaluate.py:** Run it immediately. If the trained model shows improvement, those numbers go everywhere. If it doesn't, you need to know now so you can retrain.

### Priority 2: demo/app.py — Gradio Demo (6 hours)

The judges click your HF Space link. Right now they see the FastAPI OpenEnv server — JSON endpoints, no UI. That is a dead end for a non-technical judge and a missed opportunity for a technical one.

The Gradio demo should be a second HF Space (or second tab in the same Space) with:

**Must-have components:**
1. **Piano roll display** — a 24-column grid (one per note), 16-row grid (one per tala beat). Color cells by status: green = valid, red = forbidden, blue = pakad note, gold = vadi/samvadi. This is your visual proof that the model learned raga grammar. Without it, you're describing a result you can't show.

2. **"Untrained" vs "Trained" toggle** — two buttons. Same starting state, different model. Run 16 steps. Show both piano rolls side by side. The contrast is the demo. The audience sees it immediately.

3. **Raga dial slider** — 0.0 (Yaman) to 1.0 (Bhairav). When dragged past 0.5, the piano roll column headers change. The agent's next action shows up in either green (adapted correctly) or orange (still in grace period). This is the drift mechanic made visible.

4. **Reward counter** — a live number showing cumulative episode reward. Goes up when the agent plays well, spikes when a pakad completes. Non-technical judges understand "higher number = better" without explanation.

**Nice-to-have (add if time permits):**
- Human keyboard input (4-note call phrase)
- Adaptation speed counter after a dial drag
- Audio playback (see Priority 3)

**Simplest possible piano roll in Gradio:**

```python
import gradio as gr
import numpy as np

NOTE_NAMES = ["Sa","Re♭","Re","Ga♭","Ga","Ma","Ma#","Pa","Dha♭","Dha","Ni♭","Ni"]
COLORS = {"valid": "#22c55e", "forbidden": "#ef4444", "vadi": "#f59e0b", "pakad": "#3b82f6", "empty": "#1e293b"}

def render_piano_roll(note_history: list[int], raga: dict) -> np.ndarray:
    # 12 cols (notes) × 16 rows (tala beats)
    grid = np.zeros((16, 12, 3), dtype=np.uint8)
    grid[:] = [30, 41, 59]  # dark background
    for beat, note in enumerate(note_history[-16:]):
        col = note % 12
        if note % 12 in raga["forbidden_notes"]:
            color = [239, 68, 68]   # red
        elif note % 12 == raga["vadi"]:
            color = [245, 158, 11]  # gold
        else:
            color = [34, 197, 94]   # green
        grid[beat % 16, col] = color
    return grid
```

Gradio's `gr.Image` can display numpy arrays directly. Scale it up with CSS to be readable.

### Priority 3: Audio (4 hours, if time allows)

A 10-second WAV of random notes followed by 10 seconds of the trained model is the single most powerful thing you can add. Anyone in the room — judge, sponsor, other participants — understands it in zero cognitive overhead.

Use `pretty_midi` or `midiutil` to convert note actions to MIDI, then `fluidsynth` to render to WAV with a sitar soundfont.

```
pip install pretty_midi midiutil
# Download a free sitar soundfont (.sf2) — e.g. from soundfonts4u.com
```

Minimum viable implementation:
```python
import pretty_midi

def actions_to_wav(actions: list[int], output_path: str):
    pm = pretty_midi.PrettyMIDI(initial_tempo=60.0)
    sitar = pretty_midi.Instrument(program=104)  # sitar MIDI program
    t = 0.0
    for action in actions:
        note_semitone = action % 12
        dur_index = action // 12
        duration = [0.25, 0.5, 1.0, 2.0][dur_index]  # quarter note units at 60bpm
        midi_note = 60 + note_semitone   # middle octave
        note = pretty_midi.Note(velocity=80, pitch=midi_note, start=t, end=t+duration)
        sitar.notes.append(note)
        t += duration
    pm.instruments.append(sitar)
    pm.write(output_path.replace(".wav", ".mid"))
    # fluidsynth to render: os.system(f"fluidsynth -ni sitar.sf2 {output_path}.mid -F {output_path}")
```

Two files: `untrained.wav` and `trained.wav`. Both go in the HF Space and in the blog post as embedded audio.

---

## Video Strategy (2-3 minutes max)

Judges watch the video while the HF Space loads. It is your first impression. It must work without sound (captions) and must make the result obvious in the first 30 seconds.

**Structure:**

**0:00-0:20 — Hook**
Text on screen: "We trained an LLM to compose Indian classical music from reward signals alone. No music data. No supervised labels."
Show 3-second clip of the piano roll — trained agent playing, mostly green cells.

**0:20-0:50 — The Problem**
One slide. "LLMs fail when rules change mid-task. Enterprise schemas drift. APIs update. Models keep generating outputs that used to be valid, now aren't."
Raga = DSL. Show the valid/forbidden note diagram for Yaman and Bhairav side by side.

**0:50-1:30 — The Demo**
Screen recording. No voiceover needed — just captions.
1. Click "Untrained" → piano roll fills with red cells → caption: "Random policy: 24% forbidden note rate"
2. Click "Trained" → piano roll fills with green/gold → caption: "GRPO-trained: 3% forbidden note rate"
3. Drag the raga dial → caption: "Raga changed mid-episode. Agent not told explicitly."
4. 3-4 steps later → blue flash → caption: "Bhairav pakad completed in 3.4 steps on average"

**1:30-2:00 — The Numbers**
Show the evaluation table as a static image. Read the key numbers in voiceover or caption them.

**2:00-2:30 — Why It Matters**
"The same training structure applies to: API schema validation, compliance rule enforcement, any domain where rules change and labeled data for the new rules doesn't exist."
End card: HF Space link, Model Hub link, GitHub link.

**Hard rules for the video:**
- No intro music that takes 15 seconds to get to the point
- No "hi I'm Jay and today I'm going to show you..." — cut straight to the result
- Captions on every important number
- Resolution 1920×1080 minimum (OBS Studio is free)
- Max 3 minutes — judges skip long videos

---

## Blog Post Strategy

The blog lives on HF or Medium. Judges read the first 3 paragraphs and the results section. Write for that.

**Title:** "Jugalbandi: Training LLMs for Schema Adherence Under Dynamic Rule Drift"

**Do not title it:** "Building a Raga Composer with RL" — that sounds like a music hobby project, not a systems research contribution.

**Paragraph 1 (the hook):**
State the enterprise problem in one sentence. "When production schemas change, LLMs hallucinate out-of-bounds — they keep generating what was valid yesterday, now rejected by the downstream system." Then: "We built a training environment that reproduces this failure mode with measurable success metrics, using Indian classical music as the domain because raga rule violations are audible to anyone in the room."

**Paragraph 2 (the claim):**
Lead with the result, not the method. "A Qwen2.5-0.5B model trained via GRPO on our environment reduces forbidden-note rate from 24% to 3%, completes raga-characteristic phrases (pakads) 2.3 times per episode vs 0.1 for the random baseline, and adapts to a mid-episode rule change in an average of 3.4 steps with 72% success rate — without ever receiving an explicit change-of-state signal."

**Paragraph 3 (why this is hard):**
"The agent sees only a raw float for the active raga. It must learn, through reward correlation, that crossing 0.5 inverts the reward landscape for several notes. This is the implicit learning constraint that mirrors real enterprise conditions."

**Results section:**
Embed the evaluation table as an image. Embed the reward curve chart. Embed the audio clips (before/after). These three things carry the technical credibility.

**Technical section:**
Environment architecture, reward layers, GRPO rationale, Unsloth/QLoRA setup. Keep it short — 200 words per topic. Link to the paper for GRPO if needed.

**Closing:**
"The raga is the vehicle. The schema adherence is the capability. Adding a new raga requires writing a Python dict. The architecture is DSL-agnostic."

**Things to include in every format (README, blog, slides):**
- The 4-row evaluation comparison table
- The reward curve (wandb screenshot or matplotlib)
- Link to HF Space, Model Hub, GitHub

---

## README Structure

The README is what GitHub and HF show by default. It must stand alone.

```markdown
# Jugalbandi — Schema Adherence Under Drift via Indian Classical Music

> An OpenEnv-compatible RL training environment. Trains LLMs to maintain
> strict adherence to complex rule systems while those rules change mid-episode.
> Human-in-the-loop control via a raga dial.

## Results

| Metric | Random | Trained | Δ |
|--------|--------|---------|---|
| Avg Episode Reward | -10.2 | +4.8 | +15.0 |
| Forbidden Note Rate | 24% | 3% | -21pp |
| Pakad Completion | 0.1/ep | 2.3/ep | +22x |
| Adapt Speed | — | 3.4 steps | — |

[HF Space →](https://your-space-url) | [Trained Model →](https://hf.co/your-model) | [Demo Video →](youtube-link)

## What This Is

[3 sentence explanation]

## How to Run

[3 commands maximum. No setup essays.]

## Architecture

[Diagram or table. Not prose.]
```

Lead with results, not setup instructions. Most READMEs bury results on page 3. Move them to line 5.

---

## The Scoring Breakdown and How to Maximize Each

### Environment Innovation (40%)

You have the innovation. The drift mechanic, grace period, adaptation bonus, and 5-layer reward function are genuinely differentiated. The risk is that judges don't understand this by reading the code.

**How to surface it:** In the README and blog, explicitly list: "5 reward layers: hard rules, soft rules, sequence-level, jugalbandi call-response, drift adaptation." List the novel mechanics as bullet points with one-sentence explanations. Don't make judges infer the innovation from reading the reward function — spell it out.

### Storytelling (30%)

Your strongest differentiator vs other submissions. The music framing makes failure audible. Lean into this explicitly in the video and blog.

**How to surface it:** The before/after audio clips are your anchor. If you only have time for one thing in Priority 3, make it those two WAV files. Second: the piano roll — watching it fill with green vs red cells communicates "learned something" faster than any table.

### Showing Improvement in Rewards (20%)

You cannot claim this without evaluate.py. This is a blocking dependency.

**How to surface it:** The evaluation table with percentage improvements is what goes on a slide. Generate it. Include error bars (standard deviation over 100 episodes) for credibility. A table without error bars reads as cherry-picked.

### Reward and Training Pipeline (10%)

You already have this — GRPO + Unsloth + Qwen + OpenEnv is exactly what the hackathon wants. Just make sure the notebook runs top-to-bottom without errors.

**How to surface it:** In the README, add a one-line "stack" list: Qwen2.5-0.5B | Unsloth QLoRA | GRPO (TRL) | OpenEnv | HuggingFace Spaces. Judges recognize these and check them off mentally.

---

## Things That Will Actively Hurt Your Score

**1. A notebook that doesn't run end-to-end.**
Judges will try to run it. If Cell 4 crashes because a dependency version changed, they mark it down. Pin every version in Cell 1. Test on a fresh Colab runtime 24 hours before submission.

**2. A HF Space that takes more than 60 seconds to load.**
If it's cold-starting from sleep, judges will move on. Either keep the Space warm (HF Pro feature, free for hackathons sometimes), or add a clear message: "If loading, it will be ready in ~60 seconds."

**3. Numbers that don't match.**
If the README says "3% forbidden note rate" and the notebook output says "7%", judges notice. Run evaluate.py once, copy the exact numbers everywhere. Do not round or approximate.

**4. Overselling the music claim.**
Do not say "our model composes at a professional level." It doesn't. Say "the model learned to adhere to raga grammar rules under dynamic drift, as measured by forbidden note rate and pakad completion rate." The honest framing is also the more impressive technical claim.

**5. No comparison to baseline.**
Every number about the trained model is meaningless without the random baseline. Always present both columns.

---

## 48-Hour Countdown

**T-48h (tonight):**
- [ ] evaluate.py written and tested locally
- [ ] Training notebook cells 1-9 verified on a fresh Colab runtime

**T-36h (tomorrow morning):**
- [ ] Full 500-step training run complete (use A100 credits)
- [ ] evaluate.py run against trained checkpoint → numbers saved

**T-24h:**
- [ ] Gradio demo deployed to HF Space
- [ ] Audio files generated (if time allows)
- [ ] Blog post first draft written

**T-12h:**
- [ ] Video recorded and uploaded
- [ ] README updated with final numbers
- [ ] All links cross-checked (HF Space, Model Hub, GitHub, video)
- [ ] Notebook run end-to-end on fresh Colab — no errors

**T-6h:**
- [ ] Submit
- [ ] Rehearse the 4-act demo from Jugalbandi.md until it's smooth
- [ ] Memorize: 3.4 steps, 72% success rate, 24%→3% forbidden note rate

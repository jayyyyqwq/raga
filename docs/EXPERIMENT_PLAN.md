# Jugalbandi — Experiment Plan (Phase 4)

**Pre-registered:** 2026-09-11, against commit `06656a5` (see each result file's own `fingerprint.git_sha` for the exact commit it was produced against — this document's own git history is the ultimate paper trail).

**Status when this was written:** no model has ever been trained. The four scripted baselines below (`random-uniform`, `random-valid`, `safe-set-cycle`, `scripted-oracle`) have been run for real, on the current codebase, and their numbers in this document are real measurements, not targets. Everything involving a trained model (`base-zeroshot`, `grpo-ORACLE`, `grpo-DIAL`, `grpo-HIDDEN`) is still pending — see §9.

**Why "pre-registered"**: everything in §4–§7 — which policies get compared, exactly how each metric is computed, which statistical tests are used — is locked in *before* any GPU training run happens. This is standard practice in research specifically to prevent a very human failure mode: running an experiment, looking at what came out, and then quietly choosing the metric or the comparison that makes the result look best. If this document needs to change after a training run has happened, that's a decision worth being honest about in whatever the change looks like (a dated addendum, not a silent edit).

---

## 1. Who this document is for, and how to read it

This is written for two audiences at once, because in practice they're often the same person at different times: someone who wants to understand *what the experiment actually tests* without needing a machine learning background, and someone who needs the exact, locked-down specification to run it correctly. Sections 2–3 are the plain-language part — if a term is unfamiliar, it's defined there before it's used anywhere else in this document. Sections 4 onward get precise, because a pre-registration document that isn't precise doesn't do its job.

If you've never touched reinforcement learning or trained a model before, read §2 and §3 first, in order, before anything else. They build on each other.

---

## 2. The question, in one paragraph

Jugalbandi is a made-up (but rule-accurate) musical game: an AI plays one note at a time, trying to stay within the grammar of an Indian classical raga — a raga is a specific set of allowed notes, a "home" note, and characteristic phrases, a bit like how a Western piece "in C major" has rules about which notes sound right. Partway through a performance, *the rules change* — the raga switches, in the middle of the piece, and what was correct a moment ago is now a mistake. A human musician mid-performance would notice this from how the music suddenly feels wrong and adjust. The question this experiment asks is: **can a small AI model learn to notice that same thing — purely from the pattern of "that last note went well" / "that last note went badly" — without ever being told in words that anything changed?** If it can, that's a real, general skill (noticing your assumptions broke, from feedback alone, without being told) demonstrated in a controlled, measurable setting. If it can't, that's still a useful, honest result — it tells us this capability doesn't show up for free at this model size, in this setting.

---

## 3. Terms used throughout this document

Read these in order — later ones build on earlier ones.

**Step.** One decision: the AI picks one note (with a duration) to play next.

**Episode.** One full performance: 64 steps, always. An episode always starts fresh (no notes played yet) and always runs the same length.

**Action.** The specific choice made at one step — literally a number from 0 to 95 that decodes into "which note" and "how long to hold it."

**Reward.** A single number, given after every step, saying how good that one action was — positive for a good musical choice (e.g. landing on the raga's most important note at the right moment), negative for a bad one (e.g. playing a note that's forbidden in the current raga). This is the *only* feedback the AI gets — nobody tells it the rules directly, it has to work out what's good from these numbers alone.

**Policy.** Whatever is deciding the actions. This can be a simple fixed rule (e.g. "always play the same four notes in a cycle"), pure randomness, or a trained AI model. Every "thing being compared" in this experiment is a policy.

**Reinforcement learning (RL).** A way of training an AI by having it try actions and giving it reward, rather than showing it "correct answers" directly (that second approach is called supervised learning). Over many attempts, the AI adjusts to favor actions that tend to earn more reward. This project uses a specific RL method called **GRPO** (Group Relative Policy Optimization): for a given situation, the model is asked to produce several different attempts, all of those attempts get scored, and the model is nudged toward whichever attempts scored *better than the group's average* for that situation. No extra "value network" is needed to predict future reward, which is why it fits on modest hardware.

**Fine-tuning / LoRA.** Starting from an existing, already-capable small language model (this project uses Qwen2.5-0.5B-Instruct, "0.5B" meaning about half a billion internal numbers) and nudging a small fraction of its internal numbers so it gets better at this specific task, rather than training a model from nothing. LoRA is the specific cheap technique used to do that nudging without needing to touch (or store) the whole model.

**Drift.** The mid-episode rule change described in §2 — the raga switches from Yaman to Bhairav or vice versa, at some step within the episode. Not every episode necessarily has a drift event (see "control condition" below).

**Arm.** How much the model is *told* directly, in the text it reads, about the current raga and whether it just changed. This is the core experimental variable — see §4.

**Baseline.** A simple, non-learned policy run through the exact same evaluation as the trained models, so a trained model's score means something relative to a floor and a ceiling, not just relative to itself. See §5.

**In-context vs. in-weights.** Two different ways a model could "know" something. *In-weights* means the knowledge is baked into the model's trained parameters — e.g. it might learn during training "switches tend to happen partway through," a general statistical pattern, without reacting to any specific episode's actual events. *In-context* means the model works something out live, within a single conversation/episode, from what it's currently observing — e.g. noticing "my last three actions all got penalized in a way they didn't used to" and changing behavior *because of that specific episode's* evidence, not because of a general pattern memorized in advance. This experiment is specifically trying to test the in-context version, because that's the more impressive and more useful capability — see §3's last paragraph in `updatedplan.md` for the full reasoning; the short version is captured in §4 below.

**Seed.** A fixed starting number for anything involving randomness, so that "random" choices are actually reproducible — running the same code with the same seed twice produces bit-for-bit identical results. Used so every policy in this experiment can be compared on the exact same set of situations.

**Censored (statistics term).** Used in §7.2. It means "we don't know the exact answer because the thing we were waiting for never happened before we stopped watching" — e.g. if we're measuring "how many steps until the model adapts to the new raga" and the episode ends before it ever does, we don't know if it would have adapted at step 65 or never — that observation is censored, not simply "no adaptation." Ignoring censored cases (just throwing them out) quietly makes a policy look better than it is, because a policy that adapts slowly, when it adapts at all, gets treated the same as one that adapts fast.

---

## 4. The three arms, shown concretely

All three arms see the exact same underlying information about the music — same note history, same rhythm position, same "how long since a phrase was completed," etc. The *only* difference between arms is which of that information gets put into the words of the prompt the model reads. Here is the actual difference, using real output from this codebase (`raaga_env/prompting.py`):

**ORACLE** — sees everything, including the raga's name in plain English and an explicit flag when the rules just changed:

```text
Active raga: bhairav (dial=0.75  GRACE PERIOD - rules just changed)
Steps since last rule change: 0.0 (normalised)
Tala position: beat 0/16
Recent notes (oldest first): Sa(quarter)
Partner's call phrase: none yet
Human call tension: 0.00 (how unresolved their phrase was)
Pakad drought: 0.00 (0=just played a phrase, 1=very long since last phrase)
Vadi drought: 0.00 (0=vadi just played, 1=long since)
Last action: Ma(sixteenth) -> outcome: penalised (-2.00)
Choose action (0-95):
```

**DIAL** — sees a raw number (0.0–1.0) that happens to encode which raga is active, but never the raga's name and never a "rules changed" flag:

```text
Raga dial: 0.75
Tala position: beat 0/16
Recent notes (oldest first): Sa(quarter)
Partner's call phrase: none yet
Human call tension: 0.00 (how unresolved their phrase was)
Pakad drought: 0.00 (0=just played a phrase, 1=very long since last phrase)
Vadi drought: 0.00 (0=vadi just played, 1=long since)
Last action: Ma(sixteenth) -> outcome: penalised (-2.00)
Choose action (0-95):
```

**HIDDEN** — sees none of that. No raga name, no dial number, no "rules changed" flag, nothing that identifies which rule set is active:

```text
Tala position: beat 0/16
Recent notes (oldest first): Sa(quarter)
Partner's call phrase: none yet
Human call tension: 0.00 (how unresolved their phrase was)
Pakad drought: 0.00 (0=just played a phrase, 1=very long since last phrase)
Vadi drought: 0.00 (0=vadi just played, 1=long since)
Last action: Ma(sixteenth) -> outcome: penalised (-2.00)
Choose action (0-95):
```

(This is exact, real output from `raaga_env/prompting.py` — not a hand-written illustration, except that the `Last action:` line is supplied for illustration; regenerated 2026-10-07. The previous version of these examples showed `ṉNi(quarter), ṉSa(sixteenth), …` for the same state — that was the prompt note-decoding bug itself, visible in this document: madhya Sa rendered a semitone flat as ṉNi, and listed *first* instead of last. Empty history slots at episode start are no longer rendered at all (§7.9), so only the one real note appears. The `ṉ` prefix marks a note in the lower octave; `Ni`, `Sa` etc. are the seven scale-degree names, the Indian-classical equivalent of do-re-mi.)

Notice the one line every arm shares: `Last action: Ma(sixteenth) -> outcome: penalised (-2.00)`. That's the "did it go well or badly" feedback described in §2 and §3 — it's rule-agnostic by design (it never says "penalised because Ma is forbidden in Bhairav"), so showing it doesn't give away the answer. This line is what makes the HIDDEN and DIAL arms' task possible at all: without it, the model reading the HIDDEN prompt above has *no way whatsoever* to know that the last action was a mistake, since nothing else in the text changed. This is Claim B, resolved as the project's chosen research question (see `updatedplan.md` §9 decision 2): **can the model use only that one feedback line to notice the rules changed and adjust, with no raga name and no dial number to lean on?**

`grpo-DIAL` is the model trained and evaluated on the DIAL arm's prompt; `grpo-HIDDEN` is trained and evaluated on HIDDEN's; `grpo-ORACLE` on ORACLE's. HIDDEN is the interesting one — it's what §3's "in-context inference" question is actually asking about. DIAL is a useful middle point: the dial number technically contains the same information as the raga name, just not spelled out in words, so comparing DIAL against HIDDEN checks whether *seeing the raw number helps even without seeing the word*. ORACLE is the easiest possible version, included as an upper bound / sanity check, not as the paper's headline result.

---

## 5. The policies being compared

Every policy below is run through the *exact same* 200-episode evaluation (§6) and scored with the *exact same* metrics (§7), so their numbers are directly comparable.

| Policy | What it does | Role | Status |
|---|---|---|---|
| `random-uniform` | Picks one of the 96 possible actions completely at random, no awareness of which notes are even legal | Absolute floor — everything else should beat this by a wide margin | **Run.** See §8 |
| `random-valid` | Picks randomly, but only among notes that are actually legal in whichever raga is *currently* active | Honest floor — without this, a model that only ever learned one raga's scale could look artificially impressive | **Run.** See §8 |
| `safe-set-cycle` | Cycles through four specific notes (Sa, Ga, Pa, Ni) that happen to never be forbidden in *either* raga | Names a specific way a trained model could cheat: play only those four notes forever, and every violation-based score looks perfect while demonstrating zero real understanding of drift | **Run.** See §8 |
| `scripted-oracle` | Plays randomly-but-legally most of the time, except it "cheats" by being told directly the instant the raga switches, and immediately plays the new raga's shortest signature phrase | Ceiling — shows the fastest any policy could possibly adapt, since it's given information no real policy (baseline or trained) is allowed to have | **Run.** See §8 |
| `base-zeroshot` | The same base model (Qwen2.5-0.5B-Instruct) used for training, but *without* any of this project's training applied to it | Shows what training actually changes, versus what the base model could already do out of the box | **Pending — needs the model-inference stack (§9)** |
| `grpo-ORACLE` | The trained model, evaluated on the ORACLE arm's prompt | Upper bound for what training buys, given the easiest possible information | **Pending — needs a GPU training run (§9)** |
| `grpo-DIAL` | The trained model, evaluated on the DIAL arm's prompt | "What the project currently claims to do" | **Pending — needs a GPU training run (§9)** |
| `grpo-HIDDEN` | The trained model, evaluated on the HIDDEN arm's prompt | **The actual research question** — see §4 | **Pending — needs a GPU training run (§9)** |

The four scripted policies are cheap (minutes, no GPU) and are run first specifically because they're what makes a trained model's score *interpretable*. A `grpo-HIDDEN` score in isolation means very little — "3.2" could be a great result or a terrible one, there's no way to tell without something to compare it to. Reported next to `random-valid` (the honest floor) and `scripted-oracle` (the ceiling), the same number tells you immediately whether training bought anything.

---

## 6. The evaluation protocol: one fixed, shared set of 200 episodes

Every policy in §5 is scored on the identical 200 episodes — same starting raga, same exact step at which the raga switches, same random seed. This matters more than it might sound: if two policies were instead each evaluated on their *own* random 200 episodes, a difference in score could just mean one policy got an "easier" batch of episodes by chance, not that it's actually better. Giving every policy the identical 200 conditions removes that confound entirely — any score difference has to come from the policy itself. (This is called a "paired" comparison, and it also makes the statistics in §8 meaningfully more sensitive — roughly a threefold gain in the ability to detect a real difference, for zero extra cost, since the episode set only needs to be generated once.)

The 200 episodes are generated once, deterministically, by `eval/episodes.py`, from a fixed master seed (`20260910`) — running the generator again produces the identical 200 episodes, verified by test. Exactly 100 start in Yaman and 100 start in Bhairav (an even split), and every episode contains exactly one raga switch, at a step chosen uniformly between step 16 and step 48 of the 64-step episode.

(Training uses a *different* episode generator, `training/train_grpo.py`'s `sample_drift_schedule()`, which includes a 20%-of-episodes "no switch at all" control condition — that's there so the model being trained can't just learn "a switch always happens near the middle" as a shortcut. Evaluation doesn't need that control condition, because every metric in §7 is specifically about what happens around a switch, and a policy's behavior on a non-switching stretch is already captured by the "before the switch" portion of every evaluation episode.)

---

## 7. The metrics, defined precisely

Every metric below is implemented once, in `eval/metrics.py`, and every number quoted anywhere in this project must come from that implementation — not recomputed by hand in some other script, which is exactly how the earlier version of this project ended up with numbers that didn't actually match what the code did (see `updatedplan.md` finding F12).

### 7.1 `valid_raga_adherence` — did it break the rules?

**Plain language:** out of all the notes played, what fraction were legal in whichever raga was active *at that moment*? Reported three ways instead of one overall number, because a single average hides the entire effect this experiment cares about:
- **pre_switch** — before the raga changes at all
- **grace** — the 3 steps immediately after a switch (during this short window, mistakes are penalized much more gently — a few in-character seconds to "notice," instead of "get it right the very first note or lose a lot of points")
- **post_grace** — everything after that grace window, where full penalties apply again

**A concrete example, from real measurements (§8):** `scripted-oracle` scores exactly 1.0 (perfect) in all three segments — it never plays a forbidden note in whatever the active raga is at that moment, before or after the switch (it does commit ascending-rule violations, which this metric doesn't count — §7.1b). `random-uniform`, which doesn't even try to pick legal notes, scores around 0.87 in every segment — it's *breaking rules at almost the same rate before and after the switch*, which makes sense: it isn't reacting to the switch at all, it's just guessing randomly the whole time regardless.

**Fine print (audit, 2026-10-07):** "legal" here means *no forbidden-note violation* — that is the whole definition (`eval/metrics.py`). A note that is simply outside the raga's scale (`out_of_raga`, e.g. Ma# in Bhairav or komal Re in Yaman) or an ascending-only rule break (`aaroha_violation`) does **not** count against this metric. That matters most exactly where this experiment looks: after a Yaman→Bhairav switch, carrying over Ma# is an out-of-raga error this metric cannot see. Kept unchanged because it is the pre-registered definition — see §7.1b for the stricter companion.

### 7.1b `hard_rule_adherence` — protocol amendment (2026-10-07)

**Added before any trained-model number was computed with either adherence metric**, alongside §7.1, not replacing it. Same three-way split; a step counts as a violation if it triggered *any* hard rule (`forbidden_note`, `out_of_raga`, or `aaroha_violation`). Motivation: §7.1 scores `random-uniform` at 0.875 while §7.1b scores it at 0.559 (after the §7.9 fixes) — 44% of its notes break a hard rule, and only the forbidden-note fraction (12.5%) is visible to §7.1. Both are reported for every policy; any claim about rule adherence should cite §7.1b.

### 7.2 `drift_adaptation_speed` — how fast did it recover?

**Plain language:** counting from the exact step the raga switches, how many more steps does it take before the model plays one of the new raga's signature phrases (a "pakad" — a short sequence of notes considered especially characteristic of that raga, similar to a recognizable riff)? Playing a pakad is used as the marker for "has clearly adjusted," rather than just "stopped breaking rules," because a policy could avoid rule violations by accident (e.g. `safe-set-cycle` in §5) without demonstrating any real adaptation.

Some episodes never see a new pakad before they end — that's a **censored** observation (see §3's glossary entry), and it's tracked separately as a **censoring rate**, never silently dropped. Reported as: the **median** number of steps among episodes that *did* adapt, plus the **censoring rate** (what fraction never adapted at all within the episode). Reporting only the median of the episodes that succeeded, without saying how many never succeeded, is specifically flagged in this project's own history (`updatedplan.md` F12) as the easiest way to accidentally manufacture a fast-looking number that doesn't represent the whole picture — so both numbers are always reported together, never one without the other.

**A concrete example, from real measurements (§8):** `scripted-oracle` has a median of 2.5 steps and 0% censoring — it always adapts, and fast (its 4-note pakad can complete as early as step 3–4 after the switch). `random-valid`, which never deliberately seeks out a pakad, has 99% censoring — in 198 of 200 episodes, it simply never happened to stumble onto a full signature phrase before the episode ended. Whatever `grpo-HIDDEN`'s eventual number is, it should land somewhere between these two, and *how close it lands to which end tells the whole story*.

### 7.3 `adaptation_success_rate@K` (K = 5, 10, 20 steps)

**Plain language:** out of every episode that had a switch, what fraction had adapted (played a new-raga pakad) within K steps of the switch? Reported as three numbers (for K=5, 10, 20) rather than one, because the *shape* of the curve — does it climb fast and then flatten, or climb slowly and steadily — is more informative than any single cutoff.

### 7.4 `post_switch_violation_decay` — the recovery curve itself

**Plain language:** for each possible number of steps since the switch (0, 1, 2, … up to 15), what fraction of episodes had a rule violation at exactly that offset, averaged across all 200 episodes? This produces a curve, not a single number — a model that's genuinely adapting should show a curve that starts high right after the switch and decays toward zero; a model with no real adaptation should show a roughly flat curve at whatever its baseline violation rate is. This is arguably the single most informative metric in this whole document, because it's the most direct visual evidence of "did it actually get better after the switch, and how quickly."

### 7.5 `pakad_rate` — musicality, independent of drift

**Plain language:** how many complete signature phrases does the policy play per episode, on average? Not specifically about drift — this is a general "is it playing real, recognizable raga phrases at all, or just wandering" measure.

### 7.6 `jugalbandi_coherence` — responsiveness to the human's call

**Plain language:** every 8 steps, a simulated "human" injects a short 4-note phrase, and the model gets extra reward for responding to it in a musically coherent way (resolving unresolved tension, contrasting direction with what the human played, or — added 2026-10, see `RETRAIN_PLAN.md` — landing on the exact swara the human's phrase ended on). This metric averages that call-response reward across every one of those human "turns" in the evaluation set. It's the metric behind this project's secondary claim — that a rule-schema environment like this one produces genuinely more *interesting*, humanlike output than a plain JSON-schema-adherence task would.

**2026-10 correction:** `eval.rollout.rollout()` never actually submitted a call before this date — `jugalbandi_coherence` was silently `NaN`/meaningless for every policy ever run through the eval harness, baselines included, regardless of how well a policy would have answered a real call. Fixed (`rollout()` gained a `call_phrase_fn` parameter; every call site in this harness now passes one). §8's table below is the first real data for this metric.

### 7.6b `call_echo_rate` — a plainer companion to the above

**Fine print (audit, 2026-10-07):** before this date, `JugalbandiEnv` flagged `call_requested` on the episode's final step too, so the denominator counted 8 calls per 64-step episode when only 7 can ever be submitted — every `call_echo_rate` was understated by a factor of 7/8 and capped at 0.875, and `jugalbandi_coherence` likewise. Fixed in the env; §8's numbers are regenerated.

**Plain language:** out of every human "turn," what fraction did the policy land on the human's last note at least once? 0 to 1, more directly readable than `jugalbandi_coherence`'s reward average. **Read §8's baseline numbers before trusting any trained model's score here** — a policy that has no idea a call happened still echoes it some of the time by chance alone (a call's last note is one of roughly 7 valid swaras, and a policy gets 8 notes per turn to possibly land on it), and that chance floor turns out to be surprisingly high.

### 7.7 `action_validity_rate` — did the model even answer correctly?

**Plain language:** out of everything the model tried to output, what fraction was a well-formed, in-range action, as opposed to gibberish or a number outside 0–95? This only applies to a real language model's output — the four scripted baselines in §8 always produce a valid action by construction (they're code, not free-form text), so their `action_validity_rate` is reported as exactly 1.0 rather than computed. It becomes meaningful once `base-zeroshot` and the `grpo-*` policies are evaluated.

### 7.8 `safe_set_occupancy` — the specific cheat-detector

**Plain language:** what fraction of steps land on one of the four notes (Sa, Ga, Pa, Ni) that are never forbidden in *either* raga? A model that quietly converges on always playing only these four notes would score perfectly on §7.1 (never breaks a rule) while demonstrating exactly zero real adaptation to drift — this metric exists specifically to catch that. **A concrete example:** `safe-set-cycle`, by construction, scores exactly 1.0 here. If a trained `grpo-*` policy's score on this metric climbs anywhere near 1.0, that is a warning sign to investigate before trusting its other scores, not a result to celebrate.

---

### 7.9 Reward and environment amendments (2026-10-07 audit)

These change the environment itself, so every number measured before this date is superseded. No trained-model result existed under the old definitions, so nothing pre-registered about a trained model was lost. Every change has a regression test.

- **Leap penalty floored at −1.0.** The penalty was unbounded (−0.2 per semitone past 7, so a two-octave leap cost −3.2, more than a forbidden note's −2.0). That broke the same invariant F11 already enforces for the drought penalties: neglecting good practice must never cost more than breaking a rule. It was also the largest single reward term for every baseline, and it is what let `safe-set-cycle` outscore `random-valid`.
- **The ascending rule is judged on the actual move into the note.** It used to use the contour of the 3 preceding notes. Under that rule, Dha→Pa (standard Yaman avaroha) after a rise was penalised, and a rising Ga→Pa went unpenalised whenever the window happened to be mixed. `direction_contrast` uses the same move direction now.
- **Call tension uses circular swara distance**, `min(d, 12−d)/6`. It used to be linear `|d|/11`, which treated Ni and Sa, a semitone apart, as maximally distant.
- **`reset()` starts from `initial_dial`** (or `options["dial"]`), not from the dial the previous episode ended on.
- **Prompt rendering:** empty note-history slots carry a sentinel and are no longer shown as fake ṉSa notes. A genuine all-Sa call is distinguished from "no call yet" via tension. The note line is now labelled `Recent notes (oldest first):`.
- **Training continuation:** the 8-step Monte-Carlo continuation now replays the source episode's remaining drift switch and its partner calls (seeded from the row's `mc_seed`). Before, it silently dropped them.
- **Evaluation precision:** on GPU, `eval/llm_policy.py` loads the base model in 4-bit NF4, matching training.

---

## 8. What we've actually measured so far

The four scripted baselines, regenerated 2026-10-07 after every fix in §7.1b, §7.6b and §7.9. The `fingerprint` in each `eval/results/*.json` holds content hashes of every file a number depends on. Those hashes, not `git_sha`, are the staleness check: a result committed together with its code necessarily records the *parent* commit, with `git_dirty: true`.

| Policy | mean reward | adherence §7.1 | hard-rule adherence §7.1b | adaptation speed (median, censoring) | success@5/10/20 | pakad rate | safe-set occupancy | jugalbandi coherence | call-echo rate |
|---|---|---|---|---|---|---|---|---|---|
| `random-uniform` | -57.05 | 0.875 | 0.559 | 35 steps, 99.5% censored | 0.0 / 0.0 / 0.0 | 0.005 | 0.333 | 0.918 | 0.467 |
| `random-valid` | -18.03 | 1.000 | 0.963 | 20.0 steps, 99% censored | 0.005 / 0.005 / 0.005 | 0.025 | 0.562 | 1.528 | 0.684 |
| `safe-set-cycle` | -27.49 | 1.000 | 0.875 | — , 100% censored | 0.0 / 0.0 / 0.0 | 0.000 | **1.000** | 1.449 | 0.474 |
| `scripted-oracle` | -4.43 | 1.000 | 0.965 | 2.5 steps, 0% censored | 1.0 / 1.0 / 1.0 | 1.020 | 0.562 | 1.525 | 0.661 |

Paired comparisons over the 200 shared episodes (`paper/data/baseline_stats.json`: paired bootstrap 95% CI and Wilcoxon signed-rank):

| Comparison | Δ return | 95% CI | Wilcoxon p | wins |
|---|---|---|---|---|
| scripted-oracle − safe-set-cycle | 23.06 | [21.48, 24.61] | 5.0e-34 | 194/200 |
| scripted-oracle − random-valid | 13.60 | [13.22, 14.01] | 1.4e-34 | 200/200 |
| safe-set-cycle − random-valid | −9.46 | [−11.04, −7.87] | 2.7e-20 | 42/200 |
| random-valid − random-uniform | 39.02 | [37.44, 40.59] | 1.4e-34 | 200/200 |

**The ordering is now the intended one:** `scripted-oracle` > `random-valid` > `safe-set-cycle` > `random-uniform`. Before §7.9, the never-adapting `safe-set-cycle` beat raga-informed `random-valid` by 10.06 and trailed the perfectly adapting oracle by only 4.74. The old reward rewarded the F13 exploit over honest play. The two §7.9 reward fixes reverse that without adding any new reward term. Now the oracle beats the safe set by 23.06 and wins 194 of 200 paired episodes.

**Where the oracle's advantage comes from:** mostly the pakad-drought term (−11.57 vs −20.61 for random-valid). One pakad resets the drought counter for the rest of the episode. The rest is the +3.0 adaptation bonus and +0.5 for the pakad itself.

**`random-valid` and `scripted-oracle` are raga-informed.** Both read the active raga off the observation's dial dimension, so their perfect §7.1 adherence uses information no HIDDEN-arm model has. Their sub-1.0 §7.1b scores come entirely from `aaroha_violation`, since neither tries to avoid rising into Pa in Yaman.

**Call-response metrics have high chance floors.** None of these policies listens to the call, yet `call_echo_rate` is 0.47–0.68. `jugalbandi_coherence` is dominated by `direction_contrast` (≈ +7.5 per episode for *every* policy, call-blind or not), so it is ≈ 1.45–1.53 regardless. A trained model's echo rate means something only if it is clearly above 0.684, and coherence must never be cited without the echo rate beside it.

**Every mean reward is still negative**, including the ceiling. Leap and drought penalties are floored per step but still accrue across 64 steps. Ordering and paired differences are what matter.

**The harness separates the cases it should.** `safe-set-cycle` has perfect §7.1 adherence but 100% censoring, 1.000 safe-set occupancy, and now a visible §7.1b cost (0.875: every rising Ga→Pa in Yaman). `scripted-oracle` adapts in a median of 2.5 steps with 0% censoring.

---

## 9. What's done, what's pending, what's still an open decision

**Done, and verified by an automated test suite (currently 151 tests passing across the whole project):**
- The F11 drought-penalty fix (Phase 5.1) and the retightened reward bounds it enables
- The fixed, shared 200-episode evaluation set (§6)
- All four scripted baseline policies (§5) and their real, post-fix results (§8)
- Every metric in §7, implemented once in `eval/metrics.py`, including `call_echo_rate` (§7.6b)
- The call-injection fix so the jugalbandi metrics actually measure something (§7.6's correction)
- Result fingerprinting, so a stale result can never silently masquerade as current (§8's table) —
  now also covers `eval/rollout.py` and `eval/evaluate.py` themselves, not just the reward/metric
  definitions, after the call-injection fix above proved a harness-behaviour change could otherwise
  go undetected by the fingerprint that exists specifically to catch this
- `eval/llm_policy.py` + `eval/evaluate_llm.py`: the runner for a trained checkpoint this section
  used to say didn't exist yet ("their runner lives with the training code once a checkpoint exists
  to evaluate" — eval/evaluate.py's own module docstring). One now does; not yet run against a real
  adapter (needs the GPU stack, same as training)
- This document

**Pending — needs an actual GPU, which this setup work did not have access to:**
- `base-zeroshot`: needs `torch` + `transformers` installed and the base model downloaded
- `grpo-ORACLE`, `grpo-DIAL`, `grpo-HIDDEN`: each needs a real training run via `training/train_grpo.py --arm {oracle,dial,hidden}`, on Colab, Kaggle, or a cloud GPU — see that script's module docstring and `training/train_grpo.ipynb` for the runnable notebook
- Once a checkpoint exists: `python -m eval.evaluate_llm --adapter-repo ... --arm hidden --name grpo-hidden-v2` — the full 200-episode comparison against this section's baselines, the thing a single inference-check episode (`training/train_grpo.ipynb` Cell 8) was never meant to substitute for

**Still an open decision, not this document's to make:**
- **Compute source and seed count** (`updatedplan.md` §9 decision 3). The full design in §10 below assumes 5 training seeds per trained arm; if compute is constrained, this drops to 2–3 seeds, which is a real decision with a real cost (fewer seeds means less statistical power in §11) and needs to be made explicitly, not discovered by running out of GPU-hours partway through.
- **Venue and deadline** (`updatedplan.md` §9 decision 4), which determines whether the listening study (Phase 6) and additional ragas (Phase 7) are in scope for this round at all.

---

## 10. The full run, once compute is available

Five training seeds per trained arm (pending the decision in §9): `grpo-ORACLE`, `grpo-DIAL`, `grpo-HIDDEN` × seeds {1, 2, 3, 4, 5} = 15 training runs. Each is evaluated on the identical 200-episode set from §6, exactly like the four baselines already were, using the same `eval/metrics.py` formulas.

Estimated compute, per `updatedplan.md` §6: roughly 25–45 GPU-hours for the 15 training runs, plus another 3–6 GPU-hours for evaluating everything. A single free Colab session isn't enough to run all of this in one sitting — plan on multiple sessions, or a paid GPU tier.

---

## 11. How results will be compared, statistically

**Why not just compare the averages?** With only a handful of training seeds (5, or fewer if compute is constrained — see §9), a plain average is noisy — a lucky or unlucky seed can shift it a lot. Two extra precautions are used instead of a plain average-vs-average comparison:

**Paired comparison** (already set up by §6): since every policy runs on the identical 200 episodes, a comparison between two policies can look at each episode individually ("on episode #47, did `grpo-HIDDEN` or `random-valid` do better?") rather than only comparing two overall averages. This removes episode-to-episode luck from the comparison entirely and is substantially more sensitive as a result.

**Bootstrap confidence intervals**, computed on those paired differences: instead of reporting one single "the difference is 4.2," a bootstrap resamples the 200 paired episodes thousands of times (with replacement) and reports the range that difference plausibly falls in — e.g. "the difference is 4.2, and it's very unlikely to be outside 2.1–6.3." That range is the actual honest answer; a single number without it can hide how uncertain the estimate really is.

**Wilcoxon signed-rank test**, used specifically because there are only 5 training seeds (or fewer). A t-test — the more commonly known statistical test — assumes the underlying differences roughly follow a bell-curve distribution, which isn't a reasonable assumption to make confidently from just 5 data points. The Wilcoxon test makes a much weaker assumption (only about the *ranking* of the differences, not their exact distribution shape), which is why it's the appropriate choice here — and using a t-test on 5 points instead would be a real, avoidable mistake, worth flagging as one if it shows up in this project's own analysis code later.

**Effect sizes are reported alongside p-values**, not instead of them. A p-value only says "this difference is unlikely to be pure chance" — it says nothing about whether the difference is big enough to matter musically or practically. Both numbers are needed.

---

## 12. The expected shape of the result, and how to report whatever actually happens

The likely ordering, based on everything measured so far and the structure of the task: `scripted-oracle` > `grpo-ORACLE` > `grpo-DIAL` ≥ `grpo-HIDDEN` ≥ `random-valid`.

**If `grpo-HIDDEN` fails to beat `random-valid`** — meaning training on the HIDDEN arm didn't teach the model anything the honest floor didn't already have — that is explicitly a valid, reportable, *publishable* outcome, not a failed experiment, provided everything in this document was actually followed: the baselines bracket the result (§8), the metrics are precise (§7), and the comparison is statistically sound (§11). The finding in that case becomes: a clean environment that separates "the model was told the rule changed" from "the model had to work it out," plus evidence that the second, harder skill doesn't emerge for free at this model size, in this setting, with this much training. That is a real, well-supported thing to know, and it's a more honest contribution than a result that looks better only because a baseline was missing or a metric was chosen after the fact — which is precisely the failure mode `updatedplan.md`'s finding F12 already caught once in this project's history.

Whatever the eventual result, `safe_set_occupancy` and `post_switch_violation_decay` (§7.4, §7.8) should be used to characterize *how* it succeeded or failed concretely, rather than describing it only in prose.

---

## 13. Commands

Running the scripted baselines (already done — see `eval/results/`; re-run only if the code they depend on has changed, in which case the old result files should be treated as stale and regenerated, not edited):

```bash
venv/Scripts/python.exe -m eval.evaluate --all
```

Running one specific baseline:

```bash
venv/Scripts/python.exe -m eval.evaluate scripted-oracle
```

Running a training arm, once on a GPU (Colab, or any machine with `requirements-train.txt` installed):

```bash
python training/train_grpo.py --arm hidden --seed 1 --steps 300 --run grpo-hidden-seed1
```

Repeat with `--arm oracle`, `--arm dial`, and every seed in §10 once the compute-source decision in §9 is made.

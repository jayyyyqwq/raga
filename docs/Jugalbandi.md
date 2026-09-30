# Jugalbandi: The Pitch


---

## The One-Sentence Pitch

We built an RL environment that trains LLMs to maintain strict adherence to a complex Domain-Specific Language while the rules of that DSL change mid-execution, with a human-in-the-loop dictating those changes. The DSL happens to be Indian classical music grammar, which makes the agent's success or failure instantly audible to anyone in the room.

---

## What the Product Actually Is

Jugalbandi is an OpenEnv-compatible RL training environment. An LLM policy sits inside it and learns, through reward signals alone, to:

1. Stay strictly within a formal rule system (a raga, acting as a DSL)
2. Respond to a human's input phrase in a structured, grammatically valid way
3. Instantly adapt when a human-controlled dial changes the active rule system mid-episode
4. Do all of the above without ever being told explicitly that the rules changed -- it must infer the new grammar from a single raw float value in its observation vector

The underlying capability being trained is schema adherence under drift. The raga is the vehicle for demonstrating it, because unlike JSON schema violations on a log file, a raga violation is instantly perceptible to human ears.

---

## Why This Targets Real Frontier Problems

### The Enterprise Problem Being Solved

Frontier models fail in production when API contracts change, compliance rules update, or a downstream system starts rejecting calls that used to be valid. The model keeps generating what it learned during training, and it hallucinates out of bounds. Every major AI lab has a version of this problem. Patronus AI's sub-theme is explicitly about schema drift in consumer workflows. Scale AI and Snorkel AI are both pushing on evolving requirements and expert-in-the-loop adaptation.

### How Jugalbandi Maps to the Problem

| Enterprise problem | Jugalbandi equivalent |
|--------------------|----------------------|
| API schema with strict field types | Raga with strict valid-note set |
| Forbidden operations (e.g. PII access patterns) | Forbidden notes (natural Ma in Yaman) |
| Context-dependent validation (valid in GET, forbidden in POST) | Direction-sensitive rules (Pa valid descending, forbidden ascending) |
| Required fields / mandatory patterns | Pakads (characteristic phrases that must appear) |
| Schema version change mid-workflow | Raga dial shifts mid-episode |
| Expert overriding a rule | Human plays a phrase the agent must resolve |
| Compliance audit trail | Piano roll with color-coded rule status |

This is not a stretched analogy. The underlying RL problem structure is identical. We built the environment in a domain where failure is audible so the demo is legible to a non-specialist audience.

---

## The Environment in Plain English

**The episode:** Up to 64 steps. Each step, the agent picks one musical note from a set of possible actions. The environment gives it a reward based on how well that note follows the currently active raga's rules.

**The human turn:** Every 8 steps, a human plays a 4-note phrase through a simple piano keyboard interface. This phrase goes into the agent's observation vector as the "call." The agent's next 4 notes are its "response." The response is rewarded based on how well it grammatically resolves the human's phrase.

**The drift mechanic:** A slider in the UI, ranging from 0.0 to 1.0, controls the active raga. 0.0 to 0.49 means Yaman. 0.5 to 1.0 means Bhairav. These two ragas have partially overlapping but fundamentally different rule sets. The human can drag this slider at any moment.

**The implicit learning constraint:** The agent never receives a flag saying "raga changed." It only sees the raw float value of the slider in its observation. It must learn, through reward correlation, that when the float crosses 0.5, the reward landscape inverts for several notes. This is deliberate. It mirrors real enterprise conditions where models are not given explicit change-of-state signals.

**The grace period:** When the slider moves, the environment activates a 3-step window where penalties for playing newly-forbidden notes are reduced by 80%. This rewards agents that gracefully transition rather than aggressively punishing agents that cannot be psychic. This is our answer to the "how does your agent handle abrupt rule changes" question.

**The adaptation bonus:** If the agent plays a pakad (characteristic phrase) of the new raga within 5 steps of the slider moving, it receives a 3x reward multiplier. This is the behavior we're optimizing for.

---

## The Observation Space

A 22-dimensional continuous vector:
- 8 dimensions for the agent's last 4 notes played
- 8 dimensions for the human's last call phrase
- 1 dimension for melodic direction (neutral / ascending / descending)
- 1 dimension for position in the rhythmic cycle
- 1 dimension for tension metric (how unresolved the human left the line)
- 1 dimension for the raw raga dial value (the implicit schema signal)
- 1 dimension for pakad drought (steps since last characteristic phrase)
- 1 dimension for steps since last schema switch (for grace period awareness)

The raw dial value being an unlabeled float is the key design decision. The agent has to learn what that number means by correlating it with reward outcomes. This is the technical claim that proves we're training real adaptation, not hardcoded rule-switching.

---

## The Reward Structure

The reward function has three layers:

**Hard rules** produce immediate large penalties. Playing a forbidden note in the current raga returns -2.0 and ends reward computation for that step. This is the schema violation floor.

**Soft rules** produce smaller graded rewards. Emphasizing the raga's most important note (vadi) gives +0.4. Emphasizing the secondary note gives +0.3. Large melodic leaps incur penalties proportional to the interval size. Repeating the same note three times in a row incurs -0.3, which prevents the trivial "always play Sa" degenerate policy.

**Sequence-level rewards** fire when multi-note patterns complete. Completing a pakad (characteristic phrase) gives +1.0. Landing the vadi note on beat 1 of the rhythmic cycle gives +0.8. These require the agent to plan across multiple steps, not just pick locally optimal notes.

**Jugalbandi rewards** layer on top during the call-response mechanic. Resolving high-tension human phrases by landing on the vadi gives up to +0.5. Contrasting the human's melodic direction gives +0.3. These reward coherent dialogue rather than just valid solo playing.

**Drift-specific rewards** activate during and after slider changes. The grace period reduces hard penalties by 80% for 3 steps. The adaptation bonus gives +3.0 for playing a new-raga pakad within 5 steps. This is the specific mechanism that trains schema drift handling.

---

## Why This Wins the Hackathon

### Environment Innovation (40% of score)

Every environment in the hackathon will have a reward function. Ours has five distinct reward layers interacting with each other, a two-turn Markov game structure, and a reward distribution that inverts mid-episode based on an implicit signal. The combination has not existed on OpenEnv before, and our prior-work search confirmed no RL environment for Indian classical music exists in any framework. The direction-sensitive rules plus the schema drift plus the implicit dial learning together form a reward structure more complex than any coding environment we've seen submitted to these hackathons historically.

### Storytelling (30% of score)

Half the room at Scaler grew up hearing this music. When the demo plays a recognizable phrase and we say "the model learned this from reward signals alone, no music data, no supervised labels, just the reward function," the room feels it. A code-refactoring demo does not produce that reaction. Our storytelling moat is the perceptibility of success and failure.

### Showing Improvement in Rewards (20% of score)

We have four distinct metrics that show improvement, not one: episode reward, forbidden-note rate, pakad completion rate, and adaptation speed after schema drift. **[TARGET, not yet measured — no model has been trained as of this writing.]** The four scripted baselines that calibrate what these charts should look like (`random-uniform`, `random-valid`, `safe-set-cycle`, `scripted-oracle`) have been run for real — see `docs/EXPERIMENT_PLAN.md` §8 for their actual numbers, and §7 for exactly how each metric is computed. Once `grpo-ORACLE`/`DIAL`/`HIDDEN` are trained (see that document's §9–10), their real numbers replace this paragraph — a specific number belongs here only once it has actually been measured.

### Reward and Training Pipeline (10% of score)

We use Unsloth for efficient LoRA fine-tuning of Qwen-0.5B. We use GRPO (not PPO) because our reward structure is sparse and episodic, which GRPO handles without needing a value network. We deploy the environment on HuggingFace Spaces as an OpenEnv-compliant HTTP server. We push the fine-tuned model to HuggingFace Hub as a public artifact. Everything in the stack is exactly what the hackathon sponsors want to see used.

---

## The Demo That Wins the Room

**Setup:** Laptop projecting a Gradio interface. Split screen: human piano keyboard on the left, agent's response piano roll on the right, raga dial at the top, live reward counter and metric tiles at the bottom.

**Act 1 -- Solo baseline (30 seconds):** Click "Generate Untrained." Random notes play. Red squares flash. Forbidden notes everywhere. Sounds chaotic. Audience laughs.

**Act 2 -- Trained solo (30 seconds):** Click "Generate Trained." Recognizable Yaman phrase plays. Green and blue squares. Pakad completion flashes. Audience visibly reacts.

**Act 3 -- Jugalbandi (45 seconds):** Click 4 notes on the human keyboard. Click "Respond." Agent plays a coherent response. Tension resolves. Point to the metrics on screen — **whatever `jugalbandi_coherence` and `valid_raga_adherence` actually read for the deployed model** (`docs/EXPERIMENT_PLAN.md` §7.1, §7.6 define these precisely); do not recite a memorized number here, read the live panel.

**Act 4 -- The drift moment (45 seconds):** Start another jugalbandi turn. While the agent is mid-response, drag the raga slider from 0.2 to 0.8. Point at the piano roll: "see that orange note? That's the grace period -- we don't punish it heavily because the schema just changed. Now watch the next few steps." If the agent lands on Ma (Bhairav's vadi) and completes a Bhairav pakad, that's the adaptation bonus firing live. **Cite `drift_adaptation_speed`'s real median-and-censoring-rate from the trained model's evaluation** (`docs/EXPERIMENT_PLAN.md` §7.2), not a specific step count and success rate memorized in advance — those numbers do not exist yet for any trained arm, and reciting invented ones here is exactly the failure mode `updatedplan.md` finding F12 documents.

That is the demo. Four acts, three minutes. No slides during demo, just the interface.

---

## Questions We Will Be Asked (And How to Answer)

### The Frontier Relevance Question

**Q:** "This is cool, but how does this actually advance LLM capabilities for things AI labs care about?"

**A:** "A raga is a DSL with strict validity rules, direction-sensitive context, required patterns, and schema drift. That's the same structural shape as API schemas, compliance rule engines, and any domain with formal constraints. We trained an LLM to adhere to a shifting DSL from reward alone, with no supervised data. That capability transfers directly to production systems where schemas change and labeled data doesn't exist for the new schema. We chose music as the domain because it makes rule violations audible, which is a demo advantage, not a capability limitation."

### The Implicit Learning Question

**Q:** "How does the agent know the raga has changed?"

**A:** "For the DIAL and HIDDEN arms specifically, it doesn't get an explicit flag — see `docs/EXPERIMENT_PLAN.md` §4 for exactly what each arm's prompt does and doesn't contain. It has one more channel than a bare float, though: a rule-agnostic 'was my last action rewarded or penalised' line (`StepFeedback`, `updatedplan.md` Phase 2.3), which is what makes *in-context* adaptation possible at all — without it, reward is never in the model's context, so there's nothing for it to correlate against within a single episode. The honest evidence this works, once it exists, is `drift_adaptation_speed` for `grpo-HIDDEN` sitting meaningfully closer to `scripted-oracle`'s ceiling than to `random-valid`'s floor (`docs/EXPERIMENT_PLAN.md` §5, §8) — reported with its censoring rate, not a cherry-picked step count. If `grpo-HIDDEN` doesn't clear that bar, that is itself the honest, reportable result (§12 of that document), not a reason to quietly move the goalposts."

### The Exploit Question

**Q:** "How does the agent avoid just repeating the safest note forever to game the reward?"

**A:** "Three mechanisms. First, a repetition penalty of -0.3 for any note appearing 3+ times consecutively. Second, pakad drought -- the agent is penalized for not completing any characteristic phrase for too long, which forces multi-note exploration. Third, vadi drought -- same mechanism for not emphasizing the raga's king note. These exist to close a specific, *named* exploit, not a vaguely-remembered one: there is a four-note set — Sa, Ga, Pa, Ni — that is penalised in neither raga (`updatedplan.md` finding F13), so a policy confined to it would score a perfect `valid_raga_adherence` while demonstrating zero real adaptation to drift. `safe_set_occupancy` (`docs/EXPERIMENT_PLAN.md` §7.8) measures exactly this directly, on every policy we evaluate, rather than relying on having happened to notice it in a training log. No model had been trained when these penalties were designed — they anticipate this failure mode, they weren't observed and then patched, and we don't say otherwise."

### The Algorithm Choice Question

**Q:** "Why GRPO over PPO?"

**A:** "GRPO doesn't require a value network. Our rewards are sparse and episodic -- pakad completions are rare early in training, adaptation bonuses only fire after schema changes. PPO's value network would struggle to assign credit across 32-to-64 step episodes with this reward shape. GRPO works directly from reward comparisons between rollouts, which is structurally cleaner for our problem."

### The Evidence Question

**Q:** "What's your evidence the agent is genuinely adapting and not just getting lucky?"

**A:** "This is exactly what `docs/EXPERIMENT_PLAN.md` is designed to answer honestly, and the answer isn't in yet — no model has been trained. The protocol: `drift_adaptation_speed` measured across a fixed, shared set of 200 evaluation episodes with pre-generated, randomized switch timing (§6–§7.2 of that document), reported as a median *and* a censoring rate, paired against `scripted-oracle` (the real ceiling for how fast adaptation can physically happen — currently a median of 2.5 steps, 0% censored, measured for real) and `random-valid` (the honest floor — currently 99% censored, i.e. it essentially never stumbles onto a pakad by chance). The specific comparison this question is really asking about — does withholding the raga name and the dial value change adaptation speed — is the `grpo-ORACLE` vs `grpo-DIAL` vs `grpo-HIDDEN` comparison itself (§4–§5 of that document), not a separate ablation. There is no fabricated step count to quote here; the real one, once measured, replaces this paragraph."

### The Sub-Theme Question

**Q:** "Which sub-theme bonus are you targeting?"

**A:** "Primarily Snorkel AI's Simulated Experts-in-the-Loop -- the human plays phrases, drags the slider, the environment simulates expert behavior with changing requirements. Secondary alignment with Patronus AI's Consumer Workflows with Schema Drift -- we implement schema drift as first-class environment behavior, not a post-hoc wrapper. Both sub-themes are directly addressed, not approximately gestured at."

### The Scalability Question

**Q:** "Can this extend beyond two ragas?"

**A:** "The rule engine is a dict. Adding a new raga is adding a new dict entry with valid_notes, forbidden_notes, aaroha, avaroha, vadi, and pakads. The slider becomes a discrete selector or a continuous multidimensional space. We built the architecture around 200+ raga extensibility from day one, even though the demo ships with two. More broadly, extending to non-music DSLs means writing a new rule dict with whatever validity predicates that DSL has. The environment structure is domain-agnostic."

---

## Technical Advantages Over Obvious Competitors

### Versus Code-Refactoring Environments
Those environments exist. SWE-bench exists. SWE-agent exists. Every hackathon sees three of them. Ours doesn't compete with them -- it occupies a completely different slot in the environment space.

### Versus Tool-Use Environments
Tool-use environments optimize correctness of a single action given context. Ours optimizes long-horizon adherence to a constraint system with dynamic shifts. These are orthogonal capabilities. An agent good at ours is testing a different skill than an agent good at tool-use.

### Versus Other Music RL Environments
RaveForce is electronic beat-based, no grammar. Prior raga work uses Markov chains and supervised LSTMs, not RL with reward shaping. No prior work combines raga grammar + RL + schema drift + call-response structure. We verified this with targeted literature search.

### Versus Generic Schema Adherence Environments
None exist on OpenEnv. If a team submits one, it will be a JSON validation environment with no perceptible failure mode for the audience. We have the same underlying capability claim with a demo that reaches everyone in the room, not just engineers squinting at log output.

---

## What We're Deliberately Not Claiming

We are not claiming the model will compose at the level of a human classical musician. That's not the point. We are claiming the model learned to adhere to a formal rule system under drift, and that the capability transfers to other formal rule systems.

We are not claiming this replaces enterprise schema-validation tools. We are claiming it trains the kind of model behavior that those tools require.

We are not claiming raga is the only or best domain for this. We are claiming it is the most demo-legible domain for this capability.

Being precise about what we are and aren't claiming is a credibility move. Overclaiming in Q&A will get punished.

---

## Related Work

Added per `updatedplan.md` Phase 8.3 — the project had zero citations before this. This situates Jugalbandi's two research threads (the drift formalism, and the raga domain) against existing work, rather than presenting either as appearing from nowhere.

**The drift formalism:**
- Contextual MDPs — Hallak, Di Castro and Mannor (2015). The formalism for the ORACLE/DIAL arms: a policy conditioned on an observed context variable that determines the reward/transition function.
- Hidden-Parameter MDPs — Doshi-Velez and Konidaris (2016). The formalism for the HIDDEN arm: the context variable is real but unobserved, and must be inferred.
- Meta-RL and in-context adaptation — Duan et al. (2016), RL²; Wang et al. (2016).
- In-context RL from trajectories — Laskin et al. (2022), Algorithm Distillation; Lee et al. (2023), Decision-Pretrained Transformer.
- Non-stationary MDPs and change-point detection — Padakandla et al.; Da Silva et al.
- Reward hacking — Skalse et al. (2022) — the framing this project uses for F13's raga-agnostic safe set and for the reward-ablation work in `updatedplan.md` Phase 5.
- GRPO — Shao et al. (2024), DeepSeekMath.

**The music domain:**
- Bhatkhande's Kramik Pustak Malika (KPM) Vol. I — the primary source for every raga rule in `raaga_env/ragas.py` (see `docs/CLASSICAL_MUSIC_AUDIT.md` for the full sourcing and cross-referencing).
- Computational raga recognition — Chordia and Rae.
- Motif spotting in Hindustani music — Ross and Rao.

**Venue realism** (`updatedplan.md` §8.4): this is workshop-paper scope, not a main-track submission — an ICML/NeurIPS workshop on agents or evaluation, ISMIR for the music-and-ML angle, or an ICLR Tiny Papers-style track. A two-raga case study is exactly the kind of contribution a workshop scope makes appropriate; scaling to more ragas (`updatedplan.md` Phase 7) is explicitly gated on Phase 4 producing a result first, not attempted preemptively to look more complete.

---

## Hackathon Sub-Theme Alignment

**Primary target: Snorkel AI -- Simulated Experts-in-the-Loop.** Direct hit. The human is the expert. The slider represents the expert's changing requirements. The agent must satisfy both simultaneously. This is the sub-theme's exact description.

**Secondary target: Patronus AI -- Consumer Workflows with Schema Drift.** Direct structural alignment. The raga rules are the schema, the slider is the drift, the grace period is the real-world adaptation window that actual production systems need.

**Tertiary fit: Scale AI -- Long-horizon instruction following in business settings.** Applies loosely -- our 64-step episodes with mid-episode drift and required sequence-level patterns are a long-horizon instruction following problem. We don't pitch this primarily, but if asked about Scale AI alignment we can frame it.

**Fallback: Wild Card -- Impress Us.** The hackathon explicitly rewards out-of-box submissions that add value to LLM training on a task. If the judges don't want to give us a sub-theme prize, the wild card category fits cleanly.

---

## What the Blog Post Will Say

Title: "Jugalbandi: Training LLMs for Schema Adherence Under Drift, via Indian Classical Music"

Opening frame: Schema drift is an unsolved frontier problem. Enterprise systems change, models hallucinate out of bounds, no labeled data exists for the new schema. We built an RL environment that addresses this structurally, using Indian classical music as the demo domain because violations are audible.

Core claim (rewritten per `updatedplan.md` Phase 8.2, to state the actual research question instead of a specific measured outcome — because until `grpo-HIDDEN` is trained and evaluated, there is no measured outcome to state):

> We present Jugalbandi, an RL environment for schema adherence under mid-episode constraint drift, instantiated on Hindustani raga grammar so that violations are audible to human listeners. We use it to separate two capabilities usually conflated in the drift literature: conditioned policy learning over an observed context variable, and regime inference from reward feedback alone. We report results for a 0.5B policy trained with GRPO across three observation arms, together with a listening study establishing that the environment's rule violations are perceptible.

Results section: paired comparison across `random-uniform`, `random-valid`, `safe-set-cycle`, `scripted-oracle`, `base-zeroshot`, `grpo-ORACLE`, `grpo-DIAL`, `grpo-HIDDEN` on the fixed 200-episode evaluation set, with bootstrap confidence intervals and a Wilcoxon signed-rank test across training seeds (`docs/EXPERIMENT_PLAN.md` §5, §11) — not a single reward curve and a single adaptation-speed number. If `grpo-HIDDEN` fails to clear `random-valid`, that is reported as the finding, not hidden behind a friendlier metric (§12). Link to HF Space and fine-tuned model once they exist.

Closing: The capability transfers. Rule dicts are swappable. The architecture is DSL-agnostic. The music is the demo, the schema adherence is the product.

---

## The Name

Jugalbandi is the Hindustani word for a musical duet, literally "entwined twins." Two musicians in call-and-response, each shaping what the other plays next. The name carries the call-response mechanic and the cultural rooting in a single word. It's memorable and searchable -- nobody else at the hackathon will have it, nobody else has submitted anything called Jugalbandi to HuggingFace Spaces.

---

## Pre-Pitch Checklist

- [ ] Can I state the one-sentence pitch without looking at notes
- [ ] Can I explain why raga = DSL in under 30 seconds
- [ ] Can I answer the frontier relevance question confidently
- [ ] Can I answer the implicit learning question with the adaptation speed number
- [ ] Can I execute the 4-act demo without tool failure
- [ ] Is the Gradio UI responsive on a backup laptop if mine fails
- [ ] Is the reward curve chart on slide 4 at a reading distance visible to row 10
- [ ] Do I have the adaptation speed stat memorized to one decimal place
- [ ] Can I say "schema drift" and "expert-in-the-loop" naturally in the pitch, not forced
- [ ] Have I rehearsed the slider-drag demo moment so it lands at exactly the right line

# Jugalbandi — Related Work & Paper Draft

Compiled from the project's own docs (`docs/Jugalbandi.md` §Related Work, `updatedplan.md` §8.3,
`docs/EXPERIMENT_PLAN.md`, `docs/CLASSICAL_MUSIC_AUDIT.md`) plus the graphify knowledge graph
(`graphify-out/GRAPH_REPORT.md`), cross-checked against Hugging Face's paper index where an arXiv ID
could be verified. This is a study aid for writing the actual paper, not the paper itself — see §4 for
the format and §5 for a pre-draft skeleton.

**Status note (inherited from `docs/EXPERIMENT_PLAN.md`):** no model has been trained as of this
writing. Every citation below supports either the *formalism* (what problem is being tested) or the
*method* (GRPO, base model, music-domain grounding) — none of them are stand-ins for the project's own
missing empirical result.

---

## 1. Reference Papers

### 1.1 Drift formalism — what kind of MDP is this?

| # | Citation | arXiv | Role |
|---|---|---|---|
| 1 | Hallak, Di Castro & Mannor. "Contextual Markov Decision Processes." (2015) | 1502.02259 (unverified — not in HF's paper index; check before citing) | Formalizes a policy conditioned on an observed context variable that determines the reward/transition function. This is exactly the **ORACLE** and **DIAL** arms: dial value is context, observed, and the policy is allowed to condition on it directly. |
| 2 | Doshi-Velez & Konidaris. "Hidden Parameter Markov Decision Processes: A Semiparametric Regression Approach for Discovering Latent Task Parametrizations." (2013/2016) | 1308.3513 (unverified — not in HF's paper index; check before citing) | Formalizes the case where the context variable is real but **unobserved** and must be inferred from interaction. This is exactly the **HIDDEN** arm — the raga identity is a latent parameter of the reward function, not something in the prompt. |
| 3 | Padakandla, K. J. & Bhatnagar. Non-stationary MDPs / change-point detection in RL. | not verified | Adjacent framing: drift-as-non-stationarity rather than drift-as-latent-context. Useful for situating why this project chose the (H-)MDP framing over a change-point-detection framing — the raga switch is a genuine regime change, not gradual drift. |
| 4 | Da Silva, Basso, Bazzan & Engel. "Dealing with Non-Stationary Environments using Context Detection." (ICML 2006) | not verified | Same adjacent cluster as #3 — classical context-detection RL, predates the modern meta-RL framing. |

**Why this cluster matters for the paper:** the project's core contribution claim is that it
*separates* two capabilities the drift literature usually conflates — conditioned policy learning
(cMDP, #1) versus regime inference from reward alone (HP-MDP, #2). The three arms (ORACLE/DIAL/HIDDEN)
are literally an ablation across this exact axis. This is the single most important citation pair in
the paper — Related Work should open with it, not bury it.

### 1.2 In-context adaptation — can a policy infer the regime *within an episode*?

| # | Citation | arXiv | Role |
|---|---|---|---|
| 5 | Duan, Schulman, Chen, Bartlett, Sutskever & Abbeel. "RL²: Fast Reinforcement Learning via Slow Reinforcement Learning." (2016) | 1611.02779 (unverified — not in HF's paper index; check before citing) | Original meta-RL framing: an RNN policy trained across many MDPs learns to adapt within an episode via its hidden state, no explicit context signal. Directly relevant to the HIDDEN arm's claim. |
| 6 | Wang, Kurth-Nelson, Tirumala, Soyer, Leibo, Munos, Blundell, Kumaran & Botvinick. "Learning to Reinforcement Learn." (2016) | 1611.05763 (unverified) | Companion paper to #5, same meta-RL claim from a neuroscience/cognitive framing. |
| 7 | Laskin, Wang, Oh, Parker-Holder, Tirumala, Liu, Kondiparthi, Guez, ... "In-Context Reinforcement Learning with Algorithm Distillation." (2022) | **2210.14215** (verified) | Trains a causal transformer on *learning histories* (not single-episode trajectories) so it performs in-context RL — improving policy across an episode purely from the sequence of (state, action, reward) it has seen, no weight updates. This is the closest existing method-level analogue to what `grpo-HIDDEN` needs to demonstrate, even though Jugalbandi trains via GRPO fine-tuning rather than Algorithm Distillation's sequence-model pretraining. |
| 8 | Lee, Xie, Pacchiano, Chandak, Finn, Nachum & Brunskill. "Supervised Pretraining Can Learn In-Context Reinforcement Learning" (Decision-Pretrained Transformer). (2023) | **2306.14892** (verified) | Shows a transformer supervised on optimal-action labels (not RL) can still perform in-context exploration/adaptation at test time. Relevant as an alternative training recipe to consider discussing/contrasting against GRPO in the Method or Discussion section. |

**Why this cluster matters:** it is the direct literature this project's HIDDEN-arm result (once
measured) will be compared against conceptually. None of these use LLM policies fine-tuned with GRPO on
a single fixed task family — they use purpose-built sequence models trained across *many* tasks. That
gap (LLM + GRPO + single task-family + in-context regime inference) is worth stating explicitly as the
paper's positioning relative to this cluster, not just a citation dump.

### 1.3 Reward hacking / degenerate policies

| # | Citation | arXiv | Role |
|---|---|---|---|
| 9 | Skalse, Howe, Krasheninnikov & Krueger. "Defining and Characterizing Reward Hacking." (2022) | **2209.13085** (verified) | General framework for what counts as reward hacking. This is the citation backing the project's own finding **F13** — the "safe set" exploit (Sa/Ga/Pa/Ni, never forbidden in either raga) — and the reasoning behind floored drought penalties (finding **F11**) and `safe_set_occupancy` as a dedicated metric. |

**Why this cluster matters:** the project already has *real, measured* evidence of a related but
distinct phenomenon — not an agent discovering an exploit, but a reward function whose unbounded
penalty terms made a non-adapting policy nearly tie a perfectly-adapting one (`docs/EXPERIMENT_PLAN.md`
§8: `scripted-oracle` -22.24 vs `safe-set-cycle` -22.34). Skalse et al.'s framework is the right lens to
formally characterize *why* that near-tie is concerning even though no policy has "hacked" anything yet
— it's a latent hacking opportunity the fixed baselines already expose.

### 1.4 RL algorithm and base model

| # | Citation | arXiv | Role |
|---|---|---|---|
| 10 | Shao, Wang, Zhu, Xu, Song, Bi, Zhang, Zhang, Li, Wu & Guo. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." (2024) | **2402.03300** (verified) | Introduces GRPO (Group Relative Policy Optimization) — the training algorithm this project uses in `training/train_grpo.py`. No value network needed, fits the sparse/episodic reward shape described in `docs/Jugalbandi.md` §"Why GRPO over PPO." |
| 11 | Qwen Team. "Qwen2.5 Technical Report." (2024) | **2412.15115** (verified) | Describes the base model family (Qwen2.5-0.5B-Instruct) that every `grpo-*` arm fine-tunes via LoRA. Cite for model provenance/capability claims in the Method section. |

### 1.5 Music domain — raga grammar and computational modeling

| # | Citation | Venue | Role |
|---|---|---|---|
| 12 | Bhatkhande, V. N. *Kramik Pustak Malika*, Vol. I. | Primary Hindustani theory source (pre-modern) | The ground-truth source for every raga rule in `raaga_env/ragas.py` (scale, forbidden notes, vadi/samvadi, pakads), cross-referenced against the Parrikar raga archive — see `docs/CLASSICAL_MUSIC_AUDIT.md` for the full sourcing methodology. This is a primary source, not a paper; cite as a musicological reference, not an ML citation. |
| 13 | Chordia, P. & Rae, A. "Raag Recognition Using Pitch-Class and Pitch-Class Dyad Distributions." | ISMIR 2007 | Computational raga *recognition* (classification), not generation — establishes that raga identity is recoverable from pitch-class statistics. Relevant contrast: this project generates under raga constraints via RL, rather than classifying pre-existing audio. |
| 14 | Ross, J. C. & Rao, P. "Detecting Melodic Motifs from Audio for Hindustani Classical Music." | ISMIR 2012 | Motif ("pakad") spotting from audio. Relevant as prior art for the *concept* of a pakad as a computationally detectable unit — this project's `match_pakad()` is a symbolic, rule-based analogue of the same idea, operating on generated note sequences rather than recorded audio. |
| 15 | Han, D. & Lee, K. "RaveForce: An Open-Source Reinforcement Learning Environment for Electronic Music." | ISMR / demo paper | Nearest prior RL-for-music environment. Beat/timbre-based, no formal grammar or drift concept — this is the project's explicit point of differentiation (`docs/Jugalbandi.md` §"Versus Other Music RL Environments"). Verify exact venue/year before citing. |

**Gap this project fills, stated plainly for the paper:** no prior work combines (a) a formal,
rule-checkable musical grammar, (b) RL with reward shaping rather than supervised/Markov-chain
generation, (c) mid-episode regime drift, and (d) a call-response (two-agent) structure. Each prior work
above covers at most one or two of these four axes.

### 1.6 Adjacent environments (for the "why not X" section)

- **SWE-bench / SWE-agent** — long-horizon code-editing benchmarks. Cited in `docs/Jugalbandi.md` as
  the "obvious competitor" category this project deliberately doesn't compete with; useful in Related
  Work only as a one-line contrast ("optimizes a different axis: single correct-patch generation, not
  sustained adherence to a shifting constraint system").
- **OpenEnv framework** (Meta PyTorch Hackathon infrastructure) — not a research paper, cite as
  infrastructure/venue context only.

---

## 2. What's still missing before this is a defensible Related Work section

1. **Verify the four unconfirmed arXiv IDs** (#1, #2, #5, #6) directly against arXiv or Semantic
   Scholar before the paper is written — HF's paper index simply doesn't cover papers from
   2013–2016 that never had an associated HF model/dataset, so "not found" here is a coverage gap in
   the lookup tool, not evidence the ID is wrong. Still: don't paste an unverified ID into a
   submission without checking it against arxiv.org directly.
2. **RaveForce's exact citation** (#15) needs a real venue/year lookup — this project's own docs cite
   it by name only, with no year.
3. **No formal citation yet for GRPO's *original* theoretical grounding** vs. PPO — DeepSeekMath (#10)
   introduces GRPO in an applied setting; if the paper wants to justify the choice more rigorously than
   "no value network needed," a PPO citation (Schulman et al. 2017) belongs alongside #10 for contrast.
4. **The listening study (Phase 6 of `updatedplan.md`) has no literature backing yet** — if that
   section survives into the final paper, it needs a 2AFC/psychophysics methodology citation (e.g.
   standard MOS/2AFC protocol papers from speech or music evaluation literature), currently absent from
   every project doc.

---

## 3. How the citation clusters map onto the paper's actual claims

| Paper claim (from `updatedplan.md` §8.2's contribution statement) | Supporting cluster |
|---|---|
| "an RL environment for schema adherence under mid-episode constraint drift" | §1.1 (cMDP / HP-MDP framing) |
| "separates conditioned policy learning ... from regime inference from reward feedback alone" | §1.1 + §1.2 |
| "instantiated on Hindustani raga grammar so that violations are audible" | §1.5 |
| "a 0.5B policy trained with GRPO across three observation arms" | §1.4 |
| "a listening study establishing that the environment's rule violations are perceptible" | §2 item 4 (gap — not yet backed) |
| Reward design avoiding the safe-set exploit (F13) | §1.3 |

---

## 4. Target format

Per `docs/Jugalbandi.md` §"Venue realism" and `updatedplan.md` §8.4: **workshop-paper scope**, not a
main-track submission. Candidate venues: an ICML/NeurIPS workshop on agents or evaluation, ISMIR
(music-and-ML angle), or an ICLR Tiny-Papers-style track. A two-raga case study is appropriately scoped
for a workshop; do not attempt to inflate it to main-track scope before Phase 4 (actual training runs)
produces a result.

**Standard workshop-paper skeleton (4–8 pages + references, roughly what ICML/NeurIPS/ISMIR workshops
expect):**

1. **Abstract** (~150 words) — one paragraph, the contribution statement from `updatedplan.md` §8.2.
2. **Introduction** (~0.75–1 page) — the schema-drift-in-production motivation
   (`docs/Jugalbandi.md` §"Why This Targets Real Frontier Problems"), the raga-as-DSL analogy, and the
   paper's actual research question (Claim B) stated as a question, not a foregone conclusion.
3. **Related Work** (~0.5–0.75 page) — the four clusters in §1.1–1.4 above, plus §1.5 for the music
   angle. Keep this tight; a workshop paper does not need an exhaustive survey, it needs to show the
   two specific gaps this work fills (§1.1's ablation axis, §1.5's method-domain combination).
4. **Environment** (~1–1.5 pages) — `JugalbandiEnv`: action/observation space, the two ragas, the
   5-layer reward function, the drift mechanic (dial, grace period, adaptation bonus), the call-response
   mechanic. This section can draw almost directly from `docs/summary.md` §2.1–2.6 and
   `docs/Jugalbandi.md`'s "Environment in Plain English" section.
5. **Experimental Design** (~1 page) — the three arms (ORACLE/DIAL/HIDDEN), the four baseline
   policies, the fixed 200-episode evaluation protocol, and the metrics (§7 of
   `docs/EXPERIMENT_PLAN.md`), condensed.
6. **Results** (~1–2 pages) — baseline table (already real, `docs/EXPERIMENT_PLAN.md` §8) plus the
   trained-arm results once available, paired bootstrap CIs and Wilcoxon signed-rank tests per §11 of
   that document. `post_switch_violation_decay` as a figure (the recovery curve), not just a table.
7. **Discussion / Limitations** (~0.5 page) — two ragas only (explicitly scoped, not hidden), no
   listening study yet if Phase 6 hasn't completed, single model size (0.5B), reward-shape caveats (the
   unbounded melodic-leap penalty, §4 of `docs/summary.md`).
8. **Conclusion** (~0.25 page).
9. **References.**

---

## 5. Pre-draft

*This is a first-pass draft skeleton with real content where the project already has it, and explicit
placeholders — never invented numbers — where it doesn't. Per this project's own standing rule
(`updatedplan.md` finding F12: a specific number belongs in a document only once it has actually been
measured), every `[PENDING]` marker below must be replaced with a real measurement, not a guess, before
submission.*

---

### Jugalbandi: Training LLMs for Schema Adherence Under Drift, via Indian Classical Music

**Abstract**

Frontier language models degrade when the rules governing a task change mid-deployment — an API
contract updates, a compliance rule set is revised, a schema version bumps — and no labeled data yet
exists for the new regime. We study this failure mode in a controlled setting by casting Hindustani
raga grammar as a formal rule system (a "DSL") and training a small language model, via reinforcement
learning, to remain compliant with it while the active rule set drifts mid-episode. We introduce
Jugalbandi, an OpenEnv-compatible environment in which a policy plays one musical note per step under a
raga's grammar (valid notes, direction-sensitive rules, characteristic phrases), while a human-controlled
signal periodically switches the active raga without ever stating so explicitly. We use this environment
to separate two capabilities usually conflated in the non-stationary-RL literature: policy learning
conditioned on an *observed* context variable (a contextual MDP), and regime inference from reward
feedback *alone*, with the context variable hidden (a hidden-parameter MDP). We train a 0.5B-parameter
policy with GRPO across three observation arms differing only in how much of this context is exposed in
the prompt, and evaluate all arms against four non-learned baselines on a fixed, paired 200-episode
evaluation set. [PENDING: one-sentence summary of the actual result, once `grpo-ORACLE/DIAL/HIDDEN` are
trained and evaluated per `docs/EXPERIMENT_PLAN.md` §8–§11.]

**1. Introduction**

Production language model systems routinely operate against rule systems that change without warning:
an API schema adds a required field, a compliance policy forbids a previously-valid action, a
downstream validator starts rejecting output it used to accept. A model trained before the change keeps
generating what it learned, and the result is a silent, hard-to-detect failure mode rather than a
crash. This problem is structurally a *non-stationary Markov decision process*: the reward and/or
transition function changes at some point during an episode, and the question is whether a policy can
detect and adapt to that change from experience alone, without being told in words that anything is
different.

We study this question in a domain chosen specifically because failure is *audible*: Hindustani
classical music, where a raga defines a strict grammar of valid notes, structurally important notes
(vadi/samvadi), direction-sensitive rules, and required characteristic phrases (pakads). A listener with
no music-theory training can hear when a performance leaves the raga's grammar, in a way that a JSON
schema violation in a log file cannot be perceived without reading the log. We build Jugalbandi, an
RL environment in which an agent plays notes under a raga's rules while the active raga can switch
mid-episode, controlled by a value the agent is never told the meaning of in so many words.

Our central methodological contribution is separating two capabilities the drift-adaptation literature
usually treats together: **conditioned policy learning**, where the context variable determining the
active rule set is directly observable (a contextual MDP, Hallak et al. 2015), and **regime inference**,
where that variable is real but hidden and must be inferred from reward correlation alone (a
hidden-parameter MDP, Doshi-Velez & Konidaris 2016). We operationalize this as three observation
"arms" — ORACLE (raga name and switch flag shown explicitly), DIAL (only a raw, unlabeled float shown),
and HIDDEN (neither) — trained and evaluated identically otherwise.

[PENDING: closing paragraph stating the actual finding once measured. Do not draft this before
`grpo-HIDDEN` has a real number — see `docs/EXPERIMENT_PLAN.md` §12 for the pre-registered space of
possible honest outcomes, including "the harder skill did not emerge," which is an explicitly valid
result to report here.]

**2. Related Work**

See §1.1–§1.5 above for the annotated list. In prose form for the paper:

Non-stationary and context-dependent RL has two dominant formalizations depending on whether the
context is observed. Contextual MDPs (Hallak et al., 2015) assume the policy can condition directly on
a context variable that determines the reward and transition dynamics — this matches our ORACLE and
DIAL arms exactly, differing only in *how* the context is encoded (name vs. raw number). Hidden-parameter
MDPs (Doshi-Velez & Konidaris, 2016) instead treat the context as a real but latent variable the policy
must infer — this is our HIDDEN arm. Meta-RL work (RL², Duan et al. 2016; Wang et al. 2016) and
in-context RL more broadly (Algorithm Distillation, Laskin et al. 2022; Decision-Pretrained
Transformer, Lee et al. 2023) demonstrate that sequence models can perform this kind of within-episode
inference without weight updates at test time, though typically via purpose-built sequence-model
pretraining across many tasks rather than GRPO fine-tuning of a general-purpose LLM on a single task
family — the setting we study.

Reward hacking (Skalse et al., 2022) provides the framework for a concrete failure mode this
environment's reward function must avoid: a "safe set" of notes (Sa, Ga, Pa, Ni) that is never
forbidden in either raga we use, which a degenerate policy could occupy exclusively to score perfectly
on rule-adherence metrics while demonstrating no real adaptation. Our baselines include a
`safe-set-cycle` policy constructed explicitly to probe this, and a dedicated `safe_set_occupancy`
metric to detect it in any trained policy.

On the music side, prior computational work on raga addresses recognition (Chordia & Rae, 2007) and
motif spotting from audio (Ross & Rao, 2012), not generation under grammatical constraint via RL.
RaveForce (Han & Lee) is, to our knowledge, the only prior RL environment for music, but is
beat/timbre-based with no formal grammar or drift concept. No prior work combines a checkable formal
grammar, RL-based generation, mid-episode regime drift, and a call-response interaction structure.

**3. Environment**

[Draw from `docs/summary.md` §2.1–2.6 — action space (`Discrete(96)`), observation space (22-dim for
`JugalbandiEnv`), the two ragas (Yaman, Bhairav) and their sourcing (`docs/CLASSICAL_MUSIC_AUDIT.md`),
the 5-layer reward function, the `DriftManager` (dial → raga mapping, grace period, adaptation bonus),
and the call-response mechanic. This section is close to ready to write directly from existing docs —
it describes what's built and tested, not a pending result.]

**4. Experimental Design**

[Draw from `docs/EXPERIMENT_PLAN.md` §4–§7: the three arms with real example prompts, the four
non-learned baselines plus the three pending `grpo-*` arms and `base-zeroshot`, the fixed 200-episode
paired evaluation protocol, and the eight precisely-defined metrics. Also close to ready — the
protocol is pre-registered and locked.]

**5. Results**

Baseline results (real, measured, `docs/EXPERIMENT_PLAN.md` §8):

| Policy | mean reward | adherence | adaptation speed (median, censoring) | success@5/10/20 | pakad rate | safe-set occ. |
|---|---|---|---|---|---|---|
| random-uniform | -68.58 | 0.875 | —, 99.5% | 0/0/0 | 0.005 | 0.333 |
| random-valid | -37.41 | 1.000 | 20.0, 99% | .005/.005/.005 | 0.025 | 0.562 |
| safe-set-cycle | -22.34 | 1.000 | —, 100% | 0/0/0 | 0.000 | 1.000 |
| scripted-oracle | -22.24 | 1.000 | 2.5, 0% | 1/1/1 | 1.020 | 0.562 |
| base-zeroshot | [PENDING] | | | | | |
| grpo-ORACLE | [PENDING] | | | | | |
| grpo-DIAL | [PENDING] | | | | | |
| grpo-HIDDEN | [PENDING] | | | | | |

[PENDING: `post_switch_violation_decay` curve figure, paired bootstrap CIs, Wilcoxon signed-rank
results across seeds per §11 of `docs/EXPERIMENT_PLAN.md`, once training runs exist.]

**6. Discussion and Limitations**

The baseline results already surface one open finding worth foregrounding regardless of the trained
arms' outcome: `scripted-oracle` (perfect, instantaneous drift adaptation) beats `safe-set-cycle`
(zero adaptation, by construction) by a very narrow margin on mean reward (-22.24 vs. -22.34) even
after the F11 drought-penalty floor fix. This suggests the reward function may currently under-reward
genuine fast adaptation relative to a degenerate but rule-compliant strategy — a hypothesis, not yet a
conclusion, that Phase 5.2/5.3 ablations are designed to test.

Limitations: two ragas only, a single 0.5B model size, an unbounded per-step melodic-leap penalty not
yet given the same treatment as the drought-penalty fix, and (if Phase 6 has not completed by
submission) no human-listening validation that rule violations are perceptible to the audiences this
motivation section invokes.

**7. Conclusion**

[PENDING — one paragraph restating the actual finding, written last.]

**References**

[Compile in the target venue's citation format from §1's table — do not hand-format until the venue is
fixed, since ICML/NeurIPS/ISMIR/ICLR templates differ.]

# Graph Report - D:\Raaga_trial_1  (2026-10-08)

## Corpus Check
- 42 files · ~79,122 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 487 nodes · 873 edges · 46 communities detected
- Extraction: 64% EXTRACTED · 36% INFERRED · 0% AMBIGUOUS · INFERRED: 314 edges (avg confidence: 0.72)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]

## God Nodes (most connected - your core abstractions)
1. `DriftSchedule` - 52 edges
2. `Trajectory` - 47 edges
3. `rollout()` - 33 edges
4. `Jugalbandi (project)` - 24 edges
5. `runAiTurn()` - 22 edges
6. `build_dataset()` - 18 edges
7. `_cycling_policy()` - 17 edges
8. `compute_all_metrics()` - 15 edges
9. `Policy` - 14 edges
10. `safe_set_cycle_policy()` - 13 edges

## Surprising Connections (you probably didn't know these)
- `Cycles Sa, Ga, Pa, Ni (madhya register, sixteenth notes) forever.     Names the` --uses--> `DriftSchedule`  [INFERRED]
  D:\Raaga_trial_1\eval\policies.py → D:\Raaga_trial_1\eval\rollout.py
- `DriftSchedule` --uses--> `One drift event per training episode, sampled uniformly over     DRIFT_WINDOW,`  [INFERRED]
  D:\Raaga_trial_1\eval\rollout.py → D:\Raaga_trial_1\training\train_grpo.py
- `DriftSchedule` --uses--> `Each row is one decision point sampled from a reference episode: walk     a fre`  [INFERRED]
  D:\Raaga_trial_1\eval\rollout.py → D:\Raaga_trial_1\training\train_grpo.py
- `DriftSchedule` --uses--> `Returns a TRL reward_funcs-compatible callable bound to `arm`. GRPO     scores`  [INFERRED]
  D:\Raaga_trial_1\eval\rollout.py → D:\Raaga_trial_1\training\train_grpo.py
- `Runs `policy` over the fixed eval set (or its first `limit`     episodes), each` --uses--> `DriftSchedule`  [INFERRED]
  D:\Raaga_trial_1\eval\evaluate_llm.py → D:\Raaga_trial_1\eval\rollout.py

## Hyperedges (group relationships)
- **Full Jugalbandi Reward Function (5 layers)** — reward_hard_rules, reward_soft_rules, reward_sequence_level, reward_jugalbandi, reward_drift_specific [EXTRACTED 1.00]
- **22-dim Observation Vector** — obs_dim_agent_last4_notes, obs_dim_human_call_phrase, obs_dim_melodic_direction, obs_dim_rhythmic_position, obs_dim_tension, obs_dim_raga_dial, obs_dim_pakad_drought, obs_dim_steps_since_switch [EXTRACTED 1.00]
- **Required Submission Artifacts** — artifact_openenv_latest, artifact_training_script, artifact_reward_loss_plots, artifact_hf_blog_or_video, artifact_hf_space, artifact_readme, artifact_openenv_yaml [EXTRACTED 1.00]

## Communities

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (74): Runs `policy` over the fixed eval set (or its first `limit`     episodes), each, Runs policy_factory(episode) over every episode in the fixed,     shared eval se, action_validity_rate(), adaptation_success_rate_at_k(), AdaptationSpeed, _adherence_split(), AdherenceSplit, call_echo_rate() (+66 more)

### Community 1 - "Community 1"
Cohesion: 0.07
Nodes (28): applyTheme(), beginYourTurn(), buildEscalationStrip(), delay(), endPerformance(), finishYourTurn(), highlightEscalation(), init() (+20 more)

### Community 2 - "Community 2"
Cohesion: 0.04
Nodes (52): Colab Notebook train_grpo.ipynb, HuggingFace Space Host, OpenEnv (latest release), openenv.yaml manifest, Unsloth/TRL training script (Colab), RaagaRL Build Guide, Environment Innovation (40%), Reward and Training Pipeline (10%) (+44 more)

### Community 3 - "Community 3"
Cohesion: 0.08
Nodes (33): random_uniform_policy(), random_valid_policy(), Plays random-valid normally, but the instant the dial switches, it     "cheats", Uniform over all 96 actions, with no awareness of which notes are     even legal, Uniformly samples a note valid in the *currently active* raga (read     directly, Uniform over all 96 actions, with no awareness of which notes are     even legal, Uniformly samples a note valid in the *currently active* raga (read     directly, Cycles Sa, Ga, Pa, Ni (madhya register, sixteenth notes) forever.     Names the (+25 more)

### Community 4 - "Community 4"
Cohesion: 0.17
Nodes (25): Run one full episode in-process, from a fresh reset, and return every     step', rollout()'s sibling for continuing from a mid-episode snapshot instead     of a, Shared step loop for rollout() and rollout_from_state() — the one     place a S, rollout(), rollout_from_state(), _run_steps(), StepRecord, step() (+17 more)

### Community 5 - "Community 5"
Cohesion: 0.15
Nodes (25): build_arg_parser(), main(), Runs `policy` over the fixed eval set (or its first `limit`     episodes), each, run_llm_over_eval_set(), _build_policy(), LLMPolicyStats, make_llm_policy(), Loads `adapter_repo` (a local path or HF Hub repo id — anything     `PeftModel. (+17 more)

### Community 6 - "Community 6"
Cohesion: 0.11
Nodes (24): hard_violation_decay(), main(), paired_bootstrap_ci(), Paper statistics for the four scripted baselines (EXPERIMENT_PLAN.md §11).  Re, Two-sided Wilcoxon signed-rank test, normal approximation with tie     correcti, wilcoxon_signed_rank(), main(), Runs policy_factory(episode) over every episode in the fixed,     shared eval se (+16 more)

### Community 7 - "Community 7"
Cohesion: 0.13
Nodes (23): set_dial(), CALL_EVERY=8, so any snapshot taken at step_count >= 8 should have     had at l, Extends the leak-safety guarantee (test_build_dataset_produces_     expected_co, The whole premise of Claim B: whether an action is good depends on     the (inv, test_build_dataset_hidden_arm_call_injection_does_not_leak_raga_name(), test_build_dataset_oracle_arm_names_the_opening_raga(), test_build_dataset_produces_expected_columns(), test_build_dataset_snapshots_include_feedback_except_at_step_zero() (+15 more)

### Community 8 - "Community 8"
Cohesion: 0.1
Nodes (23): Adaptation Bonus (3x for new-raga pakad within 5 steps), Bhairav forbidden: natural Re & natural Dha, Bhairav komal Re and komal Dha, Bhairav Pakads, Bhairav samvadi: Sa (0), Bhairav vadi: Ma (5), Bhairav valid notes {0,1,4,5,7,8,11}, Drift Mechanic (slider 0.0-1.0) (+15 more)

### Community 9 - "Community 9"
Cohesion: 0.17
Nodes (13): BaseModel, CallRequest, DialRequest, health(), infer(), InferRequest, _load_inference_model(), Human submits a 4-note call phrase. Agent will respond on next steps. (+5 more)

### Community 10 - "Community 10"
Cohesion: 0.12
Nodes (8): The mount is at "/" — this is what guarantees /reset, /step etc.     never get s, Calls are swaras 0-11; anything else used to reach set_call() and     then crash, The whole point of Claim B's channel: it says outcome, never why., Confirms _last_step_feedback is exactly the shape /infer feeds into     render_p, test_api_routes_still_take_priority_over_the_static_mount(), test_call_rejects_out_of_range_swaras(), test_last_step_feedback_integrates_with_render_prompt_for_every_arm(), test_last_step_feedback_is_rule_agnostic()

### Community 11 - "Community 11"
Cohesion: 0.17
Nodes (3): onBeat(), Tabla, TalaMandala

### Community 12 - "Community 12"
Cohesion: 0.22
Nodes (5): EvalEpisode, generate_eval_episodes(), Deterministically generates the shared eval set: n episodes, split as     evenly, test_eval_episodes_different_seed_gives_a_different_set(), test_eval_episodes_is_deterministic()

### Community 13 - "Community 13"
Cohesion: 0.18
Nodes (11): Context-dependent validation (GET/POST), Implicit learning from raw float dial, Obs: 8 dims agent's last 4 notes, Obs: melodic direction (1 dim), Obs: pakad drought (1 dim), Obs: raw raga dial value (1 dim), Obs: rhythmic cycle position (1 dim), Obs: steps since schema switch (1 dim) (+3 more)

### Community 14 - "Community 14"
Cohesion: 0.39
Nodes (1): InputStrip

### Community 15 - "Community 15"
Cohesion: 0.6
Nodes (5): analyzeResponse(), findContourMatch(), findEcho(), findTransposition(), signsOf()

### Community 16 - "Community 16"
Cohesion: 0.33
Nodes (1): Drone

### Community 17 - "Community 17"
Cohesion: 0.67
Nodes (3): fetch_instrument(), main(), Re-downloads the FluidR3_GM note samples into ui/samples/<instrument>/.  The sam

### Community 18 - "Community 18"
Cohesion: 0.67
Nodes (1): Mechanical pre-compile checks for paper/main.tex: every \\cite key exists in ref

### Community 19 - "Community 19"
Cohesion: 0.67
Nodes (0): 

### Community 20 - "Community 20"
Cohesion: 1.0
Nodes (0): 

### Community 21 - "Community 21"
Cohesion: 1.0
Nodes (0): 

### Community 22 - "Community 22"
Cohesion: 1.0
Nodes (0): 

### Community 23 - "Community 23"
Cohesion: 1.0
Nodes (0): 

### Community 24 - "Community 24"
Cohesion: 1.0
Nodes (0): 

### Community 25 - "Community 25"
Cohesion: 1.0
Nodes (0): 

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (0): 

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (1): Human submits a 4-note call phrase. Agent will respond on next steps.

### Community 28 - "Community 28"
Cohesion: 1.0
Nodes (1): GRPO calls this with a batch of (prompt, completion) pairs.     We parse the com

### Community 29 - "Community 29"
Cohesion: 1.0
Nodes (1): Seed dataset: reset env, collect n prompts from random rollouts.     GRPO will t

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (1): Reward and Loss Plots

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (1): HF Blog or <2min YouTube Video

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (1): README

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (1): HuggingFace TRL

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (1): Git SHA plus content hashes of everything a reported number depends     on: the

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (1): A policy is any callable mapping (observation, info) to a legal action     index

### Community 36 - "Community 36"
Cohesion: 1.0
Nodes (1): When to move the raga dial during an episode. `switches` maps a step     index t

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (1): Shared step loop for rollout() and rollout_from_state() — the one     place a St

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (1): Run one full episode in-process, from a fresh reset, and return every     step's

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (1): rollout()'s sibling for continuing from a mid-episode snapshot instead     of a

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (1): Lazily load base model + LoRA adapter. Cached after first call.

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (1): Human submits a 4-note call phrase. Agent will respond on next steps.

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (1): Runs the trained policy on the environment's *current* observation and     retur

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (1): CALL_EVERY=8, so any snapshot taken at step_count >= 8 should have     had at le

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (1): Extends the leak-safety guarantee (test_build_dataset_produces_     expected_col

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (1): The whole premise of Claim B: whether an action is good depends on     the (invi

## Knowledge Gaps
- **110 isolated node(s):** `Deterministically generates the shared eval set: n episodes, split as     evenly`, `Git SHA plus content hashes of everything a reported number depends     on: the`, `What eval.metrics.action_validity_rate needs. A Trajectory only     contains st`, `Returns an eval.rollout.Policy-shaped callable. `fallback_action` is     substi`, `Loads `adapter_repo` (a local path or HF Hub repo id — anything     `PeftModel.` (+105 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 20`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 21`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 23`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (1 nodes): `Human submits a 4-note call phrase. Agent will respond on next steps.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (1 nodes): `GRPO calls this with a batch of (prompt, completion) pairs.     We parse the com`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (1 nodes): `Seed dataset: reset env, collect n prompts from random rollouts.     GRPO will t`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `Reward and Loss Plots`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `HF Blog or <2min YouTube Video`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `README`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (1 nodes): `HuggingFace TRL`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `Git SHA plus content hashes of everything a reported number depends     on: the`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `A policy is any callable mapping (observation, info) to a legal action     index`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `When to move the raga dial during an episode. `switches` maps a step     index t`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `Shared step loop for rollout() and rollout_from_state() — the one     place a St`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `Run one full episode in-process, from a fresh reset, and return every     step's`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `rollout()'s sibling for continuing from a mid-episode snapshot instead     of a`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `Lazily load base model + LoRA adapter. Cached after first call.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `Human submits a 4-note call phrase. Agent will respond on next steps.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `Runs the trained policy on the environment's *current* observation and     retur`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `CALL_EVERY=8, so any snapshot taken at step_count >= 8 should have     had at le`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `Extends the leak-safety guarantee (test_build_dataset_produces_     expected_col`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `The whole premise of Claim B: whether an action is good depends on     the (invi`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `step()` connect `Community 4` to `Community 9`, `Community 1`, `Community 7`?**
  _High betweenness centrality (0.119) - this node is a cross-community bridge._
- **Why does `runAiTurn()` connect `Community 1` to `Community 9`, `Community 11`, `Community 4`, `Community 15`?**
  _High betweenness centrality (0.119) - this node is a cross-community bridge._
- **Why does `rollout()` connect `Community 4` to `Community 0`, `Community 3`, `Community 5`, `Community 6`?**
  _High betweenness centrality (0.108) - this node is a cross-community bridge._
- **Are the 49 inferred relationships involving `DriftSchedule` (e.g. with `Runs policy_factory(episode) over every episode in the fixed,     shared eval se` and `Runs `policy` over the fixed eval set (or its first `limit`     episodes), each`) actually correct?**
  _`DriftSchedule` has 49 INFERRED edges - model-reasoned connections that need verification._
- **Are the 44 inferred relationships involving `Trajectory` (e.g. with `Runs policy_factory(episode) over every episode in the fixed,     shared eval se` and `Runs `policy` over the fixed eval set (or its first `limit`     episodes), each`) actually correct?**
  _`Trajectory` has 44 INFERRED edges - model-reasoned connections that need verification._
- **Are the 28 inferred relationships involving `rollout()` (e.g. with `run_policy_over_eval_set()` and `run_llm_over_eval_set()`) actually correct?**
  _`rollout()` has 28 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `runAiTurn()` (e.g. with `.setTurn()` and `.setEnabled()`) actually correct?**
  _`runAiTurn()` has 12 INFERRED edges - model-reasoned connections that need verification._
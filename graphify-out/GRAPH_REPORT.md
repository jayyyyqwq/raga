# Graph Report - .  (2026-10-05)

## Corpus Check
- 0 files · ~99,999 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 406 nodes · 721 edges · 28 communities detected
- Extraction: 70% EXTRACTED · 30% INFERRED · 0% AMBIGUOUS · INFERRED: 216 edges (avg confidence: 0.76)
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

## God Nodes (most connected - your core abstractions)
1. `DriftSchedule` - 30 edges
2. `rollout()` - 29 edges
3. `Jugalbandi (project)` - 24 edges
4. `runAiTurn()` - 22 edges
5. `build_dataset()` - 21 edges
6. `Trajectory` - 19 edges
7. `safe_set_cycle_policy()` - 13 edges
8. `log()` - 12 edges
9. `compute_all_metrics()` - 12 edges
10. `step()` - 11 edges

## Surprising Connections (you probably didn't know these)
- `A plausible human call phrase for training: 4 swaras drawn uniformly     from th` --uses--> `DriftSchedule`  [INFERRED]
  training\train_grpo.py → eval\rollout.py
- `One drift event per training episode, sampled uniformly over     DRIFT_WINDOW, f` --uses--> `DriftSchedule`  [INFERRED]
  training\train_grpo.py → eval\rollout.py
- `Each row is one decision point sampled from a reference episode: walk     a fres` --uses--> `DriftSchedule`  [INFERRED]
  training\train_grpo.py → eval\rollout.py
- `Returns a TRL reward_funcs-compatible callable bound to `arm`. GRPO     scores o` --uses--> `DriftSchedule`  [INFERRED]
  training\train_grpo.py → eval\rollout.py
- `runAIResponse()` --calls--> `step()`  [INFERRED]
  C:\Raaga_trial_1\ui\app.js → openenv_server\server.py

## Hyperedges (group relationships)
- **Full Jugalbandi Reward Function (5 layers)** — reward_hard_rules, reward_soft_rules, reward_sequence_level, reward_jugalbandi, reward_drift_specific [EXTRACTED 1.00]
- **22-dim Observation Vector** — obs_dim_agent_last4_notes, obs_dim_human_call_phrase, obs_dim_melodic_direction, obs_dim_rhythmic_position, obs_dim_tension, obs_dim_raga_dial, obs_dim_pakad_drought, obs_dim_steps_since_switch [EXTRACTED 1.00]
- **Required Submission Artifacts** — artifact_openenv_latest, artifact_training_script, artifact_reward_loss_plots, artifact_hf_blog_or_video, artifact_hf_space, artifact_readme, artifact_openenv_yaml [EXTRACTED 1.00]

## Communities

### Community 0 - "Community 0"
Cohesion: 0.07
Nodes (63): action_validity_rate(), adaptation_success_rate_at_k(), AdaptationSpeed, AdherenceSplit, compute_all_metrics(), drift_adaptation_speed(), jugalbandi_coherence(), MetricReport (+55 more)

### Community 1 - "Community 1"
Cohesion: 0.07
Nodes (30): $(), applyTheme(), beginYourTurn(), buildEscalationStrip(), delay(), endPerformance(), finishYourTurn(), highlightEscalation() (+22 more)

### Community 2 - "Community 2"
Cohesion: 0.04
Nodes (50): Colab Notebook train_grpo.ipynb, HuggingFace Space Host, OpenEnv (latest release), openenv.yaml manifest, Unsloth/TRL training script (Colab), RaagaRL Build Guide, Environment Innovation (40%), Reward and Training Pipeline (10%) (+42 more)

### Community 3 - "Community 3"
Cohesion: 0.09
Nodes (32): LoRA / QLoRA, CALL_EVERY=8, so any snapshot taken at step_count >= 8 should have     had at le, Extends the leak-safety guarantee (test_build_dataset_produces_     expected_col, The whole premise of Claim B: whether an action is good depends on     the (invi, test_build_dataset_hidden_arm_call_injection_does_not_leak_raga_name(), test_build_dataset_oracle_arm_names_the_opening_raga(), test_build_dataset_produces_expected_columns(), test_build_dataset_snapshots_include_feedback_except_at_step_zero() (+24 more)

### Community 4 - "Community 4"
Cohesion: 0.14
Nodes (22): Protocol, Policy, rollout()'s sibling for continuing from a mid-episode snapshot instead     of a, A policy is any callable mapping (observation, info) to a legal action     index, Shared step loop for rollout() and rollout_from_state() — the one     place a St, rollout_from_state(), _run_steps(), StepRecord (+14 more)

### Community 5 - "Community 5"
Cohesion: 0.1
Nodes (23): Adaptation Bonus (3x for new-raga pakad within 5 steps), Bhairav forbidden: natural Re & natural Dha, Bhairav komal Re and komal Dha, Bhairav Pakads, Bhairav samvadi: Sa (0), Bhairav vadi: Ma (5), Bhairav valid notes {0,1,4,5,7,8,11}, Drift Mechanic (slider 0.0-1.0) (+15 more)

### Community 6 - "Community 6"
Cohesion: 0.15
Nodes (17): main(), Runs policy_factory(episode) over every episode in the fixed,     shared eval se, run_and_save(), run_policy_over_eval_set(), save_result(), compute_fingerprint(), _git_sha(), _hash_file() (+9 more)

### Community 7 - "Community 7"
Cohesion: 0.19
Nodes (17): BaseModel, CallRequest, DialRequest, health(), infer(), InferRequest, _load_inference_model(), Human submits a 4-note call phrase. Agent will respond on next steps. (+9 more)

### Community 8 - "Community 8"
Cohesion: 0.14
Nodes (6): The mount is at "/" — this is what guarantees /reset, /step etc.     never get s, The whole point of Claim B's channel: it says outcome, never why., Confirms _last_step_feedback is exactly the shape /infer feeds into     render_p, test_api_routes_still_take_priority_over_the_static_mount(), test_last_step_feedback_integrates_with_render_prompt_for_every_arm(), test_last_step_feedback_is_rule_agnostic()

### Community 9 - "Community 9"
Cohesion: 0.18
Nodes (4): Orb, Orbs, loopPulse(), tween()

### Community 10 - "Community 10"
Cohesion: 0.29
Nodes (2): noteToHz(), Sitar

### Community 11 - "Community 11"
Cohesion: 0.18
Nodes (11): Context-dependent validation (GET/POST), Implicit learning from raw float dial, Obs: 8 dims agent's last 4 notes, Obs: melodic direction (1 dim), Obs: pakad drought (1 dim), Obs: raw raga dial value (1 dim), Obs: rhythmic cycle position (1 dim), Obs: steps since schema switch (1 dim) (+3 more)

### Community 12 - "Community 12"
Cohesion: 0.22
Nodes (5): EvalEpisode, generate_eval_episodes(), Deterministically generates the shared eval set: n episodes, split as     evenly, test_eval_episodes_different_seed_gives_a_different_set(), test_eval_episodes_is_deterministic()

### Community 13 - "Community 13"
Cohesion: 0.39
Nodes (1): InputStrip

### Community 14 - "Community 14"
Cohesion: 0.47
Nodes (2): onBeat(), TalaMandala

### Community 15 - "Community 15"
Cohesion: 0.6
Nodes (5): analyzeResponse(), findContourMatch(), findEcho(), findTransposition(), signsOf()

### Community 16 - "Community 16"
Cohesion: 0.33
Nodes (1): Drone

### Community 17 - "Community 17"
Cohesion: 0.67
Nodes (2): _get(), _post()

### Community 18 - "Community 18"
Cohesion: 0.67
Nodes (3): fetch_instrument(), main(), Re-downloads the FluidR3_GM note samples into ui/samples/<instrument>/.  The sam

### Community 19 - "Community 19"
Cohesion: 1.0
Nodes (1): Reward and Loss Plots

### Community 20 - "Community 20"
Cohesion: 1.0
Nodes (1): HF Blog or <2min YouTube Video

### Community 21 - "Community 21"
Cohesion: 1.0
Nodes (1): README

### Community 22 - "Community 22"
Cohesion: 1.0
Nodes (1): HuggingFace TRL

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
Nodes (0): 

## Knowledge Gaps
- **74 isolated node(s):** `Human submits a 4-note call phrase. Agent will respond on next steps.`, `GRPO calls this with a batch of (prompt, completion) pairs.     We parse the com`, `Seed dataset: reset env, collect n prompts from random rollouts.     GRPO will t`, `Jugalbandi Pitch Document`, `Hackathon Problem Statements` (+69 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 19`** (1 nodes): `Reward and Loss Plots`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 20`** (1 nodes): `HF Blog or <2min YouTube Video`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 21`** (1 nodes): `README`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (1 nodes): `HuggingFace TRL`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 23`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `build_dataset()` connect `Community 3` to `Community 0`, `Community 4`, `Community 7`?**
  _High betweenness centrality (0.373) - this node is a cross-community bridge._
- **Why does `Jugalbandi (project)` connect `Community 2` to `Community 3`, `Community 5`?**
  _High betweenness centrality (0.325) - this node is a cross-community bridge._
- **Why does `Unsloth (efficient LoRA FT)` connect `Community 3` to `Community 2`?**
  _High betweenness centrality (0.295) - this node is a cross-community bridge._
- **Are the 27 inferred relationships involving `DriftSchedule` (e.g. with `run_policy_over_eval_set()` and `Runs policy_factory(episode) over every episode in the fixed,     shared eval se`) actually correct?**
  _`DriftSchedule` has 27 INFERRED edges - model-reasoned connections that need verification._
- **Are the 24 inferred relationships involving `rollout()` (e.g. with `run_policy_over_eval_set()` and `.reset()`) actually correct?**
  _`rollout()` has 24 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `runAiTurn()` (e.g. with `step()` and `infer()`) actually correct?**
  _`runAiTurn()` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `build_dataset()` (e.g. with `random_valid_policy()` and `Policy`) actually correct?**
  _`build_dataset()` has 13 INFERRED edges - model-reasoned connections that need verification._
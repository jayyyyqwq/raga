# Graph Report - C:\Raaga_trial_1  (2026-04-24)

## Corpus Check
- 6 files · ~16,731 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 142 nodes · 170 edges · 15 communities detected
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 8 edges (avg confidence: 0.82)
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

## God Nodes (most connected - your core abstractions)
1. `Jugalbandi (project)` - 24 edges
2. `runAIResponse()` - 11 edges
3. `Meta PyTorch OpenEnv Hackathon` - 11 edges
4. `Sitar` - 9 edges
5. `22-dim Observation Space` - 8 edges
6. `Raga Bhairav` - 8 edges
7. `Tabla` - 7 edges
8. `Raga Yaman` - 7 edges
9. `Drift Mechanic (slider 0.0-1.0)` - 6 edges
10. `OpenEnv Framework` - 6 edges

## Surprising Connections (you probably didn't know these)
- `runAIResponse()` --calls--> `step()`  [INFERRED]
  C:\Raaga_trial_1\ui\app.js → C:\Raaga_trial_1\openenv_server\server.py
- `Jugalbandi (project)` --references--> `Theme 5: Wild Card - Impress Us`  [EXTRACTED]
  Jugalbandi.md → problem statements.md
- `Jugalbandi (project)` --implements--> `OpenEnv (latest release)`  [EXTRACTED]
  Jugalbandi.md → problem statements.md
- `Jugalbandi (project)` --implements--> `Unsloth/TRL training script (Colab)`  [EXTRACTED]
  Jugalbandi.md → problem statements.md
- `Jugalbandi (project)` --implements--> `HuggingFace Space Host`  [EXTRACTED]
  Jugalbandi.md → problem statements.md

## Hyperedges (group relationships)
- **Full Jugalbandi Reward Function (5 layers)** — reward_hard_rules, reward_soft_rules, reward_sequence_level, reward_jugalbandi, reward_drift_specific [EXTRACTED 1.00]
- **22-dim Observation Vector** — obs_dim_agent_last4_notes, obs_dim_human_call_phrase, obs_dim_melodic_direction, obs_dim_rhythmic_position, obs_dim_tension, obs_dim_raga_dial, obs_dim_pakad_drought, obs_dim_steps_since_switch [EXTRACTED 1.00]
- **Required Submission Artifacts** — artifact_openenv_latest, artifact_training_script, artifact_reward_loss_plots, artifact_hf_blog_or_video, artifact_hf_space, artifact_readme, artifact_openenv_yaml [EXTRACTED 1.00]

## Communities

### Community 0 - "Community 0"
Cohesion: 0.07
Nodes (31): Colab Notebook train_grpo.ipynb, HuggingFace Space Host, OpenEnv (latest release), openenv.yaml manifest, Unsloth/TRL training script (Colab), RaagaRL Build Guide, Required mandatory patterns, Gradio UI (+23 more)

### Community 1 - "Community 1"
Cohesion: 0.1
Nodes (23): Adaptation Bonus (3x for new-raga pakad within 5 steps), Bhairav forbidden: natural Re & natural Dha, Bhairav komal Re and komal Dha, Bhairav Pakads, Bhairav samvadi: Sa (0), Bhairav vadi: Ma (5), Bhairav valid notes {0,1,4,5,7,8,11}, Drift Mechanic (slider 0.0-1.0) (+15 more)

### Community 2 - "Community 2"
Cohesion: 0.13
Nodes (15): Environment Innovation (40%), Reward and Training Pipeline (10%), Showing Improvement in Rewards (20%), Storytelling (30%), Meta PyTorch OpenEnv Hackathon, Hackathon Problem Statements, Scale AI (sub-theme sponsor), Scale AI: Long-horizon instruction following (+7 more)

### Community 3 - "Community 3"
Cohesion: 0.23
Nodes (9): BaseModel, CallRequest, DialRequest, Human submits a 4-note call phrase. Agent will respond on next steps., reset(), ResetRequest, set_dial(), StepRequest (+1 more)

### Community 4 - "Community 4"
Cohesion: 0.29
Nodes (8): delay(), log(), logBreakdown(), mapSemitoneToStringIndex(), onUserPluck(), runAIResponse(), updateReward(), step()

### Community 5 - "Community 5"
Cohesion: 0.29
Nodes (2): noteToHz(), Sitar

### Community 6 - "Community 6"
Cohesion: 0.18
Nodes (11): Context-dependent validation (GET/POST), Implicit learning from raw float dial, Obs: 8 dims agent's last 4 notes, Obs: melodic direction (1 dim), Obs: pakad drought (1 dim), Obs: raw raga dial value (1 dim), Obs: rhythmic cycle position (1 dim), Obs: steps since schema switch (1 dim) (+3 more)

### Community 7 - "Community 7"
Cohesion: 0.29
Nodes (7): LoRA / QLoRA, build_dataset(), env_reward(), make_prompt(), GRPO calls this with a batch of (prompt, completion) pairs.     We parse the com, Seed dataset: reset env, collect n prompts from random rollouts.     GRPO will t, Unsloth (efficient LoRA FT)

### Community 8 - "Community 8"
Cohesion: 0.36
Nodes (1): Tabla

### Community 9 - "Community 9"
Cohesion: 0.5
Nodes (4): Rationale: repetition/drought penalties prevent 'always Sa' exploit, Soft Rules reward layer, Samvadi (second-most important note), Vadi (king note)

### Community 10 - "Community 10"
Cohesion: 0.67
Nodes (0): 

### Community 11 - "Community 11"
Cohesion: 1.0
Nodes (1): Reward and Loss Plots

### Community 12 - "Community 12"
Cohesion: 1.0
Nodes (1): HF Blog or <2min YouTube Video

### Community 13 - "Community 13"
Cohesion: 1.0
Nodes (1): README

### Community 14 - "Community 14"
Cohesion: 1.0
Nodes (1): HuggingFace TRL

## Knowledge Gaps
- **57 isolated node(s):** `Human submits a 4-note call phrase. Agent will respond on next steps.`, `GRPO calls this with a batch of (prompt, completion) pairs.     We parse the com`, `Seed dataset: reset env, collect n prompts from random rollouts.     GRPO will t`, `Jugalbandi Pitch Document`, `Hackathon Problem Statements` (+52 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 11`** (1 nodes): `Reward and Loss Plots`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 12`** (1 nodes): `HF Blog or <2min YouTube Video`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 13`** (1 nodes): `README`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 14`** (1 nodes): `HuggingFace TRL`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Jugalbandi (project)` connect `Community 0` to `Community 1`, `Community 2`, `Community 9`, `Community 7`?**
  _High betweenness centrality (0.365) - this node is a cross-community bridge._
- **Why does `Meta PyTorch OpenEnv Hackathon` connect `Community 2` to `Community 0`?**
  _High betweenness centrality (0.090) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `runAIResponse()` (e.g. with `step()` and `.animatePluck()`) actually correct?**
  _`runAIResponse()` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Human submits a 4-note call phrase. Agent will respond on next steps.`, `GRPO calls this with a batch of (prompt, completion) pairs.     We parse the com`, `Seed dataset: reset env, collect n prompts from random rollouts.     GRPO will t` to the rest of the system?**
  _57 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.07 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.1 - nodes in this community are weakly interconnected._
- **Should `Community 2` be split into smaller, more focused modules?**
  _Cohesion score 0.13 - nodes in this community are weakly interconnected._
# Project #4 (capstone) — Semantic-Predictive 4D World Model

> **The fusion.** Combine the two questions the other projects answer separately —
> *"what is where & what does it mean?"* (spatial understanding) and *"what happens next?"*
> (dynamics) — into one system: a **persistent, semantically-grounded, predictive 4D scene**
> you can both **query** and **roll forward**, with a reasoning layer on top.
>
> This is the **most ambitious and most on-thesis** project — it hits all three target clusters
> at once (Reka/Odyssey *and* PerceptAI). It is a **capstone**: build #1 and #2 first as proven
> standalone pieces, then fuse. Do NOT attempt the fusion cold.

## Why this is the unification

| Layer | Question | From project |
|---|---|---|
| Reasoning | "what's here, and what happens if I do X?" | #3 / TGAER (agentic) + spatial-VLM (#3-spatial) |
| **Semantic 4D scene** | grounded objects + relations, persisting AND evolving | **the fusion** |
| ↳ spatial | "what is where & what does it mean?" | **#2** (semantic 3D/4D GS) |
| ↳ dynamics | "what happens next?" | **#1** (world model) |
| Perception | reconstruct + segment | SAM2 + 3DGS + ManiSkill renders |

The fusion point is the **"4D"**: a scene representation that is *both* **semantically grounded**
(objects/labels/relations) *and* **temporally predictive** (how it + the agent change under
actions). A counterfactual spatial query — *"if the gripper pushes the cube left, what gets
occluded / is the peg still reachable?"* — needs **both halves at once**, which is exactly what
neither cluster's models do alone.

## This is where the field (and the targets) are converging

- **Reka** JD: *"persistent 3D/4D scene representations that maintain temporal consistency"* — a
  semantic spatial scene (PerceptAI's half) that **evolves** (the world-model half). The fusion is
  literally their ask.
- **Cosmos3-Nano** (16B omnimodal WFM, ships `Cosmos3-Nano-Policy-DROID`) — NVIDIA unifying
  perception + generation + action in one model. Your capstone is the *small, honest* version of
  that same thesis; cite it as the frontier reference, not the deliverable.
- Research threads: **4D Gaussian Splatting** (3D+time — Project #2's 4D extension is the bridge),
  **dynamic / spatio-temporal scene graphs**, **object-centric / structured world models**,
  **language-conditioned world models** (UniSim/Genie + semantics).

## You have an ingredient for every layer

Reasoning → TGAER + CLIP/multimodal · Semantic grounding → segmentation + #2 ·
Spatial 3D → 3DGS (#2) · Dynamics → world-model RL (#1). Almost nobody has all four — that's the
differentiator, and why this is *your* maximal-leverage capstone rather than a generic one.

## The minimal solo-feasible fusion (the smallest thing that proves the thesis)

**Do it inside ManiSkill** — it elegantly provides all three signals for free, and reuses #1's sim
+ world model and #2's representation:
- **semantic** ← ManiSkill ground-truth segmentation masks (+ SAM2)
- **spatial 3D** ← RGBD → 3DGS reconstruction of a manipulation scene (#2)
- **dynamics** ← Project #1's learned world model (TD-MPC2 / DreamerV3) predicts the future given an action
- **ground truth** ← the sim renders the *actual* future, so every prediction is checkable

**v0 pipeline — concrete, verified, single-A100, license-clean stack** (from `project3-spatial-vlm-3d-scenegraph-SOTA.md`):
1. **Reconstruct + ground** a ManiSkill scene → object-level **scene graph**: VGGT (feed-forward 3D) → SAM2→SAGA language-splat → **ConceptGraphs** (MIT) — nodes = objects, edges = spatial relations.
2. **Predict the next state** under an action with Project #1's world model (TD-MPC2 / DreamerV3).
3. **Re-ground** the predicted state into the scene graph (what moved, what's now occluded/reachable).
4. **Reason:** **Qwen3-VL-8B** (Apache, native 3D-bbox grounding, LoRA) answers a **counterfactual spatial-temporal query** (*"after this push, is the peg still insertable?"*), orchestrated by **TGAER** (DSPy).
5. **Eval:** a small benchmark of such queries with **sim ground-truth futures** → accuracy + calibration + a temporal-consistency metric.

**Headline result:** the system answers spatial questions *about a predicted future* better than (a) a spatial-VLM with no dynamics and (b) a world model with no semantic grounding — i.e. the fusion earns its keep. That single ablation *is* the contribution.

## Enhancements — the strongest version on one A100 80GB

The v0 above proves the thesis; these raise the ceiling to the *strongest demonstrable* project on a single card (all VRAM-feasible if you **pipeline components sequentially** rather than co-resident — the bottleneck is integration effort, not memory):

**v1 (the target — close the loop to *action*):**
- Don't stop at QA. **Language goal → TGAER reasons over the scene graph → simulates candidate plans in the world model → picks + executes** a manipulation in ManiSkill. That turns "answer a counterfactual" into the full **perceive → understand → predict → reason → act** loop — the entire physical-AI stack, end-to-end, on one GPU.
- **Eval triad:** (a) counterfactual-query accuracy vs ground-truth futures, (b) **closed-loop task success**, (c) a **temporal-consistency** metric on the predicted 4D scene — Reka's exact phrase, made quantitative. Report with ablations (no-dynamics, no-grounding) and rliable-style stats.

**Stretch (frontier signal, once v1 lands):**
- Add **NVIDIA Cosmos3-Nano** (16B omnimodal WFM, ships `Cosmos3-Nano-Policy-DROID`; Ampere/A100, ~32 GB inference, LoRA-tight) as a **frontier dynamics arm** — compare your compact world model vs Cosmos3-Nano (zero-shot DROID policy or LoRA-adapted) as the predictive layer. A **frontier-vs-reproducible-vs-classical** comparison on identical scenes is a rare, high-signal result. (Cosmos-Reason2 is the sibling spatial-VLM frontier reference for the grounding arm.)
- An **interactive video demo**: type a query → render the predicted-future splat rolled forward → scene graph updates → answer overlaid. A 60-second clip in the README is portfolio gold for a hiring manager.

**Why this is the strongest project she can do on one A100:**
- Hits **all three target clusters' headline asks at once** (Reka persistent-4D + planning · Odyssey controllable predictable scene · PerceptAI semantic scene graph + the predictive dimension they lack).
- Combines **frontier models** (VGGT · Qwen3-VL · Cosmos3-Nano) **with rigor** (ablations, ground-truth eval, temporal consistency) — depth *and* frontier-readiness, the rare pair.
- **Closed-loop embodied** — not a static demo; it perceives, predicts, reasons, and acts.
- Reuses every asset she has (segmentation, 3DGS, world-model RL, TGAER) — almost nobody combines all four.

## What is NOT solo-feasible (honesty)

- A general open-world **omnimodal foundation model** (Cosmos-scale) — datacenter-only.
- Web-scale VLM/video pretraining; real-world city-scale digital twins.
- Long-horizon photoreal 4D generation. Keep the scene **bounded** (one manipulation scene), the
  dynamics **short-horizon**, and the query set **small + ground-truthed**.

## Sequencing (capstone = last)

```
#1 dynamics (world model)  ─┐
#2 semantic 3D/4D scene    ─┤→  #4 FUSION (semantic-predictive 4D + counterfactual query)
#3 embodied reasoning      ─┘
```

Build #1 (running) and #2 first — each is a credible standalone artifact. #3 (TGAER grounded in a
world model) and the #3-spatial VLM layer are the reasoning components. **#4 fuses them.** Trying to
fuse from day one is the trap; two solid halves + a focused bridge demo is the credible path, and
the bridge is the pitch: *"I unified spatial understanding and dynamics into a queryable predictive
scene."*

## Maps to all three target clusters at once

| Cluster | What the capstone shows them |
|---|---|
| **Reka** | persistent 3D/4D scene with temporal consistency + planning over it — their exact JD |
| **Odyssey** | a controllable, predictable scene representation (the understanding side of interactive video WMs) |
| **PerceptAI** | a semantic 3D scene graph + spatial reasoning — *plus* the predictive dimension they don't have |

> Open-model choices for the spatial/VLM + dynamics components are grounded in
> `project3-spatial-vlm-3d-scenegraph-SOTA.md` (complete: VGGT / ConceptGraphs / Qwen3-VL-8B,
> license-clean, single-A100) + `project1-`/`project2-…SOTA.md`.
> Cosmos3-Nano is the frontier reference for the omnimodal-unification thesis.

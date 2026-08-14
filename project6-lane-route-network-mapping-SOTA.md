# Project #6 — Lane & Route Network Mapping: Infrastructure-Scale Graph Construction

> **The thesis, at road scale.** The same question the other projects answer at manipulation-room
> and maritime-scene scale — *how do we turn raw sensor signals into persistent, structured
> representations of reality?* — applied to **lane and route networks**: camera + LiDAR →
> vectorized lane geometry → topology graph → route-level world state.
>
> This is NOT "May Mobility interview prep." It is the **infrastructure-world-state** vertical
> of the `physical-world-meta-fm` thesis, parallel to maritime object-state, human-state, and
> urban environment-state.

## Why this is the same thesis at a different scale

```text
                RAW SIGNALS              STRUCTURED GRAPH           REASONING
                ───────────              ────────────────           ─────────
 Maritime       RGB·IR·AIS·radar    →    vessel state graph    →   route planning
 Manipulation   RGBD·proprioception →    object/scene graph    →   task planning
 Health         wearables·voice·gait →   human-state graph     →   clinical reasoning
 Urban          weather·energy·EV    →   city-risk graph       →   policy decisions
 ──────────────────────────────────────────────────────────────────────────────
 LANE/ROUTE     camera·LiDAR·GPS    →    lane topology graph   →   route planning
```

The lane-network problem is a **graph construction + temporal consistency** problem:
- **Vectorized instance segmentation** (MapTR) = your segmentation edge at road scale
- **Lane topology** (GNN over lane segments) = your scene-graph work at infrastructure scale
- **Temporal consistency** (streaming mapping) = your 4DGS temporal work at road scale
- **HD map as world state** = the calibrated representation embodied AI plans against

The methods that solve this — MapTR, TopoNet, TopoMLP, LaneSegNet — are literally the same
architecture families (transformers, GNNs, bipartite matching) you already use, just applied to
road geometry instead of room geometry.

---

## 1. Vectorized HD Map Construction

### The representation

HD maps are **vectorized**: lane boundaries as polylines, centerlines as directed curves,
drivable areas as polygons, traffic elements as points with semantics. The task is:
given multi-view camera (+ optional LiDAR), predict these vector elements with instance-level
precision.

### Method comparison

| Method | Year / Venue | arXiv | GitHub | Stars | License | Maintained | Key Idea |
|---|---|---|---|---|---|---|---|
| **HDMapNet** | 2021 / ICRA 2022 | [2107.06307](https://arxiv.org/abs/2107.06307) | [Tsinghua-MARS-Lab/HDMapNet](https://github.com/Tsinghua-MARS-Lab/HDMapNet) | ~400 | — | Stale | Foundational: BEV → semantic/instance/directional map prediction |
| **VectorMapNet** | 2022 / ICML 2023 | [2206.08920](https://arxiv.org/abs/2206.08920) | [Tsinghua-MARS-Lab/vectormapnet](https://github.com/Tsinghua-MARS-Lab/vectormapnet) | ~200 | — | Stale | End-to-end polyline prediction via auto-regressive decoder |
| **MapTR** | 2022 / ICLR 2023 ⭐ | [2208.14437](https://arxiv.org/abs/2208.14437) | [hustvl/MapTR](https://github.com/hustvl/MapTR) | ~1.5k | **MIT** | Low (2025) | Permutation-equivalent modeling; hierarchical bipartite matching |
| **MapTRv2** | 2023 / IJCV 2024 | [2308.05736](https://arxiv.org/abs/2308.05736) | [hustvl/MapTR](https://github.com/hustvl/MapTR) | (same) | **MIT** | Low (2025) | Dense supervision + one-to-many matching; **67.4 mAP** (nuScenes) |
| **StreamMapNet** | 2023 / WACV 2024 | [2308.12570](https://arxiv.org/abs/2308.12570) | [yuantianyuan01/StreamMapNet](https://github.com/yuantianyuan01/StreamMapNet) | ~100 | — | Active | Long-sequence temporal modeling; Multi-Point Attention for elongated elements |
| **GeMap** | 2023 / ECCV 2024 | [2312.03341](https://arxiv.org/abs/2312.03341) | [cnzzx/GeMap](https://github.com/cnzzx/GeMap) | ~100 | — | Active | Geometry-decoupled attention; **69.4 mAP** (nuScenes), **71.8 mAP** (AV2) |
| **HIMap** | 2024 / CVPR 2024 | [2403.08639](https://arxiv.org/abs/2403.08639) | [BritaryZhou/HIMap](https://github.com/BritaryZhou/HIMap) | ~80 | — | Active | **77.8 mAP** (nuScenes) — significant margin over prior SOTA |
| **RelMap** | 2025 | — | — | — | — | — | Class-aware spatial relation priors; **77.1 mAP** (nuScenes) |
| **KPMapNet** | 2025 | — | — | — | — | — | **75.1 mAP** (nuScenes), **74.2 mAP** (AV2) |
| **MapUnveiler** | 2025 | — | — | — | — | — | Clip-level temporal modeling; +10.7% mAP in occluded scenes |
| **MapDiffusion** | 2025 | — | — | — | — | — | Generative diffusion for uncertainty-aware mapping |
| **SuperMapNet** | 2026 | — | — | — | — | — | +14.9% mAP improvement on nuScenes |

### Benchmark numbers (nuScenes, 60×30m, Chamfer-distance AP)

| Method | mAP | Backbone | Training |
|---|---|---|---|
| HDMapNet | 23.0 | ResNet-50 | camera only |
| VectorMapNet | 40.9 | ResNet-50 | camera only |
| MapTR | 62.1 | ResNet-50 | 24 epochs |
| **MapTRv2** | **67.4** | ResNet-50 | 24 epochs |
| GeMap | 69.4 | ResNet-50 | — |
| **HIMap** | **77.8** | — | — |

### Solo-reproducibility assessment

| Method | Feasible? | VRAM | Training Time | License | Notes |
|---|---|---|---|---|---|
| HDMapNet | ✅ | ~12 GB | ~12h | — | Simplest baseline, good starting point |
| VectorMapNet | ✅ | ~15 GB | ~48h (120ep) | — | Longer training needed |
| **MapTR/MapTRv2** | ✅ | ~19 GB | ~24h (24ep) | **MIT** | Best balance of performance/maintenance |
| StreamMapNet | ✅ | ~20 GB | ~24h | — | Temporal modeling, non-overlapping splits |
| GeMap | ✅ | ~20 GB | ~24h | — | Highest AV2 performance |
| HIMap | ✅ | ~20 GB | ~24h | — | Highest nuScenes performance |

**All methods run on 1×A100 80GB.** MapTRv2 is the recommended anchor — MIT license, documented, reproducible.

---

## 2. Lane Topology / Graph Construction

### The problem

Lane detection gives you individual lane segments. **Topology** gives you how they connect:
which lanes lead into which, where they diverge, which traffic elements govern which lanes.
This is a **graph problem**: nodes = lane centerlines, edges = connectivity (predecessor/successor).

### Method comparison

| Method | Year / Venue | arXiv | GitHub | License | Key Idea | OLS (OL-v2) |
|---|---|---|---|---|---|---|
| **TopoNet** | 2023 | [2304.05277](https://arxiv.org/abs/2304.05277) | [OpenDriveLab/TopoNet](https://github.com/OpenDriveLab/TopoNet) | — | GNN-based topology reasoning; centerlines as vertices | 35.6 |
| **TopoMLP** | 2024 / ICLR 2024 | [2310.02170](https://arxiv.org/abs/2310.02170) | [wudongming97/TopoMLP](https://github.com/wudongming97/TopoMLP) | — | Simple MLP topology heads; first-detect-then-reason | 43.7 |
| **LaneSegNet** | 2024 / ICLR 2024 | — | [OpenDriveLab/LaneSegNet](https://github.com/OpenDriveLab/LaneSegNet) | — | Unified lane segment representation (centerlines + dividers + drivable areas) | — |
| **TopoLogic** | 2024 / NeurIPS 2024 | — | [Franpin/TopoLogic](https://github.com/Franpin/TopoLogic) | — | Geometric distance + similarity-based reasoning | 44.1 |
| **TopoFG** | 2025 / AAAI 2025 | [2511.12590](https://arxiv.org/abs/2511.12590) | — | — | Fine-grained representation for topology | **48.0** |
| **TopoPoint** | 2025 / NeurIPS 2025 | — | — | — | Endpoint-based topology enhancement | **48.8** |
| **SeqGrowGraph** | 2025 / ICCV 2025 | — | [MIV-XJTU/SeqGrowGraph](https://github.com/MIV-XJTU/SeqGrowGraph) | — | Lane topology as chain of graph expansions | — |

### OpenLane-V2 benchmark (the standard)

```
OLS = ½[DET_l + DET_t + √TOP_ll + √TOP_lt]
```

| Component | What it measures | Best (2026) |
|---|---|---|
| **DET_l** | Lane centerline detection (Fréchet distance AP) | ~35 |
| **DET_t** | Traffic element detection (IoU AP) | ~55 |
| **TOP_ll** | Lane-lane topology accuracy | ~31 |
| **TOP_lt** | Lane-traffic element topology accuracy | ~31 |
| **OLS** (overall) | Unified topology score | **48.8** (TopoPoint) |

**Honest framing:** best OLS is ~48 vs theoretical max ~100. Topology is still hard. The gap
is mostly in **connectivity prediction** (TOP_ll), not detection — a graph-reasoning problem,
not a perception problem.

### Solo-reproducibility

| Method | Feasible? | VRAM | Notes |
|---|---|---|---|
| TopoNet | ✅ | ~12 GB | GNN-based, needs OpenLane-V2 |
| TopoMLP | ✅ | ~15 GB | Best OLS/simplicity tradeoff |
| TopoLogic | ✅ | ~15 GB | Interpretable pipeline |
| TopoFG | ✅ | ~15 GB | Current SOTA, likely feasible |
| TopoPoint | ✅ | ~15 GB | New SOTA, NeurIPS 2025 |

---

## 3. BEV Backbones for Online Mapping

The mapping head (MapTR) needs BEV features extracted from multi-view camera. These are the
backbone options:

| Backbone | Year / Venue | GitHub | Stars | Key Idea |
|---|---|---|---|---|
| **BEVFormer** | 2022 / ECCV 2022 | [fundamentalvision/BEVFormer](https://github.com/fundamentalvision/BEVFormer) | ~1.2k | Spatial cross-attention + temporal self-attention from 3D reference points |
| **BEVDet** | 2022 | [HuangJunJie2017/BEVDet](https://github.com/HuangJunJie2017/BEVDet) | ~1k | Forward projection (LSS) to lift 2D → BEV |
| **StreamPETR** | 2023 / ICCV 2023 | [exiawsh/StreamPETR](https://github.com/exiawsh/StreamPETR) | ~400 | Object-centric temporal modeling; first online method comparable to LiDAR |
| **SparseBEV** | 2023 / ICCV 2023 | [MCG-NJU/SparseBEV](https://github.com/MCG-NJU/SparseBEV) | ~470 | High-performance sparse 3D detection from multi-camera |

**Recommendation:** BEVFormer or StreamPETR as the BEV feature extractor, feeding into MapTRv2
as the mapping head. This is the standard recipe in recent papers.

---

## 4. Datasets & Benchmarks

### Primary datasets

| Dataset | Scale | Sensors | HD Map | Key Metric | URL |
|---|---|---|---|---|---|
| **nuScenes** | 1000 scenes, 20s each | 6 cam + 32-beam LiDAR + 5 radar | Rasterized + vectorized (v1.3) | mAP (Chamfer) | [nuscenes.org](https://www.nuscenes.org/) |
| **Argoverse 2** | 1000 3D-annotated scenes | 7 cam + 2 stereo + 2×32-beam LiDAR | Vectorized (lanes + connectivity) | mAP, mAVE | [argoverse.org/av2](https://argoverse.org/av2.html) |
| **OpenLane-V2** | 2000 scenes | Multi-view (from AV2 + nuScenes) | 3D centerlines + topology | OLS, DET_l, TOP_ll | [github.com/OpenDriveLab/OpenLane-V2](https://github.com/OpenDriveLab/OpenLane-V2) |

### New datasets (2025–2026)

| Dataset | Scale | Key Innovation |
|---|---|---|
| **HRDX** (Honda, 2026) | 40+ hours, ~1,400 km, 128-beam LiDAR + aerial imagery | 10 vector map classes, 20+ attributes; aerial pretraining improves +3.41 mAP |
| **KITScenes Multimodal** (2026) | 62 km², 9 cam + 7 LiDAR + 3× 4D radar | Full Lanelet2 topology; production-grade maps |
| **MapDR** (CVPR 2025) | 10K+ scenes, 18K driving rules | Traffic regulation extraction + lane correspondence |

### Synthetic / auto-labeling

| Dataset | Use |
|---|---|
| **HG-Lane** (2026) | Diffusion-based adverse weather generation (snow/rain/fog/night) |
| **SLEDGE** | 450K frames, generative HD map + traffic simulation |
| **ControlMap** (2026) | Controllable HD map generation via SD map conditioning |

---

## 5. Foundation Models for Mapping

### The frontier

| Model | What it does | Status |
|---|---|---|
| **EMMA** (Waymo, 2024) | Multimodal LLM (Gemini-based); lane waypoints as text polylines; joint perception + planning | Proprietary, not reproducible |
| **SDTagNet** (NeurIPS 2025) | First to leverage SD map (OpenStreetMap) textual annotations via NLP features; +5.9 mAP | Open, promising |
| **CleanMAP** (CVPR 2025 Workshop) | Distilling VLMs for confidence-driven crowdsourced HD map updates | Workshop paper |
| **AutoVLA** (2025) | VLA with adaptive reasoning + RL fine-tuning for driving | Research |
| **Senna** (hustvl) | Bridging VLMs and end-to-end driving | Research |

### The gap

No open foundation model does what EMMA does for mapping. The 2026 opportunity:
**use a VLM to auto-label lane geometry from dashcam video**, then train a fast mapping head
(MapTRv2) on the auto-labeled data. This bridges your segmentation expertise (SAM2 auto-labeling)
into the lane domain.

---

## 6. Mapping to the physical-world-meta-fm Thesis

### The stack at infrastructure scale

```
camera·LiDAR·GPS
  └─ BEV backbone (BEVFormer / StreamPETR)
       └─ vectorized lane geometry (MapTRv2)
            └─ lane topology graph (TopoMLP / TopoLogic)
                 └─ route-level world state (persistent, predictive)
                      └─ reasoning layer (VLM / frontier model)
```

### How it maps to existing projects

| Existing Project | Infrastructure-Scale Equivalent |
|---|---|
| **#1** (world model: `(z_t, a_t) → z_{t+1}`) | Lane-level predictive model: predict future lane geometry under construction/traffic |
| **#2** (3DGS semantic lifting) | Lift 2D lane annotations into persistent 3D HD map field |
| **#3** (scene graphs + spatial VLM) | Lane topology graph + route-level spatial queries ("is this lane reachable from X?") |
| **#4** (semantic-predictive 4D) | Predictive HD map that evolves over time (construction, closures, detours) |

### The differentiation

Almost nobody combines:
1. **Vectorized mapping** (MapTR) at production quality
2. **Topology reasoning** (GNN over lane graphs)
3. **Temporal consistency** (streaming/4D lane tracking)
4. **VLM grounding** (route-level natural language queries over the lane graph)

Item 4 is the gap — and your VLM + scene-graph expertise is the natural bridge.

---

## 7. Concrete Minimal First Milestone

**Goal:** smallest end-to-end version that proves the skill — BEV features → vectorized lane
geometry → topology graph — with credible numbers on a standard benchmark.

### v0 (the must-ship core): MapTRv2 on nuScenes

| Step | What | Tooling | Output |
|---|---|---|---|
| 1 | Set up **nuScenes** dataset + devkit | nuScenes devkit | 1000 scenes, 6-camera images + map annotations |
| 2 | Train **MapTRv2** (R50 backbone, 24 epochs) | [hustvl/MapTR](https://github.com/hustvl/MapTR) (MIT) | **~67 mAP** on nuScenes val; inference FPS |
| 3 | Evaluate on **Argoverse 2** | MapTRv2 config for AV2 | **~64 mAP** on AV2 val |
| 4 | Add **TopoMLP** topology head | [wudongming97/TopoMLP](https://github.com/wudongming97/TopoMLP) | **~44 OLS** on OpenLane-V2 subset_B |
| 5 | Package + document | README with mAP table, inference speed, VRAM usage | Reproducible baseline |

### v1 (the differentiator): topology + temporal

- Add **StreamMapNet** temporal modeling for streaming lane mapping
- Add **TopoLogic** or **TopoFG** for improved topology reasoning
- Evaluate on **OpenLane-V2** subset_A (Argoverse 2 derived)
- Report OLS breakdown (DET_l, DET_t, TOP_ll, TOP_lt)

### v2 (the bridge to VLM): auto-labeling + spatial queries

- Use **SAM2 + depth estimation** to auto-label lane geometry from nuScenes dashcam video
- Train MapTRv2 on auto-labeled data → measure the gap vs GT-labeled
- Fine-tune **Qwen3-VL-8B** (Apache, LoRA) to answer route-level queries over the lane graph
  ("which lane goes to the highway?", "is there a left-turn lane at the next intersection?")
- This bridges your segmentation expertise + VLM expertise into the lane domain

### Compute budget (1×A100 80GB)

| Step | VRAM | Time |
|---|---|---|
| MapTRv2 training (24ep, R50) | ~19 GB | ~24h |
| TopoMLP training | ~15 GB | ~12h |
| Inference / eval | ~8 GB | minutes |
| **Total v0** | **~19 GB peak** | **~2 days** |

---

## 8. Honest Difficulty Assessment

**Lowest risk / highest signal:**
- MapTRv2 on nuScenes — well-trodden, MIT license, documented, reproducible on 1×A100.
- The vectorized mapping task is mature; reproducing ~67 mAP is a credibility bar.

**Medium:**
- Topology (TopoMLP/TopoLogic) — the OLS metric is harder to improve, but reproducing
  baseline numbers is tractable. The graph-reasoning component is the genuinely new skill.

**Higher:**
- Temporal/streaming mapping (StreamMapNet) — needs careful data pipeline work.
- VLM auto-labeling — bridging SAM2 → lane geometry is non-trivial (lanes are thin,
  elongated, and occluded; SAM2 is optimized for objects, not linear features).

**Out of scope (mention as future direction):**
- Foundation models for mapping (EMMA-class) — datacenter-only, proprietary.
- City-scale HD map construction — production infrastructure, not a research project.
- Real-time deployment on vehicle hardware — systems engineering, not ML research.

---

## 9. Key Papers (prioritized reading)

### Tier 0 — Core canon
1. **MapTR** — Li et al., ICLR 2023 Spotlight — [2208.14437](https://arxiv.org/abs/2208.14437)
2. **MapTRv2** — Li et al., IJCV 2024 — [2308.05736](https://arxiv.org/abs/2308.05736)
3. **TopoMLP** — Wu et al., ICLR 2024 — [2310.02170](https://arxiv.org/abs/2310.02170)
4. **OpenLane-V2** — Wang et al., NeurIPS 2023 — the benchmark paper

### Tier 1 — Strong baselines
5. **StreamMapNet** — Tian et al., WACV 2024 — [2308.12570](https://arxiv.org/abs/2308.12570)
6. **TopoLogic** — NeurIPS 2024
7. **TopoFG** — AAAI 2025 — [2511.12590](https://arxiv.org/abs/2511.12590)
8. **TopoPoint** — NeurIPS 2025

### Tier 2 — BEV backbones
9. **BEVFormer** — Li et al., ECCV 2022 — [2203.17270](https://arxiv.org/abs/2203.17270)
10. **StreamPETR** — Wang et al., ICCV 2023

### Tier 3 — Frontier / foundation
11. **EMMA** — Waymo, 2024 — [2410.23262](https://arxiv.org/abs/2410.23262)
12. **SDTagNet** — NeurIPS 2025
13. **HRDX** — Honda, 2026 (dataset)

---

## 10. What is NOT solo-feasible (honesty)

- **EMMA / proprietary VLM for mapping** — Waymo internal, requires Gemini API access.
- **City-scale HD map production** — production infrastructure, not research.
- **Real-time vehicle deployment** — systems engineering, ROS 2 + CAN bus + safety validation.
- **Multi-GPU distributed training** — MapTRv2 trains fine on 1×A100; don't overclaim infra.
- **New sensor modalities (4D radar, thermal)** — data acquisition is the bottleneck, not models.

---

> **Bottom line:** this project slots into `physical-world-meta-fm` as the infrastructure-scale
> vertical — the same thesis (raw signals → structured graph → reasoning) applied to lane and
> route networks. The core method (MapTRv2 + TopoMLP) is reproducible on 1×A100 in ~2 days.
> The differentiator is bridging VLM grounding into the lane domain (route-level spatial queries
> over the topology graph), which connects directly to your segmentation + VLM expertise.

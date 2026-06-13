# Spatial VLMs + 3D Scene Graphs + Spatial Reasoning — SOTA (June 2026)

> Portfolio research brief for a **"Foundation Model for Physical AI / Spatial Intelligence"** VLM research-engineer role (PerceptAI-style: VLMs grounding language in 3D scene graphs / Gaussian-splat scenes, open-vocab spatial queries, agentic spatial reasoning, digital twins).
>
> **Verification:** all model names, repos, arXiv IDs, HF IDs, licenses, and maintenance status below were checked **live against arxiv.org / github.com / huggingface.co during June 2026**. Items I could not fully verify are flagged inline and collected in [§9](#9-honesty--what-i-could-not-fully-verify). Single-A100 VRAM figures are **inferred from param count + published recipes** unless a repo states an explicit number (the few that do are noted).
>
> **Candidate fit (context for the recommendations):** strong in production CV (panoptic/semantic **segmentation** at scale), CLIP/multimodal embeddings, distributed A100 training, eval-pipeline rigor, an agentic-LLM reasoning framework (TGAER, DSPy). Adjacency: a planned **semantic 3D/4D Gaussian Splatting** project (SAM2 masks lifted into 3DGS). Honest gaps: web-scale VLM **pretraining** (Megatron/DeepSpeed multi-node, LLaVA/Qwen-VL scale), publications.

---

## TL;DR

- **Best starting spatial VLM (1×A100 80GB, LoRA):** **Qwen3-VL-8B-Instruct** — Apache-2.0, *verified live*, has **native 3D bounding-box grounding** built in. Robotics-flavored alternatives: **RoboBrain 2.5** (Apache-2.0, on Qwen3-VL) or **NVIDIA Cosmos-Reason2** (commercial-friendly).
- **Best 3D scene-graph repo to build on:** **ConceptGraphs** (MIT, inference-only, posed RGB-D, single-GPU, largest community). Robotics slant → MIT-SPARK **Khronos**+**Clio**; LLM-reasoning slant → **3DGraphLLM**.
- **Best language-grounded-splat / open-vocab-3D-seg to pair (segmentation edge):** **LangSplat** (only language-splat with an explicit 24 GB spec; canonical baseline). Faster: **OccamLGS**; permissive license: **SAGA**.
- **3D-into-VLM approach the JD wants:** the 2025-26 frontier feeds **RGB video into a feed-forward geometry encoder** (VGGT / DUSt3R-family) fused with a 2D encoder — **Spatial-MLLM, VLM-3R, VG-LLM, Ross3D**. Splat-token VLMs (**SplatTalk, GaussianVLM**) are the youngest, most "3D-native" frontier.
- **Recon front-end:** **VGGT** (CVPR 2025 Best Paper; now has a **commercial** checkpoint too) for quality; **MoGe-2 / MapAnything-apache** for permissive-license single-GPU work.
- **Training-infra honesty:** LoRA/QLoRA VLM finetune + DeepSpeed ZeRO-Offload / FSDP2-offload are **genuinely solo-demonstrable**. Real FSDP/ZeRO **sharding** needs ≥2 GPUs. **Megatron multi-node** is *not* honestly demonstrable solo — claim conceptual understanding only.

---

## 1. Spatial VLMs

What does spatial reasoning / grounding, open weights, size, license, hardware. **Verified:** Qwen3-VL-8B-Instruct exists (Apache-2.0, 9B, model card explicitly advertises *"3D grounding for spatial reasoning and embodied AI"*).

| Model | What it does | Open wts | Sizes | License | HF ID / GitHub / arXiv | LoRA on 1×A100 80GB? | Maintained |
|---|---|---|---|---|---|---|---|
| **Qwen3-VL** ⭐ | General VLM; **native 3D box grounding** + 2D grounding; 256K–1M ctx; Visual Agent | Y | dense 2B/4B/8B/32B; MoE 30B-A3B, 235B-A22B | **Apache-2.0** (uniform) | `Qwen/Qwen3-VL-8B-Instruct` · [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) · [2511.21631](https://arxiv.org/abs/2511.21631) | **Y** (2/4/8B; 32B QLoRA; 235B no) | Active (first rel. 2025-09-23) |
| **RoboBrain 2.5** ⭐ | Embodied "brain": perception+reasoning+planning; SOTA spatial benches | Y | 4B/8B (on Qwen3-VL) | **Apache-2.0** | [FlagOpen/RoboBrain2.0](https://github.com/FlagOpen/RoboBrain2.0) · 2.0=[2507.02029](https://arxiv.org/abs/2507.02029); 2.5≈2601.14352 *(unverified ID)* | **Y** | Very active (2.5 ~Mar 2026) |
| **Cosmos-Reason** (NVIDIA) | Physical-AI spatio-temporal/physics reasoning, embodied CoT | Y | R1≈8B; R2 2B/8B/32B | **NVIDIA Open Model License** (commercial OK) | `nvidia/Cosmos-Reason1-7B` · [nvidia-cosmos/cosmos-reason1](https://github.com/nvidia-cosmos/cosmos-reason1) · [2503.15558](https://arxiv.org/abs/2503.15558) | **Y** (8B) | Active (R2 ~Mar 2026) |
| **RoboRefer** | First 3D-aware VLM for multi-step **spatial referring**, depth encoder | Y | 2B / 8B (NVILA) | **Apache-2.0** | `Zhoues/RoboRefer-8B-SFT` · [Zhoues/RoboRefer](https://github.com/Zhoues/RoboRefer) · [2506.04308](https://arxiv.org/abs/2506.04308) | **Y** | Active |
| **Spatial-MLLM** | Spatial intelligence from **pure 2D video** (dual 2D-semantic + VGGT-3D encoder) | Y | ~5–6B (Qwen2.5-VL-3B) | **MIT** | `Diankun/Spatial-MLLM-*` · [diankun-wu/Spatial-MLLM](https://github.com/diankun-wu/Spatial-MLLM) · [2505.23747](https://arxiv.org/abs/2505.23747) | **Y** | Active (NeurIPS'25) |
| **Molmo / Molmo2** (Ai2) | Open VLM with **pointing/grounding** | Y | 1B/7B/72B; Molmo2-8B | **Apache-2.0** | `allenai/Molmo-7B-D-0924` · [allenai/molmo](https://github.com/allenai/molmo) · [2409.17146](https://arxiv.org/abs/2409.17146) | **Y** (1/7/8B; 72B no) | Molmo2-8B ~Mar 2026 |
| **MolmoAct** (Ai2) | Action Reasoning Model: robot actions + visual motion trace | Y | 7B (Qwen2.5) | **Apache-2.0** (research) | `allenai/MolmoAct-7B-D-0812` · [allenai/molmoact](https://github.com/allenai/molmoact) · [2508.07917](https://arxiv.org/abs/2508.07917) | Maybe (weights **FP32** → 2× mem) | Active |
| **PaliGemma 2** (Google) | Versatile transfer VLM (VQA/detect/segment/OCR) | Y | 3B/10B/28B × {224,448,896} | **Gemma license** (commercial OK, not OSI) | `google/paligemma2-3b-pt-224` · [2412.03052](https://arxiv.org/abs/2412.03052) | **Y** (3B/10B; best-documented LoRA/QLoRA) | Active |
| **Qwen2.5-VL** | General VLM, strong 2D grounding (box+point JSON) | Y | 3B/7B/32B/72B | **Mixed**: 7/32B Apache; **3B=NC**; 72B custom | `Qwen/Qwen2.5-VL-7B-Instruct` · [QwenLM/Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) · [2502.13923](https://arxiv.org/abs/2502.13923) | **Y** (3/7B; 32B QLoRA) | Superseded by Qwen3-VL |
| **SpatialRGPT** (NVIDIA/UCSD) | Region-grounded (box/mask) spatial reasoning + per-region depth | Y | 8B (of 3/7/8B) | **Apache-2.0** | `a8cheng/SpatialRGPT-VILA1.5-8B` · [AnjieCheng/SpatialRGPT](https://github.com/AnjieCheng/SpatialRGPT) · [2406.01584](https://arxiv.org/abs/2406.01584) | Likely | **Stale** (last ~Dec 2024); succ. 4D-RGPT (NC) |
| **RoboPoint** (UW) | Outputs 2D **points** = spatial affordances | Y | 7B/13B (+LoRA ckpts) | **Apache-2.0** | `wentao-yuan/robopoint-v1-vicuna-v1.5-13b` · [wentaoyuan/RoboPoint](https://github.com/wentaoyuan/RoboPoint) · [2406.10721](https://arxiv.org/abs/2406.10721) | **Y** (LoRA ckpts exist) | Light (~Jul 2025) |
| **SpatialBot** (BAAI-DCAI) | RGB **+ depth** for precise spatial understanding | Y | 3B (Phi-2+SigLIP) | code Apache; wts CC-BY-4.0 | `RussRobin/SpatialBot-3B` · [BAAI-DCAI/SpatialBot](https://github.com/BAAI-DCAI/SpatialBot) · [2406.13642](https://arxiv.org/abs/2406.13642) | **Y** (trivial) | Moderate (ICRA'25) |
| **SpaceLLaVA** (remyx) | LLaVA-1.5 LoRA-tuned on SpatialVLM-style synthetic VQA | Y | 13B (+lite/4B) | **Apache-2.0** | `remyxai/SpaceLLaVA` · [remyxai/VQASynth](https://github.com/remyxai/VQASynth) | **Y** (it *was* made via LoRA) | Low/moderate |
| **LLaVA-3D** | 3D pos-emb on 2D CLIP patches → 3D QA/grounding | Y | 7B | **Ambiguous** (HF Apache vs GH CC-BY-NC-SA → treat NC) | `ChaimZhu/LLaVA-3D-7B` · [ZCMax/LLaVA-3D](https://github.com/ZCMax/LLaVA-3D) · [2409.18125](https://arxiv.org/abs/2409.18125) | Likely | Moderate (ICCV'25) |
| **SpatialLM** (Manycore) | Point cloud → structured 3D layout (walls/objects/boxes) | Y | 0.5B / 1B | **NC** (via Sonata encoder CC-BY-NC) | `manycore-research/SpatialLM1.1-Llama-1B` · [manycore-research/SpatialLM](https://github.com/manycore-research/SpatialLM) · [2506.07491](https://arxiv.org/abs/2506.07491) | Trivial (16–24 GB) | Active (1.1 Sep'25) |
| **3D-LLM** | First LLM taking 3D point clouds (BLIP-2/Q-Former) | Y (Drive, no HF) | ~3–12B | **MIT** | [UMass-Foundation-Model/3D-LLM](https://github.com/UMass-Foundation-Model/3D-LLM) · [2307.12981](https://arxiv.org/abs/2307.12981) | Plausible | **Abandoned** (~Feb 2024) |
| **SpatialVLM** (Google) | Synthetic-data pipeline for 3D spatial QA (distance/size) | **No** (paper) | — | — | [spatial-vlm.github.io](https://spatial-vlm.github.io) · [2401.12168](https://arxiv.org/abs/2401.12168) | N/A | Lives on via Gemini Robotics-ER (closed) |

**Closed / API-only (not finetunable):** **Gemini Robotics 1.5** (VLA, partner-only, [2510.03342](https://arxiv.org/abs/2510.03342)); **Gemini Robotics-ER 1.6** (embodied-reasoning VLM, **API-only**); **Gemini Robotics On-Device** (Trusted-Tester, no weights). Sizes undisclosed; none on HF.

**Naming note:** "SpaceVLM" as a distinct model **does not exist** — it conflates with Google's SpatialVLM.

### Verdict — best starting points for a solo dev (1×A100 80GB, LoRA / light finetune)

1. **Qwen3-VL-8B-Instruct** ⭐ — *the pick.* Verified live: Apache-2.0, 9B, model card explicitly ships **3D bounding-box grounding** on top of Qwen2.5-VL's 2D grounding. Dense 4B/8B LoRA-finetune comfortably on one A100. Best capability × clean license × current SOTA lineage × tooling maturity.
2. **RoboBrain 2.5 (4B/8B)** or **Cosmos-Reason2 (2B/8B)** — if the project is explicitly **embodied/robotics** spatial reasoning. Both LoRA on one A100; RoboBrain is Apache-2.0 and the most actively maintained embodied "brain"; Cosmos is commercial-friendly (NVIDIA Open Model License).
3. **PaliGemma 2 (3B/10B)** or **Molmo-7B/Molmo2-8B** — lowest-friction grounding baseline. PaliGemma 2 has the most documented single-A100 LoRA/QLoRA recipes and built-in detect/segment; Molmo (Apache-2.0) is the cleanest if **pointing** is the core task.

**Niche (all single-A100 LoRA-able):** RoboRefer-2B/8B (spatial referring), Spatial-MLLM (spatial-from-2D-video), SpatialBot-3B (RGB+depth).

**Explicitly pretrain-only / datacenter-only / off-limits:** SpatialVLM (never released); Qwen3-VL-235B / Qwen-VL-72B / Molmo-72B (multi-node); 3D-LLM (abandoned, no HF); all Gemini Robotics (closed); 32B variants (LoRA only with 4-bit quant).

---

## 2. Open-Vocabulary 3D Scene Graphs

> License caveat: several strong-on-paper repos (**BBQ, OpenGraph, DovSG**) ship with **no LICENSE file** → legally all-rights-reserved, unsafe for a public portfolio. **Open3DSG** is **AGPL-3.0 + archived**. Last-activity is GitHub `pushed_at` at time of research; it will drift.

| Method | Inputs | Repo | arXiv | License | Last activity | Solo-reproducible |
|---|---|---|---|---|---|---|
| **ConceptGraphs** ⭐ | Posed RGB-D | [concept-graphs/concept-graphs](https://github.com/concept-graphs/concept-graphs) | [2309.16650](https://arxiv.org/abs/2309.16650) | **MIT** | 2025-10-16 · 886★ | **Y** — inference-only; needs external LLM |
| **HOV-SG** | Posed RGB-D | [hovsg/HOV-SG](https://github.com/hovsg/HOV-SG) | [2403.17846](https://arxiv.org/abs/2403.17846) | **MIT** | 2026-01-19 · 485★ | **Y** (A100 ample; ~128 GB RAM for GT) |
| **3DGraphLLM** | Point cloud + inst-seg → LLM | [CognitiveAISystems/3DGraphLLM](https://github.com/CognitiveAISystems/3DGraphLLM) | [2412.18450](https://arxiv.org/abs/2412.18450) | **MIT** | 2026-03-23 · 121★ | **Y** (single A100; weights released) |
| **Khronos** (MIT-SPARK) | RGB-D + pose (ROS) | [MIT-SPARK/Khronos](https://github.com/MIT-SPARK/Khronos) | [2402.13817](https://arxiv.org/abs/2402.13817) | **BSD-3** | 2026-05-12 · 506★ | **Y** (light-GPU real-time SLAM backbone) |
| **Clio** (MIT-SPARK) | RGB-D + poses (ROS) / JSON | [MIT-SPARK/Clio](https://github.com/MIT-SPARK/Clio) | [2404.13696](https://arxiv.org/abs/2404.13696) | **BSD-2** | 2025-09-01 · 244★ | **Y** (task-driven; needs ROS) |
| **SG-Nav** | Posed RGB-D (sim) | [bagh2178/SG-Nav](https://github.com/bagh2178/SG-Nav) | [2410.08189](https://arxiv.org/abs/2410.08189) | **MIT** | 2025-09-16 · 339★ | **Y** (zero-shot 3DSG + LLM nav) |
| **BBQ** | Posed RGB-D | [linukc/BeyondBareQueries](https://github.com/linukc/BeyondBareQueries) | [2406.07113](https://arxiv.org/abs/2406.07113) | **None** ⚠ | 2025-07-16 · 40★ | Y (10/16 GB) but no reuse rights |
| **OpenGraph** | RGB+LiDAR (outdoor) | [BIT-DYN/OpenGraph](https://github.com/BIT-DYN/OpenGraph) | [2403.09412](https://arxiv.org/abs/2403.09412) | **None** ⚠ | 2025-12-03 · 154★ | Likely; no reuse rights |
| **Open3DSG** | Point cloud + 2D frames | [boschresearch/Open3DSG](https://github.com/boschresearch/Open3DSG) | [2402.12259](https://arxiv.org/abs/2402.12259) | **AGPL-3.0** ⚠ | archived · 160★ | Partial (4×V100 + ~300 GB preproc) |
| **SceneGraphFusion** | RGB-D incremental | [ShunChengWu/SceneGraphFusion](https://github.com/ShunChengWu/SceneGraphFusion) | [2103.14898](https://arxiv.org/abs/2103.14898) | BSD-2 | 2022-07-26 · 192★ | Y but **closed-set + dormant** |

**2025-26 leads (paper-only / not deep-verified):** **OpenFunGraph** ([2503.19199](https://arxiv.org/abs/2503.19199), CVPR'25, **no code found**), functional/affordance scene graphs (**FunGraph** [2503.07909](https://arxiv.org/abs/2503.07909)).

**Verdict:** **ConceptGraphs** is the best solo base — inference-only (no 3D training, no labeled SG data), ordinary posed RGB-D, single GPU, **MIT** (safe to fork/showcase), largest community. Robotics slant → **Khronos**+**Clio** (BSD, MIT-SPARK, most-maintained). LLM-reasoning slant → **3DGraphLLM** (freshest learnable-graph-into-LLM). Avoid BBQ/OpenGraph/DovSG (no license) and Open3DSG (AGPL + archived) for a public portfolio.

---

## 3. Open-Vocab 3D Segmentation / Grounding (her segmentation edge → 3D)

> Most 3DGS-derived repos carry the **Gaussian-Splatting License** (Inria + Max Planck, **non-commercial**, shows as `NOASSERTION` on GitHub). Fine for a research portfolio, not for product.

| Method | Repo | arXiv | License | Last activity | 1×A100 / consumer |
|---|---|---|---|---|---|
| **LangSplat** ⭐ | [minghanqin/LangSplat](https://github.com/minghanqin/LangSplat) | [2312.16084](https://arxiv.org/abs/2312.16084) | Gaussian-Splatting (NC) | README Oct'25 · ~1.1k★ | **Y — 24 GB VRAM (documented)** |
| **SAGA** | [Jumpat/SegAnyGAussians](https://github.com/Jumpat/SegAnyGAussians) | [2312.00860](https://arxiv.org/abs/2312.00860) | **Apache-2.0** | 2025-03-25 · 972★ | **Y — RTX 3090 24 GB (paper)** |
| **Gaussian Grouping** | [lkeab/gaussian-grouping](https://github.com/lkeab/gaussian-grouping) | [2312.00732](https://arxiv.org/abs/2312.00732) | **Apache-2.0** | 2024-07-04 · 1,013★ | Y (~24 GB inferred) |
| **OccamLGS** | [insait-institute/OccamLGS](https://github.com/insait-institute/OccamLGS) | [2412.01807](https://arxiv.org/abs/2412.01807) | NOASSERTION (research) | 2025-11-18 · 67★ | **Y — training-free, ~100× faster than LangSplat** |
| **Feature-3DGS** | [ShijieZhou-UCLA/feature-3dgs](https://github.com/ShijieZhou-UCLA/feature-3dgs) | [2312.03203](https://arxiv.org/abs/2312.03203) | Gaussian-Splatting (NC) | 2024-10-17 · ~667★ | Y (consumer OK typical) |
| **OpenGaussian** | [yanmin-wu/OpenGaussian](https://github.com/yanmin-wu/OpenGaussian) | [2406.02058](https://arxiv.org/abs/2406.02058) | Gaussian-Splatting (NC) | README May'25 · ~211★ | A100 likely |
| **SAM2Point** | [ZiyuGuo99/SAM2Point](https://github.com/ZiyuGuo99/SAM2Point) | [2408.16768](https://arxiv.org/abs/2408.16768) | **Apache-2.0** | 2024-09-11 · 356★ | **Y — training-free** (SAM2 most faithful 3D lift) |
| **OpenMask3D** | [OpenMask3D/openmask3d](https://github.com/OpenMask3D/openmask3d) | [2306.13631](https://arxiv.org/abs/2306.13631) | **MIT** | 2023-12-15 · 260★ | A100 Y |
| **OpenScene** | [pengsongyou/openscene](https://github.com/pengsongyou/openscene) | [2211.15654](https://arxiv.org/abs/2211.15654) | **Apache-2.0** | 2023-10-27 · 829★ | A100 Y (1×A100 40GB train) |
| **OpenNeRF** | [opennerf/opennerf](https://github.com/opennerf/opennerf) | [2404.03650](https://arxiv.org/abs/2404.03650) | **MIT** | 2025-04-23 · 146★ | A100 Y; consumer likely (Nerfstudio) |
| **Chorus** (CVPR'26 Oral) | [GaussianWorld/Chorus](https://github.com/GaussianWorld/Chorus) | [2512.17817](https://arxiv.org/abs/2512.17817) | CC-BY-SA-4.0 | Dec'25 · 20★ | A100 Y (22–48 GiB by scene) |
| **OpenSplat3D** | [VisualComputingInstitute/opensplat3d](https://github.com/VisualComputingInstitute/opensplat3d) | [2506.07697](https://arxiv.org/abs/2506.07697) | Gaussian-Splatting (NC) | 2026-03-06 · 24★ | A100 likely |
| **Dr. Splat** (CVPR'25) | [kaist-ami/Dr-Splat](https://github.com/kaist-ami/Dr-Splat) | [2502.16652](https://arxiv.org/abs/2502.16652) | Gaussian-Splatting (NC) | 2025-06-10 · 103★ | Likely Y |
| **Segment-then-Splat** (NeurIPS'25) | [luyr/Segment-then-Splat](https://github.com/luyr/Segment-then-Splat) | [2503.22204](https://arxiv.org/abs/2503.22204) | MIT (code repo) | code 29★ | A100 likely |

**SAM2-into-3D landscape:** **SAM2Point** (voxelize → multi-directional "videos" → SAM2 video seg, training-free, most faithful); **Gaussian Grouping** + **SAGA** lift SAM masks into per-Gaussian identity features (SAGA cleanest hardware story + Apache); text-grounded successors **OccamLGS / Dr. Splat / OpenSplat3D / Chorus** all benchmark vs LangSplat.

**Verdict:** **LangSplat** to pair with the scene graph — genuinely open-vocab (CLIP+SAM), the **only** language-splat with an explicit modest spec (24 GB), canonical baseline. Faster/2026-fresh → **OccamLGS** (training-free, ~100×). Permissive license + interactive → **SAGA** (Apache, RTX 3090). For the **SAM2-lifted-into-3DGS** project the candidate already plans, **SAGA / Gaussian Grouping / SAM2Point** are the direct lineage.

---

## 4. 3D-into-VLM Integration (the JD's headline)

Five mechanism families. **2025-26 trend: away from explicit 3D inputs → toward RGB-video with geometry recovered internally by a VGGT/DUSt3R-style encoder.**

| System | 3D input | Injection mechanism | arXiv | GitHub | License | Active? |
|---|---|---|---|---|---|---|
| **Spatial-MLLM** | RGB video only | Dual encoder: 2D-semantic + **VGGT-init spatial** → connector → Qwen2.5-VL | [2505.23747](https://arxiv.org/abs/2505.23747) | [diankun-wu/Spatial-MLLM](https://github.com/diankun-wu/Spatial-MLLM) | **MIT** | Y |
| **VLM-3R** | Monocular RGB video | Geometry encoder → implicit 3D + camera tokens (Spatial-Visual-View fusion) | [2505.20279](https://arxiv.org/abs/2505.20279) | [VITA-Group/VLM-3R](https://github.com/VITA-Group/VLM-3R) | Apache (DUSt3R dep NC) | Y (CVPR'26) |
| **VG-LLM** | RGB video only | VGGT geometry encoder fused at patch level → Qwen2.5-VL | [2505.24625](https://arxiv.org/abs/2505.24625) | [LaVi-Lab/VG-LLM](https://github.com/LaVi-Lab/VG-LLM) | no LICENSE ⚠ | Y |
| **Ross3D** | Posed multi-view RGB(-D) | **No new encoder** — cross-view + BEV reconstruction *objectives* | [2504.01901](https://arxiv.org/abs/2504.01901) | [Haochen-Wang409/ross3d](https://github.com/Haochen-Wang409/ross3d) | **Apache-2.0** | Y |
| **Video-3D LLM** | Posed RGB-D video | Posed-depth 3D position encoding on video tokens | [2412.00493](https://arxiv.org/abs/2412.00493) | [LaVi-Lab/Video-3D-LLM](https://github.com/LaVi-Lab/Video-3D-LLM) | **Apache-2.0** | Y |
| **GPT4Scene** | RGB video + rendered BEV | No arch change — BEV image + consistent object-ID markers (works w/ GPT-4o) | [2501.01428](https://arxiv.org/abs/2501.01428) | [Qi-Zhangyang/GPT4Scene-and-VLN-R1](https://github.com/Qi-Zhangyang/GPT4Scene-and-VLN-R1) | **Apache-2.0** | Y |
| **SplatTalk** | Posed RGB → feed-forward 3DGS | 3D-Language Gaussian field → "3D tokens" → frozen LLM (zero-shot 3D VQA) | [2503.06271](https://arxiv.org/abs/2503.06271) | [ngailapdi/SplatTalk](https://github.com/ngailapdi/SplatTalk) | **MIT** | Y (no HF wts) |
| **GaussianVLM** | 3DGS from video | Per-Gaussian language features → dual sparsifier → ~132 scene tokens | [2507.00886](https://arxiv.org/abs/2507.00886) | [amhalacheva/GaussianVLM](https://github.com/amhalacheva/GaussianVLM) | **MIT** | early-access |
| **LLaVA-3D** | Posed multi-view RGB-D | "3D patches": CLIP patches + depth/pose 3D pos-emb (ODIN-style) | [2409.18125](https://arxiv.org/abs/2409.18125) | [ZCMax/LLaVA-3D](https://github.com/ZCMax/LLaVA-3D) | unclear ⚠ | Y |
| **PointLLM** | Raw colored point cloud | Point-BERT/ULIP-2 encoder + MLP projector → tokens | [2308.16911](https://arxiv.org/abs/2308.16911) | [InternRobotics/PointLLM](https://github.com/InternRobotics/PointLLM) | CC-BY-NC-SA-4.0 | Y |
| **3DGraphLLM** | 3D scene graph | Learnable scene-graph tokens fed directly into LLM | [2412.18450](https://arxiv.org/abs/2412.18450) | [CognitiveAISystems/3DGraphLLM](https://github.com/CognitiveAISystems/3DGraphLLM) | MIT | Y |

**Approach families & tradeoffs:**
1. **Point-cloud encoder + projector** (PointLLM/GPT4Point/Robin3D) — strong object-scale; needs real point clouds; NC licenses.
2. **Posed-depth 2D→3D lifting / 3D pos-emb** (3D-LLM, LLaVA-3D, Video-3D LLM) — reuses pretrained 2D VLMs, scene-scale; needs accurate depth+poses.
3. **Gaussian-splat tokens** (SplatTalk, GaussianVLM) — most 3D-native; youngest frontier; no HF weights yet.
4. **Feed-forward geometry encoder fused with 2D encoder, NO explicit 3D input** (Spatial-MLLM, VG-LLM, VLM-3R) — RGB-video-only; lowest input requirements; **dominates 2025-26**.
5. **No arch change — supervision/visual prompting** (Ross3D objectives, GPT4Scene BEV+ID markers) — cheapest to deploy; shallower 3D.

### Feed-forward 3D reconstruction front-ends (images → point cloud + poses, no per-scene optim)

| Model | What it does | GitHub | arXiv | License | HF | 1×GPU |
|---|---|---|---|---|---|---|
| **VGGT** ⭐ | Cameras+depth+point maps+tracks, <1s | [facebookresearch/vggt](https://github.com/facebookresearch/vggt) | [2503.11651](https://arxiv.org/abs/2503.11651) | NC + **`VGGT-1B-Commercial` ckpt (commercial, no military)** | `facebook/VGGT-1B` | 1.26B, single A100 |
| **MoGe-2** | Metric point maps + depth/normal/FOV, single img | [microsoft/MoGe](https://github.com/microsoft/MoGe) | [2507.02546](https://arxiv.org/abs/2507.02546) | **MIT** | `Ruicheng/moge-2-vitl` | consumer (RTX 3090) |
| **MapAnything** | Universal metric multi-view recon, 12+ tasks | [facebookresearch/map-anything](https://github.com/facebookresearch/map-anything) | [2509.13414](https://arxiv.org/abs/2509.13414) | code Apache; `-apache` wts **Apache** / other NC | `facebook/map-anything-apache` | single A100 |
| **Pi3 (π³)** | Permutation-equivariant poses+points, no ref view | [yyfz/Pi3](https://github.com/yyfz/Pi3) | [2507.13347](https://arxiv.org/abs/2507.13347) | code BSD; wts NC (HF says BSD — conflict ⚠) | `yyfz233/Pi3` | A100 |
| **Fast3R** | 1000+ images one forward pass | [facebookresearch/fast3r](https://github.com/facebookresearch/fast3r) | [2501.13928](https://arxiv.org/abs/2501.13928) | FAIR Noncommercial | `jedyang97/Fast3R_ViT_Large_512` | **1500 imgs / 78.6 GB single A100** |
| **CUT3R** | Online/recurrent stateful metric recon | [CUT3R/CUT3R](https://github.com/CUT3R/CUT3R) | [2501.12387](https://arxiv.org/abs/2501.12387) | CC-BY-NC-SA-4.0 | Drive | near-constant mem |
| **DUSt3R** | Pairwise dense stereo, no calib | [naver/dust3r](https://github.com/naver/dust3r) | [2312.14132](https://arxiv.org/abs/2312.14132) | CC-BY-NC-SA-4.0 | `naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt` | pairs/small sets |
| **MASt3R** | DUSt3R + matching, metric | [naver/mast3r](https://github.com/naver/mast3r) | [2406.09756](https://arxiv.org/abs/2406.09756) | CC-BY-NC-SA-4.0 | `naver/MASt3R_ViTLarge_..._metric` | pairs |
| **Aether** | Geometry-aware **world model**: 4D recon + action-cond video pred | [InternRobotics/Aether](https://github.com/InternRobotics/Aether) | [2503.18945](https://arxiv.org/abs/2503.18945) | **MIT** | `AetherWorldModel/AetherV1` | needs A100 80GB |

**VGGT won CVPR 2025 Best Paper (verified).** It is the center of gravity; VGGT-Long ([2507.16443](https://arxiv.org/abs/2507.16443), RTX 4090) / FastVGGT / StreamVGGT are the variant family. **License is the load-bearing axis:** permissive = MoGe-2 (MIT), Aether (MIT), MapAnything-`apache`, VGGT-Commercial ckpt; NC = vanilla VGGT, all NAVER (DUSt3R/MASt3R), CUT3R, Fast3R, Spann3R.

**Practical picks:** single image → **MoGe-2** (MIT, 3090-class); multi-view commercial-clean → **MapAnything-apache** (single A100, most-maintained); best raw quality → **VGGT** (use Commercial ckpt to stay clean); streaming → CUT3R.

---

## 5. Benchmarks (spatial reasoning / 3D-QA)

| Benchmark | Measures | Input | Metric | URL |
|---|---|---|---|---|
| **ScanQA** | Free-form QA over ScanNet | 3D scan | EM@1, CIDEr | [2112.10482](https://arxiv.org/abs/2112.10482) · [github](https://github.com/ATR-DBI/ScanQA) |
| **SQA3D** | Situated (pose-aware) QA | 3D scan | Acc / EM | [2210.07474](https://arxiv.org/abs/2210.07474) · [site](https://sqa3d.github.io/) |
| **OpenEQA** (Meta) | Open-vocab embodied QA | Egocentric RGB-D | LLM-Match (GPT-4 judge) | [site](https://open-eqa.github.io/) · [github](https://github.com/facebookresearch/open-eqa) |
| **3DSRBench** | 3D reasoning (height/orient/loc/multi-obj) | 2D + multi-view synth | MCQ (+Circular/FlipEval) | [site](https://3dsrbench.github.io/) · [2412.07825](https://arxiv.org/abs/2412.07825) |
| **SpatialRGPT-Bench** | Region-grounded directions + metric distance | Image + region masks | Success Rate | [2406.01584](https://arxiv.org/abs/2406.01584) |
| **SpatialBench** (BAAI) | Depth/proximity/reach/size/count | RGB-D | Acc / EM | [github](https://github.com/BAAI-DCAI/SpatialBot) · [2406.13642](https://arxiv.org/abs/2406.13642) |
| **BLINK** (spatial subsets) | Spatial relation / rel-depth / multi-view | 2D single/multi | MCQ | [site](https://zeyofu.github.io/blink/) · [2404.12390](https://arxiv.org/abs/2404.12390) |
| **VSI-Bench** (NYU/Fei-Fei) | Visual-spatial intelligence | Egocentric video | MCA + MRA | [site](https://vision-x-nyu.github.io/thinking-in-space.github.io/) · [2412.14171](https://arxiv.org/abs/2412.14171) |
| **MMSI-Bench** (ICLR'26) | Multi-image spatial reasoning, 10 tasks | Multi-image | MCQ | [site](https://runsenxu.com/projects/MMSI_Bench/) · [2505.23764](https://arxiv.org/abs/2505.23764) |
| **SPAR-Bench** | 3D spatial, 20 tasks, 3 difficulties | Single+multi+video | Acc + MRA | [2503.22976](https://arxiv.org/abs/2503.22976) · [github](https://github.com/fudan-zvg/spar) |
| **All-Angles Bench** | Multi-view consistency, 6 tasks | Multi-view | Acc | [site](https://danielchyeh.github.io/All-Angles-Bench/) · [2504.15280](https://arxiv.org/abs/2504.15280) |
| **STI-Bench** (ICCV'25) | Quantitative spatial-temporal | Real video | MRA + MCQ | [site](https://mint-sjtu.github.io/STI-Bench.io/) · [2503.23765](https://arxiv.org/abs/2503.23765) |
| **OmniSpatial** | Comprehensive spatial, 50 subcats | Single image | MCQ | [site](https://qizekun.github.io/omnispatial/) · [2506.03135](https://arxiv.org/abs/2506.03135) |
| **ViewSpatial-Bench** | Ego/allo perspective-taking | Single image | MCQ | [site](https://zju-real.github.io/ViewSpatial-Page/) · [2505.21500](https://arxiv.org/abs/2505.21500) |
| **Spatial457** (CVPR'25) | 6D spatial reasoning, 5 levels | Synthetic | Acc + RPDR | [github](https://github.com/XingruiWang/Spatial457) · [2502.08636](https://arxiv.org/abs/2502.08636) |

**Categories:** genuinely-3D-input (ScanQA, SQA3D, OpenEQA, SpatialRGPT-Bench, SpatialBench); 3D-reasoning-from-2D (3DSRBench, BLINK, OmniSpatial, ViewSpatial, Spatial457); video/multi-view (VSI-Bench, STI-Bench, SPAR, All-Angles, MMSI).

**What a credible portfolio result reports (honest framing — "reproduce + ablate, don't claim SOTA"):**
- **ScanQA:** EM@1 ~22–26, CIDEr ~85–95 (LEO / Chat-Scene class) — above baseline, not SOTA.
- **SQA3D:** ~50–54% overall **with per-question-type breakdown**.
- **OpenEQA:** run a strong VLM through the official harness; report LLM-Match in the ~40s–50s. It's an eval harness, not a training target.
- **MMSI-Bench:** ~28–31% for an open model is honest (chance = 25%); >35% from an open model is a red flag.
- **Best reproducible picks for a demo:** **ViewSpatial-Bench** (huge fine-tune headroom, 43→82), **SpatialBench** (small RGB-D, clean RGB-vs-RGB-D delta story), **VSI-Bench/STI-Bench** (the credible contribution is correctly implementing **MRA** + frame-budget ablations and *reproducing* frontier numbers, not beating them).

---

## 6. Training Infrastructure — Honest Solo-vs-Lab Reality

The JD names **Megatron-LM, DeepSpeed, FSDP**. What that means on **one A100 80GB**:

| Stack | Repo | Solo-on-1-A100 reality |
|---|---|---|
| **Megatron-LM / Core** | [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) (v0.17.1, May'26) | **NOT honestly demonstrable solo.** TP/PP/CP/EP value is multi-node only (benchmarked to 6,144 H100s). Has `pretrain_vlm.py` for multimodal. Claim *conceptual understanding + code familiarity*, never "trained at scale". |
| **DeepSpeed** | [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed) (v0.19.1, May'26) | **ZeRO-Offload / ZeRO-Infinity** = genuine single-GPU win (fit 10B+ by offloading optimizer/weights to CPU/NVMe). ZeRO-3 sharding is a no-op on 1 GPU. LoRA+DeepSpeed via PEFT = clean solo demo. |
| **PyTorch FSDP2** | [FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) | Demonstrable solo: **CPU/NVMe offload + activation checkpointing + FSDP2+QLoRA**. Real *sharding* needs ≥2 GPUs. `FULL_SHARD`≈ZeRO-3, `SHARD_GRAD_OP`≈ZeRO-2. |

**Practical VLM-finetune stacks (all current/maintained, support Qwen-VL/LLaVA + LoRA/QLoRA on 1×A100):**

| Tool | Repo | Note |
|---|---|---|
| **LLaMA-Factory** | [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | Broadest VLM coverage; DeepSpeed+FSDP backends |
| **ms-swift** | [modelscope/ms-swift](https://github.com/modelscope/ms-swift) | 300+ multimodal models incl. Qwen3-VL |
| **Unsloth** | [unslothai/unsloth](https://github.com/unslothai/unsloth) | Fastest/leanest single-GPU; ~90% less VRAM |
| **Axolotl** | [axolotl-ai-cloud/axolotl](https://github.com/axolotl-ai-cloud/axolotl) | YAML-driven; QLoRA Qwen2-VL-7B ≈ 38 GB (blog ⚠) |
| **TRL + PEFT** | [huggingface/trl](https://github.com/huggingface/trl) | Most transparent; wraps `transformers.Trainer` |

**Claim honestly:**
- ✅ "Fine-tuned Qwen3-VL with QLoRA on a single A100" — **hands-on, true, centerpiece.**
- ✅ "Used DeepSpeed ZeRO-Offload / FSDP2 offload + activation checkpointing to train beyond single-GPU memory" — **true & demonstrable solo.**
- ⚠ "Demonstrated FSDP/ZeRO **sharding** on a 2–8 GPU node" — **true only if you rent the node** (cheap, high-leverage; converts "read about" → "ran it").
- ⛔ "Trained with Megatron at scale" — **don't.** Overclaiming Megatron is the easiest interview trap. Ceiling: "familiar with Megatron-Core TP/PP/CP for multi-node VLM pretraining; understand trade-offs."

---

## 7. Key Papers (prioritized)

1. **VGGT** ([2503.11651](https://arxiv.org/abs/2503.11651), 2025) — CVPR'25 **Best Paper**; feed-forward all-3D-attributes-in-one-pass; the recon front-end everyone builds on.
2. **Qwen3-VL** ([2511.21631](https://arxiv.org/abs/2511.21631), 2025) — open SOTA VLM with **native 3D grounding**; your base model.
3. **Spatial-MLLM** ([2505.23747](https://arxiv.org/abs/2505.23747), 2025) — dual 2D-semantic + VGGT-3D encoder; the dominant "3D-from-2D-video" recipe; MIT.
4. **VLM-3R** ([2505.20279](https://arxiv.org/abs/2505.20279), 2025) — VLM + instruction-aligned 3D reconstruction from monocular video (CVPR'26).
5. **ConceptGraphs** ([2309.16650](https://arxiv.org/abs/2309.16650), 2023) — the open-vocab 3D scene-graph foundation; MIT, still the most-forked base.
6. **LangSplat** ([2312.16084](https://arxiv.org/abs/2312.16084), 2023) — CLIP+SAM language field on 3DGS; the open-vocab-splat reference everyone benchmarks.
7. **SpatialVLM** ([2401.12168](https://arxiv.org/abs/2401.12168), 2024) — the synthetic-spatial-VQA-data idea that seeded the whole subfield.
8. **SpatialRGPT** ([2406.01584](https://arxiv.org/abs/2406.01584), 2024) — region-grounded spatial reasoning + per-region depth; defined the region-VQA paradigm.
9. **VSI-Bench / "Thinking in Space"** ([2412.14171](https://arxiv.org/abs/2412.14171), 2024) — the headline video spatial-intelligence benchmark; MRA metric.
10. **OpenEQA** (Meta, CVPR'24, [site](https://open-eqa.github.io/)) — embodied-QA eval harness; the credibility benchmark for a scene-understanding demo.
11. **HOV-SG** ([2403.17846](https://arxiv.org/abs/2403.17846), 2024) — hierarchical (floor→room→object) open-vocab scene graphs; MIT.
12. **RoboBrain 2.0** ([2507.02029](https://arxiv.org/abs/2507.02029), 2025) — unified embodied perception+reasoning+planning brain; the "Physical AI" archetype the JD describes.

---

## 8. Concrete Solo-Feasible Portfolio Project

**Title:** *"Open-vocab 3D scene graph + spatial-QA from a phone video, on one A100."*

**Thesis:** reconstruct a real scene from a casual video, lift SAM2 masks into a language-grounded 3D representation, build an open-vocab 3D scene graph, and answer open-vocab spatial queries with a 3D-aware VLM — wrapped in a rigorous eval harness. This bridges the candidate's **segmentation + planned 3DGS** strengths directly into the JD's "VLMs grounding language in 3D scene graphs / splat scenes + open-vocab spatial queries + agentic reasoning" headline.

**Pipeline (all single-A100, all permissive-or-research-OK):**

```
phone video
  └─ VGGT (Commercial ckpt) or MapAnything-apache   → posed point cloud + cameras   [feed-forward, no per-scene optim]
       └─ SAM2 masks → SAGA / Gaussian Grouping       → language-grounded 3DGS        [her segmentation edge → 3D]
            └─ ConceptGraphs (MIT)                     → open-vocab 3D object/relation scene graph
                 └─ 3DGraphLLM  OR  Qwen3-VL-8B (LoRA) → answer open-vocab spatial queries over the graph
                      └─ eval harness                  → ScanQA + SQA3D + OpenEQA + VSI-Bench (MRA), reproduce-not-beat
```

**Smallest end-to-end v0 (1–2 weeks):**
- One ScanNet/Replica scene (skip live capture). **VGGT → ConceptGraphs (MIT, out-of-box) → Qwen3-VL-8B zero-shot** answering 10–20 hand-written spatial queries ("what's left of the sofa?", "how far is the fridge from the table?"). No training. Deliverable: a notebook + short video + a small eval table on SQA3D / a ConceptGraphs query set.

**v1 (the differentiator, +2–4 weeks):**
- Add the **SAM2 → SAGA language-splat** branch (the candidate's 3DGS adjacency). LoRA-finetune **Qwen3-VL-8B** (QLoRA, Unsloth/LLaMA-Factory) on SpatialVLM-style synthetic QA generated from the scene graph (VQASynth-style). Report ScanQA EM@1/CIDEr, SQA3D per-type, **OpenEQA LLM-Match**, and a **RGB-only vs RGB+3D-graph ablation** — the single most credible result.

**Rough compute:** v0 = a few A100-hours (inference + recon). v1 LoRA on Qwen3-VL-8B = ~1–3 A100-days for a few epochs of synthetic QA + eval sweeps. Total well within a single 80 GB card.

**"Infra credibility" add-on (cheap, high-leverage):** run the LoRA finetune through **DeepSpeed ZeRO-Offload** and **FSDP2** once, and rent a **2×A100 node for one day** to show **real FSDP sharding** — converts the JD's infra keywords from "read" to "ran".

**Explicitly NOT solo-feasible (be upfront):**
- Web-scale **VLM pretraining** (LLaVA/Qwen-VL/Molmo scale) — datacenter-only.
- **Megatron multi-node 3D-parallel** training — lab-scale; claim conceptual only.
- Beating frontier closed models (Gemini Robotics-ER, GPT-5) on hard spatial benches — the honest goal is *reproduce + ablate + clean eval*, not SOTA.

### Mapping to the PerceptAI-style JD

| JD theme | This project demonstrates it via |
|---|---|
| VLMs grounding language in **3D scene graphs** | ConceptGraphs / 3DGraphLLM + Qwen3-VL |
| **Gaussian-splat scenes** | SAM2 → SAGA / Gaussian Grouping language-splat (her 3DGS adjacency) |
| **Open-vocab spatial queries** | CLIP/SAM2 open-vocab + Qwen3-VL 3D grounding (her CLIP/segmentation edge) |
| **Agentic spatial reasoning** | DSPy/TGAER agent loop over the scene-graph query interface (her existing framework) |
| **Digital twins** | the posed point cloud + language-splat + scene graph *is* a queryable twin |
| **VLM training at scale** (gap) | honest LoRA/QLoRA + DeepSpeed-Offload/FSDP2 demo; conceptual Megatron |

---

## 9. Honesty — What I Could NOT Fully Verify

- **Single-A100 LoRA VRAM numbers are inferred** from param count + recipes for nearly every VLM; explicit per-GPU figures are rare. Explicit VRAM was found only for: LangSplat (24 GB), SAGA (RTX 3090 24 GB), OpenScene (1×A100 40 GB train), Chorus (22–48 GiB), BBQ (10/16 GB), Fast3R (1500 imgs/78.6 GB A100), MoGe-2 (RTX 3090-class).
- **RoboBrain 2.5 arXiv ID (2601.14352)** and exact dense Qwen3-VL per-model release dates were not individually pinned (Sept–Oct 2025 rollout window + flagship 2025-09-23 confirmed; 235B confirmed).
- **License conflicts (treat as research-only until you check upstream):** LLaVA-3D (HF Apache vs GitHub CC-BY-NC-SA), VLM-3R & SpatialLM (Apache code but NC dependency/encoder), Pi3 (repo NC vs HF BSD), VG-LLM / 3D-LLaVA (no LICENSE file → all-rights-reserved by default).
- **No-license repos** (legally not reusable for a public portfolio): BBQ, OpenGraph, DovSG.
- **MolmoAct weights are FP32** → ~2× memory vs bf16; factor into single-GPU planning.
- **VGGT licensing was updated during this research:** beyond the NC original, a gated **`VGGT-1B-Commercial`** checkpoint now exists (commercial OK, no military) — verified live on the repo README.
- **Last-activity / star counts** are point-in-time GitHub `pushed_at` snapshots and will drift.
- **Paper-only / no released code or weights:** OpenFunGraph (no repo); 3D-LLM, GPT4Point, Robin3D, SplatTalk, GaussianVLM, CUT3R, Spann3R (no released HF weights — a negative that can't be exhaustively proven).
- **Benchmark SOTA numbers** come from published comparison tables (vary by `w/objects` vs `w/o objects` splits and re-eval harnesses), not live leaderboards. VSI-Bench frontier figures use a stricter re-eval harness ([2508.13142](https://arxiv.org/abs/2508.13142)) not directly comparable to the original repo.
- **Axolotl VRAM/speed figures** and the FSDP-vs-ZeRO performance characterization are from secondary blogs/Medium, not primary benchmarks — directional only.

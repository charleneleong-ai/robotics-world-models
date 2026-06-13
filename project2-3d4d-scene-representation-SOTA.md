# Semantic 3D/4D Scene Reconstruction — SOTA Research (mid-2026)

> **Project:** Semantic 3D/4D scene reconstruction with 3D Gaussian Splatting — reuse production segmentation expertise to lift 2D semantics into a 3D Gaussian field, with a rigorous eval harness; extend to dynamic (4D) scenes for temporal consistency.
> **Target role:** World Models / 3D-4D generative / embodied AI (robotics + world-models lab, Reka).
> **Author background:** Production panoptic/semantic segmentation at scale (medical WSI, 100+ classes), distributed A100 training, CLIP/multimodal embeddings, eval-pipeline rigor. *No prior neural rendering.*
> **Research date:** 2026-06-12. All repo names, paper titles, arxiv IDs, and maintenance status cross-checked against the live web (not training data alone).

---

## TL;DR — Top Recommendations

| Decision | Recommendation | Why |
|---|---|---|
| **Representation** | **3D Gaussian Splatting (3DGS)** as the substrate; add **Mip-Splatting** for anti-aliasing, **2DGS** only if you need mesh/geometry | Field has decisively landed on 3DGS for static scenes: real-time render, minutes-to-train, explicit/editable. Every semantic + 4D method builds on it. |
| **Framework / repo** | **Nerfstudio** to start (one-command `ns-train`, web viewer, NeRF+3DGS), drop to **gsplat** when you write your own loop | Both Apache-2.0 (portfolio-safe), actively maintained, gsplat is the de-facto CUDA backend. Avoid INRIA repo as a *start* (non-commercial license + conda/CUDA fragility). |
| **Semantic method** | **OpenSplat3D** (cleanest maintained repo, purpose-built SAM→3DGS instance lifting), with **SAGA** / **LangSplat** as battle-tested fallbacks | OpenSplat3D is genuinely 2025-current, 0 open issues, `just setup`, ScanNet++/LERF eval built in. SAM2 is the converged 2D backbone. |
| **4D / dynamic** | **Shape-of-Motion** (monocular, maintained, explicit motion trajectories) for the headline; **4DGS/Deformable-3DGS on D-NeRF synthetic** to prototype | Monocular = phone video, no multi-cam rig. Persistent 3D trajectories are the visual "temporal consistency" story. Cite MonoDyGauBench as honesty check. |
| **First milestone** | **gsplat/Splatfacto reconstruction of one Mip-NeRF360 scene → SAM2 masks lifted via a Gaussian feature/identity field → mIoU + PSNR/SSIM/LPIPS eval harness** | Smallest end-to-end slice that proves the differentiator (2D-seg → 3D lift) with credible numbers. ~1 GPU-week. |

---

## 1. Representation Choice (2026)

### Verdict
For **static scenes**, the field has decisively landed on **3D Gaussian Splatting (3DGS)** and its descendants. NeRF has been displaced as the default for novel-view synthesis: 3DGS gives real-time rendering (60–100+ FPS), training in minutes instead of hours, and an explicit, editable, point-like representation. NeRF-family methods survive mainly in niches (implicit compactness, some inverse-rendering/relighting). Industry adoption confirms the shift — **OpenUSD added Gaussian Splatting support in April 2026**, pulling GS into Unreal, NVIDIA Omniverse, and Houdini pipelines.

**Recommended starting representation for a solo dev from a 2D-segmentation background: vanilla 3DGS** (via `gsplat` / Splatfacto in Nerfstudio). It's the best-documented, most-tutorialized, hardware-friendly entry point, and nearly every newer variant is a drop-in modification of it — so you learn the substrate everything else builds on. Add **Mip-Splatting** once aliasing bothers you; reach for **2DGS** only when you need accurate surface/mesh geometry rather than pretty renders.

### Method comparison

| Method | Year | Improves over predecessor | Maintained? | Tradeoffs | URL |
|---|---|---|---|---|---|
| **NeRF (+ Mip-NeRF 360 / Instant-NGP)** | 2020–22 | Original implicit radiance field; Instant-NGP made it fast-ish | Legacy for static NVS | Slow render, no real-time, compact storage, hard to edit | [arxiv 2003.08934](https://arxiv.org/abs/2003.08934) |
| **3DGS (vanilla)** | 2023 (SIGGRAPH) | Explicit anisotropic Gaussians + tile rasterizer → real-time + fast train | Reference stable, not very active; ecosystem moved to gsplat | Real-time, high quality, but high memory/storage, floaters, aliasing | [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) · [arxiv 2308.04079](https://arxiv.org/abs/2308.04079) |
| **gsplat (library)** | 2024–26 | Production CUDA rasterizer: ~10% faster train, ~4× less memory, MCMC, 2DGS, LiDAR | **Very active** (v1.5.3 Jul 2025; HiGS+LiDAR 2026) | The de-facto backend, not a method itself; needs NVIDIA GPU | [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) · [arxiv 2409.06765](https://arxiv.org/abs/2409.06765) |
| **Splatfacto (Nerfstudio)** | 2024–26 | Friendly training loop + web viewer + data pipeline on gsplat | Active | Easiest on-ramp; `-W` variant for in-the-wild photo sets | [docs.nerf.studio/splat](https://docs.nerf.studio/nerfology/methods/splat.html) |
| **Brush** | 2024–26 (v0.3) | Cross-platform 3DGS in Rust/wgpu+Burn — Mac/Win/Linux/Android/browser, **no CUDA** | Active | Often faster than gsplat; smaller ecosystem; **not PyTorch** | [ArthurBrussee/brush](https://github.com/ArthurBrussee/brush) |
| **Mip-Splatting** | 2024 (CVPR) | 3D smoothing + 2D Mip filter → alias-free across zoom/scale | Stable | Fixes 3DGS's biggest visual flaw; minor overhead; standard add-on | [autonomousvision/mip-splatting](https://github.com/autonomousvision/mip-splatting) · [arxiv 2311.16493](https://arxiv.org/abs/2311.16493) |
| **2DGS** | 2024 (SIGGRAPH) | Flat 2D surfels instead of 3D ellipsoids → far better surface/normal & meshing | Active; in gsplat | Best **geometry/mesh**; slightly lower pure-NVS sharpness | [hbb1/2d-gaussian-splatting](https://github.com/hbb1/2d-gaussian-splatting) · [arxiv 2403.17888](https://arxiv.org/abs/2403.17888) |
| **Scaffold-GS / Octree-GS** | 2024 (CVPR) | Anchors Gaussians to a structured/LOD grid → fewer redundant splats, large-scene robust | Stable | More compact in large/unbounded scenes; extra structure | [city-super/Scaffold-GS](https://github.com/city-super/Scaffold-GS) |
| **Compression/pruning line** (Speedy-Splat, MEGS², FastGS) | 2025–26 | Hessian pruning, SG compaction → big memory/storage cuts | Active research | Addresses memory weakness; mix-and-match, not standalone | [Speedy-Splat](https://arxiv.org/abs/2412.00578) · [MEGS² 2509.07021](https://arxiv.org/pdf/2509.07021) |
| **Feed-forward / generative** (VGGT, Fast3R, AnySplat, DepthSplat) | 2025–26 | Predict Gaussians from unposed images in **one forward pass** — no SfM | Very active frontier | Seconds, no COLMAP; lower fidelity than per-scene optim, less mature | [VGGT 2503.11651](https://arxiv.org/abs/2503.11651) · [Awesome-Feed-Forward-3D](https://github.com/ziplab/Awesome-Feed-Forward-3D) |

### Why this ordering for a solo dev
- **3DGS is the lingua franca.** Mip-Splatting, 2DGS, Scaffold-GS, and the compression line are all *modifications of the same Gaussian primitive + rasterizer*. Learn 3DGS once; the variants are incremental.
- **Coming from 2D segmentation,** the explicit per-Gaussian representation (position, scale, rotation, opacity, SH color) is far easier to inspect, mask, and edit than a NeRF's opaque MLP — and 2D segmentation models compose naturally with GS for feature/semantic lifting.
- **Don't start at the frontier.** Feed-forward methods (VGGT et al.) are the exciting 2025–26 direction and great for "no-COLMAP, instant reconstruction" demos, but fidelity and tooling lag per-scene optimization — a phase-2 extension, not your foundation.

> *Currency note:* No released "Mip-Splatting v2" or mainstream "3DGS-DR" repo displacing 3DGS was found as of June 2026. Real 2025–26 momentum is in (a) compression/pruning of 3DGS and (b) feed-forward/generative reconstruction.

---

## 2. Frameworks & Repos

### Best starting point for a solo PyTorch dev
**Nerfstudio.** `pip install nerfstudio` → one `ns-train` command → built-in real-time web viewer, supports both NeRF and 3DGS, **Apache-2.0** (commercially safe). When you need your own training loop or to squeeze memory/speed, drop to **gsplat** (same org, Apache-2.0, the CUDA backend powering Nerfstudio). **Avoid the original INRIA repo as a starting point** — it's the quality reference but its license is non-commercial research only and setup is conda/CUDA-fragile.

**CUDA pain point (all PyTorch/CUDA repos):** the recurring failure is a mismatch between installed CUDA toolkit and the one PyTorch was built against. The ecosystem standardized on **CUDA 11.8** (11.6 has known issues; 12.x works but often needs prebuilt wheels). gsplat ships **prebuilt wheels keyed to (Python, PyTorch, CUDA) triples** — use those to skip on-the-fly `nvcc` compilation, the #1 source of install grief. Rust-based **Brush** sidesteps CUDA entirely (WebGPU).

| Repo | Stars | Last activity | License | Framework | GPU / install | Verdict |
|------|-------|---------------|---------|-----------|---------------|---------|
| **Nerfstudio** | ~11.7k | Active (commits into 2025) | Apache-2.0 | Python / PyTorch | NVIDIA GPU, CUDA 11.7/11.8; `pip install nerfstudio` | **BEST starting point.** End-to-end NeRF + 3DGS, one-command train, live viewer, portfolio-safe. |
| **gsplat** | ~5.2k | **Very active** (commits Jun 2026; v1.5.3 Jul 2025) | Apache-2.0 | Python + CUDA/C++ | NVIDIA GPU; ~4× less VRAM than INRIA; prebuilt wheels | Best *library* — you write the loop. The "level 2" pick. Safest long-term bet. |
| **Original 3DGS** (graphdeco-inria) | ~22.3k | Last major Oct 2024 | **Non-commercial research** | Python + CUDA | CC 7.0+; **24 GB VRAM** paper quality; conda | Highest baseline quality (~28.4 dB) and the benchmark. **Not a starting point** (license + fragility). |
| **threestudio** | ~7k | Slowing (Aug 2024) | Apache-2.0 | Python / PyTorch | 6 GB min; DeepFloyd ~15 GB | **Different problem:** text/image→3D *generation*, not reconstruction. |
| **NVIDIA Kaolin** | ~5.1k | Active (v0.18 Aug 2025) | Apache-2.0 (+ some NSCL) | Python + CUDA | NVIDIA GPU | General 3D DL toolkit (building block), not a 3DGS pipeline. |
| **Kaolin-Wisp** | ~1.5k | **Stale** (v0.1.2 May 2023) | NVIDIA non-commercial | Python / PyTorch | depends on Kaolin | NeRF research framework. **Avoid** — superseded by Nerfstudio/gsplat. |
| **Brush** | ~4.7k | Active (v0.3 Sep 2025) | Apache-2.0 | **Rust** / WebGPU + Burn | **No CUDA**; any GPU, Mac/Win/Linux/Android/browser | Best zero-setup / vendor-neutral / browser demo. **But Rust, not PyTorch.** |
| **gaussian-splatting-lightning** | ~1.1k | Moderate (v0.12.1 Apr 2025) | Permissive | PyTorch **Lightning** | PyTorch 2.0.1, CUDA 11.8, multi-GPU | Good if you want Lightning structure + many derived algos. Niche vs Nerfstudio. |

**Quality is essentially a tie** across INRIA (~28.4 dB), gsplat (~28.1), and Nerfstudio (~27.9) PSNR on standard benchmarks — pick on ergonomics/license, not quality. **gsplat is the most alive repo** (commits through June 2026). **Brush is the wildcard** — a browser-deployable 3DGS demo is a *strong* portfolio differentiator, at the cost of leaving PyTorch.

---

## 3. Semantic / Panoptic 3D — The Differentiator

The field has fully migrated NeRF→3DGS; **Panoptic Lifting** is the only notable NeRF-based method still cited. This is the section that maps directly onto production-segmentation expertise.

| Method | What it does | Backbone | Year / Venue | Maintained? | Reproducible? | Links |
|---|---|---|---|---|---|---|
| **Panoptic Lifting** | Lifts 2D panoptic (Mask2Former) masks into a multi-view 3D panoptic field | **NeRF** | 2023 CVPR | No (frozen Sep 2023) | Hard (Mask2Former+detectron2+old CUDA) | [2212.09802](https://arxiv.org/abs/2212.09802) · [repo](https://github.com/nihalsid/panoptic-lifting) |
| **Gaussian Grouping** | Per-Gaussian identity codes supervised by SAM masks (+DEVA) for seg+edit | 3DGS | 2024 ECCV | No (abandoned Jul 2024) | Partial (~68 open issues) | [2312.00732](https://arxiv.org/abs/2312.00732) · [repo](https://github.com/lkeab/gaussian-grouping) |
| **LangSplat** | Distills CLIP into 3D Gaussians; ~199× faster open-vocab query vs LERF | 3DGS | 2024 CVPR | No (static May 2024) | Yes, widely reproduced | [2312.16084](https://arxiv.org/abs/2312.16084) · [repo](https://github.com/minghanqin/LangSplat) |
| **Feature-3DGS** | Distills arbitrary 2D features (SAM/LSeg) into a 3DGS feature field | 3DGS | 2024 CVPR | Light (Oct 2024) | Yes, with patience (CUDA recompile per dim) | [2312.03203](https://arxiv.org/abs/2312.03203) · [repo](https://github.com/ShijieZhou-UCLA/Feature-3DGS) |
| **SAGA** | Distills SAM masks into per-Gaussian affinity; promptable 3D seg ~4 ms | 3DGS | 2025 AAAI | Stable (Mar 2025) | Yes — most battle-tested promptable | [2312.00860](https://arxiv.org/abs/2312.00860) · [repo](https://github.com/Jumpat/SegAnyGAussians) |
| **OpenGaussian** | Point-level open-vocab 3D; 3D instance feats + CLIP codebook | 3DGS | 2024 NeurIPS | Light (May 2025) | Yes but finicky (4-stage) | [2406.02058](https://arxiv.org/abs/2406.02058) · [repo](https://github.com/yanmin-wu/OpenGaussian) |
| **Gaga** | 3D-aware memory bank associates 2D masks across views → consistent instances | 3DGS | 2026 TMLR | **Yes (May 2026, MIT)** | Likely; less battle-tested | [2404.07977](https://arxiv.org/abs/2404.07977) · [repo](https://github.com/weijielyu/Gaga) |
| **PanoGS** | Language tri-plane + graph-cut super-primitives → 3D panoptic open-vocab | 3DGS | 2025 CVPR | Released | Yes (no pretrained ckpts) | [2503.18107](https://arxiv.org/abs/2503.18107) · [repo](https://github.com/zhaihongjia/PanoGS) |
| **Dr. Splat** | Direct CLIP-embedding registration (no rendering) + Product Quantization | 3DGS | 2025 CVPR Highlight | Active | Partial (eval TBA) | [2502.16652](https://arxiv.org/abs/2502.16652) · [repo](https://github.com/kaist-ami/Dr-Splat) |
| **OpenSplat3D** | Feature-splatting + SAM masks + contrastive + VLM → text-driven open-vocab 3D **instance** seg | 3DGS | 2025 CVPRW | **Yes (clean, 0 issues)** | **Yes — cleanest install** (`just setup`, ScanNet++ eval) | [2506.07697](https://arxiv.org/abs/2506.07697) · [repo](https://github.com/VisualComputingInstitute/opensplat3d) |
| **Segment then Splat** | Inverts pipeline: split Gaussians per-object *before* recon; **built on SAM2** | 3DGS | 2025 NeurIPS | Released | Likely | [2503.22204](https://arxiv.org/abs/2503.22204) · [repo](https://github.com/vulab-AI/Segment-then-Splat) |
| **SADG / TRASE** | Tracker-free lifting of SAM masks into **dynamic** 3DGS (contrastive) | 3DGS (dynamic) | 2026 3DV | **Yes (Apr 2026)** | Yes — best maintained dynamic-semantic | [2411.19290](https://arxiv.org/abs/2411.19290) · [repo](https://github.com/yunjinli/SADG-SegmentAnyDynamicGaussian) |

**Paper-only / too new for code (watch):** CAGS, PanopticSplatting ([2503.18073](https://arxiv.org/abs/2503.18073)), SegSplat, EmbodiedSplat ([2603.04254](https://arxiv.org/abs/2603.04254)), OnlinePG — feed-forward / online semantic 3DGS is the 2026 frontier for streaming/embodied robotics.

### Recommendation
- **Best maintained + reproducible to lift 2D masks into a Gaussian field: OpenSplat3D.** Cleanest live repo (0 open issues, `just setup` + `just download_ckpts`, COLMAP/glomap preproc, ScanNet++/LERF-Mask/LERF-OVS eval out of the box), peer-reviewed (CVPRW 2025), purpose-built for exactly this task.
- **Fallbacks for max community validation:** **SAGA** (most-reproduced promptable) or **LangSplat** (canonical language-field baseline). Both stable-but-unmaintained — pin an old CUDA/PyTorch env.
- **Avoid as a primary dependency:** Gaussian Grouping (abandoned, 68 issues), Panoptic Lifting (NeRF + detectron2 hell), Click-Gaussian (no code), SA4D (skeleton).
- **2D segmentation backbone: SAM2.** The 2025–26 literature converged on SAM2 over Mask2Former — Mask2Former instance labels flicker across viewpoints, breaking cross-view association; SAM2's video/tracking mode gives view-consistent masks that lift cleanly. Use **Mask2Former only if you need closed-vocab panoptic class labels**; **Grounded-SAM** only when you need text-prompted open-vocab mask generation.
- **Datasets:** **ScanNet++** (primary — high-fidelity real indoor, dense panoptic GT, OpenSplat3D native eval, doubles as NVS benchmark) + **LERF-OVS / LERF-Mask** (open-vocab text-query showcase). Add **Replica** or **3D-OVS** for a controlled sanity check.

**Suggested stack:** SAM2 (masks + tracking) → OpenSplat3D (lift into 3DGS instance field) → eval on ScanNet++ (instance/panoptic) + LERF-OVS (open-vocab query). Fully reproducible today, peer-reviewed + actively-maintained, lands on the 2026 SAM2-into-3DGS frontier.

---

## 4. 4D / Dynamic Scenes & Temporal Consistency

Three architectural axes — understand this before choosing:
- **Deformation field** — canonical Gaussians + MLP/voxel field warps them per-timestep (4DGS-Wu, Deformable-3DGS). Compact, naturally coherent, but MLP can blur fast motion.
- **Per-timestep / tracked Gaussians** — Gaussians physically move with explicit trajectories (Dynamic-3DGS, Shape-of-Motion). Gives 6-DOF tracking "for free," heavier.
- **4D primitives** — Gaussians lifted to 4D (x,y,z,t) (Spacetime Gaussians). Best raw quality for multi-view rigs, highest parameter count.

| Method | Year / Venue | Approach | Input | Maintained / Reproducible | Difficulty | Links |
|---|---|---|---|---|---|---|
| **4DGS (Wu et al.)** | 2024 CVPR | Deformation field (HexPlane voxel + MLP) | **Multi-view** / D-NeRF synthetic | ~3.7k★; stable & widely reproduced (aging) | Medium | [2310.08528](https://arxiv.org/abs/2310.08528) · [repo](https://github.com/hustvl/4DGaussians) |
| **Deformable-3DGS** | 2024 CVPR | Deformation field (MLP warps canonical) | **Monocular** (D-NeRF) | ~1.2k★; reproducible, clean | Medium | [2309.13101](https://arxiv.org/abs/2309.13101) · [repo](https://github.com/ingra14m/Deformable-3D-Gaussians) |
| **Dynamic-3DGS (Luiten)** | 2024 3DV | Per-timestep tracked Gaussians (6-DOF) | **Multi-view rig** (PanopticSports) | Reproducible but **hard data req** | Hard (data) | [2308.09713](https://arxiv.org/abs/2308.09713) · [repo](https://github.com/JonathonLuiten/Dynamic3DGaussians) |
| **Spacetime Gaussians (STG)** | 2024 CVPR | 4D primitives (temporal opacity + poly motion) | **Multi-view video** | ~815★; best multi-view quality; **heavy GPU (24–48 GB)** | Hard (compute) | [repo](https://github.com/oppo-us-research/SpacetimeGaussians) |
| **Shape-of-Motion** | 2024 → **ICCV 2025 Highlight** | Per-point SE(3) motion bases / persistent trajectories | **Monocular** (casual video) | ~1.3k★, actively developed; **best-maintained monocular** | Medium-Hard | [2407.13764](https://arxiv.org/abs/2407.13764) · [repo](https://github.com/vye16/shape-of-motion) |
| **MoSca** | 2024 | 4D motion scaffolds + deformation-graph fusion | **Monocular** | ~366★; **limited future maintenance** (declared) | Hard (setup) | [2405.17421](https://arxiv.org/abs/2405.17421) · [repo](https://github.com/JiahuiLei/MoSca) |
| **MonoDyGauBench** | **TMLR 2025** | Benchmark (not a method) — apples-to-apples monocular eval | Monocular eval | Active. Finding: monocular 4DGS *fast but brittle* | N/A | [repo](https://github.com/lynl7130/MonoDyGauBench_code) |
| **Feed-forward 4D** (ReconDrive, WorldSplat, L4GM, Forge4D) | 2024–26 | Regress Gaussians in one pass, no per-scene optim | Varies (driving / sparse-view / image) | **Research-grade / domain-locked** | Very hard | [ReconDrive 2603.07552](https://arxiv.org/abs/2603.07552) · [WorldSplat 2509.23402](https://arxiv.org/abs/2509.23402) |

### Recommendation
**Best maintained + reproducible: [Shape-of-Motion](https://github.com/vye16/shape-of-motion)** (ICCV 2025 Highlight). Strongest combo of (a) actively maintained, (b) **monocular** — only a phone video, no multi-camera rig, (c) produces *explicit persistent 3D motion trajectories*, exactly the "temporal consistency" story a portfolio wants (render the motion bases / point tracks, not just novel views). Catch: a preprocessing stack (monocular depth, masks, 2D point tracks) to stand up first — budget more time on data prep than training.

- **Realistic for a solo dev (single 12–24 GB GPU):** Shape-of-Motion (monocular) + Deformable-3DGS / 4DGS-Wu on **synthetic D-NeRF** scenes (clean, fast, reproducible deformation-field demo). Start synthetic to get a pipeline, then a real monocular clip with Shape-of-Motion.
- **Research-grade-hard:** Spacetime Gaussians, Dynamic-3DGS — need multi-view rigs and/or 48 GB GPUs. Not a good solo fit.
- **Not solo-friendly yet:** feed-forward / generative 4D — where the field is heading in 2025–26, but strong open releases are domain-locked (autonomous driving) or lack stable general-purpose code. Mention as "future direction."

**Pragmatic plan:** prototype on D-NeRF synthetic with 4DGS/Deformable-3DGS (1–2 days to a render), then the headline demo with Shape-of-Motion on one real monocular clip, citing **MonoDyGauBench (TMLR 2025)** as the honesty check — its "fast but brittle" finding lets you frame results credibly rather than overclaiming.

---

## 5. Datasets & Evaluation

### Static novel-view datasets

| Dataset | For | Scale / contents | Download |
|---|---|---|---|
| **Mip-NeRF360** | The de-facto NVS benchmark; unbounded 360° real scenes | 9 scenes (bicycle, garden, kitchen, room, …), ~100–300 imgs/scene | [jonbarron.info/mipnerf360](https://jonbarron.info/mipnerf360/) |
| **Tanks & Temples** | Real-world recon/NVS; 3DGS reports **Truck**, **Train** | 21 scenes + ground-truth laser scans | [tanksandtemples.org/download](https://www.tanksandtemples.org/download/) |
| **Deep Blending** | Real indoor; 3DGS reports **Dr Johnson**, **Playroom** | SIGGRAPH-Asia 2018 IBR capture | [project page](http://visual.cs.ucl.ac.uk/pubs/deepblending/datasets.html) |
| **NeRF Synthetic ("Blender")** | Clean synthetic object sanity test | 8 objects (lego, chair, …), 100 train / 200 test | [NeRF page](https://www.matthewtancik.com/nerf) |
| **ScanNet** | Classic indoor RGB-D **semantic + instance** seg | 1,513 scans, 2.5M frames, 20-class | [scan-net.org](http://www.scan-net.org/) |
| **ScanNet++ (v2)** | High-fidelity successor: joint **NVS + long-tail semantic** | 1,000+ scenes, sub-mm laser + 33MP DSLR + iPhone, 1,000+ classes | [scannetpp.mlsg.cit.tum.de](https://scannetpp.mlsg.cit.tum.de/scannetpp/) · [toolkit](https://github.com/scannetpp/scannetpp) |
| **Replica** | Photorealistic synthetic indoor (SLAM / NVS / semantic) | 18 scenes, dense mesh + semantic/instance | [facebookresearch/Replica-Dataset](https://github.com/facebookresearch/Replica-Dataset) |

### Dynamic (4D) datasets

| Dataset | For | Scale | Source |
|---|---|---|---|
| **DyCheck (iPhone)** | Honest **monocular** dynamic — held-out *true* novel-view validation (exposes overfitting) | 7 casual iPhone clips + synced validation cams | [hangg7.com/dycheck](https://hangg7.com/dycheck/) |
| **Neural 3D Video / Neu3D** | Standard **multi-view** dynamic NVS | 6 cooking scenes, 18–21 cams, 300 frames | [facebookresearch/Neural_3D_Video](https://github.com/facebookresearch/Neural_3D_Video) |
| **HyperNeRF** | Topology-varying deformable scenes | Real 1–2 moving phone cams | [hypernerf.github.io](https://hypernerf.github.io/) |
| **D-NeRF** | Synthetic **monocular** deformable sanity check | 8 synthetic animations, 50–200 frames | [albertpumarola/D-NeRF](https://github.com/albertpumarola/D-NeRF) |
| **Panoptic Sports** | Dynamic NVS **+ dense long-term 3D tracking** | 6 CMU-Panoptic sports clips, 31 cams | [dynamic3dgaussians.github.io](https://dynamic3dgaussians.github.io/) |

### Metrics

**Novel-view photometric:** **PSNR** ↑ (pixel MSE, dB; easy to game, saturates on synthetic) · **SSIM** ↑ (structural, 0–1) · **LPIPS** ↓ (learned perceptual — the metric reviewers weight most for sharpness).

**Semantic 3D:** **mIoU** ↑ (headline 3D seg number) · **PQ = SQ × RQ** ↑ (panoptic quality; mAP/mAP@50 for instance).

**Temporal / 4D** (on top of per-frame metrics): **tOF** ↓ (optical-flow MAE vs GT sequence) · **Temporal LPIPS / tLP** ↓ (perceptual frame-to-frame stability) · **Warping error** ↓ (flow-warp frame *t*→*t+1*, residual vs actual). Caveat: per-frame PSNR/SSIM/LPIPS reward "sharp but jittery" — temporal metrics catch the flicker. **Report both for 4D.**

### What a credible portfolio result reports

**Static NVS — vanilla 3DGS baseline (30k iters) to reproduce/beat:**

| Dataset | PSNR | SSIM | LPIPS | Meaning |
|---|---|---|---|---|
| **Mip-NeRF360** (9-scene avg) | **~27.2** | **~0.815** | **~0.21** | Reproducing this with your own pipeline is the credibility bar. Tuned 2025–26 variants reach ~27.5–28.0 / ~0.83 / ~0.18. |
| **Tanks & Temples** (Truck, Train) | ~23.1–23.7 | ~0.84–0.85 | ~0.18 | — |
| **Deep Blending** | ~29.4–29.6 | ~0.90 | ~0.24 | — |
| **NeRF Synthetic** | ~33+ | ~0.97 | ~0.03 | Near-saturated — sanity check only. |

Also report **training time** (3DGS: tens of minutes on one consumer GPU) and **render FPS** (100+ FPS) — part of the 3DGS story.

**Dynamic / 4D:** N3DV ~31–32 dB PSNR with a 4DGS-class method (+ ≥1 temporal metric); DyCheck iPhone ~16–17 on masked co-visible regions is *honest* (true held-out views matter more than the number); Panoptic Sports ~28+ PSNR with dense tracking is the Dynamic-3DGS reference (~28.7 PSNR, 850 FPS).

**Semantic 3D:** ScanNet v2 strong methods land **~70–79% mIoU**; the "working 3D seg pipeline" bar is ~70% mIoU. **ScanNet++** is the more impressive frontier — its long-tail 1,000+ classes make modest mIoU credible, and it doubles as an NVS benchmark.

**Bottom line:** a self-trained **3DGS at ~27 PSNR / ~0.81 SSIM / ~0.21 LPIPS on Mip-NeRF360** (with train-time + FPS), **plus either** a 4D result on N3DV/DyCheck with a temporal metric **or** a ~70% mIoU semantic result on ScanNet/ScanNet++, is a clear 2026 signal of competence.

---

## 6. Key Papers (prioritized, must-read first)

### Tier 1 — Landmark foundations
1. **3D Gaussian Splatting for Real-Time Radiance Field Rendering** — Kerbl et al., 2023 (SIGGRAPH). *The paper that started the field: explicit Gaussians + differentiable rasterizer give NeRF-quality views at >100 FPS.* [2308.04079](https://arxiv.org/abs/2308.04079)
2. **NeRF: Representing Scenes as Neural Radiance Fields** — Mildenhall et al., 2020 (ECCV). *The conceptual ancestor — coordinate-MLP volumetric radiance fields.* [2003.08934](https://arxiv.org/abs/2003.08934)
3. **Instant-NGP: Multiresolution Hash Encoding** — Müller et al., 2022 (SIGGRAPH). *Cut NeRF training from hours to seconds; influenced all acceleration work.* [2201.05989](https://arxiv.org/abs/2201.05989)
4. **Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields** — Barron et al., 2022 (CVPR). *Solved unbounded 360° scenes; its dataset is the standard eval for NeRF and 3DGS.* [2111.12077](https://arxiv.org/abs/2111.12077)

### Tier 2 — Key 3DGS improvements
5. **Mip-Splatting: Alias-Free 3DGS** — Yu et al., 2024 (CVPR). *3D + 2D Mip filters kill aliasing/zoom artifacts — the default multi-scale fix.* [2311.16493](https://arxiv.org/abs/2311.16493)
6. **2D Gaussian Splatting for Geometrically Accurate Radiance Fields** — Huang et al., 2024 (SIGGRAPH). *2D surfels → accurate surface/mesh reconstruction.* [2403.17888](https://arxiv.org/abs/2403.17888)
7. **Scaffold-GS: Structured 3D Gaussians** — Lu et al., 2024 (CVPR). *Anchor-based view-adaptive Gaussians cut redundancy/storage; basis for compact/large-scale systems.* [2312.00109](https://arxiv.org/abs/2312.00109)

### Tier 3 — Landmark semantic 3D
8. **LangSplat: 3D Language Gaussian Splatting** — Qin et al., 2024 (CVPR). *Distills CLIP into a SAM-structured 3D language field; reference for language-grounded 3DGS.* [2312.16084](https://arxiv.org/abs/2312.16084)
9. **Gaussian Grouping: Segment and Edit Anything in 3D** — Ye et al., 2024 (ECCV). *Learnable identity codes lift SAM masks into 3D instances + editing.* [2312.00732](https://arxiv.org/abs/2312.00732)
10. **Feature 3DGS: Distilled Feature Fields** — Zhou et al., 2024 (CVPR). *Generic recipe to distill any 2D foundation feature (SAM/CLIP/LSeg) into 3DGS.* [2312.03203](https://arxiv.org/abs/2312.03203)
11. **OpenGaussian: Point-Level Open-Vocabulary Understanding** — Wu et al., 2024 (NeurIPS). *Pushes open-vocab to crisp point-level 3D consistency.* [2406.02058](https://arxiv.org/abs/2406.02058)
12. **Panoptic Lifting** — Siddiqui et al., 2023 (CVPR). *The landmark "lift 2D masks into a consistent 3D panoptic field" formulation; foundational pre-3DGS.* [2212.09802](https://arxiv.org/abs/2212.09802)

### Tier 4 — Landmark 4D / dynamic
13. **Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis** — Luiten et al., 2024 (3DV). *First 3DGS→dynamic; unifies 6-DOF dense tracking + NVS.* [2308.09713](https://arxiv.org/abs/2308.09713)
14. **Deformable 3D Gaussians** — Yang et al., 2024 (CVPR). *Canonical Gaussians + time-conditioned deformation MLP — the dominant monocular-dynamic pattern.* [2309.13101](https://arxiv.org/abs/2309.13101)
15. **4D Gaussian Splatting for Real-Time Dynamic Scene Rendering** — Wu et al., 2024 (CVPR). *HexPlane spatio-temporal encoder → real-time compact dynamic; most-cited 4DGS baseline.* [2310.08528](https://arxiv.org/abs/2310.08528)
16. **Shape of Motion: 4D Reconstruction from a Single Video** — Wang et al., 2025 (ICCV Highlight). *Strongest 2025 casual-monocular 4D — SE(3) motion bases + depth/track priors → persistent world-frame trajectories.* [2407.13764](https://arxiv.org/abs/2407.13764)

### Tier 5 — 2025-26 breakthroughs: feed-forward 3D & Gaussian world models (robotics-relevant)
17. **VGGT: Visual Geometry Grounded Transformer** — Wang et al., 2025 (CVPR Best Paper). *Single feed-forward transformer infers cameras/depth/points/tracks in <1s; reshaping feed-forward 3D.* [2503.11651](https://arxiv.org/abs/2503.11651)
18. **Long-LRM: Long-Sequence Large Reconstruction Model** — Ziwen et al., 2024. *Feed-forward wide-coverage scenes from 32 images → Gaussians in ~1s (~800× speedup).* [2410.12781](https://arxiv.org/abs/2410.12781)
19. **GWM: Scalable Gaussian World Models for Robotic Manipulation** — Lu et al., 2025 (ICCV). *3DGS world model (DiT + 3D-aware VAE) predicts future scene states under robot actions — directly bridges 3DGS and action-conditioned world models.* [2508.17600](https://arxiv.org/abs/2508.17600)

> **Recent extensions to cite as "current work" (fresher, less established):** LangSplatV2 ([2507.07136](https://arxiv.org/abs/2507.07136), 450+ FPS language 3DGS), SemanticSplat ([2506.09565](https://arxiv.org/abs/2506.09565), feed-forward semantic Gaussian fields), GAF: Gaussian Action Field ([2506.14135](https://arxiv.org/abs/2506.14135), 4D action-conditioned manipulation). For a Reka-style world-models pitch, **VGGT + GWM + GAF** are the most on-target.

---

## 7. Concrete Minimal First Milestone

**Goal:** smallest end-to-end version that proves the differentiator — 2D segmentation expertise lifted into a 3D Gaussian field — with credible, reported numbers.

### Milestone 1 (the must-ship core): static semantic 3DGS + eval harness

| Step | What | Tooling | Output / check |
|---|---|---|---|
| 1 | Reconstruct **one Mip-NeRF360 scene** (e.g. `garden` or `room`) with 3DGS | Nerfstudio `ns-train splatfacto` (gsplat backend) | PSNR/SSIM/LPIPS vs the ~27 PSNR baseline; render FPS; train time. **This alone proves you can run the substrate.** |
| 2 | Generate **2D masks** on the training views | **SAM2** (automatic + tracked), optionally Grounded-SAM for text-prompted classes | Per-view instance masks, view-consistent via SAM2 tracking |
| 3 | **Lift masks into the Gaussian field** | **OpenSplat3D** (primary) or **SAGA** / **LangSplat** (fallback) | Per-Gaussian identity/feature field; render a segmentation map from a held-out view |
| 4 | **Eval harness** (the rigor signal — your strength) | Custom: render seg from novel views, compare to GT | **mIoU** (+ PQ if panoptic) on held-out views; PSNR/SSIM/LPIPS for NVS; one combined table |
| 5 | Package | Nerfstudio web viewer / short render | Interactive 3D + click-to-select-object demo; README with the numbers table |

**Why this scopes well:** Steps 1 and 4 are pure 3DGS + eval-pipeline rigor (her existing strengths, just in a new modality). Steps 2–3 are the genuinely new neural-rendering skill, but bounded to "lift masks I already know how to produce." A reproduced ~27 PSNR baseline + a working mIoU-on-held-out-views eval is a stronger competence signal than chasing SOTA.

**Compute/time budget (single A100 or 24 GB consumer GPU):**
- 3DGS per scene: ~20–40 min train, real-time render.
- SAM2 mask generation: minutes per scene (inference only).
- OpenSplat3D lifting: ~1–3 hours per scene incl. preproc (COLMAP/glomap is the slow part).
- Eval harness build: ~1–2 days of engineering (the reusable asset).
- **Total: ~1 GPU-week including iteration and the eval harness.**

### Milestone 2 (the 4D extension — temporal consistency)
After M1 ships: take **one D-NeRF synthetic clip** through **Deformable-3DGS / 4DGS-Wu** (1–2 days to a render, minimal data wrangling), then **one real monocular phone clip** through **Shape-of-Motion**. Report per-frame PSNR/SSIM/LPIPS **+ a temporal metric (tOF or tLPIPS)**, and visualize the persistent motion trajectories. Cite **MonoDyGauBench** to frame results honestly. Budget ~1–2 GPU-weeks (Shape-of-Motion preprocessing — depth, masks, 2D tracks — dominates).

### Honest difficulty assessment
- **Lowest risk / highest signal-per-effort:** Milestone 1 static path. The 3DGS reconstruction + eval is well-trodden; the only real unknown is OpenSplat3D's preproc env (CUDA/COLMAP). Pin CUDA 11.8.
- **Medium:** semantic lifting reproducibility — have SAGA/LangSplat ready as fallbacks if OpenSplat3D's env fights you.
- **Higher:** the 4D extension — Shape-of-Motion's preprocessing stack is the time sink, not training. Start on D-NeRF synthetic to de-risk before touching real video.
- **Out of scope (mention as future direction):** feed-forward / generative reconstruction (VGGT) and Gaussian world models (GWM) — these are the most Reka-aligned topics intellectually, but building on them is phase-3, not a first milestone.

---

### Master source list
- Representation/repos: [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) · [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) · [nerfstudio-project/nerfstudio](https://github.com/nerfstudio-project/nerfstudio) · [ArthurBrussee/brush](https://github.com/ArthurBrussee/brush) · [State of Gaussian Splatting 2026](https://www.thefuture3d.com/blog/state-of-gaussian-splatting-2026/)
- Semantic: [OpenSplat3D](https://github.com/VisualComputingInstitute/opensplat3d) · [SAGA](https://github.com/Jumpat/SegAnyGAussians) · [LangSplat](https://github.com/minghanqin/LangSplat) · [Gaga](https://github.com/weijielyu/Gaga)
- 4D: [shape-of-motion](https://github.com/vye16/shape-of-motion) · [4DGaussians](https://github.com/hustvl/4DGaussians) · [Deformable-3D-Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians) · [MonoDyGauBench](https://github.com/lynl7130/MonoDyGauBench_code)
- Datasets: [Mip-NeRF360](https://jonbarron.info/mipnerf360/) · [ScanNet++](https://scannetpp.mlsg.cit.tum.de/scannetpp/) · [DyCheck](https://hangg7.com/dycheck/) · [Neural 3D Video](https://github.com/facebookresearch/Neural_3D_Video)

# fastforks

Research forks that make open-source vision models faster and lighter, **without changing their results**.
Everything is measured on one consumer GPU (RTX 5060 Ti, 16 GB) and reported as speed **and** memory,
plus startup, first-call and new-input-size cost.

Branches: `main`/`master` = untouched upstream mirror; `portable` = plain PyTorch only (exports to ONNX / TensorRT, no build step); `cuda-kernels` = on top of `portable`, custom CUDA kernels for maximum speed.
The baseline for every speed-up is **upstream + simple flags** (driver-side settings only, such as fp16/bf16 autocast, `channels_last`, `cudnn.benchmark`, TF32); upstream without flags is shown separately.
Bundle repos such as `fast-vision-models` have no tier branches: the tier is a function argument.

## Overview

Speed-up and memory of the `cuda-kernels` tier against upstream + flags, at each repo's main setting (stereo: Middlebury-H 1482x994; "(portable)" = repo has no `cuda-kernels` tier, value is for `portable`; "vs upstream" = repo has no flags row; "-" = tier does not exist; memory = peak allocated, `cuda-kernels` / flags). Full table (33 columns: every tier in ms, speed-ups vs flags and vs upstream, peak MB alloc / reserved, startup / first call / new size, quality, licences, branches, commit SHAs, date, GPU; all 63 fast-vision-models models as individual rows): **[OVERVIEW.md](https://github.com/fastforks/.github/blob/main/OVERVIEW.md)**.

| model | task | repo | licence | cuda-kernels vs flags | memory (cuda-kernels / flags) |
|---|---|---|---|---|---|
| AANet | stereo matching | [aanet](https://github.com/fastforks/aanet) | Apache-2.0; commercial: yes | 1.39 | 0.56 |
| AANet+ | stereo matching | [aanet](https://github.com/fastforks/aanet) | Apache-2.0; commercial: yes | 1.46 | 0.63 |
| BridgeDepth (rvc_pretrain) | stereo matching | [BridgeDepth](https://github.com/fastforks/BridgeDepth) | MIT; commercial: yes | 1.01 | 1.63 |
| Cas-GwcNet-c | stereo matching | [cascade-stereo](https://github.com/fastforks/cascade-stereo) | code MIT, weights not stated; commercial: yes | 1.55 | 0.71 |
| Cas-GwcNet-gc | stereo matching | [cascade-stereo](https://github.com/fastforks/cascade-stereo) | code MIT, weights not stated; commercial: yes | 2.18 | 0.23 |
| CFNet | stereo matching | [CFNet](https://github.com/fastforks/CFNet) | MIT; commercial: yes | 1.60 | 0.26 |
| CoEx | stereo matching | [coex](https://github.com/fastforks/coex) | GPL-3.0 (copyleft); commercial: custom | 1.24 | 0.43 |
| CREStereo (ETH3D ckpt, 20 it) | stereo matching | [CREStereo-Pytorch](https://github.com/fastforks/CREStereo-Pytorch) | none (upstream has no licence); commercial: no | 5.81 | 0.39 |
| CroCo-Stereo | stereo matching | [croco](https://github.com/fastforks/croco) | CC BY-NC-SA 4.0; commercial: no | 1.43 (portable) | - |
| DEFOM-Stereo | stereo matching | [DEFOM-Stereo](https://github.com/fastforks/DEFOM-Stereo) | MIT; commercial: yes | 5.67 | 0.23 |
| Fast-ACVNet | stereo matching | [Fast-ACVNet](https://github.com/fastforks/Fast-ACVNet) | MIT; commercial: yes | 1.31 | 0.26 |
| Fast-FoundationStereo (23-36-37, 8 it) | stereo matching | [Fast-FoundationStereo](https://github.com/fastforks/Fast-FoundationStereo) | NVIDIA licence (research only); commercial: no | 2.21 | 0.69 |
| FoundationStereo (ViT-L, 32 it) | stereo matching | [FoundationStereo](https://github.com/fastforks/FoundationStereo) | NVIDIA licence (research only); commercial: no | 2.44 | 0.37 |
| GwcNet | stereo matching | [GwcNet](https://github.com/fastforks/GwcNet) | MIT; commercial: yes | 1.20 | 0.30 |
| DLNR | stereo matching | [High-frequency-Stereo-Matching-Network](https://github.com/fastforks/High-frequency-Stereo-Matching-Network) | MIT; commercial: yes | 3.44 | 0.23 |
| IGEV-Stereo (sceneflow, 32 it) | stereo matching | [IGEV](https://github.com/fastforks/IGEV) | code MIT, weights not stated; commercial: yes | 3.22 | 0.69 |
| LAS2-H | stereo matching | [LiteAnyStereo](https://github.com/fastforks/LiteAnyStereo) | MIT; commercial: yes | 1.24 | 0.98 |
| LAS2-M | stereo matching | [LiteAnyStereo](https://github.com/fastforks/LiteAnyStereo) | MIT; commercial: yes | 1.17 | 0.72 |
| MC-Stereo (Scene Flow ckpt, 32 it) | stereo matching | [MC-Stereo](https://github.com/fastforks/MC-Stereo) | none (all rights reserved); commercial: no | - | - |
| MoCha-Stereo (V1, 32 it) | stereo matching | [MoCha-Stereo](https://github.com/fastforks/MoCha-Stereo) | code MIT, weights not re-hosted; commercial: yes | 2.86 | 0.48 |
| MonSter | stereo matching | [MonSter](https://github.com/fastforks/MonSter) | MonSter Model License v1.0 (research-oriented); commercial: custom | 4.81 | 0.22 |
| MonSter++ | stereo matching | [MonSter-plusplus](https://github.com/fastforks/MonSter-plusplus) | MonSter-plusplus Model License v1.0 (research-oriented); commercial: custom | 4.72 | 0.23 |
| RT-MonSter++ | stereo matching | [MonSter-plusplus](https://github.com/fastforks/MonSter-plusplus) | MonSter-plusplus Model License v1.0 (research-oriented); commercial: custom | 1.47 | 0.42 |
| NMRF-Stereo | stereo matching | [NMRF](https://github.com/fastforks/NMRF) | MIT; commercial: yes | 2.03 | 0.09 |
| PromptStereo | stereo matching | [PromptStereo](https://github.com/fastforks/PromptStereo) | MIT; commercial: yes | 2.01 | 0.24 |
| PSMNet | stereo matching | [PSMNet](https://github.com/fastforks/PSMNet) | MIT; commercial: yes | 2.30 | 0.35 |
| RAFT-Stereo (Middlebury ckpt, 32 it) | stereo matching | [RAFT-Stereo](https://github.com/fastforks/RAFT-Stereo) | code MIT, weights not stated; commercial: yes | 3.72 | 0.28 |
| S2M2-S | stereo matching | [s2m2](https://github.com/fastforks/s2m2) | CC BY-NC 4.0 (code and weights); commercial: no | 2.36 | 0.62 |
| S2M2-XL | stereo matching | [s2m2](https://github.com/fastforks/s2m2) | CC BY-NC 4.0 (code and weights); commercial: no | 2.13 | 1.13 |
| Selective-Stereo | stereo matching | [Selective-Stereo](https://github.com/fastforks/Selective-Stereo) | MIT; commercial: yes | 6.21 | 0.10 |
| StereoAnywhere | stereo matching | [stereoanywhere](https://github.com/fastforks/stereoanywhere) | CC BY-NC-SA 4.0 (+ Univ. Bologna terms); commercial: no | 5.24 | 0.31 |
| UniMatch (stereo) | stereo matching (also flow/depth) | [unimatch](https://github.com/fastforks/unimatch) | MIT; commercial: yes | 1.53 (portable) | - |
| WAFT-Stereo B | stereo matching | [WAFT-Stereo](https://github.com/fastforks/WAFT-Stereo) | BSD-3-Clause; commercial: yes | 3.98 | 0.17 |
| WAFT-Stereo S | stereo matching | [WAFT-Stereo](https://github.com/fastforks/WAFT-Stereo) | BSD-3-Clause; commercial: yes | 3.91 | 0.10 |
| MapAnything | feed-forward multi-view reconstruction | [map-anything](https://github.com/fastforks/map-anything) | Apache-2.0; commercial: yes | 3.21 | 0.31 |
| MVSAnywhere | multi-view depth | [mvsanywhere](https://github.com/fastforks/mvsanywhere) | Niantic non-commercial research licence; commercial: no | 1.00 (portable) | - |
| MVSFormer++ | multi-view stereo | [MVSFormerPlusPlus](https://github.com/fastforks/MVSFormerPlusPlus) | Apache-2.0; commercial: yes | 1.69 | 0.27 |
| PatchmatchNet | multi-view stereo | [PatchmatchNet](https://github.com/fastforks/PatchmatchNet) | MIT; commercial: yes | 7.44 | 0.41 |
| VGGT | feed-forward multi-view reconstruction | [vggt](https://github.com/fastforks/vggt) | VGGT licence (Meta; no non-commercial clause, AUP applies); commercial: custom | 2.08 | 0.28 |
| VGGT-Omega | feed-forward multi-view reconstruction | [vggt-omega](https://github.com/fastforks/vggt-omega) | FAIR Noncommercial Research License v1; commercial: no | 1.96 | 0.25 |
| Depth-Anything-3 (DA3MONO-LARGE) | monocular / any-view depth | [Depth-Anything-3](https://github.com/fastforks/Depth-Anything-3) | code Apache-2.0; weights per checkpoint (tested DA3MONO-LARGE: Apache-2.0); commercial: yes | 2.43 | 0.47 |
| Depth-Anything-V2 ViT-L | monocular depth | [Depth-Anything-V2](https://github.com/fastforks/Depth-Anything-V2) | code Apache-2.0; ViT-L weights CC BY-NC 4.0 (ViT-S Apache-2.0); commercial: no | 3.57 | 0.79 |
| Depth-Anything-V2 ViT-S | monocular depth | [Depth-Anything-V2](https://github.com/fastforks/Depth-Anything-V2) | code Apache-2.0; ViT-S weights Apache-2.0; commercial: yes | 5.86 | 0.79 |
| Metric3D v2 ViT-L | metric monocular depth | [Metric3D](https://github.com/fastforks/Metric3D) | BSD-2-Clause (weights: see upstream); commercial: yes | 2.18 (portable) | - |
| MiDaS dpt_beit_large_512 | monocular depth | [MiDaS](https://github.com/fastforks/MiDaS) | MIT; commercial: yes | 3.17 (portable) | - |
| Depth Pro | metric monocular depth | [ml-depth-pro](https://github.com/fastforks/ml-depth-pro) | Apple licence (see LICENSE); commercial: custom | 1.87 | 0.95 |
| MoGe-2 ViT-L normal | monocular geometry | [MoGe](https://github.com/fastforks/MoGe) | MIT; commercial: yes | 1.76 | 1.08 |
| UniDepth V2 ViT-L | metric monocular depth | [UniDepth](https://github.com/fastforks/UniDepth) | CC BY-NC 4.0; commercial: no | 2.10 | 0.45 |
| UniK3D ViT-L | monocular 3D | [UniK3D](https://github.com/fastforks/UniK3D) | CC BY-NC-SA 4.0; commercial: no | 2.14 | 0.44 |
| Detectron2 (Mask R-CNN) | detection / instance segmentation | [detectron2](https://github.com/fastforks/detectron2) | Apache-2.0; commercial: yes | 1.58 | 0.68 |
| SAM 2 Hiera-L | promptable segmentation (image) | [sam2](https://github.com/fastforks/sam2) | Apache-2.0 (checkpoints: upstream terms); commercial: yes | 1.51 | 0.70 |
| SAM 3 | promptable concept segmentation | [sam3](https://github.com/fastforks/sam3) | SAM License (Meta, gated); commercial: custom | 1.90 | 0.58 |
| Sapiens-1B (seg) | human-centric vision (seg/pose/depth) | [sapiens](https://github.com/fastforks/sapiens) | Sapiens licence (CC BY-NC 4.0); commercial: no | 1.87 | 0.56 |
| SAM ViT-B | promptable segmentation (image) | [segment-anything](https://github.com/fastforks/segment-anything) | Apache-2.0 (checkpoints: upstream terms); commercial: yes | 2.44 | 0.20 |
| SAM ViT-H | promptable segmentation (image) | [segment-anything](https://github.com/fastforks/segment-anything) | Apache-2.0 (checkpoints: upstream terms); commercial: yes | 1.80 | 0.47 |
| DINOv2 ViT-L | backbone kernels library | [fastbb](https://github.com/fastforks/fastbb) | Apache-2.0 (+ CUTLASS BSD-3, DINOv2 Apache-2.0); commercial: yes | 1.54 vs upstream (no flags tier) | - |
| ResNet-50 | backbone kernels library | [fastbb](https://github.com/fastforks/fastbb) | Apache-2.0 (+ CUTLASS BSD-3, DINOv2 Apache-2.0); commercial: yes | 2.16 vs upstream (no flags tier) | - |
| ViT-B/16 (timm) | backbone kernels library | [fastbb](https://github.com/fastforks/fastbb) | Apache-2.0 (+ CUTLASS BSD-3, DINOv2 Apache-2.0); commercial: yes | 1.50 vs upstream (no flags tier) | - |
| CoTracker | point tracking | [co-tracker](https://github.com/fastforks/co-tracker) | CC BY-NC 4.0; commercial: no | 1.86 | 0.43 |
| fused-ssim | fused L1+SSIM loss kernel | [fused-ssim](https://github.com/fastforks/fused-ssim) | MIT; commercial: yes | 2.21 vs upstream (no flags tier) | - |
| gsplat | 3D Gaussian splatting training | [gsplat](https://github.com/fastforks/gsplat) | Apache-2.0; commercial: yes | 3.24 vs upstream (no flags tier) | - |
| 27 classification models | classification | [fast-vision-models](https://github.com/fastforks/fast-vision-models); [every model in OVERVIEW.md](https://github.com/fastforks/.github/blob/main/OVERVIEW.md) | per-model weights, see repo; code Apache-2.0 | median 2.10 (range 1.12 to 3.41) | median 0.51 |
| 17 segmentation models | segmentation | [fast-vision-models](https://github.com/fastforks/fast-vision-models); [every model in OVERVIEW.md](https://github.com/fastforks/.github/blob/main/OVERVIEW.md) | per-model weights, see repo; code Apache-2.0 | median 1.75 (range 1.39 to 2.60) | median 0.53 |
| 19 backbones | backbones | [fast-vision-models](https://github.com/fastforks/fast-vision-models); [every model in OVERVIEW.md](https://github.com/fastforks/.github/blob/main/OVERVIEW.md) | per-model weights, see repo; code Apache-2.0 | median 1.32 (range 0.34 to 2.15) | median 1.09 |

## Notes

- These are research forks. Upstream licences apply to the upstream code; our own additions in new repositories are Apache-2.0.
- Much of the engineering was done with AI assistance (Claude); all results are measured, not estimated.
- Kernels are built for sm_80 - sm_120 GPUs (Ampere to Blackwell); Jetson / Thor builds compile but are untested on hardware.

Maintained by Florian Nick.

# fastforks

Research forks that make open-source vision models faster and lighter, **without changing their results**.
Everything is measured on one consumer GPU (RTX 5060 Ti, 16 GB) and reported as speed **and** memory,
plus startup, first-call and new-input-size cost.

Branches: `main`/`master` = untouched upstream mirror; `portable` = plain PyTorch only (exports to ONNX / TensorRT, no build step); `cuda-kernels` = on top of `portable`, custom CUDA kernels for maximum speed.
The baseline for every speed-up is **upstream + simple flags** (driver-side settings only, such as fp16/bf16 autocast, `channels_last`, `cudnn.benchmark`, TF32); upstream without flags is shown separately.
Bundle repos such as `fast-vision-models` have no tier branches: the tier is a function argument.

Every model below, at its main setting (stereo: Middlebury-H 1482x994; fast-vision-models: the largest setting). Speed-ups are relative to upstream + flags; memory = peak allocated, cuda-kernels / flags; "-" = tier does not exist; "n/m" = not measured; "x (vs upstream)" = repo has no flags row. Each section links to its full table in [OVERVIEW.md](https://github.com/fastforks/.github/blob/main/OVERVIEW.md) (all tiers in ms, peak MB alloc / reserved, startup / first call / new size, quality, branches, commit SHAs, all settings).


## Contents

- [Stereo matching](#stereo-matching) (34 models)
- [Multi-view stereo and feed-forward reconstruction](#multi-view-stereo-and-feed-forward-reconstruction) (6 models)
- [Monocular depth and geometry](#monocular-depth-and-geometry) (9 models)
- [Promptable and instance segmentation](#promptable-and-instance-segmentation) (6 models)
- [Semantic segmentation (fast-vision-models)](#semantic-segmentation-fast-vision-models) (17 models)
- [Classification (fast-vision-models)](#classification-fast-vision-models) (27 models)
- [Backbones](#backbones) (3 models)
- [Backbones (fast-vision-models)](#backbones-fast-vision-models) (19 models)
- [Point tracking](#point-tracking) (1 models)
- [Gaussian splatting libraries](#gaussian-splatting-libraries) (2 models)

## Stereo matching

Full table: [Stereo matching in OVERVIEW.md](https://github.com/fastforks/.github/blob/main/OVERVIEW.md#stereo-matching).

| model | repo | licence (commercial) | portable vs flags | cuda-kernels vs flags | memory (cuda-kernels / flags) | quality, metric named in cell (flags -> cuda-kernels) |
|---|---|---|---|---|---|---|
| AANet | [aanet](https://github.com/fastforks/aanet) | Apache-2.0 (yes) | - | 1.39 | 0.56 | ETH3D EPE: 0.545 -> 0.547; Midd-H EPE: 15.721 -> 15.724 |
| AANet+ | [aanet](https://github.com/fastforks/aanet) | Apache-2.0 (yes) | - | 1.46 | 0.63 | ETH3D EPE: 0.745 -> 0.755; Midd-H EPE: 15.790 -> 15.838 |
| BridgeDepth (rvc_pretrain) | [BridgeDepth](https://github.com/fastforks/BridgeDepth) | MIT (yes) | 0.99 | 1.01 | 1.63 | ETH3D EPE: 0.088 -> 0.088; Midd-H EPE: 1.238 -> 1.237 |
| Cas-GwcNet-c | [cascade-stereo](https://github.com/fastforks/cascade-stereo) | code MIT, weights not stated (yes) | 0.91 | 1.55 | 0.71 | ETH3D EPE: 0.512 -> 0.513; Midd-H EPE: 8.496 -> 8.572 |
| Cas-GwcNet-gc | [cascade-stereo](https://github.com/fastforks/cascade-stereo) | code MIT, weights not stated (yes) | 1.47 | 2.18 | 0.23 | ETH3D EPE: 0.820 -> 0.824; Midd-H EPE: 6.541 -> 6.541 |
| CFNet | [CFNet](https://github.com/fastforks/CFNet) | MIT (yes) | - | 1.60 | 0.26 | ETH3D EPE: 0.399 -> 0.397; Midd-H EPE: 15.959 -> 15.952 |
| CoEx | [coex](https://github.com/fastforks/coex) | GPL-3.0 (copyleft) (custom) | - | 1.24 | 0.43 | ETH3D EPE: 18.523 -> 18.497; Midd-H EPE: 17.603 -> 17.575 |
| CREStereo (ETH3D ckpt, 20 it) | [CREStereo-Pytorch](https://github.com/fastforks/CREStereo-Pytorch) | none (upstream has no licence) (no) | 1.51 | 5.81 | 0.39 | ETH3D EPE: 0.185 -> 0.185; Midd-H EPE: 0.755 -> 0.745 |
| CroCo-Stereo | [croco](https://github.com/fastforks/croco) | CC BY-NC-SA 4.0 (no) | 1.43 | - | - | ETH3D EPE: 0.026 -> 0.026 (portable); Midd-H EPE: 1.007 -> 1.010 (portable) |
| DEFOM-Stereo | [DEFOM-Stereo](https://github.com/fastforks/DEFOM-Stereo) | MIT (yes) | - | 5.67 | 0.23 | ETH3D EPE: 0.173 -> 0.172; Midd-H EPE: 1.462 -> 1.456 |
| DLNR | [High-frequency-Stereo-Matching-Network](https://github.com/fastforks/High-frequency-Stereo-Matching-Network) | MIT (yes) | - | 3.44 | 0.23 | ETH3D EPE: 2.624 -> 2.639; Midd-H EPE: 3.115 -> 3.119 |
| Fast-ACVNet | [Fast-ACVNet](https://github.com/fastforks/Fast-ACVNet) | MIT (yes) | - | 1.31 | 0.26 | ETH3D EPE: 0.926 -> 0.935; Midd-H EPE: 9.545 -> 9.547 |
| Fast-FoundationStereo (23-36-37, 8 it) | [Fast-FoundationStereo](https://github.com/fastforks/Fast-FoundationStereo) | NVIDIA licence (research only) (no) | 0.97 | 2.21 | 0.69 | ETH3D EPE: 0.324 -> 0.323; Midd-H EPE: 0.314 -> 0.315 |
| FoundationStereo (ViT-L, 32 it) | [FoundationStereo](https://github.com/fastforks/FoundationStereo) | NVIDIA licence (research only) (no) | 1.22 | 2.44 | 0.37 | ETH3D EPE: 0.251 -> 0.250; Midd-H EPE: 0.242 -> 0.241 |
| GwcNet | [GwcNet](https://github.com/fastforks/GwcNet) | MIT (yes) | - | 1.20 | 0.30 | ETH3D EPE: 1.579 -> 1.579; Midd-H EPE: 16.619 -> 16.623 |
| IGEV-Stereo (sceneflow, 32 it) | [IGEV](https://github.com/fastforks/IGEV) | code MIT, weights not stated (yes) | 1.22 | 3.22 | 0.69 | ETH3D EPE: 0.333 -> 0.333; Midd-H EPE: 2.635 -> 2.636 |
| LAS2-H | [LiteAnyStereo](https://github.com/fastforks/LiteAnyStereo) | MIT (yes) | - | 1.24 | 0.98 | ETH3D EPE: 0.154 -> 0.155; Midd-H EPE: 6.142 -> 6.146 |
| LAS2-M | [LiteAnyStereo](https://github.com/fastforks/LiteAnyStereo) | MIT (yes) | - | 1.17 | 0.72 | ETH3D EPE: 0.233 -> 0.233; Midd-H EPE: 6.409 -> 6.409 |
| MC-Stereo (Scene Flow ckpt, 32 it) | [MC-Stereo](https://github.com/fastforks/MC-Stereo) | none (all rights reserved) (no) | - | - | - | EPE vs GT (ETH3D -> Midd-H subset): 4.56 -> 1.14 -> - |
| MoCha-Stereo (V1, 32 it) | [MoCha-Stereo](https://github.com/fastforks/MoCha-Stereo) | code MIT, weights not re-hosted (yes) | 1.17 | 2.86 | 0.48 | EPE ETH3D -> Midd-H: 0.318 -> 0.938 vs 0.318 -> 0.939 |
| MonSter | [MonSter](https://github.com/fastforks/MonSter) | MonSter Model License v1.0 (research-oriented) (custom) | - | 4.81 | 0.22 | ETH3D EPE: 0.215 -> 0.220; Midd-H EPE: 5.255 -> 5.251 |
| MonSter++ | [MonSter-plusplus](https://github.com/fastforks/MonSter-plusplus) | MonSter-plusplus Model License v1.0 (research-oriented) (custom) | - | 4.72 | 0.23 | ETH3D EPE: 0.106 -> 0.106; Midd-H EPE: 0.934 -> 0.933 |
| NMRF-Stereo | [NMRF](https://github.com/fastforks/NMRF) | MIT (yes) | - | 2.03 | 0.09 | ETH3D EPE: 0.338 -> 0.337; Midd-H EPE: 6.961 -> 6.926 |
| PromptStereo | [PromptStereo](https://github.com/fastforks/PromptStereo) | MIT (yes) | - | 2.01 | 0.24 | ETH3D EPE: 0.116 -> 0.116; Midd-H EPE: 1.087 -> 1.079 |
| PSMNet | [PSMNet](https://github.com/fastforks/PSMNet) | MIT (yes) | - | 2.30 | 0.35 | ETH3D EPE: 1.255 -> 1.255; Midd-H EPE: 24.939 -> 24.940 |
| RAFT-Stereo (Middlebury ckpt, 32 it) | [RAFT-Stereo](https://github.com/fastforks/RAFT-Stereo) | code MIT, weights not stated (yes) | 1.37 | 3.72 | 0.28 | ETH3D EPE: 0.288 -> 0.274; Midd-H EPE: 1.069 -> 1.067 |
| RT-MonSter++ | [MonSter-plusplus](https://github.com/fastforks/MonSter-plusplus) | MonSter-plusplus Model License v1.0 (research-oriented) (custom) | - | 1.47 | 0.42 | ETH3D EPE: 0.146 -> 0.147; Midd-H EPE: 5.690 -> 5.685 |
| S2M2-S | [s2m2](https://github.com/fastforks/s2m2) | CC BY-NC 4.0 (code and weights) (no) | 0.75 | 2.36 | 0.62 | ETH3D EPE: 0.316 -> 0.193; Midd-H EPE: 0.833 -> 0.652 |
| S2M2-XL | [s2m2](https://github.com/fastforks/s2m2) | CC BY-NC 4.0 (code and weights) (no) | 0.80 | 2.13 | 1.13 | ETH3D EPE: 0.189 -> 0.149; Midd-H EPE: 0.553 -> 0.499 |
| Selective-Stereo | [Selective-Stereo](https://github.com/fastforks/Selective-Stereo) | MIT (yes) | - | 6.21 | 0.10 | ETH3D EPE: 0.403 -> 0.402; Midd-H EPE: 7.503 -> 7.449 |
| StereoAnywhere | [stereoanywhere](https://github.com/fastforks/stereoanywhere) | CC BY-NC-SA 4.0 (+ Univ. Bologna terms) (no) | - | 5.24 | 0.31 | ETH3D EPE: 0.143 -> 0.143; Midd-H EPE: n/m -> n/m |
| UniMatch (stereo) | [unimatch](https://github.com/fastforks/unimatch) | MIT (yes) | 1.53 | - | - | ETH3D EPE: 0.441 -> 0.441 (portable); Midd-H EPE: 2.786 -> 2.787 (portable) |
| WAFT-Stereo B | [WAFT-Stereo](https://github.com/fastforks/WAFT-Stereo) | BSD-3-Clause (yes) | - | 3.98 | 0.17 | ETH3D EPE: 0.107 -> 0.108; Midd-H EPE: 1.366 -> 1.365 |
| WAFT-Stereo S | [WAFT-Stereo](https://github.com/fastforks/WAFT-Stereo) | BSD-3-Clause (yes) | - | 3.91 | 0.10 | ETH3D EPE: 0.096 -> 0.096; Midd-H EPE: 1.744 -> 1.741 |

## Multi-view stereo and feed-forward reconstruction

Full table: [Multi-view stereo and feed-forward reconstruction in OVERVIEW.md](https://github.com/fastforks/.github/blob/main/OVERVIEW.md#multi-view-stereo-and-feed-forward-reconstruction).

| model | repo | licence (commercial) | portable vs flags | cuda-kernels vs flags | memory (cuda-kernels / flags) | quality, metric named in cell (flags -> cuda-kernels) |
|---|---|---|---|---|---|---|
| MapAnything | [map-anything](https://github.com/fastforks/map-anything) | Apache-2.0 (yes) | - | 3.21 | 0.31 | ETH3D metric absrel: 0.1176 -> 0.1176 |
| MVSAnywhere | [mvsanywhere](https://github.com/fastforks/mvsanywhere) | Niantic non-commercial research licence (no) | 1.00 | - | - | ETH3D AbsRel: 0.0110 -> 0.0111 (portable) |
| MVSFormer++ | [MVSFormerPlusPlus](https://github.com/fastforks/MVSFormerPlusPlus) | Apache-2.0 (yes) | - | 1.69 | 0.27 | n/m (random weights, accuracy not measured) |
| PatchmatchNet | [PatchmatchNet](https://github.com/fastforks/PatchmatchNet) | MIT (yes) | - | 7.44 | 0.41 | absrel: 0.1256 -> 0.1246 |
| VGGT | [vggt](https://github.com/fastforks/vggt) | VGGT licence (Meta; no non-commercial clause, AUP applies) (custom) | - | 2.08 | 0.28 | flags identical to upstream; cuda depth rel. diff median 0.40% (8 frames) |
| VGGT-Omega | [vggt-omega](https://github.com/fastforks/vggt-omega) | FAIR Noncommercial Research License v1 (no) | - | 1.96 | 0.25 | ETH3D absrel: 0.0062 -> 0.0062 |

## Monocular depth and geometry

Full table: [Monocular depth and geometry in OVERVIEW.md](https://github.com/fastforks/.github/blob/main/OVERVIEW.md#monocular-depth-and-geometry).

| model | repo | licence (commercial) | portable vs flags | cuda-kernels vs flags | memory (cuda-kernels / flags) | quality, metric named in cell (flags -> cuda-kernels) |
|---|---|---|---|---|---|---|
| Depth Pro | [ml-depth-pro](https://github.com/fastforks/ml-depth-pro) | Apple licence (see LICENSE) (custom) | 1.02 | 1.87 | 0.95 | Midd-H AbsRel raw: 0.1480 -> 0.1481 |
| Depth-Anything-3 (DA3MONO-LARGE) | [Depth-Anything-3](https://github.com/fastforks/Depth-Anything-3) | code Apache-2.0; weights per checkpoint (tested DA3MONO-LARGE: Apache-2.0) (yes) | 1.50 | 2.43 | 0.47 | NYU-20 AbsRel: 0.03394 -> 0.03397 |
| Depth-Anything-V2 ViT-L | [Depth-Anything-V2](https://github.com/fastforks/Depth-Anything-V2) | code Apache-2.0; ViT-L weights CC BY-NC 4.0 (ViT-S Apache-2.0) (no) | 2.35 | 3.57 | 0.79 | NYU AbsRel: 0.04388 -> 0.04388 |
| Depth-Anything-V2 ViT-S | [Depth-Anything-V2](https://github.com/fastforks/Depth-Anything-V2) | code Apache-2.0; ViT-S weights Apache-2.0 (yes) | 3.96 | 5.86 | 0.79 | NYU AbsRel: 0.05093 -> 0.05093 |
| Metric3D v2 ViT-L | [Metric3D](https://github.com/fastforks/Metric3D) | BSD-2-Clause (weights: see upstream) (yes) | 2.18 | - | - | Midd-H AbsRel raw: 0.7041 -> 0.7040 (portable) |
| MiDaS dpt_beit_large_512 | [MiDaS](https://github.com/fastforks/MiDaS) | MIT (yes) | 3.17 | - | - | Midd-H AbsRel (disparity): 0.1157 -> 0.1160 (portable) |
| MoGe-2 ViT-L normal | [MoGe](https://github.com/fastforks/MoGe) | MIT (yes) | 0.99 | 1.76 | 1.08 | NYU-20 AbsRel: 0.0321 -> 0.0321 |
| UniDepth V2 ViT-L | [UniDepth](https://github.com/fastforks/UniDepth) | CC BY-NC 4.0 (no) | 1.31 | 2.10 | 0.45 | Midd-H AbsRel raw: 0.1989 -> 0.1986 |
| UniK3D ViT-L | [UniK3D](https://github.com/fastforks/UniK3D) | CC BY-NC-SA 4.0 (no) | 1.33 | 2.14 | 0.44 | Midd-H AbsRel raw: 0.2020 -> 0.2018 |

## Promptable and instance segmentation

Full table: [Promptable and instance segmentation in OVERVIEW.md](https://github.com/fastforks/.github/blob/main/OVERVIEW.md#promptable-and-instance-segmentation).

| model | repo | licence (commercial) | portable vs flags | cuda-kernels vs flags | memory (cuda-kernels / flags) | quality, metric named in cell (flags -> cuda-kernels) |
|---|---|---|---|---|---|---|
| Detectron2 (Mask R-CNN) | [detectron2](https://github.com/fastforks/detectron2) | Apache-2.0 (yes) | - | 1.58 | 0.68 | box/mask AP: 45.38 -> 41.78 vs 45.18 -> 41.69 |
| SAM 2 Hiera-L | [sam2](https://github.com/fastforks/sam2) | Apache-2.0 (checkpoints: upstream terms) (yes) | 1.22 | 1.51 | 0.70 | IoU vs upstream (L): 1 -> 0.965 |
| SAM 3 | [sam3](https://github.com/fastforks/sam3) | SAM License (Meta, gated) (custom) | 1.21 | 1.90 | 0.58 | IoU vs GT: 0.7945 -> 0.7947 |
| SAM ViT-B | [segment-anything](https://github.com/fastforks/segment-anything) | Apache-2.0 (checkpoints: upstream terms) (yes) | 2.14 | 2.44 | 0.20 | mean mask IoU vs fp32: 0.9906 -> 0.9902 |
| SAM ViT-H | [segment-anything](https://github.com/fastforks/segment-anything) | Apache-2.0 (checkpoints: upstream terms) (yes) | 1.44 | 1.80 | 0.47 | mean mask IoU vs fp32: 0.9979 -> 0.9964 |
| Sapiens-1B (seg) | [sapiens](https://github.com/fastforks/sapiens) | Sapiens licence (CC BY-NC 4.0) (no) | - | 1.87 | 0.56 | seg mIoU vs upstream: 0.9981 -> 0.9932 |

## Semantic segmentation (fast-vision-models)

Full table: [Semantic segmentation (fast-vision-models) in OVERVIEW.md](https://github.com/fastforks/.github/blob/main/OVERVIEW.md#semantic-segmentation-fast-vision-models).

| model | repo | licence (commercial) | portable vs flags | cuda-kernels vs flags | memory (cuda-kernels / flags) | quality, pixel agreement % (flags -> cuda-kernels) |
|---|---|---|---|---|---|---|
| ddrnet-23 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (yes) | 1.02 | 1.79 | 0.55 | 99.9 -> 99.9 |
| deeplabv3-r101 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (yes) | 1.01 | 1.94 | 0.58 | 100.0 -> 100.0 |
| deeplabv3plus-r101 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (yes) | 1.17 | 2.09 | 0.46 | 100.0 -> 100.0 |
| dinov2-l-linear | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (yes) | 1.02 | 1.39 | 0.69 | 100.0 -> 99.9 |
| eomt-l-dinov2 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (custom) | 1.09 | 1.44 | 0.54 | 100.0 -> 100.0 |
| eomt-l-dinov3 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (custom) | 1.40 | 1.79 | 0.47 | 100.0 -> 100.0 |
| fcn-r101 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (yes) | 1.01 | 1.95 | 0.50 | 100.0 -> 100.0 |
| hrnet-w48 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (yes) | 1.02 | 1.62 | 0.27 | 100.0 -> 100.0 |
| mask2former-swinl | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (no) | 1.09 | 1.81 | 0.55 | 99.9 -> 99.9 |
| oneformer-swinl | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (yes) | 1.04 | 1.52 | 0.53 | 99.8 -> 99.9 |
| pidnet-l | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (yes) | 1.02 | 1.66 | 0.72 | 99.9 -> 99.9 |
| pspnet-r101 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (yes) | 1.01 | 1.65 | 0.61 | 100.0 -> 100.0 |
| segformer-b0 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (no) | 1.22 | 2.10 | 0.19 | 100.0 -> 100.0 |
| segformer-b5 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (no) | 1.25 | 1.46 | 0.09 | 100.0 -> 100.0 |
| segnet | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (n/a) | 1.19 | 1.63 | 0.47 | 97.2 -> 95.9 |
| segnext-t | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (yes) | 1.76 | 2.60 | 0.19 | 99.9 -> 99.9 |
| unet-s5 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (yes) | 1.02 | 1.75 | 0.63 | 99.9 -> 99.8 |

## Classification (fast-vision-models)

Full table: [Classification (fast-vision-models) in OVERVIEW.md](https://github.com/fastforks/.github/blob/main/OVERVIEW.md#classification-fast-vision-models).

| model | repo | licence (commercial) | portable vs flags | cuda-kernels vs flags | memory (cuda-kernels / flags) | quality, top-1 agreement % (flags -> cuda-kernels) |
|---|---|---|---|---|---|---|
| alexnet | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (custom) | 0.97 | 1.67 | 0.85 | 99.6 -> 99.2 |
| convnext_v2_b | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (yes) | 1.38 | 1.55 | 1.02 | 100.0 -> 100.0 |
| deit_b | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (yes) | 1.00 | 1.16 | 1.29 | 100.0 -> 100.0 |
| densenet121 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (custom) | 1.00 | 2.13 | 0.66 | 100.0 -> 100.0 |
| dinov2_l_lin | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (yes) | 1.04 | 1.18 | 1.44 | 100.0 -> 99.6 |
| efficientnet_b0 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (yes) | 1.02 | 1.72 | 0.29 | 100.0 -> 100.0 |
| efficientnet_b4 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (yes) | 1.01 | 1.81 | 0.23 | 100.0 -> 99.6 |
| efficientnet_v2_s | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (yes) | 1.01 | 1.86 | 0.50 | 100.0 -> 100.0 |
| efficientvit_b2 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (yes) | 1.01 | 2.75 | 0.21 | 99.6 -> 100.0 |
| eva02_l | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (yes) | 1.00 | 1.59 | 1.01 | 99.6 -> 100.0 |
| fastvit_sa24 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (custom) | 1.50 | 3.16 | 0.33 | 100.0 -> 100.0 |
| inception_v3 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (custom) | 1.01 | 2.30 | 0.51 | 99.6 -> 99.6 |
| mobilenet_v1 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (custom) | 1.01 | 2.83 | 0.29 | 99.2 -> 100.0 |
| mobilenet_v2 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (custom) | 1.03 | 2.79 | 0.21 | 100.0 -> 98.4 |
| mobilenet_v3_l | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (custom) | 1.09 | 1.91 | 0.27 | 99.6 -> 100.0 |
| mobilenet_v4_m | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (custom) | 1.04 | 2.12 | 0.34 | 100.0 -> 99.6 |
| regnety_16gf | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (yes) | 1.01 | 2.49 | 0.39 | 99.6 -> 99.2 |
| resnet152 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (custom) | 1.00 | 2.28 | 0.70 | 99.6 -> 99.6 |
| resnet18 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (custom) | 1.00 | 2.10 | 0.49 | 99.6 -> 100.0 |
| resnet50 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (custom) | 1.00 | 2.41 | 0.66 | 100.0 -> 100.0 |
| shufflenet_v2 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (custom) | 1.01 | 2.96 | 0.80 | 98.8 -> 99.6 |
| siglip2_so400m_zs | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (yes) | 1.00 | 1.14 | 0.89 | 99.6 -> 99.2 |
| siglip_so400m_ft | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (yes) | 1.00 | 1.12 | 1.17 | 99.6 -> 99.6 |
| squeezenet1_1 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (custom) | 0.98 | 3.41 | 0.38 | 99.6 -> 99.6 |
| swin_b | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (yes) | 1.25 | 1.71 | 0.91 | 100.0 -> 100.0 |
| vgg16 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (custom) | 1.00 | 2.33 | 0.43 | 100.0 -> 99.6 |
| vit_b16 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (yes) | 1.00 | 1.17 | 1.29 | 100.0 -> 100.0 |

## Backbones

Full table: [Backbones in OVERVIEW.md](https://github.com/fastforks/.github/blob/main/OVERVIEW.md#backbones).

| model | repo | licence (commercial) | portable vs flags | cuda-kernels vs flags | memory (cuda-kernels / flags) | quality, metric named in cell (flags -> cuda-kernels) |
|---|---|---|---|---|---|---|
| DINOv2 ViT-L | [fastbb](https://github.com/fastforks/fastbb) | Apache-2.0 (+ CUTLASS BSD-3, DINOv2 Apache-2.0) (yes) | - | 1.54 (vs upstream) | - | rel. L2 of patch tokens < 0.01 (cuda) |
| ResNet-50 | [fastbb](https://github.com/fastforks/fastbb) | Apache-2.0 (+ CUTLASS BSD-3, DINOv2 Apache-2.0) (yes) | - | 2.16 (vs upstream) | - | n/m |
| ViT-B/16 (timm) | [fastbb](https://github.com/fastforks/fastbb) | Apache-2.0 (+ CUTLASS BSD-3, DINOv2 Apache-2.0) (yes) | - | 1.50 (vs upstream) | - | n/m |

## Backbones (fast-vision-models)

Full table: [Backbones (fast-vision-models) in OVERVIEW.md](https://github.com/fastforks/.github/blob/main/OVERVIEW.md#backbones-fast-vision-models).

| model | repo | licence (commercial) | portable vs flags | cuda-kernels vs flags | memory (cuda-kernels / flags) | quality, feature rel. L2 (flags -> cuda-kernels) |
|---|---|---|---|---|---|---|
| dinov2-vitb14 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (yes) | 1.01 | 1.32 | 1.39 | 0.0077 -> 0.0116 |
| dinov2-vitg14 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (yes) | 1.01 | 1.22 | 1.09 | 0.0036 -> 0.0037 |
| dinov2-vitl14 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (yes) | 1.01 | 1.29 | 1.13 | 0.0066 -> 0.0073 |
| dinov2-vits14 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (yes) | 1.01 | 1.42 | 1.54 | 0.0045 -> 0.0063 |
| dinov3-convnext-base | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (custom) | 1.00 | 1.32 | 1.10 | 0.0016 -> 0.0034 |
| dinov3-convnext-large | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (custom) | 1.00 | 1.39 | 1.06 | 0.0013 -> 0.0045 |
| dinov3-convnext-small | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (custom) | 1.00 | 1.29 | 1.11 | 0.0017 -> 0.0032 |
| dinov3-convnext-tiny | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (custom) | 1.00 | 1.25 | 1.15 | 0.0024 -> 0.0052 |
| dinov3-vitb16 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (custom) | 0.96 | 1.33 | 1.20 | 0.0042 -> 0.0044 |
| dinov3-vith16plus | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (custom) | 0.96 | 1.26 | 1.05 | 0.0040 -> 0.0042 |
| dinov3-vitl16 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (custom) | 1.17 | 1.70 | 0.65 | 0.0038 -> 0.0045 |
| dinov3-vits16 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (custom) | 0.95 | 1.44 | 1.50 | 0.0034 -> 0.0032 |
| timm-convnext_tiny.fb_in22k_ft_in1k | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (yes) | 1.00 | 1.16 | 1.02 | 0.0019, top-1 agree 1.000 -> 0.0027 |
| timm-efficientnet_b0.ra_in1k | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (yes) | 1.10 | 1.31 | 0.80 | 0.0034, top-1 agree 1.000 -> 0.0041 |
| timm-mobilenetv2_100.ra_in1k | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (yes) | 1.14 | 1.59 | 0.79 | 0.0053, top-1 agree 1.000 -> 0.0068 |
| timm-regnety_032.ra_in1k | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (yes) | 1.02 | 0.34 | 1.46 | 0.0020, top-1 agree 1.000 -> 0.0020 |
| timm-resnet101.a1h_in1k | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (yes) | 1.03 | 2.05 | 0.95 | 0.0014, top-1 agree 1.000 -> 0.0050 |
| timm-resnet18.a1_in1k | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (yes) | 1.00 | 1.95 | 0.93 | 0.0005, top-1 agree 1.000 -> 0.0006 |
| timm-resnet50.a1_in1k | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | per-model weights, see repo (yes) | 1.02 | 2.15 | 0.95 | 0.0007, top-1 agree 1.000 -> 0.0013 |

## Point tracking

Full table: [Point tracking in OVERVIEW.md](https://github.com/fastforks/.github/blob/main/OVERVIEW.md#point-tracking).

| model | repo | licence (commercial) | portable vs flags | cuda-kernels vs flags | memory (cuda-kernels / flags) | quality, metric named in cell (flags -> cuda-kernels) |
|---|---|---|---|---|---|---|
| CoTracker | [co-tracker](https://github.com/fastforks/co-tracker) | CC BY-NC 4.0 (no) | - | 1.86 | 0.43 | median track deviation vs upstream (px): 0.0071 -> 0.0185 |

## Gaussian splatting libraries

Full table: [Gaussian splatting libraries in OVERVIEW.md](https://github.com/fastforks/.github/blob/main/OVERVIEW.md#gaussian-splatting-libraries).

| model | repo | licence (commercial) | portable vs flags | cuda-kernels vs flags | memory (cuda-kernels / flags) | quality, metric named in cell (flags -> cuda-kernels) |
|---|---|---|---|---|---|---|
| fused-ssim | [fused-ssim](https://github.com/fastforks/fused-ssim) | MIT (yes) | - | 2.21 (vs upstream) | - | - |
| gsplat | [gsplat](https://github.com/fastforks/gsplat) | Apache-2.0 (yes) | - | 3.24 (vs upstream) | - | PSNR: - -> 26.25 (upstream 26.24) |

## Notes

- These are research forks. Upstream licences apply to the upstream code; our own additions in new repositories are Apache-2.0.
- Much of the engineering was done with AI assistance (Claude); all results are measured, not estimated.
- Kernels are built for sm_80 - sm_120 GPUs (Ampere to Blackwell); Jetson / Thor builds compile but are untested on hardware.

Maintained by Florian Nick.

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

| model | repo | licence, commercial | ms flags -> cuda-kernels (setting) | portable vs flags | cuda-kernels vs flags | memory (cuda-kernels / flags) | quality, metric in cell (flags -> cuda-kernels) |
|---|---|---|---|---|---|---|---|
| AANet | [aanet](https://github.com/fastforks/aanet) | Apache-2.0, yes | 151.1 -> 108.5 (Midd-H) | - | 1.39 | 0.56 | ETH3D EPE: 0.545 -> 0.547; Midd-H EPE: 15.721 -> 15.724 |
| AANet+ | [aanet](https://github.com/fastforks/aanet) | Apache-2.0, yes | 189.6 -> 129.9 (Midd-H) | - | 1.46 | 0.63 | ETH3D EPE: 0.745 -> 0.755; Midd-H EPE: 15.790 -> 15.838 |
| BridgeDepth (rvc_pretrain) | [BridgeDepth](https://github.com/fastforks/BridgeDepth) | MIT, yes | 441.1 -> 435.8 (Midd-H) | 0.99 | 1.01 | 1.63 | ETH3D EPE: 0.088 -> 0.088; Midd-H EPE: 1.238 -> 1.237 |
| Cas-GwcNet-c | [cascade-stereo](https://github.com/fastforks/cascade-stereo) | MIT code, yes | 353.1 -> 227.9 (Midd-H) | 0.91 | 1.55 | 0.71 | ETH3D EPE: 0.512 -> 0.513; Midd-H EPE: 8.496 -> 8.572 |
| Cas-GwcNet-gc | [cascade-stereo](https://github.com/fastforks/cascade-stereo) | MIT code, yes | 822.2 -> 377.1 (Midd-H) | 1.47 | 2.18 | 0.23 | ETH3D EPE: 0.820 -> 0.824; Midd-H EPE: 6.541 -> 6.541 |
| CFNet | [CFNet](https://github.com/fastforks/CFNet) | MIT, yes | 633 -> 396.8 (Midd-H) | - | 1.60 | 0.26 | ETH3D EPE: 0.399 -> 0.397; Midd-H EPE: 15.959 -> 15.952 |
| CoEx | [coex](https://github.com/fastforks/coex) | GPL-3.0, custom | 49.7 -> 40.2 (Midd-H) | - | 1.24 | 0.43 | ETH3D EPE: 18.523 -> 18.497; Midd-H EPE: 17.603 -> 17.575 |
| CREStereo (ETH3D ckpt, 20 it) | [CREStereo-Pytorch](https://github.com/fastforks/CREStereo-Pytorch) | none, no | 1382 -> 238 (Midd-H) | 1.51 | 5.81 | 0.39 | ETH3D EPE: 0.185 -> 0.185; Midd-H EPE: 0.755 -> 0.745 |
| CroCo-Stereo | [croco](https://github.com/fastforks/croco) | CC BY-NC-SA, no | 3992.4 -> 2791.1 portable (Midd-H) | 1.43 | - | - | ETH3D EPE: 0.026 -> 0.026 (port.); Midd-H EPE: 1.007 -> 1.010 (port.) |
| DEFOM-Stereo | [DEFOM-Stereo](https://github.com/fastforks/DEFOM-Stereo) | MIT, yes | 3715.4 -> 655.1 (Midd-H) | - | 5.67 | 0.23 | ETH3D EPE: 0.173 -> 0.172; Midd-H EPE: 1.462 -> 1.456 |
| DLNR | [High-frequency-Stereo-Matching-Network](https://github.com/fastforks/High-frequency-Stereo-Matching-Network) | MIT, yes | 910.9 -> 264.9 (Midd-H) | - | 3.44 | 0.23 | ETH3D EPE: 2.624 -> 2.639; Midd-H EPE: 3.115 -> 3.119 |
| Fast-ACVNet | [Fast-ACVNet](https://github.com/fastforks/Fast-ACVNet) | MIT, yes | 75.3 -> 57.4 (Midd-H) | - | 1.31 | 0.26 | ETH3D EPE: 0.926 -> 0.935; Midd-H EPE: 9.545 -> 9.547 |
| Fast-FoundationStereo (23-36-37, 8 it) | [Fast-FoundationStereo](https://github.com/fastforks/Fast-FoundationStereo) | custom, no | 318.7 -> 144 (Midd-H) | 0.97 | 2.21 | 0.69 | ETH3D EPE: 0.324 -> 0.323; Midd-H EPE: 0.314 -> 0.315 |
| FoundationStereo (ViT-L, 32 it) | [FoundationStereo](https://github.com/fastforks/FoundationStereo) | custom, no | 3642 -> 1495 (Midd-H) | 1.22 | 2.44 | 0.37 | ETH3D EPE: 0.251 -> 0.250; Midd-H EPE: 0.242 -> 0.241 |
| GwcNet | [GwcNet](https://github.com/fastforks/GwcNet) | MIT, yes | 335.5 -> 279.9 (Midd-H) | - | 1.20 | 0.30 | ETH3D EPE: 1.579 -> 1.579; Midd-H EPE: 16.619 -> 16.623 |
| IGEV-Stereo (sceneflow, 32 it) | [IGEV](https://github.com/fastforks/IGEV) | MIT code, yes | 944.1 -> 293.3 (Midd-H) | 1.22 | 3.22 | 0.69 | ETH3D EPE: 0.333 -> 0.333; Midd-H EPE: 2.635 -> 2.636 |
| LAS2-H | [LiteAnyStereo](https://github.com/fastforks/LiteAnyStereo) | MIT, yes | 118.7 -> 96.1 (Midd-H) | - | 1.24 | 0.98 | ETH3D EPE: 0.154 -> 0.155; Midd-H EPE: 6.142 -> 6.146 |
| LAS2-M | [LiteAnyStereo](https://github.com/fastforks/LiteAnyStereo) | MIT, yes | 47.7 -> 40.6 (Midd-H) | - | 1.17 | 0.72 | ETH3D EPE: 0.233 -> 0.233; Midd-H EPE: 6.409 -> 6.409 |
| MC-Stereo (Scene Flow ckpt, 32 it) | [MC-Stereo](https://github.com/fastforks/MC-Stereo) | none, no | n/m | - | - | - | EPE (ETH3D -> Midd-H subset): 4.56 -> 1.14 -> - |
| MoCha-Stereo (V1, 32 it) | [MoCha-Stereo](https://github.com/fastforks/MoCha-Stereo) | MIT code, yes | 1395 -> 487 (1482x994) | 1.17 | 2.86 | 0.48 | EPE ETH3D -> Midd-H: 0.318 -> 0.938 vs 0.318 -> 0.939 |
| MonSter | [MonSter](https://github.com/fastforks/MonSter) | custom, custom | 3152.3 -> 654.9 (Midd-H) | - | 4.81 | 0.22 | ETH3D EPE: 0.215 -> 0.220; Midd-H EPE: 5.255 -> 5.251 |
| MonSter++ | [MonSter-plusplus](https://github.com/fastforks/MonSter-plusplus) | custom, custom | 3194.3 -> 676.2 (Midd-H) | - | 4.72 | 0.23 | ETH3D EPE: 0.106 -> 0.106; Midd-H EPE: 0.934 -> 0.933 |
| NMRF-Stereo | [NMRF](https://github.com/fastforks/NMRF) | MIT, yes | 416.7 -> 205.5 (Midd-H) | - | 2.03 | 0.09 | ETH3D EPE: 0.338 -> 0.337; Midd-H EPE: 6.961 -> 6.926 |
| PromptStereo | [PromptStereo](https://github.com/fastforks/PromptStereo) | MIT, yes | 3251.9 -> 1620 (Midd-H) | - | 2.01 | 0.24 | ETH3D EPE: 0.116 -> 0.116; Midd-H EPE: 1.087 -> 1.079 |
| PSMNet | [PSMNet](https://github.com/fastforks/PSMNet) | MIT, yes | 526.5 -> 229 (Midd-H) | - | 2.30 | 0.35 | ETH3D EPE: 1.255 -> 1.255; Midd-H EPE: 24.939 -> 24.940 |
| RAFT-Stereo (Middlebury ckpt, 32 it) | [RAFT-Stereo](https://github.com/fastforks/RAFT-Stereo) | MIT code, yes | 1018 -> 274 (Midd-H) | 1.37 | 3.72 | 0.28 | ETH3D EPE: 0.288 -> 0.274; Midd-H EPE: 1.069 -> 1.067 |
| RT-MonSter++ | [MonSter-plusplus](https://github.com/fastforks/MonSter-plusplus) | custom, custom | 176.1 -> 119.5 (Midd-H) | - | 1.47 | 0.42 | ETH3D EPE: 0.146 -> 0.147; Midd-H EPE: 5.690 -> 5.685 |
| S2M2-S | [s2m2](https://github.com/fastforks/s2m2) | CC BY-NC, no | 263 -> 111.4 (Midd-H) | 0.75 | 2.36 | 0.62 | ETH3D EPE: 0.316 -> 0.193; Midd-H EPE: 0.833 -> 0.652 |
| S2M2-XL | [s2m2](https://github.com/fastforks/s2m2) | CC BY-NC, no | 1457 -> 683.8 (Midd-H) | 0.80 | 2.13 | 1.13 | ETH3D EPE: 0.189 -> 0.149; Midd-H EPE: 0.553 -> 0.499 |
| Selective-Stereo | [Selective-Stereo](https://github.com/fastforks/Selective-Stereo) | MIT, yes | 2148.7 -> 345.9 (Midd-H) | - | 6.21 | 0.10 | ETH3D EPE: 0.403 -> 0.402; Midd-H EPE: 7.503 -> 7.449 |
| StereoAnywhere | [stereoanywhere](https://github.com/fastforks/stereoanywhere) | CC BY-NC-SA, no | 1801.9 -> 343.9 (KITTI 1242x375) | - | 5.24 | 0.31 | ETH3D EPE: 0.143 -> 0.143; Midd-H EPE: n/m -> n/m |
| UniMatch (stereo) | [unimatch](https://github.com/fastforks/unimatch) | MIT, yes | 792.2 -> 517.1 portable (Midd-H) | 1.53 | - | - | ETH3D EPE: 0.441 -> 0.441 (port.); Midd-H EPE: 2.786 -> 2.787 (port.) |
| WAFT-Stereo B | [WAFT-Stereo](https://github.com/fastforks/WAFT-Stereo) | custom, yes | 1634.5 -> 410.2 (Midd-H) | - | 3.98 | 0.17 | ETH3D EPE: 0.107 -> 0.108; Midd-H EPE: 1.366 -> 1.365 |
| WAFT-Stereo S | [WAFT-Stereo](https://github.com/fastforks/WAFT-Stereo) | custom, yes | 1129.3 -> 288.6 (Midd-H) | - | 3.91 | 0.10 | ETH3D EPE: 0.096 -> 0.096; Midd-H EPE: 1.744 -> 1.741 |

## Multi-view stereo and feed-forward reconstruction

Full table: [Multi-view stereo and feed-forward reconstruction in OVERVIEW.md](https://github.com/fastforks/.github/blob/main/OVERVIEW.md#multi-view-stereo-and-feed-forward-reconstruction).

| model | repo | licence, commercial | ms flags -> cuda-kernels (setting) | portable vs flags | cuda-kernels vs flags | memory (cuda-kernels / flags) | quality, metric in cell (flags -> cuda-kernels) |
|---|---|---|---|---|---|---|---|
| MapAnything | [map-anything](https://github.com/fastforks/map-anything) | Apache-2.0, yes | 1786 -> 556 (9 views) | - | 3.21 | 0.31 | ETH3D metric absrel: 0.1176 -> 0.1176 |
| MVSAnywhere | [mvsanywhere](https://github.com/fastforks/mvsanywhere) | NC, no | 193.8 -> 194.7 portable (448x672) | 1.00 | - | - | ETH3D AbsRel: 0.0110 -> 0.0111 (port.) |
| MVSFormer++ | [MVSFormerPlusPlus](https://github.com/fastforks/MVSFormerPlusPlus) | Apache-2.0, yes | 463.6 -> 273.7 (1536x1024) | - | 1.69 | 0.27 | n/m (random weights, accuracy not measured) |
| PatchmatchNet | [PatchmatchNet](https://github.com/fastforks/PatchmatchNet) | MIT, yes | 313.3 -> 42.1 (1536x1024) | - | 7.44 | 0.41 | absrel: 0.1256 -> 0.1246 |
| VGGT | [vggt](https://github.com/fastforks/vggt) | NC, custom | 952.1 -> 458.1 (8 frames) | - | 2.08 | 0.28 | flags identical to upstream; cuda depth rel. diff median 0.40% (8 frames) |
| VGGT-Omega | [vggt-omega](https://github.com/fastforks/vggt-omega) | NC, no | 1001.6 -> 510 (size A) | - | 1.96 | 0.25 | ETH3D absrel: 0.0062 -> 0.0062 |

## Monocular depth and geometry

Full table: [Monocular depth and geometry in OVERVIEW.md](https://github.com/fastforks/.github/blob/main/OVERVIEW.md#monocular-depth-and-geometry).

| model | repo | licence, commercial | ms flags -> cuda-kernels (setting) | portable vs flags | cuda-kernels vs flags | memory (cuda-kernels / flags) | quality, metric in cell (flags -> cuda-kernels) |
|---|---|---|---|---|---|---|---|
| Depth Pro | [ml-depth-pro](https://github.com/fastforks/ml-depth-pro) | custom, custom | 528.1 -> 282.3 (Midd-H) | 1.02 | 1.87 | 0.95 | Midd-H AbsRel raw: 0.1480 -> 0.1481 |
| Depth-Anything-3 (DA3MONO-LARGE) | [Depth-Anything-3](https://github.com/fastforks/Depth-Anything-3) | Apache-2.0 code, yes | 175.4 -> 72.3 (1482x994 image) | 1.50 | 2.43 | 0.47 | NYU-20 AbsRel: 0.03394 -> 0.03397 |
| Depth-Anything-V2 ViT-L | [Depth-Anything-V2](https://github.com/fastforks/Depth-Anything-V2) | Apache-2.0 code, no | 267.5 -> 75 (1024 px) | 2.35 | 3.57 | 0.79 | NYU AbsRel: 0.04388 -> 0.04388 |
| Depth-Anything-V2 ViT-S | [Depth-Anything-V2](https://github.com/fastforks/Depth-Anything-V2) | Apache-2.0 code, yes | 66.2 -> 11.3 (1024 px) | 3.96 | 5.86 | 0.79 | NYU AbsRel: 0.05093 -> 0.05093 |
| Metric3D v2 ViT-L | [Metric3D](https://github.com/fastforks/Metric3D) | BSD-2, yes | 438 -> 201.3 portable (Midd-H) | 2.18 | - | - | Midd-H AbsRel raw: 0.7041 -> 0.7040 (port.) |
| MiDaS dpt_beit_large_512 | [MiDaS](https://github.com/fastforks/MiDaS) | MIT, yes | 267 -> 84.2 portable (Midd-H) | 3.17 | - | - | Midd-H AbsRel (disparity): 0.1157 -> 0.1160 (port.) |
| MoGe-2 ViT-L normal | [MoGe](https://github.com/fastforks/MoGe) | MIT, yes | 151.3 -> 85.9 (native 1482x994) | 0.99 | 1.76 | 1.08 | NYU-20 AbsRel: 0.0321 -> 0.0321 |
| UniDepth V2 ViT-L | [UniDepth](https://github.com/fastforks/UniDepth) | CC BY-NC, no | 131.9 -> 62.7 (Midd-H) | 1.31 | 2.10 | 0.45 | Midd-H AbsRel raw: 0.1989 -> 0.1986 |
| UniK3D ViT-L | [UniK3D](https://github.com/fastforks/UniK3D) | CC BY-NC-SA, no | 134.2 -> 62.6 (Midd-H) | 1.33 | 2.14 | 0.44 | Midd-H AbsRel raw: 0.2020 -> 0.2018 |

## Promptable and instance segmentation

Full table: [Promptable and instance segmentation in OVERVIEW.md](https://github.com/fastforks/.github/blob/main/OVERVIEW.md#promptable-and-instance-segmentation).

| model | repo | licence, commercial | ms flags -> cuda-kernels (setting) | portable vs flags | cuda-kernels vs flags | memory (cuda-kernels / flags) | quality, metric in cell (flags -> cuda-kernels) |
|---|---|---|---|---|---|---|---|
| Detectron2 (Mask R-CNN) | [detectron2](https://github.com/fastforks/detectron2) | Apache-2.0, yes | 40.8 -> 25.9 (size A) | - | 1.58 | 0.68 | box/mask AP: 45.38 -> 41.78 vs 45.18 -> 41.69 |
| SAM 2 Hiera-L | [sam2](https://github.com/fastforks/sam2) | Apache-2.0, yes | 86.4 -> 57.2 (image 1297x840) | 1.22 | 1.51 | 0.70 | IoU vs upstream (L): 1 -> 0.965 |
| SAM 3 | [sam3](https://github.com/fastforks/sam3) | custom, custom | 218 -> 115 (image 1297x840) | 1.21 | 1.90 | 0.58 | IoU vs GT: 0.7945 -> 0.7947 |
| SAM ViT-B | [segment-anything](https://github.com/fastforks/segment-anything) | Apache-2.0, yes | 101.3 -> 41.5 (image 1297x840) | 2.14 | 2.44 | 0.20 | mean mask IoU vs fp32: 0.9906 -> 0.9902 |
| SAM ViT-H | [segment-anything](https://github.com/fastforks/segment-anything) | Apache-2.0, yes | 308 -> 170.7 (image 1297x840) | 1.44 | 1.80 | 0.47 | mean mask IoU vs fp32: 0.9979 -> 0.9964 |
| Sapiens-1B (seg) | [sapiens](https://github.com/fastforks/sapiens) | CC BY-NC, no | 354.2 -> 189.3 (batch 1) | - | 1.87 | 0.56 | seg mIoU vs upstream: 0.9981 -> 0.9932 |

## Semantic segmentation (fast-vision-models)

Full table: [Semantic segmentation (fast-vision-models) in OVERVIEW.md](https://github.com/fastforks/.github/blob/main/OVERVIEW.md#semantic-segmentation-fast-vision-models).

| model | repo | licence, commercial | ms flags -> cuda-kernels (setting) | portable vs flags | cuda-kernels vs flags | memory (cuda-kernels / flags) | quality, pixel agreement % (flags -> cuda-kernels) |
|---|---|---|---|---|---|---|---|
| ddrnet-23 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, yes | 12.5 -> 7 (1024x2048) | 1.02 | 1.79 | 0.55 | 99.9 -> 99.9 |
| deeplabv3-r101 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, yes | 184 -> 95 (1024x2048) | 1.01 | 1.94 | 0.58 | 100.0 -> 100.0 |
| deeplabv3plus-r101 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, yes | 211 -> 101 (1024x2048) | 1.17 | 2.09 | 0.46 | 100.0 -> 100.0 |
| dinov2-l-linear | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, yes | 497 -> 358 (1024x2048) | 1.02 | 1.39 | 0.69 | 100.0 -> 99.9 |
| eomt-l-dinov2 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, custom | 205 -> 142 (1024x2048) | 1.09 | 1.44 | 0.54 | 100.0 -> 100.0 |
| eomt-l-dinov3 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, custom | 299 -> 167 (1024x2048) | 1.40 | 1.79 | 0.47 | 100.0 -> 100.0 |
| fcn-r101 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, yes | 160 -> 82.1 (1024x2048) | 1.01 | 1.95 | 0.50 | 100.0 -> 100.0 |
| hrnet-w48 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, yes | 84.5 -> 52.3 (1024x2048) | 1.02 | 1.62 | 0.27 | 100.0 -> 100.0 |
| mask2former-swinl | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, no | 279 -> 154 (1024x2048) | 1.09 | 1.81 | 0.55 | 99.9 -> 99.9 |
| oneformer-swinl | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, yes | 335 -> 221 (1024x2048) | 1.04 | 1.52 | 0.53 | 99.8 -> 99.9 |
| pidnet-l | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, yes | 20.7 -> 12.5 (1024x2048) | 1.02 | 1.66 | 0.72 | 99.9 -> 99.9 |
| pspnet-r101 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, yes | 191 -> 116 (1024x2048) | 1.01 | 1.65 | 0.61 | 100.0 -> 100.0 |
| segformer-b0 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, no | 34.3 -> 16.3 (1024x2048) | 1.22 | 2.10 | 0.19 | 100.0 -> 100.0 |
| segformer-b5 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, no | 135 -> 92.3 (1024x2048) | 1.25 | 1.46 | 0.09 | 100.0 -> 100.0 |
| segnet | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, n/a | 144 -> 88.1 (1024x2048) | 1.19 | 1.63 | 0.47 | 97.2 -> 95.9 |
| segnext-t | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, yes | 68.7 -> 26.4 (1024x2048) | 1.76 | 2.60 | 0.19 | 99.9 -> 99.9 |
| unet-s5 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, yes | 128 -> 73 (1024x2048) | 1.02 | 1.75 | 0.63 | 99.9 -> 99.8 |

## Classification (fast-vision-models)

Full table: [Classification (fast-vision-models) in OVERVIEW.md](https://github.com/fastforks/.github/blob/main/OVERVIEW.md#classification-fast-vision-models).

| model | repo | licence, commercial | ms flags -> cuda-kernels (setting) | portable vs flags | cuda-kernels vs flags | memory (cuda-kernels / flags) | quality, top-1 agreement % (flags -> cuda-kernels) |
|---|---|---|---|---|---|---|---|
| alexnet | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, custom | 3.33 -> 1.99 (b64) | 0.97 | 1.67 | 0.85 | 99.6 -> 99.2 |
| convnext_v2_b | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, yes | 107 -> 69 (b64) | 1.38 | 1.55 | 1.02 | 100.0 -> 100.0 |
| deit_b | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, yes | 45.3 -> 38.9 (b64) | 1.00 | 1.16 | 1.29 | 100.0 -> 100.0 |
| densenet121 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, custom | 35.1 -> 16.5 (b64) | 1.00 | 2.13 | 0.66 | 100.0 -> 100.0 |
| dinov2_l_lin | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, yes | 198 -> 168 (b64) | 1.04 | 1.18 | 1.44 | 100.0 -> 99.6 |
| efficientnet_b0 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, yes | 13.3 -> 7.74 (b64) | 1.02 | 1.72 | 0.29 | 100.0 -> 100.0 |
| efficientnet_b4 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, yes | 121 -> 66.9 (b64) | 1.01 | 1.81 | 0.23 | 100.0 -> 99.6 |
| efficientnet_v2_s | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, yes | 91.1 -> 49 (b64) | 1.01 | 1.86 | 0.50 | 100.0 -> 100.0 |
| efficientvit_b2 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, yes | 55.5 -> 20.2 (b64) | 1.01 | 2.75 | 0.21 | 99.6 -> 100.0 |
| eva02_l | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, yes | 1355 -> 853 (b64) | 1.00 | 1.59 | 1.01 | 99.6 -> 100.0 |
| fastvit_sa24 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, custom | 72.1 -> 22.82 (b64) | 1.50 | 3.16 | 0.33 | 100.0 -> 100.0 |
| inception_v3 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, custom | 45 -> 19.6 (b64) | 1.01 | 2.30 | 0.51 | 99.6 -> 99.6 |
| mobilenet_v1 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, custom | 9.07 -> 3.2 (b64) | 1.01 | 2.83 | 0.29 | 99.2 -> 100.0 |
| mobilenet_v2 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, custom | 10.9 -> 3.9 (b64) | 1.03 | 2.79 | 0.21 | 100.0 -> 98.4 |
| mobilenet_v3_l | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, custom | 7.2 -> 3.76 (b64) | 1.09 | 1.91 | 0.27 | 99.6 -> 100.0 |
| mobilenet_v4_m | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, custom | 12.3 -> 5.79 (b64) | 1.04 | 2.12 | 0.34 | 100.0 -> 99.6 |
| regnety_16gf | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, yes | 102 -> 41 (b64) | 1.01 | 2.49 | 0.39 | 99.6 -> 99.2 |
| resnet152 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, custom | 66.7 -> 29.3 (b64) | 1.00 | 2.28 | 0.70 | 99.6 -> 99.6 |
| resnet18 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, custom | 9.62 -> 4.59 (b64) | 1.00 | 2.10 | 0.49 | 99.6 -> 100.0 |
| resnet50 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, custom | 30.9 -> 12.8 (b64) | 1.00 | 2.41 | 0.66 | 100.0 -> 100.0 |
| shufflenet_v2 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, custom | 5.57 -> 1.88 (b64) | 1.01 | 2.96 | 0.80 | 98.8 -> 99.6 |
| siglip2_so400m_zs | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, yes | 807 -> 711 (b64) | 1.00 | 1.14 | 0.89 | 99.6 -> 99.2 |
| siglip_so400m_ft | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, yes | 798 -> 711 (b64) | 1.00 | 1.12 | 1.17 | 99.6 -> 99.6 |
| squeezenet1_1 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, custom | 6.64 -> 1.95 (b64) | 0.98 | 3.41 | 0.38 | 99.6 -> 99.6 |
| swin_b | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, yes | 89.9 -> 52.7 (b64) | 1.25 | 1.71 | 0.91 | 100.0 -> 100.0 |
| vgg16 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, custom | 67.2 -> 28.8 (b64) | 1.00 | 2.33 | 0.43 | 100.0 -> 99.6 |
| vit_b16 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, yes | 45.3 -> 38.8 (b64) | 1.00 | 1.17 | 1.29 | 100.0 -> 100.0 |

## Backbones

Full table: [Backbones in OVERVIEW.md](https://github.com/fastforks/.github/blob/main/OVERVIEW.md#backbones).

| model | repo | licence, commercial | ms flags -> cuda-kernels (setting) | portable vs flags | cuda-kernels vs flags | memory (cuda-kernels / flags) | quality, metric in cell (flags -> cuda-kernels) |
|---|---|---|---|---|---|---|---|
| DINOv2 ViT-L | [fastbb](https://github.com/fastforks/fastbb) | Apache-2.0, yes | upstream 175.6 -> 114 (896x1344 b1) | - | 1.54 (vs upstream) | - | rel. L2 of patch tokens < 0.01 (cuda) |
| ResNet-50 | [fastbb](https://github.com/fastforks/fastbb) | Apache-2.0, yes | upstream 9.27 -> 4.29 (1024x1024 b1) | - | 2.16 (vs upstream) | - | n/m |
| ViT-B/16 (timm) | [fastbb](https://github.com/fastforks/fastbb) | Apache-2.0, yes | upstream 32.3 -> 21.6 (1024 px b1) | - | 1.50 (vs upstream) | - | n/m |

## Backbones (fast-vision-models)

Full table: [Backbones (fast-vision-models) in OVERVIEW.md](https://github.com/fastforks/.github/blob/main/OVERVIEW.md#backbones-fast-vision-models).

| model | repo | licence, commercial | ms flags -> cuda-kernels (setting) | portable vs flags | cuda-kernels vs flags | memory (cuda-kernels / flags) | quality, feature rel. L2 (flags -> cuda-kernels) |
|---|---|---|---|---|---|---|---|
| dinov2-vitb14 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, yes | 52.47 -> 39.75 (896x1344 b1) | 1.01 | 1.32 | 1.39 | 0.0077 -> 0.0116 |
| dinov2-vitg14 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, yes | 444.39 -> 364.43 (896x1344 b1) | 1.01 | 1.22 | 1.09 | 0.0036 -> 0.0037 |
| dinov2-vitl14 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, yes | 146.54 -> 113.64 (896x1344 b1) | 1.01 | 1.29 | 1.13 | 0.0066 -> 0.0073 |
| dinov2-vits14 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, yes | 23.73 -> 16.66 (896x1344 b1) | 1.01 | 1.42 | 1.54 | 0.0045 -> 0.0063 |
| dinov3-convnext-base | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, custom | 25.62 -> 19.37 (1024px b1) | 1.00 | 1.32 | 1.10 | 0.0016 -> 0.0034 |
| dinov3-convnext-large | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, custom | 49.62 -> 35.66 (1024px b1) | 1.00 | 1.39 | 1.06 | 0.0013 -> 0.0045 |
| dinov3-convnext-small | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, custom | 17.54 -> 13.55 (1024px b1) | 1.00 | 1.29 | 1.11 | 0.0017 -> 0.0032 |
| dinov3-convnext-tiny | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, custom | 10.9 -> 8.73 (1024px b1) | 1.00 | 1.25 | 1.15 | 0.0024 -> 0.0052 |
| dinov3-vitb16 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, custom | 32.31 -> 24.3 (1024px b1) | 0.96 | 1.33 | 1.20 | 0.0042 -> 0.0044 |
| dinov3-vith16plus | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, custom | 207.14 -> 164.69 (1024px b1) | 0.96 | 1.26 | 1.05 | 0.0040 -> 0.0042 |
| dinov3-vitl16 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, custom | 112.9 -> 66.38 (1024px b1) | 1.17 | 1.70 | 0.65 | 0.0038 -> 0.0045 |
| dinov3-vits16 | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, custom | 14.54 -> 10.09 (1024px b1) | 0.95 | 1.44 | 1.50 | 0.0034 -> 0.0032 |
| timm-convnext_tiny.fb_in22k_ft_in1k | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, yes | 15.41 -> 13.33 (224px b32) | 1.00 | 1.16 | 1.02 | 0.0019, top-1 agree 1.000 -> 0.0027 |
| timm-efficientnet_b0.ra_in1k | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, yes | 6.17 -> 4.7 (224px b32) | 1.10 | 1.31 | 0.80 | 0.0034, top-1 agree 1.000 -> 0.0041 |
| timm-mobilenetv2_100.ra_in1k | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, yes | 4.83 -> 3.04 (224px b32) | 1.14 | 1.59 | 0.79 | 0.0053, top-1 agree 1.000 -> 0.0068 |
| timm-regnety_032.ra_in1k | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, yes | 10.83 -> 31.61 (1024px b1) | 1.02 | 0.34 | 1.46 | 0.0020, top-1 agree 1.000 -> 0.0020 |
| timm-resnet101.a1h_in1k | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, yes | 21.52 -> 10.52 (224px b32) | 1.03 | 2.05 | 0.95 | 0.0014, top-1 agree 1.000 -> 0.0050 |
| timm-resnet18.a1_in1k | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, yes | 4.71 -> 2.42 (224px b32) | 1.00 | 1.95 | 0.93 | 0.0005, top-1 agree 1.000 -> 0.0006 |
| timm-resnet50.a1_in1k | [fast-vision-models](https://github.com/fastforks/fast-vision-models) | weights, yes | 14.12 -> 6.58 (224px b32) | 1.02 | 2.15 | 0.95 | 0.0007, top-1 agree 1.000 -> 0.0013 |

## Point tracking

Full table: [Point tracking in OVERVIEW.md](https://github.com/fastforks/.github/blob/main/OVERVIEW.md#point-tracking).

| model | repo | licence, commercial | ms flags -> cuda-kernels (setting) | portable vs flags | cuda-kernels vs flags | memory (cuda-kernels / flags) | quality, metric in cell (flags -> cuda-kernels) |
|---|---|---|---|---|---|---|---|
| CoTracker | [co-tracker](https://github.com/fastforks/co-tracker) | CC BY-NC, no | 1157 -> 622.6 (48 frames) | - | 1.86 | 0.43 | median track deviation vs upstream (px): 0.0071 -> 0.0185 |

## Gaussian splatting libraries

Full table: [Gaussian splatting libraries in OVERVIEW.md](https://github.com/fastforks/.github/blob/main/OVERVIEW.md#gaussian-splatting-libraries).

| model | repo | licence, commercial | ms flags -> cuda-kernels (setting) | portable vs flags | cuda-kernels vs flags | memory (cuda-kernels / flags) | quality, metric in cell (flags -> cuda-kernels) |
|---|---|---|---|---|---|---|---|
| fused-ssim | [fused-ssim](https://github.com/fastforks/fused-ssim) | MIT, yes | upstream 1.39 -> 0.63 (L1+SSIM fwd+bwd) | - | 2.21 (vs upstream) | - | - |
| gsplat | [gsplat](https://github.com/fastforks/gsplat) | Apache-2.0, yes | upstream 223.6 s -> 69.1 s (MipNeRF-360) | - | 3.24 (vs upstream) | - | PSNR: - -> 26.25 (upstream 26.24) |

## Notes

- These are research forks. Upstream licences apply to the upstream code; our own additions in new repositories are Apache-2.0.
- Much of the engineering was done with AI assistance (Claude); all results are measured, not estimated.
- Kernels are built for sm_80 - sm_120 GPUs (Ampere to Blackwell); Jetson / Thor builds compile but are untested on hardware.

Maintained by Florian Nick.

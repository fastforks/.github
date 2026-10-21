# fastforks

Measured inference and training improvements to upstream vision projects. Each fork documents its hardware, inputs, latency, memory and output agreement in its README. Results currently come from an RTX 5060 Ti (sm_120); expect different speeds on other GPUs.

| Tier | Meaning |
|---|---|
| `main` | Upstream source at the pinned commit. |
| Upstream + fast settings | The same upstream code with measured settings such as autocast, channels-last or `cudnn.benchmark`. This is a benchmark configuration, not another branch. |
| `portable` | Plain-PyTorch code changes; the default branch where available. See each README for the exact export coverage. |
| `cuda-kernels` | Additional custom CUDA kernels where they exist. Check each README for GPU targets, build requirements and output differences. |

The available tiers depend on the project. [SAM 2](https://github.com/fastforks/sam2) has no custom-kernel branch. [Segment Anything](https://github.com/fastforks/segment-anything) offers [`fp16-accumulate`](https://github.com/fastforks/segment-anything/tree/fp16-accumulate), which changes a PyTorch precision setting rather than adding a CUDA kernel. [MC-Stereo](https://github.com/fastforks/MC-Stereo) has no upstream licence; its [`benchmark-report`](https://github.com/fastforks/MC-Stereo/tree/benchmark-report) branch adds only measured results, not optimization code. [gsplat](https://github.com/fastforks/gsplat) and [fused-ssim](https://github.com/fastforks/fused-ssim) are CUDA training libraries and have `cuda-kernels` branches only.

Released stereo forks: [S2M2](https://github.com/fastforks/s2m2), [RAFT-Stereo](https://github.com/fastforks/RAFT-Stereo), [CREStereo-Pytorch](https://github.com/fastforks/CREStereo-Pytorch), [MoCha-Stereo](https://github.com/fastforks/MoCha-Stereo), [MC-Stereo](https://github.com/fastforks/MC-Stereo).

Released depth and segmentation forks: [Depth Anything 3](https://github.com/fastforks/Depth-Anything-3), [Depth Anything V2](https://github.com/fastforks/Depth-Anything-V2), [MoGe](https://github.com/fastforks/MoGe), [SAM 3](https://github.com/fastforks/sam3), [SAM 2](https://github.com/fastforks/sam2), [Segment Anything](https://github.com/fastforks/segment-anything).

Shared kernel library: [fastbb](https://github.com/fastforks/fastbb). The upstream licences remain with their projects; each fork README states the terms for its own changes and any checkpoint restrictions.

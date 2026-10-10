# fastforks

Research forks that make open-source vision models faster and lighter, **without changing their results**.
Everything is measured on one consumer GPU (RTX 5060 Ti, 16 GB) and reported as speed **and** memory,
plus startup, first-call and new-input-size cost.

## How every fork is organised

| branch | what it is |
|---|---|
| `main` | untouched mirror of the upstream repository |
| **`fast`** (default) | upstream-friendly speed-ups: plain PyTorch only, exports to ONNX / TensorRT, no custom kernels, no build step |
| `kernels` | on top of `fast`: custom CUDA kernels (prebuilt where possible) for the maximum speed |

One commit per improvement; each commit message carries its timing and memory numbers.
Every `README` starts with a performance block comparing upstream, `fast` and `kernels`.
Bug fixes are also offered upstream as small, separate pull requests.

## Examples of what to expect (RTX 5060 Ti, general path, any input size)

| model | upstream | `kernels` |
|---|---|---|
| RAFT-Stereo, Middlebury half-res | 1079 ms, 3.3 GB | 275 ms, 0.78 GB |
| Depth Pro, 1536 px | 531 ms | 282 ms (TensorRT 11 fp16: 299 ms) |
| 3D Gaussian splatting training, garden 7k steps | 223.8 s | 68.8 s |

Numbers come from our own result files; each repository lists its exact commands, hardware and date.

## Notes

- These are research forks. Upstream licences apply to the upstream code; our own additions in new repositories are Apache-2.0.
- Much of the engineering was done with AI assistance (Claude); commits carry a `Co-Authored-By` trailer and all results are measured, not estimated.
- Kernels are built for sm_80 – sm_120 GPUs (Ampere to Blackwell); Jetson / Thor builds compile but are untested on hardware.

Maintained by Florian Nick.

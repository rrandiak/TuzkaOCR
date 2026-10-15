# Why we run TuzkaOCR from a fork

**Fork:** `rrandiak/TuzkaOCR`, branch `fix/cuda-mem-growth` (rebased on upstream v1.9.0)

Upstream TuzkaOCR supports only CPU deployment, and it publishes only a CPU image. Its GPU
build (`Dockerfile.gpu`) runs out of video memory (VRAM) under real workloads, so pages fail.

## Why the GPU build fails

It runs the OCR models with ONNX Runtime on CUDA.

- The runtime's GPU memory pool keeps a separate block of memory for every page size it has
  seen, and never gives it back. With mixed scan sizes, VRAM keeps climbing.
- Several pages are processed at the same time, all using that one memory pool.
- On our 16 GB card this caused out-of-memory failures from 6 parallel page workers upwards.

## What the fork changes

About 180 lines of code; the CPU path is unchanged.

- **Freeing memory after each step:** the pool releases unused memory after every model run.
  This is on by default. In a test with 64 page sizes, memory held went from 4.8 GB to 0.1 GB.
- **A cap on parallel GPU work** (`TUZKAOCR_GPU_CONCURRENCY`): the CPU side can keep many pages
  in progress while VRAM stays bounded.

## Measured on v1.9.0

200 mixed-size pages, 4 page workers, 6 GB GPU:

|                   | Upstream                        | Fork                                           |
|-------------------|---------------------------------|------------------------------------------------|
| Printed pages     | about 65 failed (out of memory) | 0 failed, 4.0 GB peak                          |
| Handwritten pages | about 63 failed                 | 0 failed, 4.6 GB peak (1.8 GB with a cap of 2) |

On the production GPU box (16 GB), the tuned setting (10 page workers, cap of 4) gives about
3.9 pages/s at about 2.8 GB peak. That is about 2.7 times the throughput of the old production
setup, which ran one page at a time.

## Upkeep

Each upstream release means rebasing the fork and re-checking GPU memory; it took about an hour
for v1.9.0. The plan is to offer the fix upstream as a pull request so we can drop the fork.

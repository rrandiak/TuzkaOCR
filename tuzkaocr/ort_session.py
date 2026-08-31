"""Central ONNX Runtime CUDA provider construction.

These provider options are NOT the VRAM fix. Measurements (bench/GPU_OOM_INVESTIGATION.md §4)
falsified every provider-option lever that was tried:

  - ``cudnn_conv_algo_search``: EXHAUSTIVE vs HEURISTIC give an **identical** peak (441-shape
    layout probe: 4886 MB both ways; re-confirmed on ORT 1.25.1, 64 shapes: 4848 MB both ways).
    It is a speed knob only, with no effect on memory.
  - ``cudnn_conv_use_max_workspace``: no effect (§4b). The large buffers in the failure logs are
    real activations, not cuDNN scratch.
  - ``arena_extend_strategy``: does not bound growth either (§4d) — and flipping everything to
    kNextPowerOfTwo is actively worse on a full corpus, because power-of-two over-reservation
    accumulates across distinct shapes and never flattens.

What actually drives VRAM is that the arena **retains a block set per distinct input shape**
(§7b) and that ``page_workers`` runs many inferences concurrently on that shared arena (§6).
The two real levers both live in pipeline.py: per-run arena shrinkage (``gpu_arena_shrink``,
which turns that retention into a transient — 4848 MB → 118 MB resident, measured) and the
GPU-concurrency semaphore (``gpu_concurrency``).

The committed arena split below is nonetheless correct and should be kept (§4d): layout/role use
``kSameAsRequested``, the recognizer ``kNextPowerOfTwo``. That split measured best on the full
corpus; the reasoning is empirical, not the per-width story an earlier version of this docstring
told (§4c falsified that: the recognizer sits at ~164 MB under either strategy).

Values come from Config (env-overridable via TUZKAOCR_*), so any config stays reproducible.
"""

from __future__ import annotations


def build_providers(device: str, cuda_opts: dict | None = None) -> list:
    """Provider list for an InferenceSession. On CUDA, attach the memory-bounding options."""
    if device != "cuda":
        return ["CPUExecutionProvider"]
    return [("CUDAExecutionProvider", cuda_opts or {}), "CPUExecutionProvider"]

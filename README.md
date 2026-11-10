# cuda-wbf

A research project to implement Weighted Boxes Fusion (WBF) for object detection in CUDA and measure when it reduces postprocessing latency versus the Python reference.

**Stretch goal:** Package the implementation as an importable Python module (`import cuda_wbf`) so it can be reused in other Python projects.

## Delivery status

Implementation is organized into four milestones, each delivered as a stack of focused pull requests. The first PR in a milestone targets `main`; each subsequent PR targets the preceding branch. Review and merge each stack from the bottom up.

Milestone 1 is implemented and tested in the open CPU stack below; it has not been merged into `main`. Milestones 2–4 are planned. Entries such as M2.1 are planning identifiers, not existing GitHub PR numbers.

| Milestone | Outcome | Status | Acceptance gate |
| --- | --- | --- | --- |
| 1. Build and CPU correctness | A usable scalar C++ backend with Python bindings and oracle verification | Implemented in [#1](https://github.com/ARPM01/cuda-wbf/pull/1), [#2](https://github.com/ARPM01/cuda-wbf/pull/2), and [#3](https://github.com/ARPM01/cuda-wbf/pull/3); awaiting review and merge | CPU fixture parity |
| 2. CUDA correctness | Both CUDA backends with reusable storage and shared arithmetic | Planned; follows milestone 1 | GPU fixture parity and Compute Sanitizer |
| 3. Real-data verification and comparisons | Reproducible COCO ingestion, evaluation, and NMS baselines | Planned; follows milestone 2 | Full-image parity and COCO AP agreement |
| 4. Performance study on SHARC | Reproducible latency measurements, profiling, and a results report | Planned; follows milestone 3 | Publish measurements only after parity passes |

## Milestone 1: Build and CPU correctness

| PR | Branch → base | Scope | Validation |
| --- | --- | --- | --- |
| [#1 — Build and packaging](https://github.com/ARPM01/cuda-wbf/pull/1) | `stack/cpu-01-build` → `main` | CMake 3.24+, C++17, Python 3.11, pybind11, pinned dependencies, CUDA disabled by default, prefix installation, and wheel packaging | Configure/build/install, extension smoke test, and wheel build |
| [#2 — Scalar fusion API](https://github.com/ARPM01/cuda-wbf/pull/2) | `stack/cpu-02-fusion` → `stack/cpu-01-build` | Validated `cuda_wbf.fuse` API, upstream prefilter and sort ordering, scalar matching against current representatives, reference arithmetic, and unavailable-backend errors | API checks and seeded oracle comparisons |
| [#3 — Verification and CI](https://github.com/ARPM01/cuda-wbf/pull/3) | `stack/cpu-03-verification` → `stack/cpu-02-fusion` | `verify --cpu-only`, named fixtures, randomized parity checks, invalid-input tests, Linux/macOS CI, and a machine validation report | Installed-wheel tests, CPU verification, and a Debug sanitizer build |

The CPU stack was tested on the development machine with Python 3.11.18, GCC 13.3, and CMake 3.28.3. All **210 tests** passed against the installed wheel from outside the checkout and against a Debug build with UndefinedBehaviorSanitizer. Verification covered **20 fixtures plus 120 randomized cases across 261 images**, with exact ordered labels/counts and zero observed coordinate or score error against the pinned oracle. These are local CPU results; GPU and real-data verification remain pending.

See the [CPU validation report](https://github.com/ARPM01/cuda-wbf/blob/691a37d913836b1446ac0cfe510d8f02f724e962/docs/CPU_VALIDATION.md). The verification entry point provided by this stack is:

```bash
python -m cuda_wbf.bench verify --cpu-only
```

**Gate:** require exact output counts and ordered labels, and coordinates/scores within `atol=1e-6`, `rtol=1e-5` against `ensemble-boxes==1.0.9`. Cover empty inputs, unequal weights, score and IoU boundaries, ties, moving representatives, and the slide example.

## Milestone 2: CUDA correctness

Start this stack from `main` after the CPU stack is merged. The branch names below are proposed.

| Planned PR | Branch → base | Scope | Validation |
| --- | --- | --- | --- |
| M2.1 — CUDA runtime and serial group kernel | `stack/cuda-01-groups` → `main` | Enable optional CUDA builds; accept caller-selected architectures; share cluster arithmetic with the CPU backend; implement checked reusable workspace, separate upload/launch/download operations, error handling, and `cuda_groups` with one thread per nonempty image/class group | Build with CUDA both OFF and ON; compare the group kernel with CPU and Python fixtures; verify empty calls launch no kernel |
| M2.2 — Cooperative group kernel | `stack/cuda-02-coop` → `stack/cuda-01-groups` | Implement `cuda_coop` with one 128-thread block per group, strided representative scans, highest-IoU/lowest-index reduction, and synchronized updates | Reuse the same fixtures; test ties, moving representatives, and groups exceeding 128 live clusters |
| M2.3 — GPU verification and safety | `stack/cuda-03-verification` → `stack/cuda-02-coop` | Extend verification to both GPU backends; exercise workspace reuse and alternating call sizes; document GPU safety commands and results | Oracle parity on both kernels plus Compute Sanitizer `memcheck` and `synccheck` |

**Gate:** both CUDA variants must pass the CPU fixture contract with exact counts/labels and the same numeric tolerances. Allocation, launch, and synchronization failures must reach the caller. Pass sanitizer checks before proceeding to real-data measurements.

## Milestone 3: Real-data verification and comparisons

| Planned PR | Branch → base | Scope | Validation |
| --- | --- | --- | --- |
| M3.1 — Pinned data ingestion | `stack/data-01-ingestion` → `main` | Stage the upstream prediction archive and COCO annotations; pin filenames, source order, weights, and thresholds to upstream commit `796d4b1`; preserve row order, float32 parsing, category IDs, and empty source slots; record checksums and effective configuration | Loader fixtures for malformed rows, clipping, discarded boxes/scores, missing detections, and deterministic ordering |
| M3.2 — Full-data parity and COCO evaluation | `stack/data-02-quality` → `stack/data-01-ingestion` | Extend `verify` with prediction/annotation arguments; compare every annotation image with Python WBF; convert normalized xyxy to pixel xywh for COCOeval; report AP, AP50, AP by object size, and AR100 | Full-val2017 counts, ordered labels, and numeric parity for both CUDA variants; absolute COCO AP difference at most `1e-4` versus Python |
| M3.3 — NMS comparison methods | `stack/data-03-baselines` → `stack/data-02-quality` | Add upstream CPU hard NMS, Gaussian Soft-NMS, and TorchVision CUDA hard NMS with a separate benchmark dependency group | Check class awareness, source-weight scaling, score floors, IoU/sigma settings, and evaluation of the same full image set |

**Gate:** every image must pass custom WBF parity, including images with empty outputs. Keep NMS quality results separate from the WBF acceptance gate. Use the same detections for all methods and real predictions for all quality measurements.

## Milestone 4: Performance study on SHARC

| Planned PR | Branch → base | Scope | Validation |
| --- | --- | --- | --- |
| M4.1 — Preflight and SHARC bring-up | `stack/bench-01-preflight` → `main` | Implement `preflight`; record GPU/driver/toolchain/dependency details; confirm site scheduler, partition, modules, and a one-GPU allocation; document offline staging when needed | Confirm both GPU backends load and rerun fixture/real-data gates in the target environment |
| M4.2 — Workloads and timing | `stack/bench-02-timing` → `stack/bench-01-preflight` | Implement deterministic real-data quartile selection and seed-42 synthetic cases; run all methods on identical batches; measure host-to-host latency and separately time resident CUDA kernels using events on the execution stream | Check batch selection and timing boundaries; exclude context setup, reusable allocation, file I/O, and evaluation; record actual transfer bytes |
| M4.3 — Runner and reports | `stack/bench-03-reporting` → `stack/bench-02-timing` | Complete `run`; save `timings.csv`, `metadata.json`, `quality.json`, and `summary.md`; report p50/p95, images per second, Python-relative speedups, and crossover workloads | Validate report calculations and metadata against raw samples; prevent publishing performance results before parity passes |
| M4.4 — Measured results and profiling | `stack/bench-04-results` → `stack/bench-03-reporting` | Execute the full study on one SHARC GPU; profile representative light/heavy cases; commit the reproducible report and instructions for locating run artifacts | Review parity evidence, raw-sample provenance, transfer costs, and the resulting speedup/crossover conclusions |

Use batch sizes **1, 8, and 32**, **five warmups**, and **30 timed samples** per case. The synthetic matrix uses **128, 512, and 2,048 boxes per image**, three equally weighted sources, one versus 16 classes, and overlapping versus disjoint boxes. Pin the Python reference to one worker.

**Gate:** publish performance only after correctness and quality gates pass. Distinguish host-to-host latency from kernel-only samples and include transfer costs. A result showing no host-to-host CUDA speedup still satisfies the measurement objective.

## Working with the stacks

1. Keep each PR focused on the row above and include its own validation evidence.
2. Base the first PR of each milestone on current `main`; base later PRs on their immediate predecessor.
3. Review and merge in dependency order. The current order is **#1 → #2 → #3**.
4. After a parent merges, retarget its child to `main` and update the remaining stack. If the parent was squash-merged or rebased, rebase descendants so their diffs contain only their own changes. Use force-with-lease if rewritten branches must be published.
5. Update this README when PRs open or merge, and attach gate evidence before moving to the next milestone.

## Technical reference

Start with the [implementation plan](IMPLEMENTATION_PLAN.md). It defines the reference behavior, the two CUDA kernels, the SHARC bring-up sequence, and the COCO benchmark protocol. The plan includes a CPU-only build for development on machines without CUDA.

The repository currently contains the plan. The implementation and benchmark runner are pending.

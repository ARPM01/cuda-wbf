# cuda-wbf

A mini project to implement Weighted Boxes Fusion (WBF) for object detection in CUDA and measure when it reduces postprocessing latency versus the Python reference.

**Stretch goal:** Package the implementation as an importable Python module (`import cuda_wbf`) so it can be reused in other Python projects.

## Delivery status

Implementation is organized into four milestones, each delivered as a stack of focused pull requests. The first PR in a milestone targets `main`; each subsequent PR targets the preceding branch. Review and merge each stack from the bottom up.

Milestones 1 and 2 are implemented and tested in the open CPU and CUDA stacks below; neither stack has been merged into `main`. Milestones 3–4 are planned. Entries such as M3.1 are planning identifiers, not existing GitHub PR numbers.

| Milestone | Outcome | Status | Acceptance gate |
| --- | --- | --- | --- |
| 1. Build and CPU correctness | A usable scalar C++ backend with Python bindings and oracle verification | Implemented in [#1](https://github.com/ARPM01/cuda-wbf/pull/1), [#2](https://github.com/ARPM01/cuda-wbf/pull/2), and [#3](https://github.com/ARPM01/cuda-wbf/pull/3); awaiting review and merge | CPU fixture parity |
| 2. CUDA correctness | Both CUDA backends with reusable storage and shared arithmetic | Implemented in [#5](https://github.com/ARPM01/cuda-wbf/pull/5), [#6](https://github.com/ARPM01/cuda-wbf/pull/6), and [#7](https://github.com/ARPM01/cuda-wbf/pull/7); awaiting review and merge | GPU fixture parity and Compute Sanitizer passed locally |
| 3. Real-data verification and comparisons | Reproducible COCO ingestion, evaluation, and NMS baselines | Planned; follows milestone 2 | Full-image parity and COCO AP agreement |
| 4. Performance study on SHARC | Reproducible latency measurements, profiling, and a results report | Planned; follows milestone 3 | Publish measurements only after parity passes |

## Milestone 1: Build and CPU correctness

| PR | Branch → base | Scope | Validation |
| --- | --- | --- | --- |
| [#1 — Build and packaging](https://github.com/ARPM01/cuda-wbf/pull/1) | `stack/cpu-01-build` → `main` | CMake 3.24+, C++17, Python 3.11, pybind11, pinned dependencies, CUDA disabled by default, prefix installation, and wheel packaging | Configure/build/install, extension smoke test, and wheel build |
| [#2 — Scalar fusion API](https://github.com/ARPM01/cuda-wbf/pull/2) | `stack/cpu-02-fusion` → `stack/cpu-01-build` | Validated `cuda_wbf.fuse` API, upstream prefilter and sort ordering, scalar matching against current representatives, reference arithmetic, and unavailable-backend errors | API checks and seeded oracle comparisons |
| [#3 — Verification and CI](https://github.com/ARPM01/cuda-wbf/pull/3) | `stack/cpu-03-verification` → `stack/cpu-02-fusion` | `verify --cpu-only`, named fixtures, randomized parity checks, invalid-input tests, Linux/macOS CI, and a machine validation report | Installed-wheel tests, CPU verification, and a Debug sanitizer build |

The CPU stack was tested on the development machine with Python 3.11.18, GCC 13.3, and CMake 3.28.3. All **210 tests** passed against the installed wheel from outside the checkout and against a Debug build with UndefinedBehaviorSanitizer. Verification covered **20 fixtures plus 120 randomized cases across 261 images**, with exact ordered labels/counts and zero observed coordinate or score error against the pinned oracle. These are the CPU stack's local results. GPU fixture validation is reported under milestone 2; real-data verification remains pending.

See the [CPU validation report](https://github.com/ARPM01/cuda-wbf/blob/691a37d913836b1446ac0cfe510d8f02f724e962/docs/CPU_VALIDATION.md). The verification entry point provided by this stack is:

```bash
python -m cuda_wbf.bench verify --cpu-only
```

**Gate:** require exact output counts and ordered labels, and coordinates/scores within `atol=1e-6`, `rtol=1e-5` against `ensemble-boxes==1.0.9`. Cover empty inputs, unequal weights, score and IoU boundaries, ties, moving representatives, and the slide example.

## Milestone 2: CUDA correctness

The CUDA stack currently builds on CPU PR #3 because the CPU stack remains open. PR #5 temporarily targets `stack/cpu-03-verification`; retarget it to `main` after the CPU stack merges. PRs #6 and #7 target their immediate predecessors.

| PR | Branch → base | Scope | Validation |
| --- | --- | --- | --- |
| [#5 — CUDA runtime and serial group kernel](https://github.com/ARPM01/cuda-wbf/pull/5) | `stack/cuda-01-groups` → `stack/cpu-03-verification` | Enable optional CUDA builds; accept caller-selected architectures; share cluster arithmetic with the CPU backend; implement checked reusable workspace, separate upload/launch/download operations, error handling, and `cuda_groups` with one thread per nonempty image/class group | Build with CUDA both OFF and ON; compare the group kernel with CPU and Python fixtures; verify empty calls launch no kernel |
| [#6 — Cooperative group kernel](https://github.com/ARPM01/cuda-wbf/pull/6) | `stack/cuda-02-coop` → `stack/cuda-01-groups` | Implement `cuda_coop` with one 128-thread block per group, strided representative scans, highest-IoU/lowest-index reduction, and synchronized updates | Reuse the same fixtures; test ties, moving representatives, and groups exceeding 128 live clusters |
| [#7 — GPU verification and safety](https://github.com/ARPM01/cuda-wbf/pull/7) | `stack/cuda-03-verification` → `stack/cuda-02-coop` | Extend verification to both GPU backends; exercise workspace reuse and alternating call sizes; document GPU safety commands and results | Oracle parity on both kernels plus Compute Sanitizer `memcheck` and `synccheck` |

The installed CUDA wheel passed **504 tests** on one A100 with CUDA 12.8. CPU-only regression testing passed **211 tests**. All three backends checked the same **261 images** against the oracle with exact counts/labels and zero observed coordinate or score error. Compute Sanitizer memcheck and synccheck each passed **55 fixture/safety tests** with zero errors; memcheck also reported zero leaked bytes. An additional **53-test racecheck** pass reported zero hazards. Its selection excludes Python thread-local teardown and the CLI subprocess because the tested racecheck run stalled in thread lifecycle; both tests passed normal, memcheck, and synccheck runs.

See the [CUDA validation report](https://github.com/ARPM01/cuda-wbf/blob/9c5001bc0609091728dc3dadc276724a820af281/docs/CUDA_VALIDATION.md) for commands, environment, logs, and coverage. These shared-device correctness runs make no performance claim.

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
3. Review and merge in dependency order. The current order is **#1 → #2 → #3 → #5 → #6 → #7**.
4. After a parent merges, retarget its child to `main` and update the remaining stack. If the parent was squash-merged or rebased, rebase descendants so their diffs contain only their own changes. Use force-with-lease if rewritten branches must be published.
5. Update this README when PRs open or merge, and attach gate evidence before moving to the next milestone.

## Technical reference

Start with the [implementation plan](IMPLEMENTATION_PLAN.md). It defines the reference behavior, the two CUDA kernels, the SHARC bring-up sequence, and the COCO benchmark protocol. The plan includes a CPU-only build for development on machines without CUDA.

The first implementation milestone provides a CPU-only C++17 fusion backend. CUDA backends and the performance runner are pending.

Install conda (Miniconda or Miniforge) and a C++17 compiler such as GCC or Clang. From the repository root, create and activate the Python 3.11 environment:

```bash
conda env create --file environment.yml
conda activate cuda-wbf
```

[environment.yml](environment.yml) installs CMake 3.24+, Ninja, and the pinned runtime and development dependencies from `pyproject.toml`. It does not install the project itself. Run `conda activate cuda-wbf` in each new shell before building or testing, and use `conda deactivate` when finished.

Build and test the CPU backend in the activated environment:

```bash
cmake -S . -B build/cpu -DCMAKE_BUILD_TYPE=Release -DWBF_ENABLE_CUDA=OFF \
  -DPython_EXECUTABLE="$(command -v python)" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build/cpu --parallel
cmake --install build/cpu --prefix "$PWD/build/cpu/stage"
export PYTHONPATH="$PWD/build/cpu/stage"
python -m pytest
python -m cuda_wbf.bench verify --cpu-only
```

CMake installs the Python package and extension together directly under the supplied prefix. As an alternative, build and install a wheel in the activated environment:

```bash
unset PYTHONPATH
python -m pip install '.[dev]'
python -m pytest
python -m cuda_wbf.bench verify --cpu-only
```

Unsetting `PYTHONPATH` ensures these commands use the installed wheel rather than a previous CMake staging directory. CUDA is disabled by default for both installation methods. If an existing CMake build was configured with another Python environment, use a fresh build directory so CMake discovers the conda interpreter.

Call `cuda_wbf.fuse(images, weights, backend="cpp", iou_thr=0.7, skip_box_thr=0.001)` with one list of sources per image. Each source is a `(boxes, scores, labels)` tuple: normalized `(N, 4)` xyxy boxes, `(N,)` scores in `(0, 1]`, and `(N,)` integer labels. Keep every configured source slot, using correctly shaped empty arrays when necessary. The result contains one score-ordered tuple per image. Boxes and scores are float64; labels are int64.

Inputs must be finite, coordinates must be in `[0, 1]` with positive box area, and source weights must be positive with a finite sum. Weights are canonicalized to float64; input score precision is preserved for the reference prefilter's threshold comparison. The API rejects invalid data before calling the upstream prefilter. Class grouping and both score sorts reuse the pinned reference behavior. The native loop preserves singleton precision and rounds merged coordinates and scores as the reference does. Requests for `cuda_groups` or `cuda_coop` currently raise an actionable error.

`verify --cpu-only` requires no COCO data or GPU. It checks the pinned oracle versions, runs 20 named fixtures and 120 seeded randomized cases, and prints a JSON report with the maximum errors and environment versions. Verification requires exact output shapes and ordered labels plus `atol=1e-6`, `rtol=1e-5` for coordinates and scores. Pytest also covers invalid inputs, independent geometry expectations, repeated calls, and the installed CLI. The full data loader, CUDA verification, `preflight`, and performance `run` commands remain later milestones.

See [CPU validation results](docs/CPU_VALIDATION.md) for the tests run on the development machine and sanitizer reproduction commands.

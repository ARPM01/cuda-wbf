# CUDA milestone validation

Validated on the development machine on 2026-11-10. These are fixture and safety results. No COCO assets, detector inference, or performance measurements were used.

## Environment

| Component | Value |
| --- | --- |
| Host | `ai-n002.hpc.coe.upd.edu.ph`, Linux x86_64, kernel 6.8.0-124-generic |
| GPU | NVIDIA A100-SXM4-40GB, compute capability 8.0 |
| Device selection | `CUDA_VISIBLE_DEVICES=6`, exposed to the process as ordinal 0 |
| NVIDIA driver | 580.159.03 |
| CUDA Toolkit | 12.8.93 |
| CUDA runtime / driver API | 12080 / 13000 |
| Compute Sanitizer | 2025.1.0.0, build 35583870 |
| Host compiler / CMake | GCC 13.3.0 / CMake 3.28.3 |
| Python / pybind11 | 3.11.18 / 2.13.6 |
| Oracle dependencies | ensemble-boxes 1.0.9, NumPy 2.2.6, Numba 0.61.2 |

All devices were shared with existing work. Fixture runs used one selected device directly; no Slurm allocation was active and the configured controller did not respond. SHARC allocation bring-up and latency measurement remain later work. These results establish no speedup or latency baseline.

## Build and package checks

- Release CMake builds with CUDA OFF and ON succeeded. The tested CUDA build used explicit architecture `80`.
- A separate CUDA configuration with `CMAKE_CUDA_ARCHITECTURES=native` succeeded with the GPU visible.
- Both the default CPU wheel and the CUDA-enabled wheel built and installed into separate Python 3.11 environments.
- From `/tmp`, with `PYTHONPATH` unset, the installed CPU wheel passed **211 tests** (295 GPU tests deselected). The installed CUDA wheel passed **504 tests**, with two unavailable-backend checks skipped because those backends are present.

CUDA wheels were built with:

```bash
python -m pip wheel --no-deps --no-build-isolation . --wheel-dir build/cuda-wheels \
  -Ccmake.define.WBF_ENABLE_CUDA=ON -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=80
```

The local environments were `/tmp/cuda-wbf-py311` for builds, `/tmp/cuda-wbf-wheel-test` for the CPU wheel, and `/tmp/cuda-wbf-cuda-wheel-test` for the CUDA wheel.

## Oracle parity

`python -m cuda_wbf.bench verify` passed the same 20 named fixtures and 120 seed-42 randomized cases on every backend. Output counts and ordered labels matched exactly. Required numeric tolerances were `atol=1e-6`, `rtol=1e-5`.

| Backend | Images checked | Maximum coordinate error | Maximum score error |
| --- | --- | --- | --- |
| `cpp` | 261 | 0.0 | 0.0 |
| `cuda_groups` | 261 | 0.0 | 0.0 |
| `cuda_coop` | 261 | 0.0 | 0.0 |

The installed CUDA wheel's JSON report is saved locally at `build/cuda/verification.json`.

## Safety checks

The suite additionally covers resident relaunches, changing candidate/group counts on one workspace, empty runs with no launch, exact padded transfer accounting, overflow recovery, shared-workspace serialization, independent per-thread public workspaces, and teardown with a weak-reference assertion after garbage collection. Dedicated cases require matching beyond slot 128, breaking equal-IoU ties between slots 127 and 128, and launching 257 groups.

| Tool | Tests | Result | Successful local log |
| --- | --- | --- | --- |
| memcheck, including full leak checking | 55 named-fixture/safety tests | 0 errors; 0 bytes leaked in 0 allocations | `build/cuda/sanitizer/memcheck.1313021.log` |
| synccheck | 55 named-fixture/safety tests | 0 errors | `build/cuda/sanitizer/synccheck.1314317.log` |
| racecheck, additional kernel/workspace coverage | 53 tests | 0 hazards; 0 errors; 0 warnings | `build/cuda/sanitizer/racecheck.1328718.log` |

The full normal test run includes all randomized cases. Memcheck and synccheck include the Python thread-lifecycle and child CLI tests. On this toolchain, racecheck stalled in the public API's thread-local workspace lifecycle test, so its narrower selection excludes that test and the subprocess CLI check. The validation script makes this selection explicit. No racecheck claim is made for those two tests.

To reproduce after building and staging the CUDA package, select one GPU and run from the repository root:

```bash
export PYTHONPATH="$PWD/build/cuda/stage"
# Set CUDA_VISIBLE_DEVICES to the GPU assigned to your session before running.
python -m cuda_wbf.bench verify
bash scripts/check_cuda.sh test
bash scripts/check_cuda.sh memcheck
bash scripts/check_cuda.sh synccheck
bash scripts/check_cuda.sh racecheck
```

Set `WBF_PYTHON` to the intended interpreter if needed. The script sets the GPU-test opt-in and a single Numba worker. Sanitizers return exit code 99 on detected errors and retain per-run logs. Real-data parity, COCO AP agreement, and performance reporting remain subsequent acceptance gates.

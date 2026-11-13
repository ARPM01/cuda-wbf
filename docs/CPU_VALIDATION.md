# CPU milestone validation

Validated on the development machine on 2026-11-10. No GPU allocation or COCO assets were used.

| Component | Version |
| --- | --- |
| OS | Linux x86_64, kernel 6.8.0-124-generic, glibc 2.39 |
| Python | 3.11.18, isolated environment at `/tmp/cuda-wbf-py311` |
| Host compiler | GCC 13.3.0 |
| CMake | 3.28.3 |
| pybind11 | 2.13.6 |
| ensemble-boxes | 1.0.9 |
| NumPy | 2.2.6 |
| Numba | 0.61.2 |
| pytest | 8.3.5 |

## Results

- Release CMake configuration, build, and installation under `build/cpu/stage` passed.
- Wheel build and installation into a separate environment at `/tmp/cuda-wbf-wheel-test` passed. Tests ran from `/tmp` with `PYTHONPATH` unset and imported the installed extension from that environment's `site-packages`.
- All 210 pytest tests passed against the installed wheel.
- The Debug build with UndefinedBehaviorSanitizer passed all 210 tests without sanitizer diagnostics.
- `python -m cuda_wbf.bench verify --cpu-only` passed 20 named fixtures and 120 randomized cases (seed 42), covering 261 images. Ordered labels and output shapes matched exactly. Maximum coordinate and score errors were both **0.0**, within the required `atol=1e-6`, `rtol=1e-5`.

The fixtures cover empty inputs, multiple images/classes, source weighting and absent sources, score and IoU boundaries, candidate and output score ties, best-match and equal-IoU decisions, moving representatives, singleton precision, more than 128 live clusters, and the slide example. Pytest additionally checks invalid inputs, repeated calls of alternating sizes, version mismatch rejection, and CLI exit codes.

## Reproduction

From the repository root, create the conda environment with `conda env create --file environment.yml` if it does not exist, then follow the Release build and test commands in [README.md](../README.md#technical-reference). The results above describe the original validation environment; the conda environment provides the same Python minor version and pinned Python dependencies for reproduction.

To check undefined behavior using GCC or Clang, activate the environment and configure a separate build:

```bash
conda activate cuda-wbf
cmake -S . -B build/cpu-ubsan -DCMAKE_BUILD_TYPE=Debug -DWBF_ENABLE_CUDA=OFF \
  -DPython_EXECUTABLE="$(command -v python)" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  '-DCMAKE_CXX_FLAGS=-fsanitize=undefined -fno-sanitize-recover=all' \
  -DCMAKE_SHARED_LINKER_FLAGS=-fsanitize=undefined
cmake --build build/cpu-ubsan --parallel 4
cmake --install build/cpu-ubsan --prefix "$PWD/build/cpu-ubsan/stage"
PYTHONPATH="$PWD/build/cpu-ubsan/stage" NUMBA_NUM_THREADS=1 python -m pytest -q
```

GitHub Actions checks the wheel and CMake installation on Linux and macOS. The local results above establish the CPU fixture gate; CUDA safety, real-data parity, COCO quality, and performance remain later gates.

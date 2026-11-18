# cuda-wbf

Weighted Boxes Fusion (WBF) for object detection, packaged as `cuda_wbf`. It provides a scalar C++ backend and two optional CUDA backends.

## SHARC benchmark results

On one NVIDIA A100, `cuda_coop` lowered median host-to-host latency across all 48 workloads compared with single-worker Python WBF. It was **7.87×–15.18× faster** across the 12 real COCO workloads. Both CUDA backends matched the Python reference on all **5,000 COCO val2017 images** and achieved **55.82% COCO AP**. [Benchmark setup and results](docs/SHARC_BENCHMARK.md) · [Full timing report](docs/benchmarks/sharc-a100/summary.md)

## Build and verify

Use Python 3.11, CMake 3.24+, and a C++17 compiler. From the repository root, create the environment and install the CPU build:

```bash
conda env create --file environment.yml
conda activate cuda-wbf
python -m pip install '.[dev,evaluation]'
python -m pytest
python -m cuda_wbf.bench verify --cpu-only
```

### CUDA build

Use a driver-compatible CUDA Toolkit 11.8 or newer. Select the GPU with `CUDA_VISIBLE_DEVICES` before configuring with `native` and running verification. When building without a visible GPU, use an explicit compute capability such as `80` for A100.

```bash
cmake -S . -B build/cuda -DCMAKE_BUILD_TYPE=Release -DWBF_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native -DPython_EXECUTABLE="$(command -v python)" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build/cuda --parallel
cmake --install build/cuda --prefix "$PWD/build/cuda/stage"
export PYTHONPATH="$PWD/build/cuda/stage"
WBF_TEST_CUDA=1 python -m pytest
python -m cuda_wbf.bench verify
```

See the [CPU validation report](docs/CPU_VALIDATION.md) and [CUDA validation report](docs/CUDA_VALIDATION.md) for parity and sanitizer checks.

## Python API

Call `cuda_wbf.fuse(images, weights, backend="cpp")` with one ordered list of sources per image and one positive weight per source. Each source is a `(boxes, scores, labels)` tuple with normalized `(N, 4)` xyxy boxes, `(N,)` scores in `(0, 1]`, and `(N,)` integer labels. Keep empty source slots with the same array shapes. The function returns one score-ordered tuple per image.

Use `cuda_wbf.available_backends()` to check which backends are compiled. `cuda_groups` uses one thread per image/class group. `cuda_coop` uses one block per group to search cluster representatives cooperatively. [CUDA runtime design](docs/CUDA_RUNTIME.md)

## COCO benchmark prerequisites

Stage the [prediction archive](https://github.com/ZFTurbo/Weighted-Boxes-Fusion/releases/download/v1.0.5/benchmark.zip) at `data/benchmark.zip`. Extract `annotations/instances_val2017.json` from the [COCO 2017 annotations archive](https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip) to `data/instances_val2017.json`.

Install the evaluation dependency and the CUDA-enabled TorchVision baseline dependencies. These commands use the CUDA 12.8 wheels from the SHARC run:

```bash
python -m pip install pycocotools==2.0.11
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

See the [SHARC benchmark guide](docs/SHARC_BENCHMARK.md#full-study) for the timing and profiling commands, or the [COCO validation report](docs/COCO_VALIDATION.md) for quality checks.

## Repository structure

- [`src/`](src/) contains the scalar C++ implementation, CUDA kernels and workspace, shared fusion arithmetic, and pybind11 bindings.
- [`python/cuda_wbf/`](python/cuda_wbf/) contains the Python API, input validation, COCO evaluation, NMS baselines, and benchmark CLI.
- [`tests/`](tests/) covers API behavior, reference parity, CUDA safety, and the benchmark tooling.
- [`scripts/`](scripts/) contains the CUDA test and Compute Sanitizer runner.
- [`docs/`](docs/) contains the runtime design and validation guides. [`docs/benchmarks/sharc-a100/`](docs/benchmarks/sharc-a100/) holds the published timings, profiles, quality metrics, and run metadata.
- [`CMakeLists.txt`](CMakeLists.txt), [`pyproject.toml`](pyproject.toml), and [`environment.yml`](environment.yml) define the native build, Python package, and development environment. [`.github/workflows/`](.github/workflows/) contains the CPU CI workflow.

See the [implementation plan](IMPLEMENTATION_PLAN.md) for the algorithm and benchmark protocol.

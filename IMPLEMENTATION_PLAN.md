# CUDA WBF implementation and benchmark plan

## Goal and scope

Reproduce standard Weighted Boxes Fusion (WBF) on saved COCO detections and measure when CUDA reduces host-to-host postprocessing latency relative to single-worker Python WBF. The first GPU study runs on one SHARC GPU. Compare two CUDA variants that differ in how they search for a matching cluster. Keep detector inference and training outside the experiment.

Slides 5–9 describe the algorithm dependency and slides 10–13 outline the experiment. The 98-image Colab run on slides 15–16 is historical context. Its exact inputs and settings were unavailable for verification. Use the public upstream archive for the main study. [Project slides](https://docs.google.com/presentation/d/1wIYOvnkF0MnmgdjBX9AYkE6BM6w6X9TcZxBZdVCsZCA/edit) · [WBF paper](https://arxiv.org/abs/1910.13302)

## 1. Establish the build and reference contract

- Build a C++17 core with CMake 3.24+ and expose it through pybind11 to Python 3.11. Default `WBF_ENABLE_CUDA` to `OFF` so the scalar C++ backend and its tests build on macOS or Linux without a CUDA Toolkit. With CUDA enabled, use a driver-compatible CUDA Toolkit from 11.8 onward and let CMake accept `CMAKE_CUDA_ARCHITECTURES` from the caller. Use `native` in a GPU allocation and an explicit architecture when building without a visible GPU. [CMake CUDA architecture settings](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html)
- Pin the Python oracle to `ensemble-boxes==1.0.9`, `numpy==2.2.6`, and `numba==0.61.2`. Call standard `ensemble_boxes.weighted_boxes_fusion` per image. Support class-aware 2D WBF with `conf_type="avg"`, `allows_overflow=False`, and positive source weights. Default to `iou_thr=0.7` and `skip_box_thr=0.001`. Use the scalar C++ backend to validate the native arithmetic on CPU. Use the Python oracle for parity and the primary speed comparison. [Reference package](https://pypi.org/project/ensemble-boxes/1.0.9/) · [WBF source](https://github.com/ZFTurbo/Weighted-Boxes-Fusion/blob/master/ensemble_boxes/ensemble_boxes_wbf.py)
- Expose `cuda_wbf.fuse(images, weights, *, backend, iou_thr=0.7, skip_box_thr=0.001)`. Each image contains one `(boxes, scores, labels)` tuple per source in configured order. Use `(N, 4)` normalized `xyxy` coordinates, `(N,)` positive scores in `(0, 1]`, and `(N,)` integer labels. Preserve empty source slots. Validate shapes, finite values, positive box area, and weights and thresholds at the boundary. Return one score-ordered `(boxes, scores, labels)` tuple per image, including correctly shaped empty results. Support `cpp`, `cuda_groups`, and `cuda_coop` as backend names. Raise an actionable error when a requested backend is unavailable.
- Reuse the pinned upstream prefilter and NumPy sort to produce candidates in descending weighted score order for each image/class. Preserve its first-seen class order and its `argsort()[::-1]` tie ordering. Pack candidates into contiguous group slices. After native fusion, concatenate clusters in class order and cluster creation order, then use the same NumPy descending sort for final scores.

## 2. Implement and verify both CUDA variants

Implement the scalar C++ loop first. For each candidate, compare against the **current** representatives of the same group. Choose the highest IoU strictly greater than `iou_thr`. Break equal-IoU ties by earliest cluster creation. Create a cluster when no representative qualifies. Update the selected representative before processing the next candidate.

Share cluster math across all native backends. Keep the running sum of weighted scores in float64 and accumulate weighted coordinates in membership order, rounding each addition to float32 to match the reference. For a singleton, preserve the input coordinates and weighted score. After a merge, round the divided coordinates and mean weighted score to float32. Use float64 for IoU and final score scaling. Scale the mean by `min(cluster_size, num_sources) / sum(weights)` using all configured sources, including absent ones. Disable floating-point contraction and use precise division in the native builds. Check arithmetic and output ordering against the pinned oracle before benchmarking.

| Backend | Work assignment | Match and update |
| --- | --- | --- |
| `cuda_groups` | One CUDA thread per nonempty image/class group | Scan current representatives serially in the thread. |
| `cuda_coop` | One 128-thread block per nonempty image/class group | Scan cluster slots in strides of 128 and reduce `(IoU, cluster index)` by highest IoU and lowest index. One thread updates the winning cluster or appends a new one. Synchronize before reading the updated representative or reusing reduction storage. |

Use a shared flat layout and reusable CUDA workspace with input slices, up to one cluster slot per input box, and a valid cluster count per group. Check sizes before allocation and reset per-run state. Return empty batches without launching kernels. Surface CUDA allocation, launch, and synchronization errors to the caller. Separate upload, launch, and download internally so the benchmark can time the resident kernel using the same implementation as the public API. Keep prefiltering and sorting on the host for both variants.

## 3. Build the experiment runner

### Data and comparisons

- Stage `benchmark.zip` and COCO `instances_val2017.json`. Take the active filenames, source order, and model weights from upstream commit `796d4b1`. Use its WBF thresholds with the standard oracle. Record the asset checksums, effective configuration, and image IDs in the run metadata. [Prediction archive](https://github.com/ZFTurbo/Weighted-Boxes-Fusion/releases/download/v1.0.5/benchmark.zip) · [COCO downloads](https://cocodataset.org/#download) · [Pinned upstream configuration](https://github.com/ZFTurbo/Weighted-Boxes-Fusion/blob/796d4b1b6125ebef795cf14363dae00da78d5a1c/benchmark_coco/run_benchmark_coco.py)
- Load predictions once into memory. Preserve CSV row order within each image/source, parse boxes and scores to float32 as in the upstream adapter, reject non-finite values, discard reversed or degenerate boxes and nonpositive scores, and clip coordinates to `[0, 1]`. Preserve original COCO category IDs and a slot for each source even when it has no detections. Evaluate the full `val2017` annotation image set for every method. Convert normalized `xyxy` to pixel `xywh` only for evaluation. Use COCOeval's standard detection limits and include images with empty outputs. [COCO evaluator source](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)
- Compare the two CUDA variants and single-worker Python WBF on the same SHARC node. Run upstream CPU hard NMS and Gaussian Soft-NMS plus TorchVision CUDA hard NMS for latency and quality context. Give the NMS methods the same input detections and raw score floor of `0.001`. Use class-aware hard NMS at IoU `0.5`, with scores weighted by `source_weight / sum(weights)` on both CPU and GPU. Use Gaussian Soft-NMS with `sigma=0.1` and output score floor `0.0001`. Keep TorchVision in a separate benchmark dependency group. [TorchVision batched NMS](https://docs.pytorch.org/vision/stable/generated/torchvision.ops.batched_nms.html)

### Workloads, timing, and outputs

- Use batch sizes `1`, `8`, and `32`. For real latency tests, sort images by eligible box count and then by image ID, split into four equal-rank quartiles, and partition each quartile into complete batches. Select up to four evenly spaced batches from each quartile and batch size and cycle through them identically across methods. For synthetic tests, use seed `42` and `128`, `512`, and `2,048` boxes per image with three equally weighted sources. Cross one class versus 16 classes with overlapping versus disjoint boxes. Use real predictions for all quality measurements.
- Use five warmups and 30 timed samples per case. Pin the CPU reference to one worker. Measure host-to-host latency with a wall clock from in-memory detections through score filtering, grouping, sorting, host packing, transfers, synchronization, and host-ready outputs where applicable. Exclude context setup, reusable buffer allocation, file I/O, and COCO evaluation. Separately time the CUDA kernels on already uploaded, grouped, and sorted inputs using events on the execution stream. Label those samples as kernel-only. Record the actual transfer bytes, including any padded output. [CUDA event timing](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html)
- Save per-sample `timings.csv`, `metadata.json`, `quality.json`, and `summary.md`. Include backend, timing boundary, workload, seed or image IDs, input and output counts, transfer bytes, commit, dependency versions, and hardware details. Report p50 and p95 latency, images per second, and host-to-host speedup relative to single-worker Python WBF. Report whether and where each CUDA variant crosses below the reference p50. Include COCO AP, AP50, AP for small, medium, and large objects, and AR100. Profile representative light and heavy cases after parity passes.

## 4. Bring up on SHARC and apply acceptance gates

Implement the runner as `python -m cuda_wbf.bench` with `preflight`, `verify`, and `run` subcommands. Use the planned invocation below inside a one-GPU allocation. Discover SHARC's scheduler, partition, and module settings from the site before running. Stage the Python wheels and data on a host with network access if compute nodes are offline. Select a compatible TorchVision CUDA build for the detected driver. Keep the run commands independent of the scheduler.

```bash
# Planned commands once the implementation is present and dependencies are installed.
nvidia-smi
nvcc --version
cmake --version
python --version
cmake -S . -B build/cuda -DCMAKE_BUILD_TYPE=Release -DWBF_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native -DPython_EXECUTABLE="$(command -v python)" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build/cuda --parallel
cmake --install build/cuda --prefix "$PWD/build/cuda/stage"
export PYTHONPATH="$PWD/build/cuda/stage"
python -m cuda_wbf.bench preflight
python -m cuda_wbf.bench verify --predictions data/benchmark.zip \
  --annotations data/instances_val2017.json
python -m cuda_wbf.bench run --predictions data/benchmark.zip \
  --annotations data/instances_val2017.json --output runs/sharc
```

Have CMake install the Python package and extension together under the specified prefix. `preflight` records the GPU, compute capability, driver, `nvcc`, host compiler, CMake, Python, and dependency versions and checks that the requested GPU backends load. For CPU development, use a separate build and install directory, set `WBF_ENABLE_CUDA=OFF`, and run `verify --cpu-only` without the COCO arguments for fixture tests.

1. **CPU fixture parity.** Cover empty batches and sources, multiple images and classes, unequal weights, scores below and at the floor, IoU equal to the threshold, best-match and score ties, and the moving-representative dependency. Include the slide example at IoU threshold `0.55`, where boxes `(20,20,60,60)` and `(25,25,65,65)` with scores `0.90` and `0.60` fuse to `(22,22,62,62)` after normalizing and converting back to pixels.
2. **CUDA fixture parity and safety.** Run the same cases on both kernels, including a group with more than 128 live clusters, and alternate differently sized calls on one workspace. Run Compute Sanitizer `memcheck` and `synccheck` on the GPU fixture suite. Require exact output counts and ordered labels, and coordinates and scores within `atol=1e-6` and `rtol=1e-5` against Python.
3. **Real-data parity.** Apply the same output checks to every image in the COCO workload. Require an absolute COCO AP difference of at most `1e-4` for each custom variant versus Python. Keep NMS quality results separate from the WBF parity gate.
4. **Performance report.** Publish timings after parity passes. Identify which workloads benefit and include the transfer costs. A result showing no host-to-host speedup on the tested GPU satisfies the measurement objective.

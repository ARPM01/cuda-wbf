# SHARC benchmark setup

Use one A100 and one CPU worker. Run `sinfo -o '%P %G %t'` to see active partitions and `scontrol ping` to check the controller. The configuration visible from `ai-n002` lists the `gpu` partition on `ai-n003` with A100 GPUs. Confirm the active partition before submitting a job.

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=1 --mem=16G --time=02:00:00 --pty bash
```

Slurm sets `CUDA_VISIBLE_DEVICES` inside the allocation. For an authorized direct assignment, set it to the assigned GPU and record the assignment with the run. Use `nvidia-smi` to check whether the assigned device is idle before measurements.

Activate the Python 3.11 environment from the [README](../README.md#technical-reference). Use `nvcc --version` and `nvidia-smi` to check the toolkit and driver. If the toolkit is provided as a module, use `module avail cuda` to find its local name. Build and install the CUDA package with the README's CMake commands. Install the evaluation and CUDA TorchVision dependencies from the [Milestone 3 setup](../README.md#milestone-3-real-data-verification-and-comparisons).

For offline compute nodes, stage the environment, wheels, and both data assets on a shared filesystem from a network-enabled host. Install staged wheels with `python -m pip install --no-index --find-links /path/to/wheels`, specifying the matching CUDA versions of torch and torchvision. Keep each run's JSON and CSV artifacts on shared storage.

```bash
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m cuda_wbf.bench preflight
```

Preflight runs one actual fusion through each selected backend and prints the GPU, build, available toolchain versions, dependencies, and CPU affinity as JSON.

## Full study

```bash
python -m cuda_wbf.bench run \
  --predictions data/benchmark.zip \
  --annotations data/instances_val2017.json \
  --output runs/sharc
```

Use a new output directory each time. The command runs fixture and full-data parity and quality checks before timing. It reads both assets once and writes the per-sample CSV, metadata, quality results, and Markdown summary together when the run succeeds. Use `--allocation-note` to record a direct GPU assignment.

The real cases use equal-rank detection-count quartiles and up to four evenly spaced complete batches per quartile and batch size. The synthetic matrix uses seed 42 and crosses three detection counts, two class counts, two overlap conditions, and three batch sizes. Each method cycles through the same batches for five warmups and 30 samples after setup.

Host-to-host timing starts with in-memory detections and ends with host-ready outputs. Each batch shape is exercised before warmup to initialize contexts and reserve reusable device buffers. Kernel-only samples bracket the resident kernel with CUDA events on its execution stream. The buffer resets precede the start event. The raw CSV records input and output counts and actual host transfers. For TorchVision the download consists of the selected indices and the host gathers boxes and scores from its original arrays.

## Representative kernel profiles

Run the profiler after the main study has finished. Use the same CUDA build and device. The light case is one image with 128 overlapping boxes over 16 classes. The heavy case is a batch of 32 images with 2,048 disjoint boxes per image in one class. Each capture contains 30 resident launches per WBF kernel, with output resets and the final readback visible in the trace.

```bash
for case in light heavy; do
  nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --output "runs/sharc/profile-${case}" \
    python -m cuda_wbf.bench profile --results runs/sharc --case "$case"
  nsys stats --report cuda_gpu_kern_sum "runs/sharc/profile-${case}.nsys-rep"
done
```

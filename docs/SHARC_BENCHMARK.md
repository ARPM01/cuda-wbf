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

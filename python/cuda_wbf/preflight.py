"""Inspect the benchmark host and exercise the selected native backends."""

from importlib.metadata import PackageNotFoundError, version
import os
import platform
import socket
import subprocess

import numpy as np

from . import _native, available_backends, build_info, fuse

CUDA_BACKENDS = ("cuda_groups", "cuda_coop")


def _probe(*command):
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return process.stdout.strip()


def preflight(*, cpu_only=False):
    required = ("cpp",) if cpu_only else ("cpp", *CUDA_BACKENDS)
    if not set(required).issubset(available_backends()):
        raise RuntimeError("CUDA preflight requires both CUDA backends. Build with WBF_ENABLE_CUDA=ON.")

    image = [(np.array([[0., 0., 1., 1.]]), np.array([0.9]), np.array([1]))]
    for backend in required:
        result, = fuse([image], [1.], backend=backend)
        if not all(np.array_equal(actual, expected) for actual, expected in zip(result, image[0])):
            raise RuntimeError(f"{backend} failed the one-box preflight check")

    dependencies = {}
    for package in ("cuda-wbf", "ensemble-boxes", "numpy", "numba", "pybind11", "pycocotools", "torch", "torchvision"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = None

    return {
        "status": "passed", "host": socket.gethostname(), "platform": platform.platform(),
        "python": platform.python_version(), "build": build_info(), "dependencies": dependencies,
        "backends_checked": list(required), "device": None if cpu_only else _native.cuda_device_info(),
        "cpu_model": _cpu_model(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "tools": {
            "cmake": _probe("cmake", "--version"),
            "host_compiler": _probe("c++", "--version"),
            "nvcc": _probe("nvcc", "--version"),
            "nvidia_smi": _probe("nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total",
                                 "--format=csv,noheader"),
        },
        "environment": {key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_JOB_GPUS",
            "NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        )},
    }


def _cpu_model():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.partition(":")[2].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()

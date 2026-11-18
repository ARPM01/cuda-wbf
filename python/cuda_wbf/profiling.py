"""Capture two resident-kernel workloads with Nsight Systems."""

import hashlib
import json
from pathlib import Path

from . import _native
from .preflight import CUDA_BACKENDS
from .timing import SAMPLES, resident_workspaces
from .workloads import synthetic_workload

PROFILE_CASES = {
    "light": (128, 16, "overlapping", 1),
    "heavy": (2048, 1, "disjoint", 32),
}


def profile_case(results, case_name):
    quality = json.loads((results / "quality.json").read_text(encoding="utf-8"))
    metadata = json.loads((results / "metadata.json").read_text(encoding="utf-8"))
    if quality.get("status") != "passed" or quality.get("images") != 5000:
        raise ValueError("profiling requires a passing full COCO study")
    with Path(_native.__file__).open("rb") as stream:
        native_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if metadata.get("native_sha256") != native_hash:
        raise ValueError("profile with the same native build used by the study")
    if metadata.get("device") != json.loads(json.dumps(_native.cuda_device_info())):
        raise ValueError("profile on the same CUDA device configuration used by the study")

    import torch

    case = synthetic_workload(*PROFILE_CASES[case_name])
    (workspace, _), = resident_workspaces(case)
    for backend in CUDA_BACKENDS:
        workspace.time_kernel(backend)
        workspace.download()
    with torch.cuda.profiler.profile():
        for backend in CUDA_BACKENDS:
            with torch.cuda.nvtx.range(f"{case.name}/{backend}"):
                for _ in range(SAMPLES):
                    workspace.launch(backend)
                workspace.download()
    return {"status": "passed", "workload": case.name, "launches_per_backend": SAMPLES}

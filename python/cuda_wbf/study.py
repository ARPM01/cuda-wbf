"""Run the full benchmark after fixture and COCO verification."""

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys

import numba

from . import _native
from .coco_data import load_coco_data
from .coco_quality import verify_coco_data
from .preflight import _probe, preflight
from .reporting import write_results
from .timing import SAMPLES, WARMUPS, host_methods, measure_case
from .verification import verify_cuda
from .workloads import real_workloads, synthetic_workloads


def run_study(predictions, annotations, output, *, allocation_note=None):
    if output.exists():
        raise FileExistsError(f"choose a new output directory: {output}")
    started = datetime.now(timezone.utc).isoformat()
    environment = preflight()
    commit = _probe("git", "rev-parse", "HEAD")
    worktree_status = _probe("git", "status", "--porcelain")
    with Path(_native.__file__).open("rb") as stream:
        native_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    numba.set_num_threads(1)
    fixture_report = verify_cuda()
    print("Loading COCO annotations and predictions", file=sys.stderr)
    data = load_coco_data(predictions, annotations)
    if len(data.image_ids) != 5000:
        raise ValueError("the full study requires all 5,000 COCO val2017 images")
    quality, metadata = verify_coco_data(data, fixture_report, baselines=True)

    import torch  # The verified CUDA baseline has already loaded this extra.

    torch.set_num_threads(1)
    methods = host_methods(environment["device"]["device"])
    workloads = [*real_workloads(data), *synthetic_workloads()]
    rows = []
    for index, case in enumerate(workloads, 1):
        print(f"Timing {index}/{len(workloads)} {case.name}", file=sys.stderr, flush=True)
        rows.extend(measure_case(case, methods))
    settings = [
        {"name": case.name, "kind": case.kind, "batch_size": case.batch_size, **case.parameters,
         "batches": [{"image_ids": batch.image_ids, "input_boxes": batch.input_boxes,
                      "eligible_boxes": batch.eligible_boxes} for batch in case.batches]}
        for case in workloads
    ]
    metadata.update({
        "commit": commit, "worktree_dirty": bool(worktree_status) if worktree_status is not None else None,
        "native_sha256": native_sha256, "environment": environment, "allocation_note": allocation_note,
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(), "fixture_verification": fixture_report,
        "timing": {"warmups": WARMUPS, "samples": SAMPLES, "python_workers": numba.get_num_threads(),
                   "torch_cpu_workers": torch.get_num_threads(), "workloads": settings},
    })
    write_results(output, rows, metadata, quality)
    return {"status": "passed", "output": str(output), "workloads": len(workloads), "samples": len(rows)}

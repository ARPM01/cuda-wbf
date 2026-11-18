"""Run the full benchmark or extend a completed study with scalar timings."""

import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numba

from . import _native
from .coco_data import load_coco_data
from .coco_quality import verify_coco_data
from .preflight import _probe, preflight
from .reporting import write_results
from .timing import HOST, SAMPLES, WARMUPS, host_methods, measure_case, measure_host, scalar_wbf
from .verification import verify_cpu, verify_cuda
from .workloads import real_workloads, synthetic_workloads


def _native_hash():
    with Path(_native.__file__).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _workload_settings(workloads):
    return [
        {"name": case.name, "kind": case.kind, "batch_size": case.batch_size, **case.parameters,
         "batches": [{"image_ids": list(batch.image_ids), "input_boxes": batch.input_boxes,
                      "eligible_boxes": batch.eligible_boxes} for batch in case.batches]}
        for case in workloads
    ]


def run_study(predictions, annotations, output, *, allocation_note=None):
    if output.exists():
        raise FileExistsError(f"choose a new output directory: {output}")
    started = datetime.now(timezone.utc).isoformat()
    environment = preflight()
    commit = _probe("git", "rev-parse", "HEAD")
    worktree_status = _probe("git", "status", "--porcelain")
    native_sha256 = _native_hash()
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
    metadata.update({
        "commit": commit, "worktree_dirty": bool(worktree_status) if worktree_status is not None else None,
        "native_sha256": native_sha256, "environment": environment, "allocation_note": allocation_note,
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(), "fixture_verification": fixture_report,
        "timing": {"warmups": WARMUPS, "samples": SAMPLES, "python_workers": numba.get_num_threads(),
                   "torch_cpu_workers": torch.get_num_threads(), "workloads": _workload_settings(workloads)},
    })
    write_results(output, rows, metadata, quality)
    return {"status": "passed", "output": str(output), "workloads": len(workloads), "samples": len(rows)}


def _archived_rows(results):
    path = results / "timings.csv"
    if path.exists():
        stream = path.open(newline="", encoding="utf-8")
    else:
        stream = gzip.open(results / "timings.csv.gz", "rt", newline="", encoding="utf-8")
    with stream:
        return list(csv.DictReader(stream))


def append_cpp(results, predictions, annotations, output):
    if output.exists():
        raise FileExistsError(f"choose a new output directory: {output}")
    started = datetime.now(timezone.utc).isoformat()
    metadata = json.loads((results / "metadata.json").read_text(encoding="utf-8"))
    quality = json.loads((results / "quality.json").read_text(encoding="utf-8"))
    rows = _archived_rows(results)
    if any(row["backend"] == "cpp" for row in rows):
        raise ValueError("the previous study already contains scalar C++ timings")
    cpp_quality = quality.get("methods", {}).get("cpp", {})
    if quality.get("status") != "passed" or quality.get("images") != 5000 or cpp_quality.get("images") != 5000:
        raise ValueError("the previous study must include passing scalar C++ verification on all 5,000 COCO images")
    native_sha256 = _native_hash()
    if metadata.get("native_sha256") != native_sha256:
        raise ValueError("use the same native build as the previous study")

    timing = metadata["timing"]
    warmups, samples = timing["warmups"], timing["samples"]
    if warmups < 0 or samples <= 0 or not timing["workloads"]:
        raise ValueError("the previous study has invalid timing settings")
    reference = {}
    for row in rows:
        if row["backend"] == "python" and row["boundary"] == HOST:
            key = row["workload"], int(row["sample"])
            if key in reference:
                raise ValueError(f"duplicate Python timing sample for {key[0]} at index {key[1]}")
            reference[key] = row

    environment = preflight(cpu_only=True)
    numba.set_num_threads(1)
    fixture_report = verify_cpu()
    print("Loading COCO annotations and predictions", file=sys.stderr, flush=True)
    data = load_coco_data(predictions, annotations)
    for asset in ("predictions", "annotations"):
        if data.provenance["assets"][asset]["sha256"] != metadata["assets"][asset]["sha256"]:
            raise ValueError(f"{asset} checksum differs from the previous study")
    if data.provenance["wbf"] != metadata["wbf"]:
        raise ValueError("WBF settings differ from the previous study")
    workloads = [*real_workloads(data), *synthetic_workloads()]
    if _workload_settings(workloads) != timing["workloads"]:
        raise ValueError("reconstructed workloads differ from the previous study")
    if reference.keys() != {(case.name, sample) for case in workloads for sample in range(samples)}:
        raise ValueError("the previous study has an incomplete or unexpected set of Python samples")

    for index, case in enumerate(workloads, 1):
        print(f"Timing scalar C++ {index}/{len(workloads)} {case.name}", file=sys.stderr, flush=True)
        for row in measure_host(case, "cpp", scalar_wbf, warmups, samples):
            expected = reference[case.name, row["sample"]]
            if any(str(value) != expected[key] for key, value in row.items()
                   if key not in ("backend", "latency_ms")):
                raise ValueError(f"{case.name} sample {row['sample']} differs from the archived Python batch or output count")
            rows.append(row)

    worktree_status = _probe("git", "status", "--porcelain")
    added = len(workloads) * samples
    timing["cpp_followup"] = {
        "commit": _probe("git", "rev-parse", "HEAD"),
        "worktree_dirty": bool(worktree_status) if worktree_status is not None else None,
        "native_sha256": native_sha256, "environment": environment,
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "fixture_verification": fixture_report, "samples_added": added,
    }
    write_results(output, rows, metadata, quality)
    return {"status": "passed", "output": str(output), "workloads": len(workloads),
            "samples_added": added, "samples": len(rows)}

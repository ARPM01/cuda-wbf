"""Host-to-host and resident-kernel benchmark timings."""

from functools import partial
import json
from time import perf_counter_ns

from . import _native, fuse
from .api import _cuda_workspace, _prepare, _unpack
from .baselines import cpu_hard_nms, cpu_soft_nms, torchvision_cuda_nms
from .coco_data import IOU_THR, SCORE_FLOOR
from .preflight import CUDA_BACKENDS
from .verification import compare_image, oracle

WARMUPS = 5
SAMPLES = 30
HOST = "host_to_host"
KERNEL = "kernel_only"


def host_methods(device_index):
    cuda_nms = torchvision_cuda_nms(device_index)
    return {
        "python": lambda images, weights: [oracle(image, weights, IOU_THR, SCORE_FLOOR) for image in images],
        **{backend: partial(fuse, backend=backend, iou_thr=IOU_THR, skip_box_thr=SCORE_FLOOR)
           for backend in CUDA_BACKENDS},
        "cpu_hard_nms": lambda images, weights: [cpu_hard_nms(image, weights) for image in images],
        "cpu_soft_nms": lambda images, weights: [cpu_soft_nms(image, weights) for image in images],
        "torchvision_cuda_nms": lambda images, weights: [cuda_nms(image, weights) for image in images],
    }


def _row(case, batch, slot, backend, boundary, sample, elapsed, count, upload=0, download=0):
    return {
        "workload": case.name, "kind": case.kind, "batch_size": case.batch_size,
        "batch_slot": slot, "image_ids": json.dumps(batch.image_ids, separators=(",", ":")),
        **{key: case.parameters.get(key, "") for key in ("quartile", "seed", "boxes_per_image", "classes", "geometry")},
        "backend": backend, "boundary": boundary, "sample": sample, "latency_ms": elapsed,
        "input_boxes": batch.input_boxes, "eligible_boxes": batch.eligible_boxes, "output_boxes": count,
        "upload_bytes": upload, "download_bytes": download,
    }


def _outputs_count(outputs):
    return sum(len(result[1]) for result in outputs)


def _transfer_bytes(backend, batch, output_count):
    if backend in CUDA_BACKENDS and batch.eligible_boxes:
        stats = _cuda_workspace().stats
        return stats["upload_bytes"], stats["download_bytes"]
    if backend == "torchvision_cuda_nms":
        # Each filtered row uploads four float64 coordinates, one float64
        # score and one int64 label. The host receives int64 keep indices.
        return 48 * batch.eligible_boxes, 8 * output_count
    return 0, 0


def _samples(case, backend, run, warmups, samples):
    batches = case.batches
    # Warm each selected shape before either warmups or samples so reusable
    # CUDA allocations and first-use compilation occur outside measurements.
    for batch in batches:
        run(batch.images, case.weights)
    capacity = _cuda_workspace().stats["allocated_bytes"] if backend in CUDA_BACKENDS else None
    for index in range(warmups):
        run(batches[index % len(batches)].images, case.weights)
    for index in range(samples):
        slot = index % len(batches)
        batch = batches[slot]
        before = perf_counter_ns()
        outputs = run(batch.images, case.weights)
        elapsed = (perf_counter_ns() - before) / 1e6
        if capacity is not None and _cuda_workspace().stats["allocated_bytes"] != capacity:
            raise RuntimeError(f"{backend} allocated a reusable device buffer during timing")
        count = _outputs_count(outputs)
        upload, download = _transfer_bytes(backend, batch, count)
        yield _row(case, batch, slot, backend, HOST, index, elapsed, count, upload, download)


def resident_workspaces(case):
    for batch in case.batches:
        packed = _prepare(batch.images, case.weights, IOU_THR, SCORE_FLOOR)
        workspace = _native.CudaWorkspace()
        workspace.upload(*packed.native_args)
        yield workspace, packed


def measure_case(case, methods, warmups=WARMUPS, samples=SAMPLES):
    expected = [methods["python"](batch.images, case.weights) for batch in case.batches]
    workspaces = list(resident_workspaces(case))
    counts = []
    for (workspace, packed), outputs in zip(workspaces, expected, strict=True):
        counts.append(_outputs_count(outputs))
        for backend in CUDA_BACKENDS:
            workspace.time_kernel(backend)
            actual = _unpack(packed, *workspace.download())
            for index, (result, reference) in enumerate(zip(actual, outputs, strict=True)):
                compare_image(result, reference, f"{case.name}, {backend}, image {index}")
    for backend, run in methods.items():
        yield from _samples(case, backend, run, warmups, samples)
    for backend in CUDA_BACKENDS:
        for index in range(warmups):
            workspaces[index % len(workspaces)][0].time_kernel(backend)
        for index in range(samples):
            slot = index % len(workspaces)
            elapsed = workspaces[slot][0].time_kernel(backend)
            yield _row(case, case.batches[slot], slot, backend, KERNEL, index, elapsed, counts[slot])

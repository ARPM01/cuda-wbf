"""Deterministic fixtures and oracle parity checks, without a pytest dependency."""

from dataclasses import dataclass
from importlib.metadata import version
import platform

import numpy as np
from ensemble_boxes import weighted_boxes_fusion

from . import _native, available_backends, build_info, fuse

ATOL = 1e-6
RTOL = 1e-5
PINS = {"ensemble-boxes": "1.0.9", "numpy": "2.2.6", "numba": "0.61.2"}


@dataclass
class Case:
    name: str
    images: list
    weights: list
    iou_thr: float = 0.7
    skip_box_thr: float = 0.001


def source(boxes=(), scores=(), labels=None, dtype=np.float64):
    boxes = np.asarray(boxes, dtype=dtype).reshape(-1, 4)
    scores = np.asarray(scores, dtype=dtype)
    labels = np.zeros(len(boxes), np.int64) if labels is None else np.asarray(labels, np.int64)
    return boxes, scores, labels


def fixture_cases():
    whole = [0, 0, 1, 1]
    pair = [[0, 0, 0.5, 1], [0.125, 0, 0.625, 1], [0.25, 0, 0.75, 1]]
    many = [[(i % 16) / 16, (i // 16) / 16,
             (i % 16) / 16 + 0.02, (i // 16) / 16 + 0.02] for i in range(160)]
    return [
        Case("empty_batch", [], [1, 1]),
        Case("empty_sources", [[source(), source()]], [1, 2]),
        Case("multiple_images_classes_and_empty_slots", [
            [source([whole, whole], [0.9, 0.7], [7, 2]), source([whole], [0.8], [7])],
            [source(), source()],
            [source(), source([[0.1, 0.1, 0.3, 0.3]], [0.6], [2])],
        ], [2, 1]),
        Case("slide_example", [[
            source([[0.2, 0.2, 0.6, 0.6]], [0.9]),
            source([[0.25, 0.25, 0.65, 0.65]], [0.6]),
        ]], [1, 1], iou_thr=0.55),
        Case("score_floor", [[source([whole] * 3, [0.0005, 0.001, 0.0015], [1, 2, 3])]], [1]),
        Case("all_scores_filtered", [[source([whole], [0.0005])]], [1]),
        Case("float32_score_floor", [[source([whole], [0.5], dtype=np.float32)]], [1], skip_box_thr=0.5000000001),
        Case("iou_equal_threshold", [[source([whole, [0, 0, 0.5, 1]], [0.9, 0.8])]], [1], iou_thr=0.5),
        Case("iou_zero_disjoint", [[source([[0, 0, 0.25, 1], [0.5, 0, 0.75, 1]], [1, 1])]], [1], iou_thr=0),
        Case("iou_one_identical", [[source([whole, whole], [1, 1])]], [1], iou_thr=1),
        Case("best_match", [[source([[0, 0, 0.4, 1], [0.6, 0, 1, 1], [0.2, 0, 1, 1]], [0.9, 0.8, 0.7])]], [1], iou_thr=0.15),
        Case("equal_iou_earliest_cluster", [[source([[0, 0, 0.375, 1], [0.625, 0, 1, 1], [0.25, 0, 0.75, 1]], [0.95, 0.8, 0.6])]], [1], iou_thr=0.1),
        Case("score_ties_and_first_seen_classes", [[source(many[:40], [0.5] * 40, [9, 2] * 20)]], [1]),
        Case("moving_representative_includes", [[source(pair, [0.9, 0.8, 0.7])]], [1], iou_thr=0.4),
        Case("moving_representative_excludes", [[source(pair, [0.9, 0.8, 0.7])]], [1], iou_thr=0.5),
        Case("absent_source_weight", [[source([whole], [0.9]), source(), source()]], [2, 1, 4]),
        Case("cluster_size_over_source_count", [[source([whole] * 3, [0.9, 0.8, 0.7])]], [1]),
        Case("more_than_128_clusters", [[source(many, np.linspace(0.1, 0.9, len(many))), source()]], [1, 1]),
        Case("singleton_precision", [[source([[0.123456789123, 0.1, 0.765432198765, 0.9]], [0.987654321987])]], [1]),
        Case("unequal_fractional_weights", [[source([whole], [0.9]), source([whole], [0.6]), source()]], [0.3, 2.7, 0.9]),
    ]


def randomized_cases(count=120, seed=42):
    rng = np.random.default_rng(seed)
    for case_index in range(count):
        sources = 1 + case_index % 4
        images = []
        dtype = np.float32 if case_index % 2 else np.float64
        for _ in range(1 + case_index % 3):
            image = []
            for _ in range(sources):
                size = int(rng.integers(0, 150))
                xy = rng.uniform(0, 0.7, (size, 2))
                wh = rng.uniform(0.02, 0.3, (size, 2))
                boxes = np.concatenate([xy, xy + wh], axis=1)
                scores = (rng.choice([0.0005, 0.001, 0.25, 0.5, 1], size) if case_index % 3 == 0
                          else rng.uniform(0.0001, 1, size))
                labels = rng.integers(0, 1 if case_index % 5 == 0 else 5, size)
                image.append(source(boxes, scores, labels, dtype))
            images.append(image)
        weights = [1.0] * sources if case_index % 2 else rng.uniform(0.1, 3, sources).tolist()
        yield Case(f"random_{seed}_{case_index}", images, weights, iou_thr=[0, 0.3, 0.7, 1][case_index % 4])


def check_case(case, backend="cpp"):
    actual = fuse(case.images, case.weights, backend=backend,
                  iou_thr=case.iou_thr, skip_box_thr=case.skip_box_thr)
    if len(actual) != len(case.images):
        raise AssertionError(f"{case.name}: incorrect number of output images")
    max_box_error = max_score_error = 0.0
    for image_index, (image, result) in enumerate(zip(case.images, actual)):
        expected = weighted_boxes_fusion(
            *zip(*image), weights=np.asarray(case.weights, np.float64),
            iou_thr=case.iou_thr, skip_box_thr=case.skip_box_thr,
            conf_type="avg", allows_overflow=False,
        )
        context = f"{case.name}, image {image_index}"
        for output, reference in zip(result, expected):
            if output.shape != reference.shape:
                raise AssertionError(f"{context}: output shape {output.shape} != {reference.shape}")
        np.testing.assert_array_equal(result[2], expected[2], err_msg=context)
        for output, reference in zip(result[:2], expected[:2]):
            np.testing.assert_allclose(output, reference, atol=ATOL, rtol=RTOL, err_msg=context)
        if not (result[0].dtype == result[1].dtype == np.float64 and result[2].dtype == np.int64):
            raise AssertionError(f"{context}: incorrect output dtypes")
        if not (np.diff(result[1]) <= 0).all():
            raise AssertionError(f"{context}: outputs are not ordered by descending score")
        max_box_error = max(max_box_error, float(np.abs(result[0] - expected[0]).max(initial=0)))
        max_score_error = max(max_score_error, float(np.abs(result[1] - expected[1]).max(initial=0)))
    return max_box_error, max_score_error


def _dependencies():
    dependencies = {name: version(name) for name in PINS}
    if dependencies != PINS:
        raise RuntimeError(f"Oracle version mismatch: expected {PINS}, installed {dependencies}")
    return dependencies


def _verify_backend(backend, fixtures_only=False):
    dependencies = _dependencies()
    fixtures = fixture_cases()
    random_cases = [] if fixtures_only else list(randomized_cases())
    errors = [check_case(case, backend=backend) for case in fixtures + random_cases]
    return {
        "status": "passed", "backend": backend, "fixture_cases": len(fixtures),
        "randomized_cases": len(random_cases), "seed": 42,
        "images": sum(len(case.images) for case in fixtures + random_cases),
        "atol": ATOL, "rtol": RTOL,
        "max_coordinate_error": max(error[0] for error in errors),
        "max_score_error": max(error[1] for error in errors),
        "python": platform.python_version(), "platform": platform.platform(),
        "dependencies": dependencies, "build": build_info(),
    }


def verify_cpu(fixtures_only=False):
    return _verify_backend("cpp", fixtures_only)


def verify_cuda(fixtures_only=False):
    _dependencies()
    required = ("cpp", "cuda_groups", "cuda_coop")
    if any(backend not in available_backends() for backend in required):
        raise RuntimeError("GPU verification requires a CUDA build with both backends; use verify --cpu-only for a CPU build")
    device = _native.cuda_device_info()
    reports = [_verify_backend(backend, fixtures_only) for backend in required]
    return {"status": "passed", "device": device, "backends": reports}

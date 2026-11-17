"""Deterministic real and synthetic benchmark workloads."""

from dataclasses import dataclass
from itertools import product
import math

import numpy as np

from .coco_data import SCORE_FLOOR

BATCH_SIZES = (1, 8, 32)
BOX_COUNTS = (128, 512, 2048)
SEED = 42


def eligible_boxes(image):
    return sum(int(np.count_nonzero(scores >= SCORE_FLOOR)) for _, scores, _ in image)


@dataclass
class Batch:
    images: list
    image_ids: tuple
    input_boxes: int
    eligible_boxes: int


def _batch(images, image_ids=()):
    return Batch(images, tuple(image_ids), sum(len(source[1]) for image in images for source in image),
                 sum(eligible_boxes(image) for image in images))


@dataclass
class Workload:
    name: str
    kind: str
    batch_size: int
    batches: list
    weights: np.ndarray
    parameters: dict


def real_workloads(data, batch_sizes=BATCH_SIZES):
    counts = [eligible_boxes(image) for image in data.images]
    ranked = sorted(range(len(counts)), key=lambda i: (counts[i], data.image_ids[i]))
    for quartile, indices in enumerate(np.array_split(ranked, 4), 1):
        for size in batch_sizes:
            complete = len(indices) // size
            if not complete:
                continue
            # Include both endpoints and evenly spaced complete batches between.
            starts = np.linspace(0, complete - 1, num=min(4, complete), dtype=int) * size
            batches = []
            for start in starts:
                chosen = indices[start:start + size]
                batches.append(_batch([data.images[i] for i in chosen], [data.image_ids[i] for i in chosen]))
            yield Workload(f"real-q{quartile}-b{size}", "real", size, batches, data.weights, {"quartile": quartile})


def synthetic_workload(box_count, classes, geometry, batch_size):
    # Reinitializing the seed uses the same image prefix at each batch size.
    rng = np.random.default_rng(SEED)
    images = []
    for _ in range(batch_size):
        if geometry == "overlapping":
            boxes = np.array([0.2, 0.2, 0.8, 0.8]) + rng.uniform(-0.01, 0.01, (box_count, 4))
        else:
            side = math.isqrt(box_count - 1) + 1
            cells = rng.permutation(side * side)[:box_count]
            corners = np.column_stack((cells % side, cells // side)) / side
            boxes = np.column_stack((corners, corners + 0.75 / side))
        scores = rng.uniform(0.05, 1.0, box_count).astype(np.float32)
        labels = rng.permutation(np.arange(box_count, dtype=np.int64) % classes)
        images.append(list(zip(np.array_split(boxes.astype(np.float32), 3),
                               np.array_split(scores, 3), np.array_split(labels, 3))))
    parameters = {"seed": SEED, "boxes_per_image": box_count, "classes": classes, "geometry": geometry}
    name = f"synthetic-{box_count}-c{classes}-{geometry}-b{batch_size}"
    return Workload(name, "synthetic", batch_size, [_batch(images)], np.ones(3, dtype=np.float64), parameters)


def synthetic_workloads():
    for box_count, classes, geometry, batch_size in product(
            BOX_COUNTS, (1, 16), ("overlapping", "disjoint"), BATCH_SIZES):
        yield synthetic_workload(box_count, classes, geometry, batch_size)

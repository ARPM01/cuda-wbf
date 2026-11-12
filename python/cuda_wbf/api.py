"""Validated public API and reference-compatible host preprocessing."""

import numpy as np
from ensemble_boxes.ensemble_boxes_wbf import prefilter_boxes

from . import _native


def _empty():
    return np.empty((0, 4), np.float64), np.empty(0, np.float64), np.empty(0, np.int64)


def _real_array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numbers")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


def _threshold(value, name):
    array = _real_array(value, name)
    if array.ndim != 0 or not 0 <= array.item() <= 1:
        raise ValueError(f"{name} must be a scalar in [0, 1]")
    return array.item()


def _validate(images, weights, iou_thr, skip_box_thr):
    # Each weight belongs to a source slot, so all images use the same source order.
    # Keep the full weight sum for the final confidence calculation, even when
    # some sources have no detections for a particular image.
    weights = np.asarray(_real_array(weights, "weights"), dtype=np.float64)
    if weights.ndim != 1 or weights.size == 0 or not (weights > 0).all():
        raise ValueError("weights must be a nonempty 1D array of positive source weights")
    with np.errstate(over="ignore"):
        weight_sum = weights.sum()
    if not np.isfinite(weight_sum):
        raise ValueError("the sum of weights must be finite")
    iou_thr = _threshold(iou_thr, "iou_thr")
    skip_box_thr = _threshold(skip_box_thr, "skip_box_thr")
    validated = []
    for image_index, image in enumerate(images):
        if len(image) != weights.size:
            raise ValueError(f"image {image_index} must have {weights.size} source slots, including empty sources")
        sources = []
        for source_index, source in enumerate(image):
            context = f"image {image_index}, source {source_index}"
            if len(source) != 3:
                raise ValueError(f"{context} must be a (boxes, scores, labels) tuple")
            boxes = _real_array(source[0], f"{context} boxes")
            scores = _real_array(source[1], f"{context} scores")
            labels = np.asarray(source[2])
            if boxes.ndim != 2 or boxes.shape[1] != 4:
                raise ValueError(f"{context} boxes must have shape (N, 4)")
            if scores.shape != (len(boxes),) or labels.shape != (len(boxes),):
                raise ValueError(f"{context} scores and labels must have shape (N,)")
            if labels.dtype.kind not in "iu" or (labels.size and labels.max() > np.iinfo(np.int64).max):
                raise ValueError(f"{context} labels must be integers representable as int64")
            if not ((boxes >= 0) & (boxes <= 1)).all():
                raise ValueError(f"{context} boxes must be normalized to [0, 1]")
            if not ((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])).all():
                raise ValueError(f"{context} boxes must have positive area and ordered xyxy coordinates")
            if not ((scores > 0) & (scores <= 1)).all():
                raise ValueError(f"{context} scores must be in (0, 1]")
            # Preserve the input score dtype until prefiltering. Converting a
            # float32 score to float64 here could change a near-threshold decision.
            sources.append((boxes, scores, labels.astype(np.int64, copy=False)))
        validated.append(sources)
    return validated, weights, weight_sum, iou_thr, skip_box_thr


def fuse(images, weights, *, backend, iou_thr=0.7, skip_box_thr=0.001):
    """Fuse a batch of images with ordered source slots.

    Each source is ``(boxes, scores, labels)`` with normalized xyxy boxes of
    shape (N, 4), scores in (0, 1], and integer labels of shape (N,). Empty
    sources must retain these shapes. Returns one score-ordered tuple per
    image, with float64 boxes/scores and int64 labels. Implements standard
    class-aware WBF with conf_type='avg' and allows_overflow=False.
    """
    if backend not in ("cpp", "cuda_groups", "cuda_coop"):
        raise ValueError("backend must be 'cpp', 'cuda_groups', or 'cuda_coop'")
    if backend != "cpp":
        raise RuntimeError(f"{backend} is unavailable in this CPU build. Use backend='cpp'; CUDA backends are pending milestone 2.")
    images, weights, weight_sum, iou_thr, skip_box_thr = _validate(images, weights, iou_thr, skip_box_thr)
    # Flatten the nonempty (image, class) groups into one array for C++.
    # Each pair of offsets delimits a slice of candidate rows. For example,
    # [0, 2, 5] describes two groups, with rows [0:2] and [2:5].
    packed = []
    offsets = [0]
    # Python keeps the class label for each group and the range of groups for
    # each image. The same label in two images belongs to two different groups.
    group_labels = []
    image_groups = [0]
    for image in images:
        boxes, scores, labels = zip(*image)
        groups = prefilter_boxes(boxes, scores, labels, weights, skip_box_thr)
        for label, candidates in groups.items():
            # The prefilter applies the score cutoff, multiplies scores by source
            # weights, and sorts each class by weighted score. Keep its ordering
            # for ties. Pass only [weighted_score, x1, y1, x2, y2] to C++.
            packed.append(candidates[:, [1, 4, 5, 6, 7]])
            offsets.append(offsets[-1] + len(candidates))
            group_labels.append(label)
        image_groups.append(len(group_labels))
    if not packed:
        return [_empty() for _ in images]
    candidates = np.ascontiguousarray(np.concatenate(packed), dtype=np.float64)
    values, output_offsets = _native.fuse_groups(
        candidates, np.asarray(offsets, np.int64), len(weights), weight_sum, iou_thr
    )
    # A group can return fewer rows after fusion. The new offsets tell us how
    # many times to repeat its class label and where each image's output ends.
    labels = np.repeat(np.asarray(group_labels, np.int64), np.diff(output_offsets))
    results = []
    for first, last in zip(image_groups[:-1], image_groups[1:]):
        start, end = output_offsets[first], output_offsets[last]
        # Before the final score sort, rows are in class order, then cluster
        # creation order. Starting from that order matches the reference on ties.
        order = values[start:end, 0].argsort()[::-1]
        rows = values[start:end][order]
        results.append((rows[:, 1:].copy(), rows[:, 0].copy(), labels[start:end][order]))
    return results

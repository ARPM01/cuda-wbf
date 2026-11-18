"""Class-aware NMS comparisons on one image of COCO predictions."""

import numpy as np
from ensemble_boxes import nms, soft_nms

from .coco_data import SCORE_FLOOR

NMS_IOU = 0.5
SOFT_NMS_SIGMA = 0.1
SOFT_NMS_FLOOR = 0.0001


def _weighted_inputs(image, weights):
    weight_sum = np.sum(weights, dtype=np.float64)
    boxes, scores, labels = [], [], []
    for (source_boxes, source_scores, source_labels), weight in zip(image, weights):
        keep = source_scores >= SCORE_FLOOR
        boxes.append(source_boxes[keep].astype(np.float64))
        scores.append(source_scores[keep].astype(np.float64) * weight / weight_sum)
        labels.append(source_labels[keep])
    return boxes, scores, labels


def _empty():
    return np.empty((0, 4), np.float64), np.empty(0, np.float64), np.empty(0, np.int64)


def cpu_hard_nms(image, weights):
    boxes, scores, labels = _weighted_inputs(image, weights)
    if not any(len(source) for source in boxes):
        return _empty()
    return nms(boxes, scores, labels, iou_thr=NMS_IOU)


def cpu_soft_nms(image, weights):
    boxes, scores, labels = _weighted_inputs(image, weights)
    if not any(len(source) for source in boxes):
        return _empty()
    return soft_nms(boxes, scores, labels, method=2, sigma=SOFT_NMS_SIGMA, thresh=SOFT_NMS_FLOOR)


def torchvision_cuda_nms(device_index):
    """Check the operator on the selected device and return an image callable."""
    try:
        import torch
        from torchvision.ops import batched_nms

        device = torch.device("cuda", device_index)
        scores = torch.ones(1, device=device, dtype=torch.float64)
        boxes = torch.tensor([[0., 0., 1., 1.]], device=device, dtype=torch.float64)
        labels = torch.zeros(1, device=device, dtype=torch.int64)
        batched_nms(boxes, scores, labels, NMS_IOU).cpu()
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError(
            "TorchVision CUDA hard NMS requires matching CUDA-enabled torch and torchvision "
            f"builds and a usable GPU (see the benchmark extra): {error}"
        ) from error

    def run(image, weights):
        inputs = _weighted_inputs(image, weights)
        if not any(len(source) for source in inputs[0]):
            return _empty()
        arrays = tuple(np.concatenate(sources) for sources in inputs)
        boxes, scores, labels = (torch.from_numpy(array).to(device) for array in arrays)
        keep = batched_nms(boxes, scores, labels, NMS_IOU).cpu().numpy()
        return tuple(array[keep] for array in arrays)

    return run

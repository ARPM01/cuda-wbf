"""Full-image oracle parity and COCO detection quality."""

from contextlib import redirect_stdout
from importlib.metadata import version
import sys

from . import fuse
from .coco_data import IOU_THR, SCORE_FLOOR, load_coco_data
from .verification import ATOL, RTOL, compare_image, oracle

AP_TOLERANCE = 1e-4
BATCH_SIZE = 32
METRIC_INDEX = {"AP": 0, "AP50": 1, "AP_small": 3, "AP_medium": 4, "AP_large": 5, "AR100": 8}


def detection_rows(image_id, dimensions, result):
    width, height = dimensions
    boxes, scores, labels = result
    for (x1, y1, x2, y2), score, label in zip(boxes.astype("float64", copy=False), scores, labels):
        yield {
            "image_id": image_id, "category_id": int(label), "score": float(score),
            "bbox": [x1 * width, y1 * height, (x2 - x1) * width, (y2 - y1) * height],
        }


class CocoEvaluator:
    def __init__(self, data):
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        except ImportError as error:
            raise RuntimeError("COCO verification requires pycocotools. Install cuda-wbf[evaluation].") from error
        self.data = data
        self.coco_eval = COCOeval
        self.truth = COCO()
        self.truth.dataset = data.annotations
        self.truth.createIndex()

    def evaluate(self, results):
        detections = [
            row
            for image_id, result in zip(self.data.image_ids, results, strict=True)
            for row in detection_rows(image_id, self.data.dimensions[image_id], result)
        ]
        if detections:
            detected = self.truth.loadRes(detections)
        else:
            # loadRes inspects the first detection to choose a result format.
            detected = type(self.truth)()
            detected.dataset = {**self.truth.dataset, "annotations": []}
            detected.createIndex()
        evaluator = self.coco_eval(self.truth, detected, iouType="bbox")
        evaluator.params.imgIds = self.data.image_ids.copy()
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        return {name: float(evaluator.stats[index]) for name, index in METRIC_INDEX.items()}


def _verify_wbf(data, backends):
    outputs = {backend: [] for backend in ("python", *backends)}
    errors = {backend: (0.0, 0.0) for backend in backends}
    for start in range(0, len(data.images), BATCH_SIZE):
        images = data.images[start:start + BATCH_SIZE]
        expected = [oracle(image, data.weights, IOU_THR, SCORE_FLOOR) for image in images]
        outputs["python"].extend(expected)
        for backend in backends:
            actual = fuse(images, data.weights, backend=backend, iou_thr=IOU_THR, skip_box_thr=SCORE_FLOOR)
            if len(actual) != len(images):
                raise AssertionError(f"{backend}: incorrect output image count starting at COCO image {data.image_ids[start]}")
            for index, (result, reference) in enumerate(zip(actual, expected)):
                context = f"{backend}, COCO image {data.image_ids[start + index]}"
                box_error, score_error = compare_image(result, reference, context)
                errors[backend] = max(box_error, errors[backend][0]), max(score_error, errors[backend][1])
            outputs[backend].extend(actual)
        checked = start + len(images)
        if checked % 512 == 0 or checked == len(data.images):
            print(f"Verified WBF parity on {checked}/{len(data.images)} COCO images", file=sys.stderr)
    return outputs, errors


def verify_coco(predictions, annotations, fixture_report):
    with redirect_stdout(sys.stderr):
        reports = fixture_report.get("backends", [fixture_report])
        backends = [report["backend"] for report in reports]
        device = fixture_report.get("device")
        dependencies = dict(reports[0]["dependencies"])

        print("Loading COCO annotations and predictions", file=sys.stderr)
        data = load_coco_data(predictions, annotations)
        evaluator = CocoEvaluator(data)
        outputs, errors = _verify_wbf(data, backends)
        quality = {}
        for name, results in outputs.items():
            print(f"Evaluating {name}", file=sys.stderr)
            quality[name] = {"metrics": evaluator.evaluate(results), "images": len(results)}
            if name in errors:
                ap_diff = abs(quality[name]["metrics"]["AP"] - quality["python"]["metrics"]["AP"])
                if ap_diff > AP_TOLERANCE:
                    raise AssertionError(f"{name}: COCO AP difference {ap_diff} exceeds {AP_TOLERANCE}")
                quality[name].update(
                    absolute_ap_difference=ap_diff, max_coordinate_error=errors[name][0],
                    max_score_error=errors[name][1],
                )

    build = reports[0]
    metadata = {
        **data.provenance, "dependencies": {**dependencies, "pycocotools": version("pycocotools")},
        "python": build["python"], "platform": build["platform"], "build": build["build"],
        "device": device, "wbf_backends": backends, "wbf_batch_size": BATCH_SIZE,
    }
    report = {
        "status": "passed", "images": len(data.images),
        "tolerances": {"atol": ATOL, "rtol": RTOL, "absolute_ap_difference": AP_TOLERANCE},
        "methods": quality,
    }
    return report, metadata

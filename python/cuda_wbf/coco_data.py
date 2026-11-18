"""Load the pinned upstream COCO benchmark into ordered WBF inputs."""

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
from zipfile import ZipFile

import numpy as np

UPSTREAM_COMMIT = "796d4b1b6125ebef795cf14363dae00da78d5a1c"
SOURCES = (
    ("EffNetB4-preds.csv", 4),
    ("EffNetB4-mirror-preds.csv", 4),
    ("EffNetB5-preds.csv", 5),
    ("EffNetB5-mirror-preds.csv", 5),
    ("EffNetB6-preds.csv", 7),
    ("EffNetB6-mirror-preds.csv", 7),
    ("EffNetB7-preds.csv", 9),
    ("EffNetB7-mirror-preds.csv", 9),
    ("DetRS-valid.csv", 8),
    ("DetRS-mirror-valid.csv", 8),
    ("DetRS_resnet50-valid.csv", 5),
    ("DetRS_resnet50-mirror-valid.csv", 5),
    ("yolov5x_tta.csv", 10),
)
IOU_THR = 0.7
SCORE_FLOOR = 0.001
BOX_COLUMNS = ("x1", "y1", "x2", "y2")
REQUIRED_COLUMNS = {*BOX_COLUMNS, "score", "img_id", "label"}


@dataclass
class CocoData:
    annotations: dict
    image_ids: list
    dimensions: dict
    images: list
    weights: np.ndarray
    provenance: dict


def _csv_id(value):
    try:
        return int(value)
    except ValueError:
        # Some upstream CSVs encode category IDs as decimals such as "1.0".
        try:
            number = Decimal(value)
        except InvalidOperation:
            raise ValueError(f"expected an integer ID, got {value!r}") from None
        if not (number.is_finite() and -(2**63) <= number < 2**63
                and number == number.to_integral_value()):
            raise ValueError(f"expected an integer ID, got {value!r}")
        return int(number)


def read_source(stream, filename, dimensions, categories):
    reader = csv.DictReader(stream, strict=True)
    rows = {}
    counts = dict(accepted=0, nonpositive_score=0, invalid_box=0, collapsed_after_clipping=0)
    try:
        header = reader.fieldnames or []
        if not REQUIRED_COLUMNS.issubset(header) or len(header) != len(set(header)):
            raise ValueError("missing required or duplicate CSV columns")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("wrong number of CSV fields")
            image_id = _csv_id(row["img_id"])
            category = _csv_id(row["label"])
            if image_id not in dimensions:
                raise ValueError(f"unknown image ID {image_id}")
            box = tuple(float(row[column]) for column in BOX_COLUMNS)
            score = float(row["score"])
            if not all(map(math.isfinite, (*box, score))):
                raise ValueError("nonfinite coordinate or score")
            if score > 1:
                raise ValueError("score exceeds one")
            if score <= 0 or np.float32(score) == 0:
                # EffNetB7 uses label 0 with a negative score for placeholder rows.
                counts["nonpositive_score"] += 1
                continue
            if category not in categories:
                raise ValueError(f"unknown category ID {category}")
            x1, y1, x2, y2 = box
            if x1 >= x2 or y1 >= y2:
                counts["invalid_box"] += 1
                continue
            clipped = tuple(np.float32(min(1., max(0., coord))) for coord in box)
            if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
                counts["collapsed_after_clipping"] += 1
                continue
            rows.setdefault(image_id, []).append((*clipped, score, category))
            counts["accepted"] += 1
    except (csv.Error, UnicodeError, ValueError) as error:
        raise ValueError(f"{filename}, CSV row {max(1, reader.line_num)}: {error}") from error

    predictions = {}
    for image_id, image_rows in rows.items():
        values = np.asarray(image_rows, dtype=np.float64)
        predictions[image_id] = (
            values[:, :4].astype(np.float32),
            values[:, 4].astype(np.float32),
            values[:, 5].astype(np.int64),
        )
    return predictions, counts


def load_coco_data(predictions, annotations):
    with open(annotations, "rb") as stream:
        annotation_bytes = stream.read()
    try:
        document = json.loads(annotation_bytes)
        dimensions = {row["id"]: (row["width"], row["height"]) for row in document["images"]}
        categories = {row["id"] for row in document["categories"]}
        if not dimensions or len(dimensions) != len(document["images"]):
            raise ValueError("expected a nonempty list of distinct image IDs")
        if not categories or len(categories) != len(document["categories"]):
            raise ValueError("expected a nonempty list of distinct category IDs")
        if any(type(key) is not int for key in (*dimensions, *categories)):
            raise ValueError("image and category IDs must be integers")
        if any(type(value) is not int or value <= 0 for size in dimensions.values() for value in size):
            raise ValueError("image dimensions must be positive integers")
        if not isinstance(document["annotations"], list):
            raise ValueError("expected a list of annotations")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{annotations}: invalid COCO annotations: {error}") from error

    image_ids = sorted(dimensions)
    images = [[] for _ in image_ids]
    empty = (np.empty((0, 4), np.float32), np.empty(0, np.float32), np.empty(0, np.int64))
    sources = []
    with open(predictions, "rb") as stream:
        prediction_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        with ZipFile(stream) as archive:
            for filename, weight in SOURCES:
                if archive.namelist().count(filename) != 1:
                    raise ValueError(f"{predictions}: expected exactly one {filename}")
                with archive.open(filename) as member, io.TextIOWrapper(member, encoding="utf-8", newline="") as csv_file:
                    by_image, counts = read_source(csv_file, filename, dimensions, categories)
                for image_id, image in zip(image_ids, images):
                    image.append(by_image.get(image_id, empty))
                sources.append({"filename": filename, "weight": weight, "rows": counts})

    provenance = {
        "upstream_commit": UPSTREAM_COMMIT,
        "assets": {
            "predictions": {"path": str(predictions), "sha256": prediction_hash},
            "annotations": {"path": str(annotations), "sha256": hashlib.sha256(annotation_bytes).hexdigest()},
        },
        "sources": sources,
        "image_ids": image_ids,
        "wbf": {"iou_thr": IOU_THR, "skip_box_thr": SCORE_FLOOR, "conf_type": "avg", "allows_overflow": False},
    }
    weights = np.asarray([weight for _, weight in SOURCES], dtype=np.float64)
    return CocoData(document, image_ids, dimensions, images, weights, provenance)

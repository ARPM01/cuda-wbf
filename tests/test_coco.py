import hashlib
import io
import json
from zipfile import ZipFile

import numpy as np
import pytest

from cuda_wbf.coco_data import SOURCES, load_coco_data, read_source

HEADER = "img_id,x1,y1,x2,y2,score,label\n"


@pytest.fixture
def assets(tmp_path):
    annotations = tmp_path / "instances_val2017.json"
    annotations.write_text(json.dumps({
        "images": [{"id": 13, "width": 80, "height": 40}, {"id": 5, "width": 80, "height": 40}],
        "categories": [{"id": 7, "name": "first"}, {"id": 19, "name": "second"}],
        "annotations": [{"id": 1, "image_id": 5, "category_id": 7, "iscrowd": 0,
                         "area": 400, "bbox": [0, 10, 40, 10]}],
    }))
    predictions = tmp_path / "benchmark.zip"
    with ZipFile(predictions, "w") as archive:
        for filename, _ in reversed(SOURCES):
            rows = ""
            if filename == SOURCES[0][0]:
                rows = ("5,-0.1,0.25,0.5,0.5,0.001,7\n"
                        "5,0.4,0.25,0.5,0.5,0.8,19\n"
                        "5,0.4,0.25,0.5,0.5,-100000,0\n"
                        "5,0.6,0.25,0.5,0.5,0.8,19\n"
                        "5,1.1,0.25,1.2,0.5,0.8,19\n")
            elif filename == SOURCES[-1][0]:
                rows = "5,0,0.25,0.5,0.5,0.6,7.0\n"
            archive.writestr(filename, HEADER + rows)
    return predictions, annotations


def test_pinned_loader(assets):
    data = load_coco_data(*assets)
    assert data.image_ids == [5, 13]
    assert [item["filename"] for item in data.provenance["sources"]] == [name for name, _ in SOURCES]
    assert data.weights.tolist() == [4, 4, 5, 5, 7, 7, 9, 9, 8, 8, 5, 5, 10]
    assert data.provenance["assets"]["predictions"]["sha256"] == hashlib.sha256(assets[0].read_bytes()).hexdigest()
    first, empty_image = data.images
    assert len(first) == len(empty_image) == 13
    assert all(slot[0].shape == (0, 4) for slot in first[1:-1] + empty_image)
    boxes, scores, labels = first[0]
    assert boxes.dtype == scores.dtype == np.float32
    assert scores[0] == np.float32(0.001)
    assert labels.tolist() == [7, 19]
    assert first[-1][1].tolist() == [np.float32(0.6)]
    assert data.provenance["sources"][0]["rows"] == {
        "accepted": 2, "nonpositive_score": 1, "invalid_box": 1, "collapsed_after_clipping": 1,
    }


@pytest.mark.parametrize("row, message", [
    ("5,0,0,1,1,0.1", "wrong number"),
    ("5,0,0,1,1,broken,7", "could not convert"),
    ("5,nan,0,1,1,0.1,7", "nonfinite"),
    ("5,0,0,1,1,1.01,7", "exceeds one"),
    ("99,0,0,1,1,0.1,7", "unknown image ID"),
    ("5,0,0,1,1,0.1,77", "unknown category ID"),
    ("5,0,0,1,1,0.1,7.25", "expected an integer ID"),
])
def test_invalid_csv_row_reports_location(row, message):
    with pytest.raises(ValueError, match=f"source.csv, CSV row 2: .*{message}"):
        read_source(io.StringIO(HEADER + row + "\n"), "source.csv", {5: (80, 40)}, {7})

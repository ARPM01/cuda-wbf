import os
from types import SimpleNamespace

import numpy as np
import pytest

from cuda_wbf import _native
from cuda_wbf.timing import HOST, KERNEL, host_methods, measure_case
from cuda_wbf.verification import source
from cuda_wbf.workloads import real_workloads, synthetic_workload, synthetic_workloads


def test_real_batch_selection_respects_rank_ties_and_complete_batches():
    ids = list(range(40, 0, -1))
    # The two eligibility counts create ties resolved by image ID.
    images = [[source([[0, 0, 1, 1]], [0.001 if image_id > 20 else 0.0005], [1], np.float32)] for image_id in ids]
    data = SimpleNamespace(image_ids=ids, images=images, weights=np.ones(1))
    cases = list(real_workloads(data, batch_sizes=(2, 32)))
    assert [case.name for case in cases] == ["real-q1-b2", "real-q2-b2", "real-q3-b2", "real-q4-b2"]
    assert [batch.image_ids for batch in cases[0].batches] == [(1, 2), (3, 4), (5, 6), (9, 10)]
    assert [batch.image_ids for batch in cases[-1].batches] == [(31, 32), (33, 34), (35, 36), (39, 40)]
    assert all(batch.eligible_boxes == 2 for batch in cases[-1].batches)


def test_synthetic_matrix_and_batch_prefix():
    cases = list(synthetic_workloads())
    assert len(cases) == len({case.name for case in cases}) == 36
    small = synthetic_workload(128, 16, "disjoint", 1)
    large = synthetic_workload(128, 16, "disjoint", 8)
    for source_a, source_b in zip(small.batches[0].images[0], large.batches[0].images[0], strict=True):
        for array_a, array_b in zip(source_a, source_b, strict=True):
            np.testing.assert_array_equal(array_a, array_b)
    assert large.batches[0].input_boxes == 8 * 128


@pytest.mark.cuda
@pytest.mark.skipif(os.getenv("WBF_TEST_TORCHVISION") != "1", reason="requires the CUDA TorchVision benchmark extra")
def test_cuda_timing_boundaries_and_transfer_bytes():
    case = synthetic_workload(128, 16, "disjoint", 1)
    samples = list(measure_case(case, host_methods(_native.cuda_device_info()["device"]), warmups=1, samples=2))
    assert len(samples) == 16
    assert all(row["output_boxes"] == 128 and row["latency_ms"] > 0 for row in samples)
    cuda = next(row for row in samples if row["backend"] == "cuda_coop" and row["boundary"] == HOST)
    assert cuda["upload_bytes"] == 128 * 40 + 17 * 8
    assert cuda["download_bytes"] == 128 * 40 + 16 * 8 + 4
    torch = next(row for row in samples if row["backend"] == "torchvision_cuda_nms")
    assert (torch["upload_bytes"], torch["download_bytes"]) == (128 * 48, 128 * 8)
    assert all(row["upload_bytes"] == row["download_bytes"] == 0 for row in samples if row["boundary"] == KERNEL)

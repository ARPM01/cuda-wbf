import numpy as np
import pytest

from cuda_wbf import fuse
from cuda_wbf.verification import check_case, fixture_cases, randomized_cases


@pytest.mark.parametrize("case", fixture_cases(), ids=lambda case: case.name)
def test_fixture_parity(case):
    check_case(case)


@pytest.mark.parametrize("case", list(randomized_cases()), ids=lambda case: case.name)
def test_randomized_parity(case):
    check_case(case)


def _run(name):
    case = next(case for case in fixture_cases() if case.name == name)
    return fuse(case.images, case.weights, backend="cpp", iou_thr=case.iou_thr,
                skip_box_thr=case.skip_box_thr)


def test_independent_geometry_and_scaling_expectations():
    np.testing.assert_allclose(_run("slide_example")[0][0] * 100, [[22, 22, 62, 62]], atol=1e-5)
    assert len(_run("iou_equal_threshold")[0][0]) == 2
    assert len(_run("moving_representative_includes")[0][0]) == 1
    assert len(_run("moving_representative_excludes")[0][0]) == 2
    assert len(_run("more_than_128_clusters")[0][0]) == 160
    np.testing.assert_array_equal(_run("score_floor")[0][2], [3, 2])
    np.testing.assert_allclose(_run("absent_source_weight")[0][1], [0.9 * 2 / 7], atol=1e-12)
    np.testing.assert_allclose(_run("cluster_size_over_source_count")[0][1], [0.8], atol=1e-6)
    # The third candidate overlaps both clusters; it must choose the second.
    np.testing.assert_allclose(_run("best_match")[0][0][:, 0], [0, (0.8 * 0.6 + 0.7 * 0.2) / 1.5], atol=1e-6)
    # Equal IoUs choose the first cluster, leaving the second box untouched.
    np.testing.assert_array_equal(_run("equal_iou_earliest_cluster")[0][0][0], [0.625, 0, 1, 1])


def test_singleton_is_not_rounded_to_float32():
    actual = _run("singleton_precision")[0]
    np.testing.assert_array_equal(actual[0], [[0.123456789123, 0.1, 0.765432198765, 0.9]])
    np.testing.assert_array_equal(actual[1], [0.987654321987])


def test_repeated_alternating_call_sizes():
    for name in ["more_than_128_clusters", "empty_sources", "slide_example", "empty_batch"] * 3:
        case = next(case for case in fixture_cases() if case.name == name)
        check_case(case)

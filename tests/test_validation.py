import numpy as np
import pytest

from cuda_wbf import _native, fuse


def valid():
    return np.array([[0.1, 0.1, 0.9, 0.9]]), np.array([0.8]), np.array([1])


@pytest.mark.parametrize("weights", [[], [0], [-1], [np.nan], [np.inf], [[1]], ["1"], [True], [1e308, 1e308]])
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        fuse([], weights, backend="cpp")


@pytest.mark.parametrize("name", ["iou_thr", "skip_box_thr"])
@pytest.mark.parametrize("value", [-0.01, 1.01, np.nan, np.inf, [0.5], "0.5", True])
def test_invalid_thresholds(name, value):
    with pytest.raises(ValueError):
        fuse([], [1], backend="cpp", **{name: value})


@pytest.mark.parametrize("field,value", [
    (0, np.empty((0,))), (0, [[0, 0, 1]]), (0, [[[0, 0, 1, 1]]]),
    (0, [[np.nan, 0, 1, 1]]), (0, [[0, 0, np.inf, 1]]),
    (0, [[-0.1, 0, 1, 1]]), (0, [[0, 0, 1.1, 1]]),
    (0, [[0.9, 0, 0.1, 1]]), (0, [[0, 0.9, 1, 0.1]]),
    (0, [[0, 0, 0, 1]]), (0, [[0, 0, 1, 0]]),
    (0, [[0j, 0j, 1j, 1j]]),
    (1, []), (1, [[0.5]]), (1, [np.nan]), (1, [np.inf]),
    (1, [0]), (1, [-0.1]), (1, [1.1]), (1, ["0.5"]),
    (2, []), (2, [[1]]), (2, [1.0]), (2, [True]), (2, ["1"]),
    (2, np.array([2**63], np.uint64)),
])
def test_invalid_source(field, value):
    data = list(valid())
    data[field] = value
    with pytest.raises(ValueError):
        fuse([[tuple(data)]], [1], backend="cpp")


def test_missing_source_slots_and_malformed_tuples():
    with pytest.raises(ValueError, match="source slots"):
        fuse([[valid()]], [1, 2], backend="cpp")
    with pytest.raises(ValueError, match="tuple"):
        fuse([[(valid()[0], valid()[1])]], [1], backend="cpp")


def test_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        fuse([], [1], backend="unknown")


@pytest.mark.parametrize("offsets", [[1, 1], [0, 2], [0, -1, 1], [0, 0, 1], [0]])
def test_native_rejects_invalid_offsets(offsets):
    with pytest.raises(ValueError, match="offsets"):
        _native.fuse_groups(np.array([[1., 0, 0, 1, 1]]), np.array(offsets, np.int64), 1, 1., 0.7)


def test_empty_native_groups():
    values, offsets = _native.fuse_groups(np.empty((0, 5)), np.array([0], np.int64), 1, 1., 0.7)
    assert values.shape == (0, 5)
    np.testing.assert_array_equal(offsets, [0])

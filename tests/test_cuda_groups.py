import os

import numpy as np
import pytest

from cuda_wbf import _native
from cuda_wbf.verification import check_case, fixture_cases, randomized_cases

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(
    os.getenv("WBF_TEST_CUDA") != "1", reason="set WBF_TEST_CUDA=1 in a GPU allocation")]


@pytest.mark.parametrize("case", fixture_cases() + list(randomized_cases()), ids=lambda case: case.name)
def test_group_kernel_parity(case):
    """Compare the CUDA group kernel with the Python oracle."""
    check_case(case, backend="cuda_groups")


def test_resident_launch_and_transfer_accounting():
    """Verify relaunches reuse uploaded inputs and report exact transfer bytes."""
    workspace = _native.CudaWorkspace()
    rows = np.array([[0.9, 0, 0, 0.5, 0.5], [0.6, 0, 0, 0.5, 0.5]])
    offsets = np.array([0, 2], np.int64)
    expected = _native.fuse_groups(rows, offsets, 2, 2., 0.7)
    workspace.upload(rows, offsets, 2, 2., 0.7)
    # Upload owns the resident data; changing the caller's arrays cannot change it.
    rows[:] = 0
    offsets[:] = 0
    for _ in range(3):
        workspace.launch("cuda_groups")
        actual = workspace.download()
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
    assert workspace.stats["upload_bytes"] == 2 * 5 * 8 + 2 * 8
    assert workspace.stats["download_bytes"] == 2 * 5 * 8 + 8 + 4
    assert workspace.stats["kernel_launches"] == 3


def test_workspace_growth_reuse_empty_and_state_errors():
    """Check buffer growth and reuse, empty inputs, and invalid call order."""
    workspace = _native.CudaWorkspace()
    with pytest.raises(RuntimeError, match="Upload"):
        workspace.launch("cuda_groups")
    with pytest.raises(RuntimeError, match="Launch"):
        workspace.download()
    for count in [1, 257, 5, 1024, 0, 3, 0]:
        rows = np.tile([0.8, 0, 0, 1, 1], (count, 1))
        offsets = np.array([0, count] if count else [0], np.int64)
        actual = workspace.fuse(rows, offsets, 1, 1., 0.7, "cuda_groups")
        expected = _native.fuse_groups(rows, offsets, 1, 1., 0.7)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
        if count == 1024:
            allocated = workspace.stats["allocated_bytes"]
        if count in (0, 3):
            assert workspace.stats["allocated_bytes"] == allocated
        if not count:
            assert workspace.stats["upload_bytes"] == workspace.stats["download_bytes"] == 0
    assert workspace.stats["kernel_launches"] == 5


def test_overflow_is_reported_and_workspace_can_be_reused():
    """Verify float32 overflow is reported and the workspace recovers."""
    workspace = _native.CudaWorkspace()
    offsets = np.array([0, 2], np.int64)
    with pytest.raises(OverflowError, match="float32"):
        workspace.fuse(np.tile([1e38, 0, 0, 1, 1], (2, 1)) * np.array([10, 1, 1, 1, 1]),
                       offsets, 2, 2e39, 0.7, "cuda_groups")
    rows = np.tile([0.8, 0, 0, 1, 1], (2, 1))
    actual = workspace.fuse(rows, offsets, 2, 2., 0.7, "cuda_groups")
    np.testing.assert_allclose(actual[0][:, 0], [0.8], atol=1e-6)

from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
import subprocess
import sys
import weakref

import numpy as np
import pytest

from cuda_wbf import _native
from cuda_wbf.verification import check_case, fixture_cases

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(
    os.getenv("WBF_TEST_CUDA") != "1", reason="set WBF_TEST_CUDA=1 with one GPU selected")]
BACKENDS = ["cuda_groups", "cuda_coop"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_many_groups_and_partial_group_grid(backend):
    rows = np.tile([0.8, 0, 0, 1, 1], (257, 1))
    slices = np.arange(258, dtype=np.int64)
    actual = _native.CudaWorkspace().fuse(rows, slices, 2, 3., 0.7, backend)
    expected = _native.fuse_groups(rows, slices, 2, 3., 0.7)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("backend", BACKENDS)
def test_alternating_group_counts_and_resident_relaunch(backend):
    workspace = _native.CudaWorkspace()
    launches = 0
    for count, groups in [(513, 1), (257, 257), (0, 0), (7, 3), (1, 1), (0, 0)]:
        rows = np.tile([0.8, 0, 0, 1, 1], (count, 1))
        slices = np.linspace(0, count, groups + 1, dtype=np.int64)
        expected = _native.fuse_groups(rows, slices, 3, 3., 0.7)
        workspace.upload(rows, slices, 3, 3., 0.7)
        for _ in range(2):
            workspace.launch(backend)
            actual = workspace.download()
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)
            launches += bool(groups)
            assert workspace.stats["kernel_launches"] == launches
        if groups:
            assert workspace.stats["download_bytes"] == count * 40 + groups * 8 + 4
        else:
            assert workspace.stats["upload_bytes"] == workspace.stats["download_bytes"] == 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_overflow_uniform_exit_and_recovery(backend):
    workspace = _native.CudaWorkspace()
    offsets = np.array([0, 2], np.int64)
    with pytest.raises(OverflowError, match="float32"):
        workspace.fuse(np.tile([1e39, 0, 0, 1, 1], (2, 1)), offsets, 2, 2e39, 0.7, backend)
    rows = np.tile([0.8, 0, 0, 1, 1], (2, 1))
    actual = workspace.fuse(rows, offsets, 2, 2., 0.7, backend)
    expected = _native.fuse_groups(rows, offsets, 2, 2., 0.7)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def test_complete_fuse_call_is_serialized_on_shared_workspace():
    workspace = _native.CudaWorkspace()

    def run(backend):
        for count in [1, 257, 3, 129]:
            rows = np.tile([0.8, 0, 0, 1, 1], (count, 1))
            offsets = np.array([0, count], np.int64)
            actual = workspace.fuse(rows, offsets, 2, 2., 0.7, backend)
            expected = _native.fuse_groups(rows, offsets, 2, 2., 0.7)
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(run, BACKENDS))
    reference = weakref.ref(workspace)
    del run, workspace
    gc.collect()
    assert reference() is None


def test_public_api_workspaces_are_independent_across_threads():
    cases = fixture_cases()

    def run(backend):
        for name in ["more_than_128_clusters", "empty_sources", "slide_example"] * 2:
            check_case(next(case for case in cases if case.name == name), backend)

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(run, BACKENDS))


def test_unknown_backend_does_not_launch():
    workspace = _native.CudaWorkspace()
    workspace.upload(np.array([[0.8, 0, 0, 1, 1]]), np.array([0, 1], np.int64), 1, 1., 0.7)
    with pytest.raises(ValueError, match="backend"):
        workspace.launch("invalid")
    assert workspace.stats["kernel_launches"] == 0


def test_gpu_verification_cli_from_other_directory(tmp_path):
    process = subprocess.run([sys.executable, "-m", "cuda_wbf.bench", "verify", "--fixtures-only"],
                             cwd=tmp_path, text=True, capture_output=True)
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert report["status"] == "passed"
    assert [item["backend"] for item in report["backends"]] == ["cpp", *BACKENDS]
    assert all(item["fixture_cases"] == 20 for item in report["backends"])

import json
import os
import subprocess
import sys

import pytest

from cuda_wbf import bench, verification


def test_verify_cpu_only_from_other_directory(tmp_path):
    process = subprocess.run([sys.executable, "-m", "cuda_wbf.bench", "verify", "--cpu-only"],
                             cwd=tmp_path, env={**os.environ, "NUMBA_NUM_THREADS": "1"},
                             text=True, capture_output=True)
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert report["status"] == "passed"
    assert report["fixture_cases"] == 20
    assert report["randomized_cases"] == 120
    assert report["dependencies"]["ensemble-boxes"] == "1.0.9"


def test_cli_requires_explicit_cpu_mode():
    process = subprocess.run([sys.executable, "-m", "cuda_wbf.bench", "verify"],
                             text=True, capture_output=True)
    assert process.returncode == 2
    assert "--cpu-only" in process.stderr


def test_verification_rejects_wrong_oracle_version(monkeypatch):
    monkeypatch.setattr(verification, "version", lambda name: "0.0.0")
    with pytest.raises(RuntimeError, match="Oracle version mismatch"):
        verification.verify_cpu()


def test_cli_returns_failure_on_parity_error(monkeypatch, capsys):
    def fail():
        raise AssertionError("deliberate parity mismatch")

    monkeypatch.setattr(bench, "verify_cpu", fail)
    assert bench.main(["verify", "--cpu-only"]) == 1
    assert "deliberate parity mismatch" in capsys.readouterr().err

#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root after selecting one GPU and a staged CUDA package.
WBF_PYTHON="${WBF_PYTHON:-python}"
WBF_SANITIZER_LOG_DIR="${WBF_SANITIZER_LOG_DIR:-build/cuda/sanitizer}"
export WBF_TEST_CUDA=1
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-1}"

case "${1:-test}" in
  test)
    exec "$WBF_PYTHON" -m pytest -q
    ;;
  memcheck|synccheck|racecheck)
    mkdir -p "$WBF_SANITIZER_LOG_DIR"
    WBF_SANITIZER_TOOL="$1"
    WBF_SANITIZER_EXTRA=()
    WBF_SANITIZER_SELECTION='not random_'
    if [[ "$WBF_SANITIZER_TOOL" == memcheck ]]; then
      WBF_SANITIZER_EXTRA=(--leak-check full)
    fi
    if [[ "$WBF_SANITIZER_TOOL" == racecheck ]]; then
      # Racecheck stalls during Python thread-local workspace teardown on the
      # tested toolchain. Lifecycle/CLI tests remain in test/memcheck/synccheck.
      WBF_SANITIZER_SELECTION='not random_ and not gpu_verification_cli and not public_api_workspaces'
    fi
    exec compute-sanitizer --tool "$WBF_SANITIZER_TOOL" --target-processes all \
      --error-exitcode 99 --log-file "$WBF_SANITIZER_LOG_DIR/$WBF_SANITIZER_TOOL.%p.log" \
      "${WBF_SANITIZER_EXTRA[@]}" "$WBF_PYTHON" -m pytest -q -m cuda -k "$WBF_SANITIZER_SELECTION"
    ;;
  *)
    echo "Usage: bash scripts/check_cuda.sh [test|memcheck|synccheck|racecheck]" >&2
    exit 2
    ;;
esac

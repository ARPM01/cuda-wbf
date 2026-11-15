# CUDA runtime and group kernel

The optional CUDA build retains the CPU backend. `available_backends()` reports compiled implementations; CUDA runtime errors explain the failing operation and recommend checking the device/driver or using `cpp`.

## Shared arithmetic and layout

`src/cluster_math.hpp` supplies the same append, IoU, and score-scaling functions to C++ and CUDA. Coordinate sums round each membership-ordered addition to float32; score sums and IoU use float64. Singletons retain their original representative values. Builds disable contraction and fast math; CUDA also uses precise division and preserves denormals.

Host preprocessing packs each nonempty image/class into a contiguous slice of float64 rows `(weighted_score, x1, y1, x2, y2)`, with int64 slice offsets. A group owns at most one cluster slot per candidate and an int64 valid cluster count. The group kernel assigns one thread to each slice and updates a representative before reading the next candidate. Strict IoU comparison preserves earliest-cluster tie breaking.

Output has the same padded row capacity as input. Only the first valid cluster slots within each slice are compacted on the host. The public API then restores image/class layout and performs the reference final sort.

## Workspace lifecycle

The public API caches one `_native.CudaWorkspace` per Python thread. Each workspace captures its CUDA device, owns a nondefault execution stream, and grows its device buffers only when needed. Explicit workspace calls and the complete `fuse` operation are protected by a mutex. Independent workspaces can execute independently.

The internal interface used by tests and the future timing runner separates:

1. `upload(candidates, offsets, sources, weight_sum, iou_thr)`: validates inputs at the binding, checks sizes, reserves buffers, uploads data, and waits until caller buffers can be released or changed.
2. `launch(backend)`: resets counts, arithmetic status, and padded output; launches the selected kernel on the workspace stream; checks launch errors. Repeated launches use the same resident inputs and reset cluster membership.
3. `download()`: copies padded rows, valid counts, and arithmetic status; synchronizes the stream; rejects overflow or invalid counts; returns compact host rows and offsets.

`fuse(...)` performs all three steps under one lock. Empty resident inputs allocate no new storage and launch no kernel. Destruction releases buffers and the stream on the captured device. CUDA allocation, transfer, launch, and synchronization failures are surfaced to Python.

For N candidates and G nonempty groups, the upload is `N * 5 * 8 + (G + 1) * 8` bytes. The download is `N * 5 * 8 + G * 8 + 4` bytes, including padded output and the arithmetic status integer. Both are zero for empty inputs. `workspace.stats` records these actual transfer sizes, buffer capacities, allocated bytes, input/group counts, device index, and cumulative nonempty kernel launches. Device resets are not host transfers.

## Validation of the first CUDA slice

On the development machine with CUDA 12.8 and an A100 (architecture 80), the CUDA build passed **352 tests**, with one unavailable-backend test skipped because `cuda_groups` is present. This includes the 20 CPU fixtures and 120 randomized cases against the Python oracle, resident relaunches, transfer accounting, workspace growth/reuse, empty inputs, and overflow recovery. The separate CPU-only build passed all **210 tests**.

GPU correctness tests opt in with `WBF_TEST_CUDA=1`. The cooperative kernel and Compute Sanitizer gate follow in the dependent PRs. These checks make no performance claim.

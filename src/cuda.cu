#include "cuda.hpp"
#include "cluster_math.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace wbf {
namespace {

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA ") + operation + ": " + cudaGetErrorString(error) +
                                 ". Check the GPU/driver or use backend='cpp'.");
    }
}

std::size_t bytes_for(std::size_t count, std::size_t element_size) {
    // Check the product before computing a buffer size. A wrapped size could
    // make the allocation or copy smaller than the kernel expects.
    if (count > std::numeric_limits<std::size_t>::max() / element_size) {
        throw std::overflow_error("CUDA workspace size exceeds the addressable range");
    }
    return count * element_size;
}

struct DeviceGuard {
    int previous;
    explicit DeviceGuard(int device) {
        // Device selection belongs to the calling host thread. Restore its
        // original selection when a workspace operation returns or throws.
        check_cuda(cudaGetDevice(&previous), "get current device");
        if (previous != device) {
            check_cuda(cudaSetDevice(device), "select workspace device");
        }
    }
    ~DeviceGuard() { cudaSetDevice(previous); }
};

template <typename T>
struct DeviceBuffer {
    T* data = nullptr;
    std::size_t capacity = 0;
    ~DeviceBuffer() { reset(); }
    void reset() noexcept {
        if (data) { cudaFree(data); }
        data = nullptr;
        capacity = 0;
    }
    void reserve(std::size_t required) {
        if (required <= capacity) { return; }
        // Grow geometrically to reuse allocations across calls. Bound the
        // doubling by the largest element count that fits in a byte size.
        const auto limit = std::numeric_limits<std::size_t>::max() / sizeof(T);
        if (required > limit) { throw std::overflow_error("CUDA buffer size overflow"); }
        const auto grown = capacity <= limit / 2 ? capacity * 2 : limit;
        const auto new_capacity = std::max(required, grown);
        T* replacement = nullptr;
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&replacement), bytes_for(new_capacity, sizeof(T))), "allocate workspace");
        // Allocate the replacement first so allocation failure leaves the old
        // buffer available. Upload overwrites the input, and the next launch
        // rebuilds the working state, so no device-to-device copy is needed.
        if (data) {
            const auto error = cudaFree(data);
            if (error != cudaSuccess) {
                cudaFree(replacement);
                check_cuda(error, "release old workspace buffer");
            }
        }
        data = replacement;
        capacity = new_capacity;
    }
    std::size_t bytes() const { return bytes_for(capacity, sizeof(T)); }
};

__device__ void write_group(const Cluster* clusters, double* output, std::int64_t start,
                           std::int64_t live, std::size_t sources, double weight_sum) {
    // Write the valid prefix at the group's original input offset. Each
    // group can write independently because its padded slice is disjoint.
    for (std::int64_t slot = 0; slot < live; ++slot) {
        const auto& cluster = clusters[start + slot];
        double* row = output + (start + slot) * 5;
        row[0] = scaled_score(cluster, sources, weight_sum);
        for (unsigned axis = 1; axis < 5; ++axis) { row[axis] = cluster.representative[axis]; }
    }
}

__global__ void groups_kernel(const double* candidates, const std::int64_t* offsets,
                              std::size_t groups, Cluster* clusters, double* output,
                              std::int64_t* counts, int* error, std::size_t sources,
                              double weight_sum, double iou_thr) {
    // One thread owns each image/class group. The last block may contain
    // extra threads because the launch rounds up to a full block.
    const auto group = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (group >= groups) { return; }
    const auto start = offsets[group];
    // Start with zero live clusters on every launch. A group's slice has
    // one slot per candidate, which covers the case where none can merge.
    std::int64_t live = 0;
    for (auto index = start; index < offsets[group + 1]; ++index) {
        const double* candidate = candidates + index * 5;
        // Use the next free slot as the default. A cluster must beat the
        // threshold to qualify, and strict comparison keeps the first
        // cluster in creation order when overlaps tie.
        auto best = live;
        double best_iou = iou_thr;
        for (std::int64_t slot = 0; slot < live; ++slot) {
            const double overlap = matching_iou(clusters[start + slot].representative, candidate);
            if (overlap > best_iou) {
                best_iou = overlap;
                best = slot;
            }
        }
        if (best == live) {
            // Initialize the slot before it becomes part of the live range.
            // Previous launches may have left cluster data in this buffer.
            clusters[start + live] = Cluster{};
            ++live;
        }
        // Commit the new representative before reading the next candidate.
        // That candidate's IoU depends on the box after this update.
        if (!append_cluster(clusters[start + best], candidate)) {
            // Multiple groups can overflow at once. Use an atomic write to
            // report failure through the status copied back by download().
            atomicExch(error, 1);
            return;
        }
    }
    write_group(clusters, output, start, live, sources, weight_sum);
    counts[group] = live;
}

__global__ void coop_kernel(const double* candidates, const std::int64_t* offsets,
                            Cluster* clusters, double* output, std::int64_t* counts,
                            int* error, std::size_t sources, double weight_sum, double iou_thr) {
    // One 128-thread block owns the contiguous slice for one image/class
    // group. Here lane is the thread's index in the whole block (0 through 127).
    const auto group = static_cast<std::size_t>(blockIdx.x);
    const unsigned lane = threadIdx.x;
    const auto start = offsets[group];

    // Each thread has one entry in these shared arrays for its best match.
    // The block reuses the entries for the reduction and for each candidate.
    __shared__ double best_ious[128];
    __shared__ std::int64_t best_indices[128];
    __shared__ std::int64_t live;
    __shared__ int failed;
    // Only thread 0 initializes and changes the group's live count and error
    // flag. Wait for initialization before any thread scans the group.
    if (lane == 0) { live = 0; failed = 0; }
    __syncthreads();

    // All threads work on the same candidate at a time, in input order.
    // Each candidate can change the representative used by the next one.
    for (auto index = start; index < offsets[group + 1]; ++index) {
        const double* candidate = candidates + index * 5;
        double local_iou = iou_thr;
        std::int64_t local_index = INT64_MAX;

        // A representative is the cluster's current fused box. Thread t
        // compares the candidate against slots t, t + 128, t + 256, and so on
        // up to live. For example, thread 72 checks slots 72 and 200 if both
        // exist. Together the threads visit each live slot exactly once.
        // The threshold must be exceeded. Within a thread, visiting slots
        // in ascending order and using > keeps the earliest slot on a tie.
        for (std::int64_t slot = lane; slot < live; slot += 128) {
            const double overlap = matching_iou(clusters[start + slot].representative, candidate);
            if (overlap > local_iou) { local_iou = overlap; local_index = slot; }
        }
        // Threads with no qualifying match publish (iou_thr, INT64_MAX),
        // including threads whose starting slot is already outside live.
        best_ious[lane] = local_iou;
        best_indices[lane] = local_index;
        __syncthreads();

        // Halve the number of surviving matches each round. Threads 0-63
        // first compare against entries 64-127. Threads 0-31 then compare
        // against the surviving entries 32-63, and so on until thread 0
        // holds the winner. All 128 threads wait between rounds so no
        // thread reads an entry before its previous-round update.
        // Compare by highest IoU, then lowest cluster slot, to preserve
        // the serial kernel's creation-order tie break across threads.
        for (unsigned stride = 64; stride > 0; stride /= 2) {
            if (lane < stride) {
                const double other_iou = best_ious[lane + stride];
                const auto other_index = best_indices[lane + stride];
                if (other_iou > best_ious[lane] ||
                    (other_iou == best_ious[lane] && other_index < best_indices[lane])) {
                    best_ious[lane] = other_iou;
                    best_indices[lane] = other_index;
                }
            }
            __syncthreads();
        }

        // Thread 0 commits the result. If the sentinel survived the entire
        // reduction, start a cluster at the next free slot. Appending the
        // candidate updates the running sums and the representative box.
        if (lane == 0) {
            auto best = best_indices[0];
            if (best == INT64_MAX) {
                best = live++;
                clusters[start + best] = Cluster{};
            }
            if (!append_cluster(clusters[start + best], candidate)) {
                failed = 1;
                atomicExch(error, 1);
            }
        }
        // Wait for the update before any thread moves on. This also makes
        // the error decision uniform across the block, so every thread
        // either exits here or reaches the next candidate's barriers.
        __syncthreads();
        if (failed) { return; }
    }

    // Reuse the same strided assignment to write the fused rows. Each live
    // slot has a single writer. The host uses counts[group] to compact just
    // the live prefix of this group's padded output slice after download.
    for (std::int64_t slot = lane; slot < live; slot += 128) {
        const auto& cluster = clusters[start + slot];
        double* row = output + (start + slot) * 5;
        row[0] = scaled_score(cluster, sources, weight_sum);
        for (unsigned axis = 1; axis < 5; ++axis) { row[axis] = cluster.representative[axis]; }
    }
    if (lane == 0) { counts[group] = live; }
}

}  // namespace

struct CudaWorkspace::Impl {
    mutable std::mutex mutex;
    int device = 0;
    int max_grid_x = 0;
    cudaStream_t stream = nullptr;
    cudaEvent_t start_event = nullptr, stop_event = nullptr;
    DeviceBuffer<double> candidates, output;
    DeviceBuffer<Cluster> clusters;
    DeviceBuffer<std::int64_t> offsets, counts;
    DeviceBuffer<int> error;
    std::vector<std::int64_t> host_offsets;
    std::size_t count = 0, groups = 0, sources = 0;
    double weight_sum = 0, iou_thr = 0;
    bool uploaded = false, launched = false;
    std::size_t upload_bytes = 0, download_bytes = 0;
    std::uint64_t kernel_launches = 0;

    Impl() {
        // Keep the stream and buffers on the device selected at construction,
        // even if the caller later selects a different device.
        check_cuda(cudaGetDevice(&device), "get current device");
        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, device), "query device properties");
        max_grid_x = properties.maxGridSize[0];
        check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create execution stream");
    }

    ~Impl() {
        // Drain queued work and release the allocations on their owning
        // device before destroying the stream and restoring the selection.
        int previous = device;
        cudaGetDevice(&previous);
        cudaSetDevice(device);
        if (stream) { cudaStreamSynchronize(stream); }
        candidates.reset(); output.reset(); clusters.reset(); offsets.reset(); counts.reset(); error.reset();
        if (start_event) { cudaEventDestroy(start_event); }
        if (stop_event) { cudaEventDestroy(stop_event); }
        if (stream) { cudaStreamDestroy(stream); }
        cudaSetDevice(previous);
    }

    void upload(const double* input, std::size_t input_count, const std::int64_t* input_offsets,
                std::size_t group_count, std::size_t source_count, double sum, double threshold) {
        DeviceGuard guard(device);
        // Invalidate the old run before replacing its inputs. A prior launch
        // may still be queued, so wait before resizing or writing its buffers.
        uploaded = launched = false;
        check_cuda(cudaStreamSynchronize(stream), "synchronize before upload");
        count = input_count; groups = group_count; sources = source_count; weight_sum = sum; iou_thr = threshold;
        upload_bytes = download_bytes = 0;
        // Keep the original slice boundaries on the host for compaction.
        // The device copy lets each thread find its slice during the launch.
        host_offsets.assign(input_offsets, input_offsets + groups + 1);
        if (groups) {
            const auto row_values = bytes_for(count, 5);
            candidates.reserve(row_values); output.reserve(row_values); clusters.reserve(count);
            offsets.reserve(groups + 1); counts.reserve(groups); error.reserve(1);
            const auto candidate_bytes = bytes_for(row_values, sizeof(double));
            const auto offset_bytes = bytes_for(groups + 1, sizeof(std::int64_t));
            check_cuda(cudaMemcpyAsync(candidates.data, input, candidate_bytes, cudaMemcpyHostToDevice, stream), "upload candidates");
            check_cuda(cudaMemcpyAsync(offsets.data, input_offsets, offset_bytes, cudaMemcpyHostToDevice, stream), "upload group offsets");
            // Wait for both copies before returning so the caller can change
            // or release its NumPy buffers after a successful upload.
            check_cuda(cudaStreamSynchronize(stream), "finish upload");
            upload_bytes = candidate_bytes + offset_bytes;
        }
        uploaded = true;
    }

    void launch(const std::string& backend, bool timed = false) {
        if (!uploaded) { throw std::runtime_error("Upload candidates before launching a CUDA kernel"); }
        if (backend != "cuda_groups" && backend != "cuda_coop") {
            throw std::invalid_argument("Unknown or unavailable CUDA backend: " + backend);
        }
        DeviceGuard guard(device);
        launched = false;
        if (groups) {
            // The serial kernel assigns one thread per group. The cooperative
            // kernel assigns a full block per group.
            const bool use_groups = backend == "cuda_groups";
            const auto blocks = use_groups ? (groups + 127) / 128 : groups;
            if (blocks > static_cast<std::size_t>(max_grid_x)) {
                throw std::overflow_error("Group count exceeds the CUDA launch grid limit");
            }
            // A resident relaunch reuses the uploaded rows and offsets but
            // starts with fresh status, counts, and padded output. Queue the
            // resets on the same stream so they finish before the kernel runs.
            check_cuda(cudaMemsetAsync(error.data, 0, sizeof(int), stream), "reset arithmetic status");
            check_cuda(cudaMemsetAsync(counts.data, 0, bytes_for(groups, sizeof(std::int64_t)), stream), "reset cluster counts");
            check_cuda(cudaMemsetAsync(output.data, 0, bytes_for(bytes_for(count, 5), sizeof(double)), stream), "clear padded output");
            if (timed) { check_cuda(cudaEventRecord(start_event, stream), "record kernel start"); }
            if (use_groups) {
                groups_kernel<<<static_cast<unsigned>(blocks), 128, 0, stream>>>(
                    candidates.data, offsets.data, groups, clusters.data, output.data, counts.data, error.data,
                    sources, weight_sum, iou_thr);
            } else {
                // Keep the block width in sync with coop_kernel's shared
                // arrays, scan and write strides, and first reduction stride.
                coop_kernel<<<static_cast<unsigned>(blocks), 128, 0, stream>>>(
                    candidates.data, offsets.data, clusters.data, output.data, counts.data, error.data,
                    sources, weight_sum, iou_thr);
            }
            // Check launch submission here. Download synchronizes the stream
            // to surface errors that happen while the kernel is executing.
            check_cuda(cudaGetLastError(), use_groups ? "launch cuda_groups" : "launch cuda_coop");
            ++kernel_launches;
            if (timed) { check_cuda(cudaEventRecord(stop_event, stream), "record kernel end"); }
        }
        launched = true;
    }

    float time_kernel(const std::string& backend) {
        DeviceGuard guard(device);
        if (!start_event) { check_cuda(cudaEventCreate(&start_event), "create kernel start event"); }
        if (!stop_event) { check_cuda(cudaEventCreate(&stop_event), "create kernel end event"); }
        launch(backend, true);
        if (!groups) { return 0; }
        check_cuda(cudaEventSynchronize(stop_event), "wait for timed kernel");
        float elapsed = 0;
        check_cuda(cudaEventElapsedTime(&elapsed, start_event, stop_event), "read kernel time");
        return elapsed;
    }

    Result download() {
        if (!launched) { throw std::runtime_error("Launch a CUDA kernel before downloading results"); }
        DeviceGuard guard(device);
        Result result;
        result.offsets.reserve(groups + 1);
        result.offsets.push_back(0);
        if (!groups) { return result; }
        // Copy the padded rows and their valid counts together. The stream
        // orders these copies after the kernel, and the host reads them only
        // after synchronization.
        std::vector<double> padded_output(bytes_for(count, 5));
        std::vector<std::int64_t> group_counts(groups);
        int arithmetic_error = 0;
        const auto output_bytes = bytes_for(padded_output.size(), sizeof(double));
        const auto count_bytes = bytes_for(groups, sizeof(std::int64_t));
        check_cuda(cudaMemcpyAsync(padded_output.data(), output.data, output_bytes, cudaMemcpyDeviceToHost, stream), "download padded output");
        check_cuda(cudaMemcpyAsync(group_counts.data(), counts.data, count_bytes, cudaMemcpyDeviceToHost, stream), "download cluster counts");
        check_cuda(cudaMemcpyAsync(&arithmetic_error, error.data, sizeof(int), cudaMemcpyDeviceToHost, stream), "download arithmetic status");
        check_cuda(cudaStreamSynchronize(stream), "finish kernel and download");
        download_bytes = output_bytes + count_bytes + sizeof(int);
        if (arithmetic_error) {
            throw std::overflow_error("Weighted cluster arithmetic overflows float32; use smaller source weights.");
        }
        result.rows.reserve(count);
        for (std::size_t group = 0; group < groups; ++group) {
            const auto live = group_counts[group];
            const auto start = host_offsets[group];
            // Validate the count against this group's original slice before
            // copying that slice's valid prefix into the compact result.
            if (live < 0 || live > host_offsets[group + 1] - start) {
                throw std::runtime_error("CUDA returned an invalid cluster count");
            }
            for (std::int64_t slot = 0; slot < live; ++slot) {
                Row row{};
                std::copy_n(padded_output.data() + (start + slot) * 5, 5, row.begin());
                result.rows.push_back(row);
            }
            // Record the end of this group's compact slice. This also sets
            // the start of the next group, even when padding was discarded.
            result.offsets.push_back(static_cast<std::int64_t>(result.rows.size()));
        }
        return result;
    }

    WorkspaceStats stats() const {
        // Report the number of boxes and groups that fit across all required
        // buffers. Each buffer grows independently, so the smallest limits
        // how much work fits without another allocation.
        return {std::min({candidates.capacity / 5, output.capacity / 5, clusters.capacity}),
                std::min(offsets.capacity ? offsets.capacity - 1 : 0, counts.capacity),
                candidates.bytes() + output.bytes() + clusters.bytes() + offsets.bytes() + counts.bytes() + error.bytes(),
                upload_bytes, download_bytes, count, groups, kernel_launches, device};
    }
};

CudaWorkspace::CudaWorkspace() : impl_(std::make_unique<Impl>()) {}
CudaWorkspace::~CudaWorkspace() = default;

DeviceInfo cuda_device_info() {
    DeviceInfo info{};
    check_cuda(cudaGetDevice(&info.device), "get current device");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, info.device), "query device properties");
    info.name = properties.name;
    info.major = properties.major;
    info.minor = properties.minor;
    check_cuda(cudaRuntimeGetVersion(&info.runtime_version), "query runtime version");
    check_cuda(cudaDriverGetVersion(&info.driver_version), "query driver API version");
    return info;
}

void CudaWorkspace::upload(const double* candidates, std::size_t count, const std::int64_t* offsets,
                           std::size_t groups, std::size_t sources, double weight_sum, double iou_thr) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->upload(candidates, count, offsets, groups, sources, weight_sum, iou_thr);
}

void CudaWorkspace::launch(const std::string& backend) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->launch(backend);
}

float CudaWorkspace::time_kernel(const std::string& backend) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->time_kernel(backend);
}

Result CudaWorkspace::download() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->download();
}

Result CudaWorkspace::fuse(const double* candidates, std::size_t count, const std::int64_t* offsets,
                          std::size_t groups, std::size_t sources, double weight_sum, double iou_thr,
                          const std::string& backend) {
    // Hold the lock across the whole sequence so another caller cannot swap
    // out the inputs between upload and launch or the outputs before download.
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->upload(candidates, count, offsets, groups, sources, weight_sum, iou_thr);
    impl_->launch(backend);
    return impl_->download();
}

WorkspaceStats CudaWorkspace::stats() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->stats();
}

}  // namespace wbf

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
    if (count > std::numeric_limits<std::size_t>::max() / element_size) {
        throw std::overflow_error("CUDA workspace size exceeds the addressable range");
    }
    return count * element_size;
}

struct DeviceGuard {
    int previous;
    explicit DeviceGuard(int device) {
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
        const auto limit = std::numeric_limits<std::size_t>::max() / sizeof(T);
        if (required > limit) { throw std::overflow_error("CUDA buffer size overflow"); }
        const auto grown = capacity <= limit / 2 ? capacity * 2 : limit;
        const auto new_capacity = std::max(required, grown);
        T* replacement = nullptr;
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&replacement), bytes_for(new_capacity, sizeof(T))), "allocate workspace");
        // No resident data is retained across uploads; new storage need not be copied.
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
    const auto group = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (group >= groups) { return; }
    const auto start = offsets[group];
    std::int64_t live = 0;
    for (auto index = start; index < offsets[group + 1]; ++index) {
        const double* candidate = candidates + index * 5;
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
            clusters[start + live] = Cluster{};
            ++live;
        }
        if (!append_cluster(clusters[start + best], candidate)) {
            atomicExch(error, 1);
            return;
        }
    }
    write_group(clusters, output, start, live, sources, weight_sum);
    counts[group] = live;
}

}  // namespace

struct CudaWorkspace::Impl {
    mutable std::mutex mutex;
    int device = 0;
    int max_grid_x = 0;
    cudaStream_t stream = nullptr;
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
        check_cuda(cudaGetDevice(&device), "get current device");
        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, device), "query device properties");
        max_grid_x = properties.maxGridSize[0];
        check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create execution stream");
    }

    ~Impl() {
        int previous = device;
        cudaGetDevice(&previous);
        cudaSetDevice(device);
        if (stream) { cudaStreamSynchronize(stream); }
        candidates.reset(); output.reset(); clusters.reset(); offsets.reset(); counts.reset(); error.reset();
        if (stream) { cudaStreamDestroy(stream); }
        cudaSetDevice(previous);
    }

    void upload(const double* input, std::size_t input_count, const std::int64_t* input_offsets,
                std::size_t group_count, std::size_t source_count, double sum, double threshold) {
        DeviceGuard guard(device);
        uploaded = launched = false;
        check_cuda(cudaStreamSynchronize(stream), "synchronize before upload");
        count = input_count; groups = group_count; sources = source_count; weight_sum = sum; iou_thr = threshold;
        upload_bytes = download_bytes = 0;
        host_offsets.assign(input_offsets, input_offsets + groups + 1);
        if (groups) {
            const auto row_values = bytes_for(count, 5);
            candidates.reserve(row_values); output.reserve(row_values); clusters.reserve(count);
            offsets.reserve(groups + 1); counts.reserve(groups); error.reserve(1);
            const auto candidate_bytes = bytes_for(row_values, sizeof(double));
            const auto offset_bytes = bytes_for(groups + 1, sizeof(std::int64_t));
            check_cuda(cudaMemcpyAsync(candidates.data, input, candidate_bytes, cudaMemcpyHostToDevice, stream), "upload candidates");
            check_cuda(cudaMemcpyAsync(offsets.data, input_offsets, offset_bytes, cudaMemcpyHostToDevice, stream), "upload group offsets");
            // The caller's NumPy buffers may be released or changed after upload returns.
            check_cuda(cudaStreamSynchronize(stream), "finish upload");
            upload_bytes = candidate_bytes + offset_bytes;
        }
        uploaded = true;
    }

    void launch(const std::string& backend) {
        if (!uploaded) { throw std::runtime_error("Upload candidates before launching a CUDA kernel"); }
        if (backend != "cuda_groups") { throw std::invalid_argument("Unknown or unavailable CUDA backend: " + backend); }
        DeviceGuard guard(device);
        launched = false;
        if (groups) {
            const auto blocks = (groups + 127) / 128;
            if (blocks > static_cast<std::size_t>(max_grid_x)) {
                throw std::overflow_error("Group count exceeds the CUDA launch grid limit");
            }
            check_cuda(cudaMemsetAsync(error.data, 0, sizeof(int), stream), "reset arithmetic status");
            check_cuda(cudaMemsetAsync(counts.data, 0, bytes_for(groups, sizeof(std::int64_t)), stream), "reset cluster counts");
            check_cuda(cudaMemsetAsync(output.data, 0, bytes_for(bytes_for(count, 5), sizeof(double)), stream), "clear padded output");
            groups_kernel<<<static_cast<unsigned>(blocks), 128, 0, stream>>>(
                candidates.data, offsets.data, groups, clusters.data, output.data, counts.data, error.data,
                sources, weight_sum, iou_thr);
            check_cuda(cudaGetLastError(), "launch cuda_groups");
            ++kernel_launches;
        }
        launched = true;
    }

    Result download() {
        if (!launched) { throw std::runtime_error("Launch a CUDA kernel before downloading results"); }
        DeviceGuard guard(device);
        Result result;
        result.offsets.reserve(groups + 1);
        result.offsets.push_back(0);
        if (!groups) { return result; }
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
            if (live < 0 || live > host_offsets[group + 1] - start) {
                throw std::runtime_error("CUDA returned an invalid cluster count");
            }
            for (std::int64_t slot = 0; slot < live; ++slot) {
                Row row{};
                std::copy_n(padded_output.data() + (start + slot) * 5, 5, row.begin());
                result.rows.push_back(row);
            }
            result.offsets.push_back(static_cast<std::int64_t>(result.rows.size()));
        }
        return result;
    }

    WorkspaceStats stats() const {
        return {std::min({candidates.capacity / 5, output.capacity / 5, clusters.capacity}),
                std::min(offsets.capacity ? offsets.capacity - 1 : 0, counts.capacity),
                candidates.bytes() + output.bytes() + clusters.bytes() + offsets.bytes() + counts.bytes() + error.bytes(),
                upload_bytes, download_bytes, count, groups, kernel_launches, device};
    }
};

CudaWorkspace::CudaWorkspace() : impl_(std::make_unique<Impl>()) {}
CudaWorkspace::~CudaWorkspace() = default;

void CudaWorkspace::upload(const double* candidates, std::size_t count, const std::int64_t* offsets,
                           std::size_t groups, std::size_t sources, double weight_sum, double iou_thr) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->upload(candidates, count, offsets, groups, sources, weight_sum, iou_thr);
}

void CudaWorkspace::launch(const std::string& backend) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->launch(backend);
}

Result CudaWorkspace::download() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->download();
}

Result CudaWorkspace::fuse(const double* candidates, std::size_t count, const std::int64_t* offsets,
                          std::size_t groups, std::size_t sources, double weight_sum, double iou_thr,
                          const std::string& backend) {
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

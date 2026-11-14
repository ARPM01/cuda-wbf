#pragma once

#include "core.hpp"

#include <memory>
#include <string>

namespace wbf {

struct WorkspaceStats {
    // Capacities describe the reusable allocations. Transfer bytes describe
    // the current upload and download. The launch count is cumulative.
    std::size_t capacity_boxes, capacity_groups, allocated_bytes;
    std::size_t upload_bytes, download_bytes, input_boxes, groups;
    std::uint64_t kernel_launches;
    int device;
};

// Captures the current CUDA device at construction and owns a stream and
// reusable device buffers on it. Each call locks this workspace, and separate
// workspaces can run concurrently.
class CudaWorkspace {
public:
    CudaWorkspace();
    ~CudaWorkspace();
    CudaWorkspace(const CudaWorkspace&) = delete;
    CudaWorkspace& operator=(const CudaWorkspace&) = delete;

    // Replace the resident input with count packed five-value rows and
    // groups + 1 slice offsets. The caller can release the input after
    // this call returns, and this upload can be used for multiple launches.
    void upload(const double* candidates, std::size_t count, const std::int64_t* offsets,
                std::size_t groups, std::size_t sources, double weight_sum, double iou_thr);

    // Reset per-run output and enqueue the selected kernel on the workspace
    // stream. A resident upload must precede the first launch.
    void launch(const std::string& backend);

    // Wait for the most recent launch and return compact rows with cumulative
    // group offsets. Turn a kernel arithmetic failure into an exception.
    Result download();

    // Run upload, launch, and download under a single lock so other callers
    // cannot replace the resident input or output between steps.
    Result fuse(const double* candidates, std::size_t count, const std::int64_t* offsets,
                std::size_t groups, std::size_t sources, double weight_sum, double iou_thr,
                const std::string& backend);
    WorkspaceStats stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace wbf

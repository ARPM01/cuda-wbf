#pragma once

#include "core.hpp"

#include <memory>
#include <string>

namespace wbf {

struct WorkspaceStats {
    std::size_t capacity_boxes, capacity_groups, allocated_bytes;
    std::size_t upload_bytes, download_bytes, input_boxes, groups;
    std::uint64_t kernel_launches;
    int device;
};

struct DeviceInfo {
    std::string name;
    int device, major, minor, runtime_version, driver_version;
};

DeviceInfo cuda_device_info();

// Owns one execution stream and grow-only device storage. Independent workspaces
// may run concurrently. Each operation, and the complete fuse call, is serialized.
class CudaWorkspace {
public:
    CudaWorkspace();
    ~CudaWorkspace();
    CudaWorkspace(const CudaWorkspace&) = delete;
    CudaWorkspace& operator=(const CudaWorkspace&) = delete;

    void upload(const double* candidates, std::size_t count, const std::int64_t* offsets,
                std::size_t groups, std::size_t sources, double weight_sum, double iou_thr);
    void launch(const std::string& backend);
    Result download();
    Result fuse(const double* candidates, std::size_t count, const std::int64_t* offsets,
                std::size_t groups, std::size_t sources, double weight_sum, double iou_thr,
                const std::string& backend);
    WorkspaceStats stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace wbf

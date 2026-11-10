#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace wbf {

// Flat candidate/output rows: weighted score, x1, y1, x2, y2 (float64).
using Row = std::array<double, 5>;

struct Result {
    std::vector<Row> rows;
    std::vector<std::int64_t> offsets;
};

Result fuse_groups(const double* candidates, std::size_t count,
                   const std::int64_t* offsets, std::size_t groups,
                   std::size_t sources, double weight_sum, double iou_thr);

}  // namespace wbf

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace wbf {

// Both input and output rows store a score followed by four box coordinates.
// The input score includes the source weight. The output score includes the
// final confidence rescaling. All five values are stored as float64.
using Row = std::array<double, 5>;

struct Result {
    std::vector<Row> rows;
    // The output rows for group g occupy [offsets[g], offsets[g + 1]).
    // The last offset is the total number of fused rows.
    std::vector<std::int64_t> offsets;
};

// candidates points to the first number in a flat array of five-value rows.
// The input offsets mark where each image/class group starts and ends.
Result fuse_groups(const double* candidates, std::size_t count,
                   const std::int64_t* offsets, std::size_t groups,
                   std::size_t sources, double weight_sum, double iou_thr);

}  // namespace wbf

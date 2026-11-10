#include "core.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace wbf {
namespace {

struct Cluster {
    Row representative{};
    double score_sum = 0;
    std::array<float, 4> coordinate_sums{};
    std::size_t size = 0;

    void append(const double* candidate) {
        score_sum += candidate[0];
        ++size;
        for (std::size_t axis = 0; axis < 4; ++axis) {
            // Match NumPy float32 += a float64 weighted coordinate, in order.
            coordinate_sums[axis] = static_cast<float>(
                static_cast<double>(coordinate_sums[axis]) + candidate[0] * candidate[axis + 1]);
            if (!std::isfinite(coordinate_sums[axis])) {
                throw std::overflow_error("Weighted coordinates overflow float32; use smaller source weights.");
            }
            representative[axis + 1] = size == 1 ? candidate[axis + 1] :
                static_cast<double>(static_cast<float>(static_cast<double>(coordinate_sums[axis]) / score_sum));
        }
        representative[0] = size == 1 ? candidate[0] :
            static_cast<double>(static_cast<float>(score_sum / static_cast<double>(size)));
        if (!std::isfinite(score_sum) || !std::isfinite(representative[0])) {
            throw std::overflow_error("Weighted scores overflow; use smaller source weights.");
        }
    }
};

double iou(const Row& a, const double* b) {
    const double width = std::max(0.0, std::min(a[3], b[3]) - std::max(a[1], b[1]));
    const double height = std::max(0.0, std::min(a[4], b[4]) - std::max(a[2], b[2]));
    const double intersection = width * height;
    const double area_a = (a[3] - a[1]) * (a[4] - a[2]);
    const double area_b = (b[3] - b[1]) * (b[4] - b[2]);
    return intersection / (area_a + area_b - intersection);
}

}  // namespace

Result fuse_groups(const double* candidates, std::size_t count,
                   const std::int64_t* offsets, std::size_t groups,
                   std::size_t sources, double weight_sum, double iou_thr) {
    Result result;
    result.rows.reserve(count);
    result.offsets.reserve(groups + 1);
    result.offsets.push_back(0);
    for (std::size_t group = 0; group < groups; ++group) {
        std::vector<Cluster> clusters;
        clusters.reserve(static_cast<std::size_t>(offsets[group + 1] - offsets[group]));
        for (auto index = offsets[group]; index < offsets[group + 1]; ++index) {
            const double* candidate = candidates + static_cast<std::size_t>(index) * 5;
            std::size_t best = clusters.size();
            double best_iou = iou_thr;
            for (std::size_t slot = 0; slot < clusters.size(); ++slot) {
                const double overlap = iou(clusters[slot].representative, candidate);
                // Strict comparison preserves the earliest cluster for equal IoU.
                if (overlap > best_iou) {
                    best_iou = overlap;
                    best = slot;
                }
            }
            if (best == clusters.size()) {
                clusters.emplace_back();
            }
            clusters[best].append(candidate);
        }
        for (const auto& cluster : clusters) {
            Row output = cluster.representative;
            // Preserve the oracle's multiply-then-divide float64 operation order.
            output[0] = output[0] * static_cast<double>(std::min(sources, cluster.size)) / weight_sum;
            result.rows.push_back(output);
        }
        result.offsets.push_back(static_cast<std::int64_t>(result.rows.size()));
    }
    return result;
}

}  // namespace wbf

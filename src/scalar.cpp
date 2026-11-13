#include "core.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace wbf {
namespace {

// Each cluster represents one fused detection. Keep its current box for IoU
// comparisons and running totals to update it when another candidate joins.
struct Cluster {
    Row representative{};
    double score_sum = 0;
    std::array<float, 4> coordinate_sums{};
    std::size_t size = 0;

    void append(const double* candidate) {
        // The first value is the weighted score. Use it as the weight for all
        // four coordinates when updating the cluster's weighted average.
        score_sum += candidate[0];
        ++size;
        for (std::size_t axis = 0; axis < 4; ++axis) {
            // The Python reference adds each float64 product to a float32
            // coordinate sum. Round back to float32 after each addition and
            // keep the same addition order to reproduce the reference.
            coordinate_sums[axis] = static_cast<float>(
                static_cast<double>(coordinate_sums[axis]) + candidate[0] * candidate[axis + 1]);
            if (!std::isfinite(coordinate_sums[axis])) {
                throw std::overflow_error("Weighted coordinates overflow float32; use smaller source weights.");
            }
            // With one member, the reference keeps the original coordinate.
            // With multiple members, it divides by the total weighted score
            // and rounds the resulting coordinate to float32.
            representative[axis + 1] = size == 1 ? candidate[axis + 1] :
                static_cast<double>(static_cast<float>(static_cast<double>(coordinate_sums[axis]) / score_sum));
        }
        // A merged cluster starts with the mean of its members' weighted
        // scores, rounded to float32. A singleton keeps its original score.
        representative[0] = size == 1 ? candidate[0] :
            static_cast<double>(static_cast<float>(score_sum / static_cast<double>(size)));
        if (!std::isfinite(score_sum) || !std::isfinite(representative[0])) {
            throw std::overflow_error("Weighted scores overflow; use smaller source weights.");
        }
    }
};

// Intersection over union (IoU) is shared area divided by total covered area.
// Clamp the shared width and height to zero for boxes that do not overlap.
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
    // Each group contains one class from one image. Start with fresh clusters
    // for each group and visit its candidates in weighted-score order.
    for (std::size_t group = 0; group < groups; ++group) {
        std::vector<Cluster> clusters;
        clusters.reserve(static_cast<std::size_t>(offsets[group + 1] - offsets[group]));
        for (auto index = offsets[group]; index < offsets[group + 1]; ++index) {
            // Move five numbers per row to find this candidate's score.
            const double* candidate = candidates + static_cast<std::size_t>(index) * 5;
            // clusters.size() is one past the last valid cluster index.
            // Use it to indicate that no cluster has qualified yet.
            std::size_t best = clusters.size();
            double best_iou = iou_thr;
            for (std::size_t slot = 0; slot < clusters.size(); ++slot) {
                const double overlap = iou(clusters[slot].representative, candidate);
                // Starting best_iou at the threshold makes the cutoff strict.
                // Using > also keeps the first cluster found when two tie.
                if (overlap > best_iou) {
                    best_iou = overlap;
                    best = slot;
                }
            }
            if (best == clusters.size()) {
                clusters.emplace_back();
            }
            // Update the representative now so the next candidate compares
            // against the newly fused box.
            clusters[best].append(candidate);
        }
        for (const auto& cluster : clusters) {
            Row output = cluster.representative;
            // Scale the mean weighted score by the member count capped at
            // the number of sources. Divide by the sum of all source weights,
            // including empty sources. Keep this operation order for parity.
            output[0] = output[0] * static_cast<double>(std::min(sources, cluster.size)) / weight_sum;
            result.rows.push_back(output);
        }
        result.offsets.push_back(static_cast<std::int64_t>(result.rows.size()));
    }
    return result;
}

}  // namespace wbf

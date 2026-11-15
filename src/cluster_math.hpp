#pragma once

#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define WBF_HD __host__ __device__
#else
#define WBF_HD
#endif

namespace wbf {

// The representative stores the cluster score followed by x1, y1, x2, y2.
// Keep running sums so each new member can update it without revisiting earlier
// candidates. The coordinate sums use float32 to match the NumPy reference.
struct Cluster {
    double representative[5]{};
    double score_sum = 0;
    float coordinate_sums[4]{};
    std::uint64_t size = 0;
};

WBF_HD inline bool finite_value(double value) {
    // This helper compiles for both CPU callers and CUDA kernel callers.
#ifdef __CUDA_ARCH__
    return isfinite(value);
#else
    return std::isfinite(value);
#endif
}

WBF_HD inline double minimum(double a, double b) { return b < a ? b : a; }
WBF_HD inline double maximum(double a, double b) { return a < b ? b : a; }

WBF_HD inline bool append_cluster(Cluster& cluster, const double* candidate) {
    // candidate[0] is already weighted by its source. Use it as the weight
    // for each coordinate and add it to the cluster's total score.
    cluster.score_sum += candidate[0];
    ++cluster.size;
    for (unsigned axis = 0; axis < 4; ++axis) {
        // Multiply and add in double, then round the stored sum to float32
        // after each candidate. Rounding only after the last addition would
        // change the result relative to NumPy's in-place float32 accumulation.
        cluster.coordinate_sums[axis] = static_cast<float>(
            static_cast<double>(cluster.coordinate_sums[axis]) + candidate[0] * candidate[axis + 1]);
        if (!finite_value(cluster.coordinate_sums[axis])) {
            // The cluster may be partially updated. The caller aborts the
            // fusion when this helper returns false.
            return false;
        }
        // A singleton keeps the original double coordinate. With multiple
        // members, divide by the total weighted score and round the average
        // to float32 before storing it in the double representative.
        cluster.representative[axis + 1] = cluster.size == 1 ? candidate[axis + 1] :
            static_cast<double>(static_cast<float>(static_cast<double>(cluster.coordinate_sums[axis]) / cluster.score_sum));
    }
    // Preserve a singleton's original weighted score. For a merged cluster,
    // round the mean weighted score to float32. Final source normalization
    // happens in scaled_score after the group has processed all candidates.
    cluster.representative[0] = cluster.size == 1 ? candidate[0] :
        static_cast<double>(static_cast<float>(cluster.score_sum / static_cast<double>(cluster.size)));
    return finite_value(cluster.score_sum) && finite_value(cluster.representative[0]);
}

WBF_HD inline double matching_iou(const double* a, const double* b) {
    // Both rows start with a score. On each axis, take the inner pair of box
    // edges and clamp their separation to zero for disjoint or touching boxes.
    const double width = maximum(0.0, minimum(a[3], b[3]) - maximum(a[1], b[1]));
    const double height = maximum(0.0, minimum(a[4], b[4]) - maximum(a[2], b[2]));
    const double intersection = width * height;
    const double area_a = (a[3] - a[1]) * (a[4] - a[2]);
    const double area_b = (b[3] - b[1]) * (b[4] - b[2]);
    // The union counts the overlapping area once, so subtract it from the
    // sum of the individual areas before dividing.
    return intersection / (area_a + area_b - intersection);
}

WBF_HD inline double scaled_score(const Cluster& cluster, std::uint64_t sources, double weight_sum) {
    // Cap the number of member boxes at the number of configured sources.
    // A source can contribute multiple boxes, so this count can differ from
    // the number of distinct sources in the cluster. Multiply the mean score
    // by the capped count, then divide by the sum of all configured source
    // weights, including the weights of sources with no detections.
    const auto represented = sources < cluster.size ? sources : cluster.size;
    return cluster.representative[0] * static_cast<double>(represented) / weight_sum;
}

}  // namespace wbf

#undef WBF_HD

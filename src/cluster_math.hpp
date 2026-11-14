#pragma once

#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define WBF_HD __host__ __device__
#else
#define WBF_HD
#endif

namespace wbf {

// Keep the current fused box for IoU comparisons and running totals for updates.
struct Cluster {
    double representative[5]{};
    double score_sum = 0;
    float coordinate_sums[4]{};
    std::uint64_t size = 0;
};

WBF_HD inline bool finite_value(double value) {
#ifdef __CUDA_ARCH__
    return isfinite(value);
#else
    return std::isfinite(value);
#endif
}

WBF_HD inline double minimum(double a, double b) { return b < a ? b : a; }
WBF_HD inline double maximum(double a, double b) { return a < b ? b : a; }

WBF_HD inline bool append_cluster(Cluster& cluster, const double* candidate) {
    cluster.score_sum += candidate[0];
    ++cluster.size;
    for (unsigned axis = 0; axis < 4; ++axis) {
        // Round each membership-ordered addition just as NumPy float32 += float64 does.
        cluster.coordinate_sums[axis] = static_cast<float>(
            static_cast<double>(cluster.coordinate_sums[axis]) + candidate[0] * candidate[axis + 1]);
        if (!finite_value(cluster.coordinate_sums[axis])) {
            return false;
        }
        // Keep the original coordinate for a singleton. Round the weighted
        // average to float32 once the cluster has multiple members.
        cluster.representative[axis + 1] = cluster.size == 1 ? candidate[axis + 1] :
            static_cast<double>(static_cast<float>(static_cast<double>(cluster.coordinate_sums[axis]) / cluster.score_sum));
    }
    // The merged score starts as the mean of the members' weighted scores.
    cluster.representative[0] = cluster.size == 1 ? candidate[0] :
        static_cast<double>(static_cast<float>(cluster.score_sum / static_cast<double>(cluster.size)));
    return finite_value(cluster.score_sum) && finite_value(cluster.representative[0]);
}

// Intersection over union is the shared area divided by the total covered area.
WBF_HD inline double matching_iou(const double* a, const double* b) {
    const double width = maximum(0.0, minimum(a[3], b[3]) - maximum(a[1], b[1]));
    const double height = maximum(0.0, minimum(a[4], b[4]) - maximum(a[2], b[2]));
    const double intersection = width * height;
    const double area_a = (a[3] - a[1]) * (a[4] - a[2]);
    const double area_b = (b[3] - b[1]) * (b[4] - b[2]);
    return intersection / (area_a + area_b - intersection);
}

WBF_HD inline double scaled_score(const Cluster& cluster, std::uint64_t sources, double weight_sum) {
    // Cap the member count at the number of sources and use the total weight
    // of all configured sources, including those with no detections.
    const auto represented = sources < cluster.size ? sources : cluster.size;
    return cluster.representative[0] * static_cast<double>(represented) / weight_sum;
}

}  // namespace wbf

#undef WBF_HD

#include "core.hpp"
#include "cluster_math.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace wbf {

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
                const double overlap = matching_iou(clusters[slot].representative, candidate);
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
            if (!append_cluster(clusters[best], candidate)) {
                throw std::overflow_error("Weighted cluster arithmetic overflows float32; use smaller source weights.");
            }
        }
        for (const auto& cluster : clusters) {
            Row output{};
            std::copy_n(cluster.representative, 5, output.begin());
            output[0] = scaled_score(cluster, sources, weight_sum);
            result.rows.push_back(output);
        }
        result.offsets.push_back(static_cast<std::int64_t>(result.rows.size()));
    }
    return result;
}

}  // namespace wbf

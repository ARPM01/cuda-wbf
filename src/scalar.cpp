#include "core.hpp"
#include "cluster_math.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace wbf {

Result fuse_groups(const double* candidates, std::size_t count,
                   const std::int64_t* offsets, std::size_t groups,
                   std::size_t sources, double weight_sum, double iou_thr) {
    // Every candidate can create at most one output row. The result offsets
    // start at zero and record the cumulative number of fused rows after
    // each group, so they index the compact output.
    Result result;
    result.rows.reserve(count);
    result.offsets.reserve(groups + 1);
    result.offsets.push_back(0);
    // Preprocessing packs each image/class group into a contiguous slice
    // in descending weighted-score order. Process each slice independently.
    for (std::size_t group = 0; group < groups; ++group) {
        std::vector<Cluster> clusters;
        clusters.reserve(static_cast<std::size_t>(offsets[group + 1] - offsets[group]));
        for (auto index = offsets[group]; index < offsets[group + 1]; ++index) {
            // Each packed row is (weighted_score, x1, y1, x2, y2).
            const double* candidate = candidates + static_cast<std::size_t>(index) * 5;
            // Use the next free slot as the initial best index. If the scan
            // leaves it selected, the candidate will start a new cluster there.
            std::size_t best = clusters.size();
            double best_iou = iou_thr;
            for (std::size_t slot = 0; slot < clusters.size(); ++slot) {
                const double overlap = matching_iou(clusters[slot].representative, candidate);
                // A match must beat the threshold. Scan in creation order
                // and replace the best match only for a strictly higher IoU,
                // so equal overlaps favor the earliest cluster.
                if (overlap > best_iou) {
                    best_iou = overlap;
                    best = slot;
                }
            }
            if (best == clusters.size()) {
                clusters.emplace_back();
            }
            // Appending can move the fused box. Update it before the next
            // candidate computes IoU, since that candidate's match can
            // depend on where the previous candidate moved the box.
            if (!append_cluster(clusters[best], candidate)) {
                throw std::overflow_error("Weighted cluster arithmetic overflows float32; use smaller source weights.");
            }
        }
        for (const auto& cluster : clusters) {
            // Copy the finished box and apply source normalization to its
            // score after all of the group's members have been assigned.
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

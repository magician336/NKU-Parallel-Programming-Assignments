#pragma once
#include "searcher.h"
#include "ivf_index.h"
#include "simd_l2.h"
#include <algorithm>
#include <limits>
#include <vector>

class SIMDSearcher : public BaseSearcher
{
private:
    const IVFPQIndex *index;
    const float *base_data;

public:
    SIMDSearcher(const IVFPQIndex *idx, const float *base)
        : index(idx), base_data(base)
    {
    }

    std::priority_queue<Candidate> search(const float *query, int top_k, int nprobe) override
    {
        auto cmp_asc = [](const Candidate &a, const Candidate &b)
        {
            return a.dist < b.dist;
        };

        // 粗量化距离
        std::vector<float> coarse_dists(index->n_lists);
        {
            MicroProfiler::Timer _t("7_Coarse_Dist");
            compute_all_L2_sqr_d96(query, index->ivf_centroids.data(), index->n_lists, coarse_dists.data());
        }

        std::vector<Candidate> coarse_cands(index->n_lists);
        for (int c = 0; c < index->n_lists; ++c)
        {
            coarse_cands[c] = {coarse_dists[c], static_cast<uint32_t>(c)};
        }

        // 选最近中心
        {
            MicroProfiler::Timer _t("8_Coarse_Sort");
            std::partial_sort(coarse_cands.begin(), coarse_cands.begin() + nprobe,
                              coarse_cands.end(), cmp_asc);
        }

        // 精确L2距离
        std::vector<Candidate> all_cands;
        all_cands.reserve(nprobe * 200);

        {
            MicroProfiler::Timer _t("9_Exact_Dist_Compute");
            for (int pi = 0; pi < nprobe; ++pi)
            {
                int list_id = static_cast<int>(coarse_cands[pi].id);
                const InvertedList &cur_list = index->lists[list_id];
                if (cur_list.total_elements == 0)
                    continue;

                for (const auto &block : cur_list.blocks)
                {
                    for (int k = 0; k < 16; ++k)
                    {
                        uint32_t id = block.ids[k];
                        if (id == 0xFFFFFFFF)
                            continue;
                        float dist = compute_L2_sqr(query, base_data + id * index->d, index->d);
                        all_cands.push_back({dist, id});
                    }
                }
            }
        }

        // Top-K
        {
            MicroProfiler::Timer _t("10_TopK_Select");
            int actual_k = std::min(top_k, static_cast<int>(all_cands.size()));
            if (actual_k == 0)
            {
                return std::priority_queue<Candidate>();
            }

            std::nth_element(all_cands.begin(), all_cands.begin() + actual_k, all_cands.end(), cmp_asc);

            std::priority_queue<Candidate> result;
            for (int i = 0; i < actual_k; ++i)
            {
                result.push(all_cands[i]);
            }
            return result;
        }
    }
};

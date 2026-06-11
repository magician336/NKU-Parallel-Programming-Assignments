#pragma once
#include "ivf_hnsw_index.h"
#include "simd_l2.h"
#include <queue>
#include <vector>
#include <algorithm>

struct Candidate
{
    float dist;
    uint32_t id;
    bool operator<(const Candidate &other) const
    {
        return dist > other.dist;
    }
};

class IVFHNSWSearcher
{
private:
    const IVFHNSWIndex *index;

public:
    IVFHNSWSearcher(const IVFHNSWIndex *idx) : index(idx) {}

    const IVFHNSWIndex *get_index() const { return index; }

    std::priority_queue<Candidate> search(
        const float *query,
        int top_k,
        int nprobe,
        const Candidate *predefined_probes = nullptr)
    {

        auto cmp_asc = [](const Candidate &a, const Candidate &b)
        {
            return a.dist < b.dist;
        };

        // 粗量化
        std::vector<Candidate> coarse_cands;
        if (predefined_probes != nullptr)
        {
            coarse_cands.resize(nprobe);
            for (int p = 0; p < nprobe; ++p)
            {
                coarse_cands[p] = predefined_probes[p];
            }
        }
        else
        {
            std::vector<float> coarse_dists(index->n_lists);
            compute_all_L2_sqr_d96(query, index->ivf_centroids.data(), index->n_lists, coarse_dists.data());
            coarse_cands.resize(index->n_lists);
            for (int c = 0; c < index->n_lists; ++c)
            {
                coarse_cands[c] = {coarse_dists[c], static_cast<uint32_t>(c)};
            }
            std::partial_sort(coarse_cands.begin(), coarse_cands.begin() + nprobe,
                              coarse_cands.end(), cmp_asc);
        }

        // 在探查的簇中搜索（只搜索本进程负责的簇）
        std::vector<Candidate> all_cands;
        all_cands.reserve(static_cast<size_t>(nprobe) * top_k);

        for (int pi = 0; pi < nprobe; ++pi)
        {
            int list_id = static_cast<int>(coarse_cands[pi].id);
            auto *local_hnsw = index->hnsw_graphs[list_id];

            if (local_hnsw == nullptr)
            {
                continue; // 本进程不负责该簇
            }

            auto local_res = local_hnsw->searchKnn(query, top_k);
            while (!local_res.empty())
            {
                auto top_item = local_res.top();
                local_res.pop();
                all_cands.push_back({top_item.first, static_cast<uint32_t>(top_item.second)});
            }
        }

        // 取全局 top-k
        if (all_cands.size() > static_cast<size_t>(top_k))
        {
            std::nth_element(all_cands.begin(), all_cands.begin() + top_k,
                             all_cands.end(), cmp_asc);
            all_cands.resize(top_k);
        }

        std::priority_queue<Candidate> final_pq;
        for (const auto &c : all_cands)
        {
            final_pq.push(c);
        }
        return final_pq;
    }
};

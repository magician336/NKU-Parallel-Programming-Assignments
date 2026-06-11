#pragma once
#include "hnsw_hnsw_index.h"
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

class HNSWHNSWSearcher
{
private:
    const HNSWHNSWIndex *index;

public:
    HNSWHNSWSearcher(const HNSWHNSWIndex *idx) : index(idx) {}

    const HNSWHNSWIndex *get_index() const { return index; }

    std::priority_queue<Candidate> search(
        const float *query,
        int top_k,
        int nprobe)
    {

        auto cmp_asc = [](const Candidate &a, const Candidate &b)
        {
            return a.dist < b.dist;
        };

        // 顶层 HNSW：找到 nprobe 个最近的分区
        index->top_hnsw->setEf(std::max(10, nprobe));
        auto top_res = index->top_hnsw->searchKnn(query, nprobe);

        std::vector<int> target_buckets;
        target_buckets.reserve(nprobe);
        while (!top_res.empty())
        {
            target_buckets.push_back(static_cast<int>(top_res.top().second));
            top_res.pop();
        }

        // 在目标分区的底层 HNSW 中搜索
        std::vector<Candidate> all_cands;
        all_cands.reserve(static_cast<size_t>(nprobe) * top_k);

        for (int bucket_id : target_buckets)
        {
            auto *local_hnsw = index->bottom_hnsws[bucket_id];
            if (local_hnsw == nullptr)
                continue;

            local_hnsw->setEf(std::max(top_k, 50));
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

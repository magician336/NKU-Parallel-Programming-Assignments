#ifndef SEARCH_PTHREAD_H
#define SEARCH_PTHREAD_H

#include "simd_utils.h"
#include "pq_quantizer.h"
#include "thread_pool.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <pthread.h>

// SIMD L2重排
inline float l2_distance_simd(const float* a, const float* b, size_t dim) {
    simd4f sum(0.0f);
    size_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        simd4f va = simd4f::loadu(a + i);
        simd4f vb = simd4f::loadu(b + i);
        simd4f diff = va - vb;
        sum = fmadd(diff, diff, sum);
    }
    float s = reduce_sum(sum);
    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        s += diff * diff;
    }
    return s;
}

struct Neighbor {
    int id;
    float distance;
    bool operator<(Neighbor const& other) const {
        return distance < other.distance;
    }
};

inline std::vector<Neighbor> merge_top_k(const std::vector<std::vector<Neighbor>>& local_results, int k) {
    std::vector<Neighbor> all;
    size_t total = 0;
    for (const auto& lr : local_results) total += lr.size();
    all.reserve(total);
    for (const auto& lr : local_results) {
        all.insert(all.end(), lr.begin(), lr.end());
    }
    if ((int)all.size() <= k) {
        std::sort(all.begin(), all.end());
        return all;
    }
    std::vector<Neighbor> heap;
    heap.reserve(k);
    for (int i = 0; i < k; ++i) {
        heap.push_back(all[i]);
    }
    std::make_heap(heap.begin(), heap.end());
    for (size_t i = k; i < all.size(); ++i) {
        if (all[i].distance < heap.front().distance) {
            std::pop_heap(heap.begin(), heap.end());
            heap.back() = all[i];
            std::push_heap(heap.begin(), heap.end());
        }
    }
    std::sort_heap(heap.begin(), heap.end());
    return heap;
}

inline std::vector<Neighbor> select_top_k(const std::vector<float>& distances, int k) {
    int n = static_cast<int>(distances.size());
    if (k > n) k = n;
    if (k <= 0) return {};

    std::vector<Neighbor> heap;
    heap.reserve(k);
    for (int i = 0; i < k; ++i) {
        heap.push_back({i, distances[i]});
    }
    std::make_heap(heap.begin(), heap.end());

    for (int i = k; i < n; ++i) {
        if (distances[i] < heap.front().distance) {
            std::pop_heap(heap.begin(), heap.end());
            heap.back() = {i, distances[i]};
            std::push_heap(heap.begin(), heap.end());
        }
    }
    std::sort_heap(heap.begin(), heap.end());
    return heap;
}

// PQ搜索(单query)
inline std::vector<Neighbor> pq_search(
    const float* base,
    const std::vector<uint8_t>& base_codes,
    const float* query,
    size_t base_num,
    size_t dim,
    int k,
    int p,
    const PQQuantizer& pq)
{
    if (p > (int)base_num) p = (int)base_num;
    if (p < k) p = k;

    // Stage1: LUT+ADC
    auto lut = pq.build_lut(query);

    std::vector<float> coarse_dist(base_num);
#if defined(USE_NEON) || defined(USE_SSE)
    size_t i = 0;
    for (; i + 4 <= base_num; i += 4) {
        simd4f sum(0.0f);
        for (int sub = 0; sub < pq.m_; ++sub) {
            int lut_offset = sub * pq.ksub_;
            simd4f vals(
                lut[lut_offset + base_codes[(i+0) * pq.m_ + sub]],
                lut[lut_offset + base_codes[(i+1) * pq.m_ + sub]],
                lut[lut_offset + base_codes[(i+2) * pq.m_ + sub]],
                lut[lut_offset + base_codes[(i+3) * pq.m_ + sub]]
            );
            sum = sum + vals;
        }
        float tmp[4];
        sum.storeu(tmp);
        coarse_dist[i+0] = tmp[0];
        coarse_dist[i+1] = tmp[1];
        coarse_dist[i+2] = tmp[2];
        coarse_dist[i+3] = tmp[3];
    }
    for (; i < base_num; ++i) {
        coarse_dist[i] = pq.adc_distance(base_codes.data() + i * pq.m_, lut);
    }
#else
    for (size_t i = 0; i < base_num; ++i) {
        coarse_dist[i] = pq.adc_distance(base_codes.data() + i * pq.m_, lut);
    }
#endif
    auto coarse_top_p = select_top_k(coarse_dist, p);

    // Stage2: SIMD L2重排
    std::vector<float> rerank_dist(p);
    std::vector<int> rerank_id(p);
    for (int i = 0; i < p; ++i) {
        int idx = coarse_top_p[i].id;
        rerank_id[i] = idx;
        rerank_dist[i] = l2_distance_simd(query, base + idx * dim, dim);
    }

    int actual_k = std::min(k, p);
    std::vector<Neighbor> heap;
    heap.reserve(actual_k);
    for (int i = 0; i < actual_k; ++i) {
        heap.push_back({rerank_id[i], rerank_dist[i]});
    }
    std::make_heap(heap.begin(), heap.end());
    for (int i = actual_k; i < p; ++i) {
        if (rerank_dist[i] < heap.front().distance) {
            std::pop_heap(heap.begin(), heap.end());
            heap.back() = {rerank_id[i], rerank_dist[i]};
            std::push_heap(heap.begin(), heap.end());
        }
    }
    std::sort_heap(heap.begin(), heap.end());
    return heap;
}

// 预构建LUT的PQ搜索
inline std::vector<Neighbor> pq_search_with_lut(
    const float* base,
    const std::vector<uint8_t>& base_codes,
    const float* query,
    const std::vector<float>& lut,
    size_t base_num,
    size_t dim,
    int k,
    int p,
    const PQQuantizer& pq)
{
    if (p > (int)base_num) p = (int)base_num;
    if (p < k) p = k;

    std::vector<float> coarse_dist(base_num);
#if defined(USE_NEON) || defined(USE_SSE)
    size_t i = 0;
    for (; i + 4 <= base_num; i += 4) {
        simd4f sum(0.0f);
        for (int sub = 0; sub < pq.m_; ++sub) {
            int lut_offset = sub * pq.ksub_;
            simd4f vals(
                lut[lut_offset + base_codes[(i+0) * pq.m_ + sub]],
                lut[lut_offset + base_codes[(i+1) * pq.m_ + sub]],
                lut[lut_offset + base_codes[(i+2) * pq.m_ + sub]],
                lut[lut_offset + base_codes[(i+3) * pq.m_ + sub]]
            );
            sum = sum + vals;
        }
        float tmp[4];
        sum.storeu(tmp);
        coarse_dist[i+0] = tmp[0];
        coarse_dist[i+1] = tmp[1];
        coarse_dist[i+2] = tmp[2];
        coarse_dist[i+3] = tmp[3];
    }
    for (; i < base_num; ++i) {
        coarse_dist[i] = pq.adc_distance(base_codes.data() + i * pq.m_, lut);
    }
#else
    for (size_t i = 0; i < base_num; ++i) {
        coarse_dist[i] = pq.adc_distance(base_codes.data() + i * pq.m_, lut);
    }
#endif
    auto coarse_top_p = select_top_k(coarse_dist, p);

    std::vector<float> rerank_dist(p);
    std::vector<int> rerank_id(p);
    for (int i = 0; i < p; ++i) {
        int idx = coarse_top_p[i].id;
        rerank_id[i] = idx;
        rerank_dist[i] = l2_distance_simd(query, base + idx * dim, dim);
    }

    int actual_k = std::min(k, p);
    std::vector<Neighbor> heap;
    heap.reserve(actual_k);
    for (int i = 0; i < actual_k; ++i) {
        heap.push_back({rerank_id[i], rerank_dist[i]});
    }
    std::make_heap(heap.begin(), heap.end());
    for (int i = actual_k; i < p; ++i) {
        if (rerank_dist[i] < heap.front().distance) {
            std::pop_heap(heap.begin(), heap.end());
            heap.back() = {rerank_id[i], rerank_dist[i]};
            std::push_heap(heap.begin(), heap.end());
        }
    }
    std::sort_heap(heap.begin(), heap.end());
    return heap;
}

// Query级线程池并行

inline std::vector<std::vector<Neighbor>> pq_search_pthread(
    const float* base,
    const std::vector<uint8_t>& base_codes,
    const float* queries,
    size_t base_num,
    size_t dim,
    int query_num,
    int k,
    int p,
    int num_threads,
    const PQQuantizer& pq)
{
    tp::set_num_threads(num_threads);
    std::vector<std::vector<Neighbor>> results(query_num);
    tp::parallel_for_static(0, static_cast<size_t>(query_num), [&](size_t q) {
        results[q] = pq_search(
            base, base_codes, queries + q * dim,
            base_num, dim, k, p, pq);
    });
    return results;
}

// LUT线程池并行

inline std::vector<float> build_lut_pthread(const float* query, const PQQuantizer& pq, int num_threads) {
    tp::set_num_threads(num_threads);
    std::vector<float> lut(pq.m_ * pq.ksub_);
    tp::parallel_for_static(0, static_cast<size_t>(pq.m_), [&](size_t sub) {
        int ksub = pq.ksub_;
        size_t dsub = pq.dsub_;
        const float* sub_query = query + sub * dsub;
        for (int c = 0; c < ksub; ++c) {
            const float* cent = pq.centroids_.data() + static_cast<int>(sub) * ksub * dsub + c * dsub;
            float dist = 0.0f;
            for (size_t d = 0; d < dsub; ++d) {
                float diff = sub_query[d] - cent[d];
                dist += diff * diff;
            }
            lut[static_cast<int>(sub) * ksub + c] = dist;
        }
    });
    return lut;
}

// Base分区线程池并行

inline std::vector<Neighbor> pq_search_pthread_partition_single(
    const float* base,
    const std::vector<uint8_t>& base_codes,
    const float* query,
    size_t base_num,
    size_t dim,
    int k,
    int p,
    const PQQuantizer& pq,
    int num_threads)
{
    if (p > (int)base_num) p = (int)base_num;
    if (p < k) p = k;

    auto lut = pq.build_lut(query);

    tp::set_num_threads(num_threads);
    std::vector<std::vector<Neighbor>> local_results(num_threads);

    tp::parallel_region([&](int t) {
        int chunk = base_num / num_threads;
        int rem = base_num % num_threads;
        int start = t * chunk + std::min(t, rem);
        int count = chunk + (t < rem ? 1 : 0);
        int end = start + count;

        std::vector<float> local_dist(count);
        for (int i = start; i < end; ++i) {
            local_dist[i - start] = pq.adc_distance(base_codes.data() + i * pq.m_, lut);
        }

        int local_p = std::min(p, count);
        auto local_top = select_top_k(local_dist, local_p);
        for (auto& nb : local_top) {
            nb.id += start;
        }
        local_results[t] = std::move(local_top);
    });

    auto coarse_top_p = merge_top_k(local_results, p);

    std::vector<float> rerank_dist(p);
    std::vector<int> rerank_id(p);
    for (int i = 0; i < p; ++i) {
        rerank_id[i] = coarse_top_p[i].id;
        rerank_dist[i] = l2_distance_simd(query, base + rerank_id[i] * dim, dim);
    }

    int actual_k = std::min(k, p);
    std::vector<Neighbor> heap;
    heap.reserve(actual_k);
    for (int i = 0; i < actual_k; ++i) {
        heap.push_back({rerank_id[i], rerank_dist[i]});
    }
    std::make_heap(heap.begin(), heap.end());
    for (int i = actual_k; i < p; ++i) {
        if (rerank_dist[i] < heap.front().distance) {
            std::pop_heap(heap.begin(), heap.end());
            heap.back() = {rerank_id[i], rerank_dist[i]};
            std::push_heap(heap.begin(), heap.end());
        }
    }
    std::sort_heap(heap.begin(), heap.end());
    return heap;
}

inline std::vector<std::vector<Neighbor>> pq_search_pthread_partition(
    const float* base,
    const std::vector<uint8_t>& base_codes,
    const float* queries,
    size_t base_num,
    size_t dim,
    int query_num,
    int k,
    int p,
    int num_threads,
    const PQQuantizer& pq)
{
    std::vector<std::vector<Neighbor>> results(query_num);
    for (int q = 0; q < query_num; ++q) {
        results[q] = pq_search_pthread_partition_single(
            base, base_codes, queries + q * dim,
            base_num, dim, k, p, pq, num_threads);
    }
    return results;
}

// Batch线程池并行

inline std::vector<std::vector<Neighbor>> pq_search_pthread_batch(
    const float* base,
    const std::vector<uint8_t>& base_codes,
    const float* queries,
    size_t base_num,
    size_t dim,
    int query_num,
    int k,
    int p,
    int num_threads,
    const PQQuantizer& pq)
{
    tp::set_num_threads(num_threads);
    std::vector<std::vector<Neighbor>> results(query_num);

    // 预构建LUT
    std::vector<std::vector<float>> luts(query_num);
    for (int q = 0; q < query_num; ++q) {
        luts[q] = pq.build_lut(queries + q * dim);
    }

    tp::parallel_for_static(0, static_cast<size_t>(query_num), [&](size_t q) {
        results[q] = pq_search_with_lut(
            base, base_codes, queries + q * dim,
            luts[q], base_num, dim, k, p, pq);
    });
    return results;
}

#endif // SEARCH_PTHREAD_H

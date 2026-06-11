#ifndef SEARCH_H
#define SEARCH_H

#include "simd_utils.h"
#include "pq_quantizer.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstddef>

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

inline float ip_distance_simd(const float* a, const float* b, size_t dim) {
    simd4f sum(0.0f);
    size_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        simd4f va = simd4f::loadu(a + i);
        simd4f vb = simd4f::loadu(b + i);
        sum = fmadd(va, vb, sum);
    }
    float s = reduce_sum(sum);
    for (; i < dim; ++i) {
        s += a[i] * b[i];
    }
    return 1.0f - s;
}

struct Neighbor {
    int id;
    float distance;
    bool operator<(Neighbor const& other) const {
        return distance < other.distance;
    }
};

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

    std::vector<float> rerank_dist(p);
    std::vector<int> rerank_id(p);
    for (int i = 0; i < p; ++i) {
        int idx = coarse_top_p[i].id;
        rerank_id[i] = idx;
        rerank_dist[i] = ip_distance_simd(query, base + idx * dim, dim);
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

inline std::vector<Neighbor> pq_search_fastscan(
    const float* base,
    const std::vector<uint8_t>& base_codes_soa,
    const float* query,
    size_t base_num,
    size_t dim,
    int k,
    int p,
    const PQQuantizer& pq)
{
    if (p > (int)base_num) p = (int)base_num;
    if (p < k) p = k;

    auto lut_u8 = pq.build_lut_u8(query);

    std::vector<float> coarse_dist(base_num);
    for (size_t i = 0; i < base_num; ++i) {
        uint32_t sum = 0;
        for (int sub = 0; sub < pq.m_; ++sub) {
            uint8_t code = base_codes_soa[sub * base_num + i];
            sum += lut_u8[sub * pq.ksub_ + code];
        }
        coarse_dist[i] = static_cast<float>(sum);
    }

    auto coarse_top_p = select_top_k(coarse_dist, p);

    std::vector<float> rerank_dist(p);
    std::vector<int> rerank_id(p);
    for (int i = 0; i < p; ++i) {
        int idx = coarse_top_p[i].id;
        rerank_id[i] = idx;
        rerank_dist[i] = ip_distance_simd(query, base + idx * dim, dim);
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

#endif // SEARCH_H

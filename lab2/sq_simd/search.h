#ifndef SEARCH_H
#define SEARCH_H

#include "simd_utils.h"
#include "sq_quantizer.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstddef>

// 串行L2距离
inline float l2_distance_serial(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// SIMD版L2距离，一次处理4个float
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

// 用最大堆选top-k（堆顶是最大距离，方便替换）
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

// SQ搜索：两阶段，先粗排再精排
inline std::vector<Neighbor> sq_search(
    const float* base,
    const std::vector<uint8_t>& base_codes,
    const float* query,
    size_t base_num,
    size_t dim,
    int k,
    int p,
    const SQQuantizer& quantizer,
    bool use_adc = true)
{
    if (p > (int)base_num) p = (int)base_num;
    if (p < k) p = k;

    // stage 1: 粗排，在量化空间算近似距离
    std::vector<float> coarse_dist(base_num);

    if (use_adc) {
        for (size_t i = 0; i < base_num; ++i) {
            coarse_dist[i] = quantizer.asymmetric_distance_simd(
                query, base_codes.data() + i * dim);
        }
    } else {
        std::vector<uint8_t> query_code(dim);
        quantizer.encode(query, query_code.data());
        for (size_t i = 0; i < base_num; ++i) {
            coarse_dist[i] = quantizer.symmetric_distance(
                query_code.data(), base_codes.data() + i * dim);
        }
    }

    auto coarse_top_p = select_top_k(coarse_dist, p);

    // stage 2: 精排，对top-p候选算精确L2距离
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

#endif // SEARCH_H

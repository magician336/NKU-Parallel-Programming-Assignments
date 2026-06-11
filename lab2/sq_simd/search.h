#ifndef SEARCH_H
#define SEARCH_H

#include "simd_utils.h"
#include "sq_quantizer.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstddef>

inline float l2_distance_serial(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

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

inline std::vector<Neighbor> sq_search(
    const float* base,
    const std::vector<uint8_t>& base_codes,
    const float* query,
    size_t base_num,
    size_t dim,
    int k,
    int p,
    const SQQuantizer& quantizer)
{
    if (p > (int)base_num) p = (int)base_num;
    // if (p < k) p = k;

    std::vector<float> coarse_dist(base_num);

    std::vector<uint8_t> query_code(dim);
    quantizer.encode(query, query_code.data());
    for (size_t i = 0; i < base_num; ++i) {
        coarse_dist[i] = quantizer.symmetric_distance_simd(
            query_code.data(), base_codes.data() + i * dim);
    }

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

inline std::priority_queue<std::pair<float, uint32_t>> sq_search_wrapper(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k)
{
    static SQQuantizer quantizer(vecdim);
    static std::vector<uint8_t> base_codes;
    static bool initialized = false;

    if (!initialized) {
        quantizer.train(base, base_number);
        base_codes = quantizer.encode_batch(base, base_number);
        initialized = true;
    }

    int p = 5;
    if (p > (int)base_number) p = (int)base_number;

    auto neighbors = sq_search(
        base, base_codes, query, base_number, vecdim,
        static_cast<int>(k), p, quantizer);

    std::priority_queue<std::pair<float, uint32_t>> result;
    for (const auto& n : neighbors) {
        result.push({n.distance, static_cast<uint32_t>(n.id)});
    }
    return result;
}

inline std::priority_queue<std::pair<float, uint32_t>> flat_search(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k)
{
    return sq_search_wrapper(base, query, base_number, vecdim, k);
}

#endif // SEARCH_H

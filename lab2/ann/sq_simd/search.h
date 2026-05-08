#ifndef SEARCH_H
#define SEARCH_H

#include "simd_utils.h"
#include "sq_quantizer.h"
#include <queue>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstddef>

// SIMD版IP距离（精排用）
inline float inner_product_simd(const float* a, const float* b, size_t dim) {
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

// 最大堆选top-k
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

// SQ搜索：两阶段，先粗排（量化空间L2）再精排（原始空间IP）
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
    if (p < k) p = k;

    // stage 1: 粗排，ADC在量化空间算近似距离
    std::vector<float> coarse_dist(base_num);
    for (size_t i = 0; i < base_num; ++i) {
        coarse_dist[i] = quantizer.asymmetric_distance_simd(
            query, base_codes.data() + i * dim);
    }
    auto coarse_top_p = select_top_k(coarse_dist, p);

    // stage 2: 精排，对top-p候选算精确IP距离
    std::vector<float> rerank_dist(p);
    std::vector<int> rerank_id(p);
    for (int i = 0; i < p; ++i) {
        int idx = coarse_top_p[i].id;
        rerank_id[i] = idx;
        rerank_dist[i] = inner_product_simd(query, base + idx * dim, dim);
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

// 包装为框架要求的flat_search接口
// 内部用SQ两阶段搜索，返回priority_queue
// 用静态变量缓存SQ量化器和编码结果，避免每次查询重复训练
std::priority_queue<std::pair<float, uint32_t>> flat_search(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k)
{
    const int p = 100;

    static SQQuantizer* g_sq = nullptr;
    static std::vector<uint8_t> g_base_codes;
    static size_t g_base_number = 0;
    static size_t g_vecdim = 0;

    if (!g_sq || g_base_number != base_number || g_vecdim != vecdim) {
        delete g_sq;
        g_sq = new SQQuantizer(vecdim);
        g_sq->train(base, base_number);
        g_base_codes = g_sq->encode_batch(base, base_number);
        g_base_number = base_number;
        g_vecdim = vecdim;
    }

    auto result = sq_search(base, g_base_codes, query, base_number, vecdim, k, p, *g_sq);

    std::priority_queue<std::pair<float, uint32_t>> q;
    for (const auto& n : result) {
        q.push({n.distance, (uint32_t)n.id});
    }
    return q;
}

#endif // SEARCH_H

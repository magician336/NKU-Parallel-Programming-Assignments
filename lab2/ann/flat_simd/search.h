#ifndef SEARCH_H
#define SEARCH_H

#include "simd_utils.h"
#include <queue>
#include <cstddef>
#include <cstdint>

// 串行IP距离：1 - dot_product
inline float inner_product_serial(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return 1.0f - sum;
}

// SIMD版IP距离，一次算4个float的点积
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

// flat_search：暴力遍历所有base向量，用SIMD加速IP距离计算
// 返回最大堆，堆顶是k个结果中最差的（距离最大）
std::priority_queue<std::pair<float, uint32_t>> flat_search(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k)
{
    std::priority_queue<std::pair<float, uint32_t>> q;
    for (size_t i = 0; i < base_number; ++i) {
        float dis = inner_product_simd(base + i * vecdim, query, vecdim);
        if (q.size() < k) {
            q.push({dis, (uint32_t)i});
        } else if (dis < q.top().first) {
            q.push({dis, (uint32_t)i});
            q.pop();
        }
    }
    return q;
}

#endif // SEARCH_H

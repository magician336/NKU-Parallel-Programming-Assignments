#ifndef SEARCH_H
#define SEARCH_H

#include "pq_quantizer.h"
#include "simd_utils.h"
#include <queue>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <limits>

// 精确IP距离（用于精排）
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
    return 1.0f - s;  // IP distance: 1 - dot
}

std::priority_queue<std::pair<float, uint32_t>> flat_search(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k)
{
    // 静态缓存PQ量化器和编码，避免每次查询重复训练
    static PQQuantizer pq;
    static std::vector<uint8_t> codes;
    static bool ready = false;
    if (!ready) {
        // 参数：m=8个子空间，每个子空间ksub=256个中心
        pq = PQQuantizer(vecdim, 8, 256);
        pq.train(base, base_number);
        codes = pq.encode_batch(base, base_number);
        ready = true;
    }

    // 为当前query构建LUT
    std::vector<float> lut = pq.build_lut(query);

    const int m = pq.m_;
    const int ksub = pq.ksub_;
    const size_t base_num = base_number;

    // ---------- 粗排：ADC批量4路SIMD ----------
    std::vector<float> coarse_dist(base_num);

    size_t i = 0;
    for (; i + 4 <= base_num; i += 4) {
        simd4f sum(0.0f);
        for (int sub = 0; sub < m; ++sub) {
            int off = sub * ksub;
            simd4f vals(
                lut[off + codes[(i + 0) * m + sub]],
                lut[off + codes[(i + 1) * m + sub]],
                lut[off + codes[(i + 2) * m + sub]],
                lut[off + codes[(i + 3) * m + sub]]
            );
            sum = sum + vals;
        }
        float tmp[4];
        sum.storeu(tmp);
        coarse_dist[i + 0] = tmp[0];
        coarse_dist[i + 1] = tmp[1];
        coarse_dist[i + 2] = tmp[2];
        coarse_dist[i + 3] = tmp[3];
    }

    // 剩余不足4个的标量处理
    for (; i < base_num; ++i) {
        float d = 0.0f;
        for (int sub = 0; sub < m; ++sub) {
            d += lut[sub * ksub + codes[i * m + sub]];
        }
        coarse_dist[i] = d;
    }

    // ---------- 取Top-p候选（精排用） ----------
    size_t p = std::min(k * 50, base_num);  // 粗排取50k个候选
    std::vector<size_t> ids(base_num);
    for (size_t j = 0; j < base_num; ++j) ids[j] = j;
    std::nth_element(ids.begin(), ids.begin() + p, ids.end(),
        [&](size_t a, size_t b) {
            return coarse_dist[a] < coarse_dist[b];
        });
    ids.resize(p);

    // ---------- 精排：精确IP距离 ----------
    std::priority_queue<std::pair<float, uint32_t>> result;
    for (size_t idx : ids) {
        float dis = inner_product_simd(query, base + idx * vecdim, vecdim);
        if (result.size() < k) {
            result.push({dis, static_cast<uint32_t>(idx)});
        } else if (dis < result.top().first) {
            result.push({dis, static_cast<uint32_t>(idx)});
            result.pop();
        }
    }
    return result;
}

#endif // SEARCH_H

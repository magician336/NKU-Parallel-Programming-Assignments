#ifndef PQ_QUANTIZER_H
#define PQ_QUANTIZER_H

#include "simd_utils.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <random>
#include <omp.h>

// 子空间KMeans
inline void kmeans_subspace(
    const float* data,
    size_t n,
    size_t d_sub,
    int k,
    float* centroids,
    unsigned seed = 42)
{
    if (n == 0 || k == 0) return;
    std::mt19937 gen(seed);

    // 随机初始化
    std::vector<int> indices(n);
    for (size_t i = 0; i < n; ++i) indices[i] = (int)i;
    std::shuffle(indices.begin(), indices.end(), gen);
    for (int c = 0; c < k; ++c) {
        int idx = indices[c % (int)n];
        std::memcpy(centroids + c * d_sub,
                    data + idx * d_sub,
                    d_sub * sizeof(float));
    }

    std::vector<int> labels(n);
    std::vector<int> counts(k);
    std::vector<float> new_centroids(k * d_sub);

    const int max_iter = 10;
    for (int iter = 0; iter < max_iter; ++iter) {
        // E-step
        for (size_t i = 0; i < n; ++i) {
            float best_dist = 1e30f;
            int best_c = 0;
            for (int c = 0; c < k; ++c) {
                float dist = 0.0f;
                for (size_t d = 0; d < d_sub; ++d) {
                    float diff = data[i * d_sub + d] - centroids[c * d_sub + d];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_c = c;
                }
            }
            labels[i] = best_c;
        }

        // M-step
        std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);
        for (size_t i = 0; i < n; ++i) {
            int c = labels[i];
            for (size_t d = 0; d < d_sub; ++d) {
                new_centroids[c * d_sub + d] += data[i * d_sub + d];
            }
            counts[c]++;
        }
        for (int c = 0; c < k; ++c) {
            if (counts[c] > 0) {
                float inv = 1.0f / counts[c];
                for (size_t d = 0; d < d_sub; ++d) {
                    new_centroids[c * d_sub + d] *= inv;
                }
            }
        }

        // 收敛检查
        float shift = 0.0f;
        for (int c = 0; c < k; ++c) {
            for (size_t d = 0; d < d_sub; ++d) {
                float diff = new_centroids[c * d_sub + d] - centroids[c * d_sub + d];
                shift += diff * diff;
            }
        }
        std::memcpy(centroids, new_centroids.data(), k * d_sub * sizeof(float));
        if (shift < 1e-6f) break;
    }
}

// PQ量化器
class PQQuantizer {
public:
    size_t dim_ = 0;
    int m_ = 0;       // 子空间数
    int ksub_ = 0;    // 每子空间聚类中心数
    size_t dsub_ = 0; // 每子空间维度

    // centroids布局
    std::vector<float> centroids_;

    PQQuantizer() = default;
    PQQuantizer(size_t dim, int m, int ksub = 256)
        : dim_(dim), m_(m), ksub_(ksub), dsub_(dim / m)
    {
    }

    // 训练
    void train(const float* base, size_t n) {
        centroids_.resize(m_ * ksub_ * dsub_);
        std::vector<float> sub_data(n * dsub_);

        for (int sub = 0; sub < m_; ++sub) {
            // 提取子空间数据
            for (size_t i = 0; i < n; ++i) {
                std::memcpy(sub_data.data() + i * dsub_,
                            base + i * dim_ + sub * dsub_,
                            dsub_ * sizeof(float));
            }
            kmeans_subspace(sub_data.data(), n, dsub_, ksub_,
                            centroids_.data() + sub * ksub_ * dsub_,
                            42 + sub);
        }
    }

    // 编码
    void encode(const float* vec, uint8_t* code) const {
        for (int sub = 0; sub < m_; ++sub) {
            const float* sub_vec = vec + sub * dsub_;
            float best_dist = 1e30f;
            int best_c = 0;
            for (int c = 0; c < ksub_; ++c) {
                const float* cent = centroids_.data() + sub * ksub_ * dsub_ + c * dsub_;
                float dist = 0.0f;
                for (size_t d = 0; d < dsub_; ++d) {
                    float diff = sub_vec[d] - cent[d];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_c = c;
                }
            }
            code[sub] = static_cast<uint8_t>(best_c);
        }
    }

    // 批量编码
    std::vector<uint8_t> encode_batch(const float* base, size_t n) const {
        std::vector<uint8_t> codes(n * m_);
        for (size_t i = 0; i < n; ++i) {
            encode(base + i * dim_, codes.data() + i * m_);
        }
        return codes;
    }

    // 批量编码(OpenMP)
    std::vector<uint8_t> encode_batch_omp(const float* base, size_t n, int num_threads) const {
        std::vector<uint8_t> codes(n * m_);
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (size_t i = 0; i < n; ++i) {
            encode(base + i * dim_, codes.data() + i * m_);
        }
        return codes;
    }

    // 单空间距离计算
    inline float compute_centroid_dist(const float* sub_query, const float* cent) const {
#if defined(USE_NEON) || defined(USE_SSE)
        simd4f sum(0.0f);
        size_t d = 0;
        for (; d + 4 <= dsub_; d += 4) {
            simd4f q = simd4f::loadu(sub_query + d);
            simd4f cn = simd4f::loadu(cent + d);
            simd4f diff = q - cn;
            sum = fmadd(diff, diff, sum);
        }
        float dist = reduce_sum(sum);
        for (; d < dsub_; ++d) {
            float diff = sub_query[d] - cent[d];
            dist += diff * diff;
        }
        return dist;
#else
        float dist = 0.0f;
        for (size_t d = 0; d < dsub_; ++d) {
            float diff = sub_query[d] - cent[d];
            dist += diff * diff;
        }
        return dist;
#endif
    }

    // 构建LUT：query到每个子空间所有中心的距离表（串行）
    std::vector<float> build_lut(const float* query) const {
        std::vector<float> lut(m_ * ksub_);
        for (int sub = 0; sub < m_; ++sub) {
            const float* sub_query = query + sub * dsub_;
            for (int c = 0; c < ksub_; ++c) {
                const float* cent = centroids_.data() + sub * ksub_ * dsub_ + c * dsub_;
                lut[sub * ksub_ + c] = compute_centroid_dist(sub_query, cent);
            }
        }
        return lut;
    }

    // 构建LUT(OpenMP)
    std::vector<float> build_lut_omp(const float* query, int num_threads) const {
        std::vector<float> lut(m_ * ksub_);
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int sub = 0; sub < m_; ++sub) {
            const float* sub_query = query + sub * dsub_;
            for (int c = 0; c < ksub_; ++c) {
                const float* cent = centroids_.data() + sub * ksub_ * dsub_ + c * dsub_;
                lut[sub * ksub_ + c] = compute_centroid_dist(sub_query, cent);
            }
        }
        return lut;
    }

    // 批量构建LUT
    // cache友好
    std::vector<std::vector<float>> build_lut_batch_omp(
        const float* queries, int query_num, int num_threads) const
    {
        std::vector<std::vector<float>> luts(query_num);
        for (int q = 0; q < query_num; ++q) {
            luts[q].resize(m_ * ksub_);
        }
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int q = 0; q < query_num; ++q) {
            for (int sub = 0; sub < m_; ++sub) {
                const float* sub_query = queries + q * dim_ + sub * dsub_;
                for (int c = 0; c < ksub_; ++c) {
                    const float* cent = centroids_.data() + sub * ksub_ * dsub_ + c * dsub_;
                    luts[q][sub * ksub_ + c] = compute_centroid_dist(sub_query, cent);
                }
            }
        }
        return luts;
    }

    // ADC距离
    float adc_distance(const uint8_t* code, const std::vector<float>& lut) const {
        float sum = 0.0f;
        for (int sub = 0; sub < m_; ++sub) {
            sum += lut[sub * ksub_ + code[sub]];
        }
        return sum;
    }
};

#endif // PQ_QUANTIZER_H

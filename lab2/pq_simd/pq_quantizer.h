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

class PQQuantizer {
public:
    size_t dim_ = 0;
    int m_ = 0;
    int ksub_ = 0;
    size_t dsub_ = 0;

    std::vector<float> centroids_;

    PQQuantizer() = default;
    PQQuantizer(size_t dim, int m, int ksub = 256)
        : dim_(dim), m_(m), ksub_(ksub), dsub_(dim / m)
    {
    }

    void train(const float* base, size_t n) {
        centroids_.resize(m_ * ksub_ * dsub_);
        std::vector<float> sub_data(n * dsub_);

        for (int sub = 0; sub < m_; ++sub) {
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

    std::vector<uint8_t> encode_batch(const float* base, size_t n) const {
        std::vector<uint8_t> codes(n * m_);
        for (size_t i = 0; i < n; ++i) {
            encode(base + i * dim_, codes.data() + i * m_);
        }
        return codes;
    }

    std::vector<uint8_t> encode_batch_soa(const float* base, size_t n) const {
        auto codes_aos = encode_batch(base, n);
        std::vector<uint8_t> codes_soa(n * m_);
        for (int sub = 0; sub < m_; ++sub) {
            for (size_t i = 0; i < n; ++i) {
                codes_soa[sub * n + i] = codes_aos[i * m_ + sub];
            }
        }
        return codes_soa;
    }

    std::vector<float> build_lut(const float* query) const {
        std::vector<float> lut(m_ * ksub_);
        for (int sub = 0; sub < m_; ++sub) {
            const float* sub_query = query + sub * dsub_;
            for (int c = 0; c < ksub_; ++c) {
                const float* cent = centroids_.data() + sub * ksub_ * dsub_ + c * dsub_;
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
                lut[sub * ksub_ + c] = dist;
#else
                float dist = 0.0f;
                for (size_t d = 0; d < dsub_; ++d) {
                    float diff = sub_query[d] - cent[d];
                    dist += diff * diff;
                }
                lut[sub * ksub_ + c] = dist;
#endif
            }
        }
        return lut;
    }

    std::vector<uint8_t> build_lut_u8(const float* query) const {
        auto lut_f = build_lut(query);
        std::vector<uint8_t> lut_u8(m_ * ksub_);
        for (int sub = 0; sub < m_; ++sub) {
            int off = sub * ksub_;
            float min_v = lut_f[off];
            float max_v = min_v;
            for (int c = 1; c < ksub_; ++c) {
                float v = lut_f[off + c];
                if (v < min_v) min_v = v;
                if (v > max_v) max_v = v;
            }
            float scale = (max_v > min_v) ? 255.0f / (max_v - min_v) : 0.0f;
            for (int c = 0; c < ksub_; ++c) {
                float v = lut_f[off + c];
                lut_u8[off + c] = static_cast<uint8_t>((v - min_v) * scale);
            }
        }
        return lut_u8;
    }

    float adc_distance(const uint8_t* code, const std::vector<float>& lut) const {
        float sum = 0.0f;
        for (int sub = 0; sub < m_; ++sub) {
            sum += lut[sub * ksub_ + code[sub]];
        }
        return sum;
    }
};

#endif // PQ_QUANTIZER_H

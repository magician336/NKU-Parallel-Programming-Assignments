#ifndef SQ_QUANTIZER_H
#define SQ_QUANTIZER_H

#include "simd_utils.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>

// SQ: 每维单独量化到 8bit unsigned int
// code[i] = round((vec[i] - min_i) / (max_i - min_i) * 255)
class SQQuantizer {
public:
    size_t dim_ = 0;
    std::vector<float> min_val_;   // 每维最小值
    std::vector<float> max_val_;   // 每维最大值
    std::vector<float> scale_;     // (max - min) / 255
    std::vector<float> inv_scale_; // 255 / (max - min)
    bool trained_ = false;

    SQQuantizer() = default;
    explicit SQQuantizer(size_t dim) : dim_(dim) {}

    // 从base向量学习每维的min/max
    void train(const float* base, size_t n) {
        min_val_.assign(dim_, +1e30f);
        max_val_.assign(dim_, -1e30f);
        for (size_t i = 0; i < n; ++i) {
            const float* vec = base + i * dim_;
            for (size_t d = 0; d < dim_; ++d) {
                min_val_[d] = std::min(min_val_[d], vec[d]);
                max_val_[d] = std::max(max_val_[d], vec[d]);
            }
        }
        scale_.resize(dim_);
        inv_scale_.resize(dim_);
        for (size_t d = 0; d < dim_; ++d) {
            float range = max_val_[d] - min_val_[d];
            if (range < 1e-8f) range = 1e-8f;
            scale_[d] = range / 255.0f;
            inv_scale_[d] = 255.0f / range;
        }
        trained_ = true;
    }

    // 单个float向量 -> uint8编码
    void encode(const float* vec, uint8_t* code) const {
        for (size_t d = 0; d < dim_; ++d) {
            float val = (vec[d] - min_val_[d]) * inv_scale_[d];
            if (val < 0.0f) val = 0.0f;
            if (val > 255.0f) val = 255.0f;
            code[d] = static_cast<uint8_t>(val + 0.5f);
        }
    }

    // 批量编码
    std::vector<uint8_t> encode_batch(const float* base, size_t n) const {
        std::vector<uint8_t> codes(n * dim_);
        for (size_t i = 0; i < n; ++i) {
            encode(base + i * dim_, codes.data() + i * dim_);
        }
        return codes;
    }

    // uint8 -> float 解码
    void decode(const uint8_t* code, float* vec) const {
        for (size_t d = 0; d < dim_; ++d) {
            vec[d] = code[d] * scale_[d] + min_val_[d];
        }
    }

    // SIMD批量解码：neon一次解16个uint8，sse一次解4个
    void decode_batch_simd(const uint8_t* code, float* out) const {
#if defined(USE_NEON)
        size_t d = 0;
        for (; d + 16 <= dim_; d += 16) {
            uint8x16_t c_u8 = vld1q_u8(code + d);
            uint8x8_t low8  = vget_low_u8(c_u8);
            uint8x8_t high8 = vget_high_u8(c_u8);

            uint16x8_t c16_0 = vmovl_u8(low8);
            uint16x8_t c16_1 = vmovl_u8(high8);

            float32x4_t cf0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(c16_0)));
            float32x4_t s0  = vld1q_f32(scale_.data() + d);
            float32x4_t m0  = vld1q_f32(min_val_.data() + d);
            vst1q_f32(out + d, vfmaq_f32(m0, cf0, s0));

            float32x4_t cf1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(c16_0)));
            float32x4_t s1  = vld1q_f32(scale_.data() + d + 4);
            float32x4_t m1  = vld1q_f32(min_val_.data() + d + 4);
            vst1q_f32(out + d + 4, vfmaq_f32(m1, cf1, s1));

            float32x4_t cf2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(c16_1)));
            float32x4_t s2  = vld1q_f32(scale_.data() + d + 8);
            float32x4_t m2  = vld1q_f32(min_val_.data() + d + 8);
            vst1q_f32(out + d + 8, vfmaq_f32(m2, cf2, s2));

            float32x4_t cf3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(c16_1)));
            float32x4_t s3  = vld1q_f32(scale_.data() + d + 12);
            float32x4_t m3  = vld1q_f32(min_val_.data() + d + 12);
            vst1q_f32(out + d + 12, vfmaq_f32(m3, cf3, s3));
        }
        for (; d < dim_; ++d) {
            out[d] = code[d] * scale_[d] + min_val_[d];
        }
#elif defined(USE_SSE)
        size_t d = 0;
        for (; d + 4 <= dim_; d += 4) {
            int val = code[d] | (code[d+1] << 8) | (code[d+2] << 16) | (code[d+3] << 24);
            __m128i tmp = _mm_cvtsi32_si128(val);
            __m128i c16 = _mm_unpacklo_epi8(tmp, _mm_setzero_si128());
            __m128i c32 = _mm_unpacklo_epi16(c16, _mm_setzero_si128());
            __m128 cf = _mm_cvtepi32_ps(c32);
            __m128 s = _mm_loadu_ps(scale_.data() + d);
            __m128 m = _mm_loadu_ps(min_val_.data() + d);
            _mm_storeu_ps(out + d, _mm_add_ps(m, _mm_mul_ps(cf, s)));
        }
        for (; d < dim_; ++d) {
            out[d] = code[d] * scale_[d] + min_val_[d];
        }
#else
        for (size_t d = 0; d < dim_; ++d) {
            out[d] = code[d] * scale_[d] + min_val_[d];
        }
#endif
    }

    // ADC：query是float，base是量化后的code
    // 粗排用IP距离近似：1 - dot(query, decoded)
    float asymmetric_distance(const float* query, const uint8_t* code) const {
        float sum = 0.0f;
        for (size_t d = 0; d < dim_; ++d) {
            float decoded = code[d] * scale_[d] + min_val_[d];
            sum += query[d] * decoded;
        }
        return 1.0f - sum;
    }

    // SIMD版ADC（IP近似）
    float asymmetric_distance_simd(const float* query, const uint8_t* code) const {
#if defined(USE_NEON) || defined(USE_SSE)
        std::vector<float> decoded(dim_);
        decode_batch_simd(code, decoded.data());
        simd4f sum(0.0f);
        size_t d = 0;
        for (; d + 4 <= dim_; d += 4) {
            simd4f q = simd4f::loadu(query + d);
            simd4f dec = simd4f::loadu(decoded.data() + d);
            sum = fmadd(q, dec, sum);
        }
        float s = reduce_sum(sum);
        for (; d < dim_; ++d) {
            s += query[d] * decoded[d];
        }
        return 1.0f - s;
#else
        return asymmetric_distance(query, code);
#endif
    }

    // SDC：query和base都量化后比较
    float symmetric_distance(const uint8_t* code_a, const uint8_t* code_b) const {
        int sum = 0;
        for (size_t d = 0; d < dim_; ++d) {
            int diff = static_cast<int>(code_a[d]) - static_cast<int>(code_b[d]);
            sum += diff * diff;
        }
        return static_cast<float>(sum);
    }
};

#endif // SQ_QUANTIZER_H

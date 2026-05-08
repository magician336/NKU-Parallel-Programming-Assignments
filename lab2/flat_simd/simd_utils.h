#ifndef SIMD_UTILS_H
#define SIMD_UTILS_H

#include <cstddef>

#if defined(__ARM_NEON) || defined(__aarch64__)
  #include <arm_neon.h>
  #define USE_NEON
#elif defined(__SSE__)
  #include <xmmintrin.h>
  #include <emmintrin.h>
  #ifdef __AVX__
    #include <immintrin.h>
    #define USE_AVX
  #endif
#include <pmmintrin.h>
  #define USE_SSE
#endif

// simd4f: 128bit寄存器，4个float
struct simd4f {
#if defined(USE_NEON)
    float32x4_t v;
#elif defined(USE_SSE)
    __m128 v;
#else
    float v[4];
#endif

    simd4f() = default;

#if defined(USE_NEON)
    explicit simd4f(float32x4_t _v) : v(_v) {}
    explicit simd4f(float f) : v(vmovq_n_f32(f)) {}
#elif defined(USE_SSE)
    explicit simd4f(__m128 _v) : v(_v) {}
    explicit simd4f(float f) : v(_mm_set1_ps(f)) {}
#else
    explicit simd4f(float f) { v[0]=v[1]=v[2]=v[3]=f; }
#endif

    static simd4f load(const float* p) {
    #if defined(USE_NEON)
        return simd4f(vld1q_f32(p));
    #elif defined(USE_SSE)
        return simd4f(_mm_load_ps(p));
    #else
        simd4f r; r.v[0]=p[0]; r.v[1]=p[1]; r.v[2]=p[2]; r.v[3]=p[3]; return r;
    #endif
    }

    static simd4f loadu(const float* p) {
    #if defined(USE_NEON)
        return simd4f(vld1q_f32(p));
    #elif defined(USE_SSE)
        return simd4f(_mm_loadu_ps(p));
    #else
        simd4f r; r.v[0]=p[0]; r.v[1]=p[1]; r.v[2]=p[2]; r.v[3]=p[3]; return r;
    #endif
    }

    void store(float* p) const {
    #if defined(USE_NEON)
        vst1q_f32(p, v);
    #elif defined(USE_SSE)
        _mm_store_ps(p, v);
    #else
        p[0]=v[0]; p[1]=v[1]; p[2]=v[2]; p[3]=v[3];
    #endif
    }

    void storeu(float* p) const {
    #if defined(USE_NEON)
        vst1q_f32(p, v);
    #elif defined(USE_SSE)
        _mm_storeu_ps(p, v);
    #else
        p[0]=v[0]; p[1]=v[1]; p[2]=v[2]; p[3]=v[3];
    #endif
    }
};

inline simd4f operator+(simd4f a, simd4f b) {
#if defined(USE_NEON)
    return simd4f(vaddq_f32(a.v, b.v));
#elif defined(USE_SSE)
    return simd4f(_mm_add_ps(a.v, b.v));
#else
    simd4f r; for(int i=0;i<4;i++) r.v[i]=a.v[i]+b.v[i]; return r;
#endif
}

inline simd4f operator-(simd4f a, simd4f b) {
#if defined(USE_NEON)
    return simd4f(vsubq_f32(a.v, b.v));
#elif defined(USE_SSE)
    return simd4f(_mm_sub_ps(a.v, b.v));
#else
    simd4f r; for(int i=0;i<4;i++) r.v[i]=a.v[i]-b.v[i]; return r;
#endif
}

inline simd4f operator*(simd4f a, simd4f b) {
#if defined(USE_NEON)
    return simd4f(vmulq_f32(a.v, b.v));
#elif defined(USE_SSE)
    return simd4f(_mm_mul_ps(a.v, b.v));
#else
    simd4f r; for(int i=0;i<4;i++) r.v[i]=a.v[i]*b.v[i]; return r;
#endif
}

// c + a * b
inline simd4f fmadd(simd4f a, simd4f b, simd4f c) {
#if defined(USE_NEON) && defined(__aarch64__)
    return simd4f(vfmaq_f32(c.v, a.v, b.v));
#elif defined(USE_AVX2)
    return simd4f(_mm_fmadd_ps(a.v, b.v, c.v));
#elif defined(USE_SSE)
    return simd4f(_mm_add_ps(c.v, _mm_mul_ps(a.v, b.v)));
#else
    simd4f r; for(int i=0;i<4;i++) r.v[i]=c.v[i]+a.v[i]*b.v[i]; return r;
#endif
}

// 水平求和
inline float reduce_sum(simd4f a) {
#if defined(USE_NEON)
    float32x2_t low  = vget_low_f32(a.v);
    float32x2_t high = vget_high_f32(a.v);
    low = vpadd_f32(low, high);
    low = vpadd_f32(low, low);
    return vget_lane_f32(low, 0);
#elif defined(USE_SSE)
    __m128 shuf = _mm_shuffle_ps(a.v, a.v, _MM_SHUFFLE(1, 0, 3, 2));
    __m128 sums = _mm_add_ps(a.v, shuf);
    shuf = _mm_shuffle_ps(sums, sums, _MM_SHUFFLE(2, 3, 0, 1));
    sums = _mm_add_ps(sums, shuf);
    return _mm_cvtss_f32(sums);
#else
    return a.v[0] + a.v[1] + a.v[2] + a.v[3];
#endif
}

#endif // SIMD_UTILS_H

#ifndef SEARCH_H
#define SEARCH_H

#include "simd_utils.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstddef>

// Inner product distance: 1 - sum(xi * yi)
// Range: (-inf, +inf), smaller is closer
inline float inner_product_serial(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return 1.0f - sum;
}

// L2 distance squared: sum((xi - yi)^2)
// Range: [0, +inf), smaller is closer
inline float l2_distance_serial(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// Distance Functions: SIMD Optimized (4 floats per register)
inline float inner_product_simd(const float* a, const float* b, size_t dim) {
    simd4f sum(0.0f);
    size_t i = 0;
    // Main loop: process 4 floats at a time
    for (; i + 4 <= dim; i += 4) {
        simd4f va = simd4f::loadu(a + i);
        simd4f vb = simd4f::loadu(b + i);
        sum = fmadd(va, vb, sum);
    }
    float s = reduce_sum(sum);
    // Tail handling for remaining elements
    for (; i < dim; ++i) {
        s += a[i] * b[i];
    }
    return 1.0f - s;
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

// Top-K Selection (min-heap based)
struct Neighbor {
    int id;
    float distance;
    bool operator<(Neighbor const& other) const {
        return distance < other.distance;
    }
};

// Select top-k smallest distances using a max-heap of size k
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

// Flat Search Wrappers
enum class DistanceType {
    L2,           // Euclidean distance squared
    InnerProduct  // 1 - dot product
};

// Serial baseline flat search
inline std::vector<Neighbor> flat_search_serial(
    const float* base,
    const float* query,
    size_t base_num,
    size_t dim,
    int k,
    DistanceType dtype = DistanceType::L2)
{
    std::vector<float> distances(base_num);
    if (dtype == DistanceType::L2) {
        for (size_t i = 0; i < base_num; ++i) {
            distances[i] = l2_distance_serial(query, base + i * dim, dim);
        }
    } else {
        for (size_t i = 0; i < base_num; ++i) {
            distances[i] = inner_product_serial(query, base + i * dim, dim);
        }
    }
    return select_top_k(distances, k);
}

// SIMD optimized flat search
inline std::vector<Neighbor> flat_search_simd(
    const float* base,
    const float* query,
    size_t base_num,
    size_t dim,
    int k,
    DistanceType dtype = DistanceType::L2)
{
    std::vector<float> distances(base_num);
    if (dtype == DistanceType::L2) {
        for (size_t i = 0; i < base_num; ++i) {
            distances[i] = l2_distance_simd(query, base + i * dim, dim);
        }
    } else {
        for (size_t i = 0; i < base_num; ++i) {
            distances[i] = inner_product_simd(query, base + i * dim, dim);
        }
    }
    return select_top_k(distances, k);
}

#endif 

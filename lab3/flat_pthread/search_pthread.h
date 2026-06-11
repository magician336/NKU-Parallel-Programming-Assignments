#ifndef SEARCH_PTHREAD_H
#define SEARCH_PTHREAD_H

#include "simd_utils.h"
#include "thread_pool.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <pthread.h>

inline float inner_product_serial(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return 1.0f - sum;
}

inline float l2_distance_serial(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

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

inline std::vector<Neighbor> merge_top_k(const std::vector<std::vector<Neighbor>>& local_results, int k) {
    std::vector<Neighbor> all;
    size_t total = 0;
    for (const auto& lr : local_results) total += lr.size();
    all.reserve(total);
    for (const auto& lr : local_results) {
        all.insert(all.end(), lr.begin(), lr.end());
    }
    if ((int)all.size() <= k) {
        std::sort(all.begin(), all.end());
        return all;
    }
    std::vector<Neighbor> heap;
    heap.reserve(k);
    for (int i = 0; i < k; ++i) {
        heap.push_back(all[i]);
    }
    std::make_heap(heap.begin(), heap.end());
    for (size_t i = k; i < all.size(); ++i) {
        if (all[i].distance < heap.front().distance) {
            std::pop_heap(heap.begin(), heap.end());
            heap.back() = all[i];
            std::push_heap(heap.begin(), heap.end());
        }
    }
    std::sort_heap(heap.begin(), heap.end());
    return heap;
}

enum class DistanceType {
    L2,
    InnerProduct
};

// 串行基线
inline std::vector<Neighbor> flat_search_serial(
    const float* base, const float* query,
    size_t base_num, size_t dim, int k,
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

// SIMD优化
inline std::vector<Neighbor> flat_search_simd(
    const float* base, const float* query,
    size_t base_num, size_t dim, int k,
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

// Query级线程池并行

inline std::vector<std::vector<Neighbor>> flat_search_pthread_query(
    const float* base, const float* queries,
    size_t base_num, size_t dim, int query_num,
    int k, int num_threads, DistanceType dtype = DistanceType::L2)
{
    tp::set_num_threads(num_threads);
    std::vector<std::vector<Neighbor>> results(query_num);
    tp::parallel_for_static(0, static_cast<size_t>(query_num), [&](size_t q) {
        results[q] = flat_search_simd(
            base, queries + q * dim,
            base_num, dim, k, dtype);
    });
    return results;
}

// Base分区线程池并行

inline std::vector<Neighbor> flat_search_pthread_partition_single(
    const float* base, const float* query,
    size_t base_num, size_t dim, int k, int p,
    int num_threads, DistanceType dtype = DistanceType::L2)
{
    if ((int)base_num <= p) {
        return flat_search_simd(base, query, base_num, dim, k, dtype);
    }

    tp::set_num_threads(num_threads);
    std::vector<std::vector<Neighbor>> local_results(num_threads);
    
    tp::parallel_region([&](int t) {
        int chunk = base_num / num_threads;
        int rem = base_num % num_threads;
        int start = t * chunk + std::min(t, rem);
        int count = chunk + (t < rem ? 1 : 0);
        int end = start + count;

        std::vector<float> local_dist(count);
        if (dtype == DistanceType::L2) {
            for (int i = start; i < end; ++i) {
                local_dist[i - start] = l2_distance_simd(query, base + i * dim, dim);
            }
        } else {
            for (int i = start; i < end; ++i) {
                local_dist[i - start] = inner_product_simd(query, base + i * dim, dim);
            }
        }

        int local_p = std::min(p, count);
        auto local_top = select_top_k(local_dist, local_p);
        for (auto& nb : local_top) {
            nb.id += start;
        }
        local_results[t] = std::move(local_top);
    });

    return merge_top_k(local_results, k);
}

inline std::vector<std::vector<Neighbor>> flat_search_pthread_partition(
    const float* base, const float* queries,
    size_t base_num, size_t dim, int query_num,
    int k, int p, int num_threads, DistanceType dtype = DistanceType::L2)
{
    std::vector<std::vector<Neighbor>> results(query_num);
    for (int q = 0; q < query_num; ++q) {
        results[q] = flat_search_pthread_partition_single(
            base, queries + q * dim, base_num, dim, k, p, num_threads, dtype);
    }
    return results;
}

#endif // SEARCH_PTHREAD_H

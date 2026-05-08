#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <cstring>
#include "search.h"

// Test Data Generation
std::vector<float> generate_random_data(size_t n, size_t dim, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    std::vector<float> data(n * dim);
    for (size_t i = 0; i < n * dim; ++i) {
        data[i] = dis(gen);
    }
    return data;
}

// Evaluation Metrics
float compute_recall(const std::vector<Neighbor>& ground_truth,
                     const std::vector<Neighbor>& result,
                     int k)
{
    int hit = 0;
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            if (result[i].id == ground_truth[j].id) {
                ++hit;
                break;
            }
        }
    }
    return (k > 0) ? (float)hit / (float)k : 0.0f;
}

int main() {
    // DEEP100K-like parameters (use smaller scale for local testing)
    const size_t base_num = 10000;   // DEEP100K: 100000
    const size_t dim = 96;           // DEEP100K: 96
    const int k = 10;
    const int query_num = 100;
    const DistanceType dtype = DistanceType::L2;

    std::cout << "========================================" << std::endl;
    std::cout << "       Flat-SIMD Experiment           " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Base vectors : " << base_num << std::endl;
    std::cout << "Dimension    : " << dim << std::endl;
    std::cout << "Top-K        : " << k << std::endl;
    std::cout << "Queries      : " << query_num << std::endl;
    std::cout << "Distance     : " << (dtype == DistanceType::L2 ? "L2" : "InnerProduct") << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // Generate test data
    auto base = generate_random_data(base_num, dim, 123);
    auto queries = generate_random_data(query_num, dim, 456);

    // Warm-up cache
    {
        volatile float dummy = 0.0f;
        for (size_t i = 0; i < base_num; ++i) {
            dummy += l2_distance_serial(queries.data(), base.data() + i * dim, dim);
        }
    }

    // Serial Baseline
    std::vector<std::vector<Neighbor>> gt_results(query_num);
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int q = 0; q < query_num; ++q) {
        gt_results[q] = flat_search_serial(base.data(), queries.data() + q * dim,
                                           base_num, dim, k, dtype);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    double time_serial = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // SIMD Optimized
    std::vector<std::vector<Neighbor>> simd_results(query_num);
    t1 = std::chrono::high_resolution_clock::now();
    for (int q = 0; q < query_num; ++q) {
        simd_results[q] = flat_search_simd(base.data(), queries.data() + q * dim,
                                           base_num, dim, k, dtype);
    }
    t2 = std::chrono::high_resolution_clock::now();
    double time_simd = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // Results
    float avg_recall = 0.0f;
    for (int q = 0; q < query_num; ++q) {
        avg_recall += compute_recall(gt_results[q], simd_results[q], k);
    }
    avg_recall /= query_num;

    float avg_latency_serial = time_serial / query_num;
    float avg_latency_simd   = time_simd   / query_num;

    std::cout << "Serial total : " << time_serial << " ms" << std::endl;
    std::cout << "SIMD total   : " << time_simd << " ms" << std::endl;
    std::cout << "Serial avg   : " << avg_latency_serial << " ms/query" << std::endl;
    std::cout << "SIMD avg     : " << avg_latency_simd << " ms/query" << std::endl;
    std::cout << "Speedup      : " << (time_serial / time_simd) << "x" << std::endl;
    std::cout << "Recall@" << k << "    : " << avg_recall << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}

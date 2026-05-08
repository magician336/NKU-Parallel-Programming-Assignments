#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <cstring>
#include "search.h"
#include "pq_quantizer.h"

// ============================================================================
// Test Data Generation
// ============================================================================

std::vector<float> generate_random_data(size_t n, size_t dim, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    std::vector<float> data(n * dim);
    for (size_t i = 0; i < n * dim; ++i) {
        data[i] = dis(gen);
    }
    return data;
}

float l2_distance_serial(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

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

// ============================================================================
// Main
// ============================================================================

int main() {
    const size_t base_num = 10000;
    const size_t dim = 96;
    const int k = 10;
    const int query_num = 100;
    const int m = 4;        // number of subspaces
    const int ksub = 256;   // centroids per subspace

    std::cout << "========================================" << std::endl;
    std::cout << "        PQ-SIMD Experiment            " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Base vectors : " << base_num << std::endl;
    std::cout << "Dimension    : " << dim << std::endl;
    std::cout << "Subspaces(m) : " << m << std::endl;
    std::cout << "Centroids    : " << ksub << std::endl;
    std::cout << "Top-K        : " << k << std::endl;
    std::cout << "Queries      : " << query_num << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // Generate data
    auto base = generate_random_data(base_num, dim, 123);
    auto queries = generate_random_data(query_num, dim, 456);

    // Train PQ
    PQQuantizer pq(dim, m, ksub);
    pq.train(base.data(), base_num);
    auto base_codes = pq.encode_batch(base.data(), base_num);

    float compression = (dim * sizeof(float)) / (float)(m * sizeof(uint8_t));
    std::cout << "PQ trained. Compression ratio: "
              << dim * sizeof(float) << " -> " << m
              << " bytes (" << compression << "x)" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // Ground truth: serial flat search
    std::vector<std::vector<Neighbor>> gt(query_num);
    for (int q = 0; q < query_num; ++q) {
        std::vector<float> dists(base_num);
        for (size_t i = 0; i < base_num; ++i) {
            dists[i] = l2_distance_serial(queries.data() + q * dim,
                                           base.data() + i * dim, dim);
        }
        gt[q] = select_top_k(dists, k);
    }

    // Baseline serial time
    double time_serial = 0.0;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int q = 0; q < query_num; ++q) {
        std::vector<float> dists(base_num);
        for (size_t i = 0; i < base_num; ++i) {
            dists[i] = l2_distance_serial(queries.data() + q * dim,
                                           base.data() + i * dim, dim);
        }
        select_top_k(dists, k);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    time_serial = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // Test different p values for latency-recall trade-off
    std::vector<int> p_values = {10, 20, 50, 100, 200, 500};

    std::cout << "ADC Strategy (LUT Build + Table Lookup + Rerank)" << std::endl;
    std::cout << std::setw(8) << "p"
              << std::setw(15) << "Latency(ms)"
              << std::setw(12) << "Recall@" << k
              << std::setw(12) << "Speedup"
              << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    for (int p : p_values) {
        auto t1 = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<Neighbor>> results(query_num);
        for (int q = 0; q < query_num; ++q) {
            results[q] = pq_search(base.data(), base_codes,
                                   queries.data() + q * dim,
                                   base_num, dim, k, p, pq);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        double time_pq = std::chrono::duration<double, std::milli>(t2 - t1).count();

        float avg_recall = 0.0f;
        for (int q = 0; q < query_num; ++q) {
            avg_recall += compute_recall(gt[q], results[q], k);
        }
        avg_recall /= query_num;

        std::cout << std::setw(8) << p
                  << std::setw(15) << std::fixed << std::setprecision(3) << (time_pq / query_num)
                  << std::setw(12) << std::fixed << std::setprecision(4) << avg_recall
                  << std::setw(12) << std::fixed << std::setprecision(2) << (time_serial / time_pq)
                  << std::endl;
    }

    std::cout << "========================================" << std::endl;

    // Optional: test different m values
    std::cout << "\n--- Effect of subspace count m ---" << std::endl;
    std::vector<int> m_values = {2, 4, 8, 12};
    for (int test_m : m_values) {
        if (dim % test_m != 0) continue;
        PQQuantizer pq_test(dim, test_m, ksub);
        pq_test.train(base.data(), base_num);
        auto codes_test = pq_test.encode_batch(base.data(), base_num);

        float comp = (dim * sizeof(float)) / (float)(test_m * sizeof(uint8_t));
        std::cout << "m=" << test_m << ": compression=" << comp << "x, ";

        // Quick recall test with p=100
        float rec = 0.0f;
        for (int q = 0; q < 20; ++q) {
            auto res = pq_search(base.data(), codes_test,
                                 queries.data() + q * dim,
                                 base_num, dim, k, 100, pq_test);
            rec += compute_recall(gt[q], res, k);
        }
        rec /= 20.0f;
        std::cout << "recall@" << k << " (p=100)=" << rec << std::endl;
    }

    return 0;
}

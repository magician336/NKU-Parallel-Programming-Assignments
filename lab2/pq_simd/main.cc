#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include "search.h"
#include "pq_quantizer.h"

template<typename T>
T* LoadData(const std::string& filename, int& num, int& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return nullptr;
    }
    in.read((char*)&num, sizeof(int));
    in.read((char*)&dim, sizeof(int));
    T* data = new T[num * dim];
    in.read((char*)data, num * dim * sizeof(T));
    in.close();
    return data;
}

float ip_distance_serial(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return 1.0f - sum;
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

int main() {
    std::string data_path = "/anndata/";
    const int k = 10;
    const int ksub = 256;

    std::cout << "PQ-SIMD Experiment" << std::endl;

    int base_num = 0, dim = 0;
    int query_num = 0, query_dim = 0;

    float* base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_num, dim);
    float* queries = LoadData<float>(data_path + "DEEP100K.query.fbin", query_num, query_dim);

    if (!base || !queries) {
        std::cerr << "Failed to load dataset!" << std::endl;
        return -1;
    }

    std::cout << "Base vectors : " << base_num << std::endl;
    std::cout << "Dimension    : " << dim << std::endl;
    std::cout << "Top-K        : " << k << std::endl;
    std::cout << "Queries      : " << query_num << std::endl;

    std::cout << "Computing ground truth..." << std::endl;
    std::vector<std::vector<Neighbor>> gt(query_num);
    for (int q = 0; q < query_num; ++q) {
        std::vector<float> dists(base_num);
        for (size_t i = 0; i < base_num; ++i) {
            dists[i] = ip_distance_serial(queries + q * dim,
                                          base + i * dim, dim);
        }
        gt[q] = select_top_k(dists, k);
    }

    std::cout << "Running baseline flat search..." << std::endl;
    double time_serial = 0.0;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int q = 0; q < query_num; ++q) {
        std::vector<float> dists(base_num);
        for (size_t i = 0; i < base_num; ++i) {
            dists[i] = ip_distance_serial(queries + q * dim,
                                          base + i * dim, dim);
        }
        select_top_k(dists, k);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    time_serial = std::chrono::duration<double, std::milli>(t2 - t1).count();

    std::vector<int> m_values = {8, 16};
    std::vector<int> p_values = {50, 100, 500};

    for (int m : m_values) {
        if (dim % m != 0) {
            std::cout << "m=" << m << " does not divide dim=" << dim << std::endl;
            continue;
        }

        std::cout << "m = " << std::setw(2) << m << " Experiment" << std::endl;
        std::cout << "Subspaces(m) : " << m << std::endl;
        std::cout << "Centroids    : " << ksub << std::endl;

        PQQuantizer pq(dim, m, ksub);
        pq.train(base, base_num);
        auto base_codes = pq.encode_batch(base, base_num);
        auto base_codes_soa = pq.encode_batch_soa(base, base_num);

        float compression = (dim * sizeof(float)) / (float)(m * sizeof(uint8_t));
        std::cout << "PQ trained. Compression ratio: " << compression << "x" << std::endl;

        std::cout << "ADC Strategy (LUT Build + Table Lookup + Rerank)" << std::endl;
        std::cout << std::setw(8) << "p"
                  << std::setw(15) << "Latency(ms)"
                  << std::setw(12) << "Recall@" << k
                  << std::setw(12) << "Speedup"
                  << std::endl;

        for (int p : p_values) {
            auto t1 = std::chrono::high_resolution_clock::now();
            std::vector<std::vector<Neighbor>> results(query_num);
            for (int q = 0; q < query_num; ++q) {
                results[q] = pq_search(base, base_codes,
                                       queries + q * dim,
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

        std::cout << "FastScan (SoA + 8-bit LUT)" << std::endl;
        std::cout << std::setw(8) << "p"
                  << std::setw(15) << "Latency(ms)"
                  << std::setw(12) << "Recall@" << k
                  << std::setw(12) << "Speedup"
                  << std::setw(12) << "vs PQ"
                  << std::endl;

        for (int p : p_values) {
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int q = 0; q < query_num; ++q) {
                pq_search(base, base_codes,
                          queries + q * dim,
                          base_num, dim, k, p, pq);
            }
            auto t0_end = std::chrono::high_resolution_clock::now();
            double time_pq_ref = std::chrono::duration<double, std::milli>(t0_end - t0).count();

            auto t1 = std::chrono::high_resolution_clock::now();
            std::vector<std::vector<Neighbor>> results(query_num);
            for (int q = 0; q < query_num; ++q) {
                results[q] = pq_search_fastscan(base, base_codes_soa,
                                                queries + q * dim,
                                                base_num, dim, k, p, pq);
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            double time_fastscan = std::chrono::duration<double, std::milli>(t2 - t1).count();

            float avg_recall = 0.0f;
            for (int q = 0; q < query_num; ++q) {
                avg_recall += compute_recall(gt[q], results[q], k);
            }
            avg_recall /= query_num;

            std::cout << std::setw(8) << p
                      << std::setw(15) << std::fixed << std::setprecision(3) << (time_fastscan / query_num)
                      << std::setw(12) << std::fixed << std::setprecision(4) << avg_recall
                      << std::setw(12) << std::fixed << std::setprecision(2) << (time_serial / time_fastscan)
                      << std::setw(12) << std::fixed << std::setprecision(2) << (time_pq_ref / time_fastscan)
                      << std::endl;
        }
    }

    delete[] base;
    delete[] queries;

    return 0;
}

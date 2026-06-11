#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sys/time.h>
#include "search_pthread.h"
#include "pq_quantizer.h"

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d) {
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "Cannot open file " << data_path << "\n";
        exit(1);
    }
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(size_t i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();
    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";
    return data;
}

float compute_recall(const std::vector<Neighbor>& ground_truth,
                     const std::vector<Neighbor>& result, int k)
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
    size_t base_num = 0, query_num = 0, gt_d = 0, dim = 0;
    std::string data_path = "/anndata/";

    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_num, dim);
    auto queries = LoadData<float>(data_path + "DEEP100K.query.fbin", query_num, dim);
    auto gt_data = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", query_num, gt_d);

    query_num = 2000;
    const int k = 10;
    const int m = 4;
    const int ksub = 256;
    const int p = 100;
    const std::vector<int> thread_counts = {1, 2, 4, 8};

    std::cerr << "Dataset: DEEP100K | Base: " << base_num
              << " | Queries: " << query_num << " | Dim: " << dim << "\n";

    std::vector<std::vector<Neighbor>> gt_results(query_num);
    for (int q = 0; q < query_num; ++q) {
        for (int i = 0; i < k; ++i) {
            gt_results[q].push_back({gt_data[q * gt_d + i], 0.0f});
        }
    }

    // 训练PQ
    std::cerr << "Training PQ (m=" << m << ", ksub=" << ksub << ")...\n";
    PQQuantizer pq(dim, m, ksub);
    auto t1 = std::chrono::high_resolution_clock::now();
    pq.train(base, base_num);
    auto t2 = std::chrono::high_resolution_clock::now();
    double train_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::cerr << train_ms << " ms\n";

    auto base_codes = pq.encode_batch(base, base_num);
    std::cerr << "Encoded " << base_num << " vectors. Compression: "
              << (float)(dim * sizeof(float)) / (m * sizeof(uint8_t)) << "x\n";

    // 串行SIMD基线
    std::cerr << "Running Serial SIMD baseline...\n";
    std::vector<std::vector<Neighbor>> serial_results(query_num);
    t1 = std::chrono::high_resolution_clock::now();
    for (int q = 0; q < query_num; ++q) {
        serial_results[q] = pq_search(base, base_codes, queries + q * dim,
                                       base_num, dim, k, p, pq);
    }
    t2 = std::chrono::high_resolution_clock::now();
    double time_serial_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    float serial_recall = 0.0f;
    for (int q = 0; q < query_num; ++q) {
        serial_recall += compute_recall(gt_results[q], serial_results[q], k);
    }
    serial_recall /= query_num;

    std::cerr << time_serial_ms << " ms total, "
              << time_serial_ms / query_num << " ms/query, recall=" << serial_recall << "\n";

    float final_recall = 0.0f;
    float final_latency_us = 0.0f;

    // Query级Pthread并行
    std::cout << "Threads | Total(ms) | Avg(ms/q) | Speedup | Recall" << std::endl;

    for (int nt : thread_counts) {
        t1 = std::chrono::high_resolution_clock::now();
        auto results = pq_search_pthread(base, base_codes, queries,
                                          base_num, dim, query_num, k, p, nt, pq);
        t2 = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

        float avg_recall = 0.0f;
        for (int q = 0; q < query_num; ++q) {
            avg_recall += compute_recall(gt_results[q], results[q], k);
        }
        avg_recall /= query_num;

        if (nt == 8) {
            final_recall = avg_recall;
            final_latency_us = (time_ms / query_num) * 1000.0f;
        }

        std::cout << std::setw(7) << nt << " | "
                  << std::setw(9) << std::fixed << std::setprecision(2) << time_ms << " | "
                  << std::setw(9) << std::fixed << std::setprecision(3) << (time_ms / query_num) << " | "
                  << std::setw(7) << std::fixed << std::setprecision(2) << (time_serial_ms / time_ms) << " | "
                  << std::setw(6) << std::fixed << std::setprecision(4) << avg_recall
                  << std::endl;
    }

    // Base分区Pthread并行
    std::cout << "Threads | Total(ms) | Avg(ms/q) | Speedup | Recall" << std::endl;

    for (int nt : thread_counts) {
        t1 = std::chrono::high_resolution_clock::now();
        auto results = pq_search_pthread_partition(base, base_codes, queries,
                                                    base_num, dim, query_num, k, p, nt, pq);
        t2 = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

        float avg_recall = 0.0f;
        for (int q = 0; q < query_num; ++q) {
            avg_recall += compute_recall(gt_results[q], results[q], k);
        }
        avg_recall /= query_num;

        if (nt == 8) {
            final_recall = avg_recall;
            final_latency_us = (time_ms / query_num) * 1000.0f;
        }

        std::cout << std::setw(7) << nt << " | "
                  << std::setw(9) << std::fixed << std::setprecision(2) << time_ms << " | "
                  << std::setw(9) << std::fixed << std::setprecision(3) << (time_ms / query_num) << " | "
                  << std::setw(7) << std::fixed << std::setprecision(2) << (time_serial_ms / time_ms) << " | "
                  << std::setw(6) << std::fixed << std::setprecision(4) << avg_recall
                  << std::endl;
    }

    std::cout << "\naverage recall: " << final_recall << "\n";
    std::cout << "average latency (us): " << final_latency_us << "\n";

    delete[] base;
    delete[] queries;
    delete[] gt_data;
    return 0;
}

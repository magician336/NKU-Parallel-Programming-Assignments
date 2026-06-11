#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sys/time.h>
#include "search_openmp.h"

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
    const DistanceType dtype = DistanceType::InnerProduct;
    const std::vector<int> thread_counts = {1, 2, 4, 8};

    std::cerr << "Dataset: DEEP100K | Base: " << base_num
              << " | Queries: " << query_num << " | Dim: " << dim << "\n";

    std::vector<std::vector<Neighbor>> gt_results(query_num);
    for (int q = 0; q < query_num; ++q) {
        for (int i = 0; i < k; ++i) {
            gt_results[q].push_back({gt_data[q * gt_d + i], 0.0f});
        }
    }

    // 预热
    {
        volatile float dummy = 0.0f;
        for (size_t i = 0; i < base_num; ++i) {
            dummy += inner_product_serial(queries, base + i * dim, dim);
        }
    }

    // 串行基线
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int q = 0; q < query_num; ++q) {
        flat_search_simd(base, queries + q * dim, base_num, dim, k, dtype);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    double time_serial_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    std::cerr << time_serial_ms << " ms total, "
              << time_serial_ms / query_num << " ms/query\n";

    float final_recall = 0.0f;
    float final_latency_us = 0.0f;

    std::cout << "Threads | Total(ms) | Avg(ms/q) | Speedup | Recall" << std::endl;

    for (int nt : thread_counts) {
        std::vector<std::vector<Neighbor>> results(query_num);
        t1 = std::chrono::high_resolution_clock::now();
        for (int q = 0; q < query_num; ++q) {
            results[q] = flat_search_openmp_partition_single(
                base, queries + q * dim, base_num, dim, k, 2 * k, nt, dtype);
        }
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

    // OpenMP Query级并行
    std::cout << "Threads | Total(ms) | Avg(ms/q) | Speedup | Recall" << std::endl;

    for (int nt : thread_counts) {
        omp_set_num_threads(nt);
        std::vector<std::vector<Neighbor>> results(query_num);

        t1 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for schedule(dynamic)
        for (int q = 0; q < query_num; ++q) {
            results[q] = flat_search_simd(base, queries + q * dim, base_num, dim, k, dtype);
        }
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

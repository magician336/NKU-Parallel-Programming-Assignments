#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <set>
#include "hnswlib/hnswlib.h"
#include "thread_pool.h"

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d) {
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "Cannot open file " << data_path << "\n";
        exit(1);
    }
    fin.read((char*)&n, 4);
    fin.read((char*)&d, 4);
    T* data = new T[n * d];
    int sz = sizeof(T);
    for (size_t i = 0; i < n; ++i) {
        fin.read(((char*)data + i * d * sz), d * sz);
    }
    fin.close();
    std::cerr << "load data " << data_path << "\n";
    std::cerr << "dimension: " << d << "  number:" << n << "\n";
    return data;
}

float compute_recall(const std::vector<std::pair<float, size_t>>& ground_truth,
                     const std::vector<std::pair<float, size_t>>& result,
                     int k)
{
    int hit = 0;
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            if (result[i].second == ground_truth[j].second) {
                ++hit;
                break;
            }
        }
    }
    return (k > 0) ? (float)hit / (float)k : 0.0f;
}

std::vector<std::pair<float, size_t>> exact_search(
    const float* base,
    const float* query,
    size_t base_num,
    size_t dim,
    int k)
{
    std::vector<std::pair<float, size_t>> dists;
    dists.reserve(base_num);
    for (size_t i = 0; i < base_num; ++i) {
        float sum = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            float diff = query[d] - base[i * dim + d];
            sum += diff * diff;
        }
        dists.push_back({sum, i});
    }
    std::nth_element(dists.begin(), dists.begin() + k, dists.end());
    std::sort(dists.begin(), dists.begin() + k);
    dists.resize(k);
    return dists;
}

// Batch并行辅助

inline std::vector<std::pair<float, size_t>> extract_from_pq(
    std::priority_queue<std::pair<float, size_t>> pq)
{
    std::vector<std::pair<float, size_t>> vec;
    vec.reserve(pq.size());
    while (!pq.empty()) {
        vec.push_back({pq.top().first, pq.top().second});
        pq.pop();
    }
    std::sort(vec.begin(), vec.end());
    return vec;
}

// Layer-0随机入口并行(Pthread)

std::vector<std::pair<float, size_t>> hnsw_search_layer0_random_parallel_pthread(
    hnswlib::HierarchicalNSW<float>& index,
    const float* query,
    int k,
    int ef,
    int num_threads)
{
    using tableint = hnswlib::tableint;
    using dist_t = float;

    size_t total_nodes = index.cur_element_count.load();

    // 随机选Layer-0入口点
    std::vector<tableint> entry_points;
    entry_points.reserve(num_threads);
    std::set<tableint> used;
    while ((int)entry_points.size() < num_threads && entry_points.size() < total_nodes) {
        tableint rand_id = rand() % total_nodes;
        if (used.insert(rand_id).second) {
            entry_points.push_back(rand_id);
        }
    }
    int actual_threads = (int)entry_points.size();

    // 各入口并行搜索
    tp::set_num_threads(actual_threads);
    std::vector<std::vector<std::pair<dist_t, tableint>>> local_results(actual_threads);

    tp::parallel_for_static(0, static_cast<size_t>(actual_threads), [&](size_t t) {
        auto pq = index.searchBaseLayerST<true>(entry_points[t], query, std::max(ef, k));
        while (!pq.empty()) {
            local_results[t].push_back({pq.top().first, pq.top().second});
            pq.pop();
        }
    });

    // 合并去重
    std::vector<std::pair<dist_t, tableint>> all;
    for (int t = 0; t < actual_threads; ++t) {
        all.insert(all.end(), local_results[t].begin(), local_results[t].end());
    }
    std::sort(all.begin(), all.end());
    all.erase(std::unique(all.begin(), all.end(),
        [](const std::pair<dist_t, tableint>& a, const std::pair<dist_t, tableint>& b) { return a.second == b.second; }), all.end());

    std::vector<std::pair<float, size_t>> result;
    result.reserve(k);
    for (size_t i = 0; i < all.size() && (int)result.size() < k; ++i) {
        result.push_back({all[i].first, (size_t)all[i].second});
    }
    return result;
}

// 评估辅助

void evaluate_strategy(
    const std::string& strategy_name,
    hnswlib::HierarchicalNSW<float>& index,
    const float* queries,
    const std::vector<std::vector<std::pair<float, size_t>>>& gt,
    size_t query_num, size_t dim, int k,
    const std::vector<int>& ef_configs,
    const std::vector<int>& thread_counts)
{
    std::cout << "EF    | Threads | Total(ms) | Avg(ms/q) | Speedup | Recall" << std::endl;

    for (int ef : ef_configs) {
        index.setEf(ef);

        // 串行基线
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int q = 0; q < (int)query_num; ++q) {
            auto pq = index.searchKnn(queries + q * dim, k);
            while (!pq.empty()) pq.pop();
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        double time_serial_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

        for (int nt : thread_counts) {
            std::vector<std::vector<std::pair<float, size_t>>> results(query_num);
            t1 = std::chrono::high_resolution_clock::now();

            if (strategy_name == "Batch-level Parallel") {
                tp::set_num_threads(nt);
                tp::parallel_for_static(0, query_num, [&](size_t q) {
                    auto pq = index.searchKnn(queries + q * dim, k);
                    results[q] = extract_from_pq(pq);
                });
            } else {  // Layer-0随机并行
                for (int q = 0; q < (int)query_num; ++q) {
                    results[q] = hnsw_search_layer0_random_parallel_pthread(index, queries + q * dim, k, ef, nt);
                }
            }

            t2 = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

            float avg_recall = 0.0f;
            for (int q = 0; q < (int)query_num; ++q) {
                avg_recall += compute_recall(gt[q], results[q], k);
            }
            avg_recall /= query_num;

            double avg_latency_us = (time_ms / query_num) * 1000.0;
            double qps = query_num / (time_ms / 1000.0);

            std::cerr << "Method: " << strategy_name
                      << " | Threads: " << nt << " | EF: " << ef
                      << " | Recall: " << avg_recall
                      << " | Latency: " << avg_latency_us << " us | QPS: " << qps << "\n";

            std::cout << std::setw(5) << ef << " | "
                      << std::setw(7) << nt << " | "
                      << std::setw(9) << std::fixed << std::setprecision(2) << time_ms << " | "
                      << std::setw(9) << std::fixed << std::setprecision(3) << (time_ms / query_num) << " | "
                      << std::setw(7) << std::fixed << std::setprecision(2) << (time_serial_ms / time_ms) << " | "
                      << std::setw(6) << std::fixed << std::setprecision(4) << avg_recall
                      << std::endl;
        }
    }
}

int main() {
    srand((unsigned)time(nullptr));

    size_t base_num = 0, query_num = 0, dim = 0;
    std::string data_path = "/anndata/";

    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_num, dim);
    auto queries = LoadData<float>(data_path + "DEEP100K.query.fbin", query_num, dim);

    query_num = 2000;
    const int k = 10;
    const int M = 16;
    const int efConstruction = 200;
    const std::vector<int> ef_configs = {50, 100, 200, 400};
    const std::vector<int> thread_counts = {1, 2, 4, 8};

    std::cerr << "Dataset: DEEP100K | Base: " << base_num
              << " | Queries: " << query_num << " | Dim: " << dim << "\n";

    // 构建HNSW索引
    std::cerr << "Building HNSW index (M=" << M << ", efConstruction=" << efConstruction << ")...\n";
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, base_num, M, efConstruction);
    for (size_t i = 0; i < base_num; ++i) {
        index.addPoint(base + i * dim, i);
    }
    std::cerr << "Index built.\n";

    std::cerr << "Computing Ground Truth...\n";
    std::vector<std::vector<std::pair<float, size_t>>> gt(query_num);
    for (int q = 0; q < query_num; ++q) {
        gt[q] = exact_search(base, queries + q * dim, base_num, dim, k);
    }

    // Batch并行评估
    evaluate_strategy("Batch-level Parallel", index, queries, gt, query_num, dim, k,
                      ef_configs, thread_counts);

    // Layer-0随机并行评估
    evaluate_strategy("Layer-0 Random Parallel", index, queries, gt, query_num, dim, k,
                      ef_configs, thread_counts);

    std::cout << "\nAll results printed above." << std::endl;

    delete[] base;
    delete[] queries;
    return 0;
}

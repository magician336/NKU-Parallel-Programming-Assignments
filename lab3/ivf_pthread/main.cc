#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <queue>
#include <ctime>

#include "profiler.h"
#include "ivf_index.h"
#include "adc_searcher.h"
#include "sdc_searcher.h"
#include "simd_searcher.h"
#include "thread_pool.h"

using namespace std;

// 数据加载

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

struct SearchResult {
    float recall;
    int64_t latency;
};

// 评估运行

void run_evaluation(int thread_count, int nprobe, BaseSearcher* searcher,
                    const float* test_query, const int* test_gt,
                    size_t test_number, size_t vecdim, size_t test_gt_d, size_t k,
                    std::ofstream& csv_file, const std::string& method_name,
                    float* out_recall = nullptr, float* out_latency = nullptr) {

    tp::set_num_threads(thread_count);
    std::vector<SearchResult> results(test_number);

    // Batch计时
    struct timeval batch_val;
    gettimeofday(&batch_val, NULL);

    tp::parallel_region([&](int tid) {
        const size_t chunk = (test_number + static_cast<size_t>(thread_count) - 1) / static_cast<size_t>(thread_count);
        const size_t start_idx = static_cast<size_t>(tid) * chunk;
        const size_t end_idx = std::min(test_number, start_idx + chunk);

        for (size_t i = start_idx; i < end_idx; ++i) {
            const unsigned long Converter = 1000 * 1000;
            struct timeval val;
            gettimeofday(&val, NULL);

            auto res = searcher->search(test_query + i * vecdim, k, nprobe);

            struct timeval newVal;
            gettimeofday(&newVal, NULL);
            int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

            std::set<uint32_t> gtset;
            for(size_t j = 0; j < k; ++j){
                gtset.insert(test_gt[j + i * test_gt_d]);
            }
            size_t acc = 0;
            auto tmp = res;
            while (!tmp.empty()) {
                if(gtset.count(tmp.top().id)) ++acc;
                tmp.pop();
            }
            results[i] = {(float)acc / k, diff};
        }
    });

    struct timeval new_batch_val;
    gettimeofday(&new_batch_val, NULL);
    const unsigned long Converter = 1000 * 1000;
    int64_t batch_diff_us = (new_batch_val.tv_sec * Converter + new_batch_val.tv_usec) - (batch_val.tv_sec * Converter + batch_val.tv_usec);

    float final_recall = 0, final_latency = 0;
    for(size_t i = 0; i < test_number; ++i) {
        final_recall += results[i].recall;
        final_latency += results[i].latency;
    }
    final_recall /= test_number;
    final_latency /= test_number;

    float qps = (float)test_number / ((float)batch_diff_us / 1000000.0f);

    std::cerr << "Method: " << method_name
              << " | Threads: " << thread_count
              << " | NProbe: " << nprobe
              << " | Recall: " << final_recall
              << " | Avg Latency: " << final_latency << " us"
              << " | QPS: " << qps << "\n";

    // 输出格式
    std::cout << std::setw(7) << thread_count << " | "
              << std::setw(9) << std::fixed << std::setprecision(2) << (batch_diff_us / 1000.0f) << " | "
              << std::setw(9) << std::fixed << std::setprecision(3) << (batch_diff_us / 1000.0f / test_number) << " | "
              << std::setw(6) << std::fixed << std::setprecision(4) << final_recall
              << std::endl;

    float avg_latency = (float)batch_diff_us / (float)test_number;
    csv_file << method_name << "," << thread_count << "," << nprobe << ","
             << final_recall << "," << avg_latency << "," << qps << "\n";

    if (out_recall) *out_recall = final_recall;
    if (out_latency) *out_latency = final_latency;

    // 打印profiler
    if (thread_count == 8) {
        std::string profile_file = "files/ivfpq_profile_t" + std::to_string(thread_count)
                                   + "_np" + std::to_string(nprobe) + ".csv";
        MicroProfiler::print_and_save(profile_file, test_number);
        MicroProfiler::reset();
    }
}


int main(int argc, char *argv[]) {
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "/anndata/";
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt_data = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

    test_number = 2000;
    const size_t k = 10;
    const int n_lists = 1024;
    const int rerank_ratio = 30;

    std::cerr << "Dataset: DEEP100K | Base: " << base_number
              << " | Queries: " << test_number << " | Dim: " << vecdim << "\n";


    // 构建索引
    IVFPQIndex index(vecdim, n_lists);
    std::cerr << "Building Index...\n";
    index.build(base, base_number);

    ADCSearcher adc_searcher(&index, base, rerank_ratio);
    SDCSearcher sdc_searcher(&index, base, rerank_ratio);

    std::ofstream csv_file("files/ivfpq_tradeoff.csv");
    csv_file << "Method,Threads,NProbe,Recall@10,Latency(us),QPS\n";

    std::vector<int> thread_configs = {1, 2, 4, 8};
    std::vector<int> nprobe_configs = {8, 16, 32, 64, 128};

    float final_recall = 0.0f;
    float final_latency_us = 0.0f;

    // ADC评估
    std::cout << "Threads | Total(ms) | Avg(ms/q) | Recall" << std::endl;

    for (int t : thread_configs) {
        for (int probe : nprobe_configs) {
            std::cerr << "\n>>> Running ADC: Threads=" << t << ", NProbe=" << probe << "\n";
            float tmp_r = 0.0f, tmp_l = 0.0f;
            MicroProfiler::reset();
            run_evaluation(t, probe, &adc_searcher, test_query, test_gt_data,
                           test_number, vecdim, test_gt_d, k, csv_file, "ADC",
                           &tmp_r, &tmp_l);
            if (t == 8 && probe == 128) {
                final_recall = tmp_r;
                final_latency_us = tmp_l;
            }
        }
    }

    // SDC评估
    std::cout << "Threads | Total(ms) | Avg(ms/q) | Recall" << std::endl;

    for (int t : thread_configs) {
        for (int probe : nprobe_configs) {
            std::cerr << "\n>>> Running SDC: Threads=" << t << ", NProbe=" << probe << "\n";
            MicroProfiler::reset();
            run_evaluation(t, probe, &sdc_searcher, test_query, test_gt_data,
                           test_number, vecdim, test_gt_d, k, csv_file, "SDC");
        }
    }

    // IVF-SIMD评估
    std::cout << "Threads | Total(ms) | Avg(ms/q) | Recall" << std::endl;

    SIMDSearcher simd_searcher(&index, base);
    for (int t : thread_configs) {
        for (int probe : nprobe_configs) {
            std::cerr << "\n>>> Running IVF-SIMD: Threads=" << t << ", NProbe=" << probe << "\n";
            MicroProfiler::reset();
            run_evaluation(t, probe, &simd_searcher, test_query, test_gt_data,
                           test_number, vecdim, test_gt_d, k, csv_file, "SIMD");
        }
    }

    csv_file.close();
    tp::shutdown_pool();

    delete[] base;
    delete[] test_query;
    delete[] test_gt_data;

    std::cout << "\naverage recall: " << final_recall << "\n";
    std::cout << "average latency (us): " << final_latency_us << "\n";

    std::cerr << "\nAll evaluations completed. Results in files/ivfpq_tradeoff.csv\n";
    return 0;
}

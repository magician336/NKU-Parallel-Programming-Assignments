#include <mpi.h>
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
#include <sys/time.h>
#include <sys/stat.h>
#include "hnswlib/hnswlib.h"
#include "thread_pool.h"

template <typename T>
T *LoadData(const std::string &filename, int &num, int &dim)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return nullptr;
    }
    in.read((char *)&num, sizeof(int));
    in.read((char *)&dim, sizeof(int));
    T *data = new T[num * dim];
    in.read((char *)data, num * dim * sizeof(T));
    in.close();
    return data;
}

float compute_recall(const std::vector<std::pair<float, size_t>> &ground_truth,
                     const std::vector<std::pair<float, size_t>> &result, int k)
{
    int hit = 0;
    for (int i = 0; i < k; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            if (result[i].second == ground_truth[j].second)
            {
                ++hit;
                break;
            }
        }
    }
    return (k > 0) ? static_cast<float>(hit) / k : 0.0f;
}

// 从 priority_queue 提取为有序 vector
inline std::vector<std::pair<float, size_t>> extract_from_pq(
    std::priority_queue<std::pair<float, size_t>> pq)
{
    std::vector<std::pair<float, size_t>> vec;
    vec.reserve(pq.size());
    while (!pq.empty())
    {
        vec.push_back({pq.top().first, pq.top().second});
        pq.pop();
    }
    std::sort(vec.begin(), vec.end());
    return vec;
}

// 精确 L2 搜索（用于 Rank 0 计算 Ground Truth）
std::vector<std::pair<float, size_t>> exact_search(
    const float *base, const float *query, size_t base_num, size_t dim, int k)
{
    std::vector<std::pair<float, size_t>> dists;
    dists.reserve(base_num);
    for (size_t i = 0; i < base_num; ++i)
    {
        float sum = 0.0f;
        for (size_t d = 0; d < dim; ++d)
        {
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

void run_hnsw_mpi_evaluation(int thread_count, int ef,
                             hnswlib::HierarchicalNSW<float> &index,
                             const float *test_query,
                             const std::vector<std::vector<std::pair<float, size_t>>> &gt,
                             size_t test_number, size_t vecdim, int k,
                             std::ofstream &csv_file, const std::string &method_name,
                             int local_base_number, int rank, int size)
{

    index.setEf(ef);

    // 结果格式：每个查询k个 (dist, global_id)
    struct ResultPair
    {
        float dist;
        int id;
    };
    std::vector<ResultPair> local_results(test_number * k);

    struct timeval start_local, end_local;
    gettimeofday(&start_local, NULL);

    if (thread_count == 1)
    {
        // 单线程串行搜索
        for (size_t q = 0; q < test_number; ++q)
        {
            auto pq = index.searchKnn(test_query + q * vecdim, k);
            int idx = 0;
            while (!pq.empty() && idx < k)
            {
                local_results[q * k + idx] = {pq.top().first,
                                              rank * local_base_number + static_cast<int>(pq.top().second)};
                pq.pop();
                idx++;
            }
            while (idx < k)
            {
                local_results[q * k + idx] = {std::numeric_limits<float>::max(), -1};
                idx++;
            }
        }
    }
    else
    {
        // Pthread batch-level 并行
        tp::set_num_threads(thread_count);
        std::vector<std::vector<std::pair<float, size_t>>> thread_results(test_number);

        tp::parallel_for_static(0, test_number, [&](size_t q)
                                {
            auto pq = index.searchKnn(test_query + q * vecdim, k);
            thread_results[q] = extract_from_pq(pq); });

        for (size_t q = 0; q < test_number; ++q)
        {
            int idx = 0;
            for (const auto &p : thread_results[q])
            {
                if (idx >= k)
                    break;
                local_results[q * k + idx] = {p.first,
                                              rank * local_base_number + static_cast<int>(p.second)};
                idx++;
            }
            while (idx < k)
            {
                local_results[q * k + idx] = {std::numeric_limits<float>::max(), -1};
                idx++;
            }
        }
    }

    gettimeofday(&end_local, NULL);
    double local_time_us = (end_local.tv_sec - start_local.tv_sec) * 1000000.0 + (end_local.tv_usec - start_local.tv_usec);

    double max_local_time_us = 0.0;
    double sum_local_time_us = 0.0;
    MPI_Reduce(&local_time_us, &max_local_time_us, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_time_us, &sum_local_time_us, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Gather 结果
    struct timeval start_gather, end_gather;
    gettimeofday(&start_gather, NULL);

    std::vector<ResultPair> global_results;
    if (rank == 0)
    {
        global_results.resize(test_number * k * size);
    }
    MPI_Gather(local_results.data(), static_cast<int>(test_number * k * sizeof(ResultPair)), MPI_BYTE,
               global_results.data(), static_cast<int>(test_number * k * sizeof(ResultPair)), MPI_BYTE,
               0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        // Merge 并计算 recall
        double total_recall = 0.0;
        for (size_t q = 0; q < test_number; ++q)
        {
            std::vector<std::pair<float, int>> all_cands;
            all_cands.reserve(size * k);
            for (int r = 0; r < size; ++r)
            {
                int offset = r * static_cast<int>(test_number * k) + static_cast<int>(q * k);
                for (size_t j = 0; j < k; ++j)
                {
                    all_cands.push_back({global_results[offset + j].dist, global_results[offset + j].id});
                }
            }
            std::sort(all_cands.begin(), all_cands.end());
            all_cands.erase(std::unique(all_cands.begin(), all_cands.end(),
                                        [](const std::pair<float, int> &a, const std::pair<float, int> &b)
                                        { return a.second == b.second; }),
                            all_cands.end());

            std::vector<std::pair<float, size_t>> result;
            for (size_t i = 0; i < all_cands.size() && result.size() < static_cast<size_t>(k); ++i)
            {
                result.push_back({all_cands[i].first, static_cast<size_t>(all_cands[i].second)});
            }

            total_recall += compute_recall(gt[q], result, k);
        }

        gettimeofday(&end_gather, NULL);
        double gather_time_us = (end_gather.tv_sec - start_gather.tv_sec) * 1000000.0 + (end_gather.tv_usec - start_gather.tv_usec);

        float final_recall = static_cast<float>(total_recall / test_number);
        float latency_per_query = static_cast<float>(max_local_time_us / test_number);
        float qps = static_cast<float>(test_number) / (max_local_time_us / 1000000.0f);
        float imbalance_ratio = (sum_local_time_us / size) > 0.0f
                                    ? static_cast<float>(max_local_time_us / (sum_local_time_us / size))
                                    : 1.0f;
        float mpi_overhead = static_cast<float>(gather_time_us / test_number);

        std::cerr << std::fixed << std::setprecision(4);
        std::cerr << "Method: " << method_name
                  << " | Threads: " << thread_count
                  << " | EF: " << ef
                  << " | Recall: " << final_recall
                  << " | Latency: " << latency_per_query << " us"
                  << " | QPS: " << qps
                  << " | Imbalance: " << imbalance_ratio
                  << " | MPIOverhead: " << mpi_overhead << " us\n";

        std::cout << std::setw(7) << thread_count << " | "
                  << std::setw(5) << ef << " | "
                  << std::setw(9) << std::fixed << std::setprecision(2) << (latency_per_query / 1000.0f) << " | "
                  << std::setw(6) << std::fixed << std::setprecision(4) << final_recall
                  << std::setw(10) << std::fixed << std::setprecision(2) << qps
                  << std::setw(12) << std::fixed << std::setprecision(3) << imbalance_ratio
                  << std::endl;

        csv_file << method_name << ","
                 << thread_count << ","
                 << ef << ","
                 << final_recall << ","
                 << latency_per_query << ","
                 << qps << ","
                 << imbalance_ratio << ","
                 << mpi_overhead << "\n";
        csv_file.flush();
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string data_path = "/anndata/";

    int base_number = 0, vecdim = 0;
    int test_number = 0, query_dim = 0;

    float *full_base = nullptr;
    float *test_query = nullptr;

    std::vector<std::vector<std::pair<float, size_t>>> gt; // 仅在 Rank 0 使用

    if (rank == 0)
    {
        std::cerr << "Rank 0 loading data...\n";
        full_base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
        test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, query_dim);

        if (!full_base || !test_query)
        {
            std::cerr << "Data files missing!\n";
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        test_number = 2000;
        const int k = 10;

        std::cerr << "Base: " << base_number << " | Queries: " << test_number
                  << " | Dim: " << vecdim << " | MPI_size: " << size << "\n";

        // 计算 Ground Truth（基于完整 base）
        std::cerr << "Computing Ground Truth...\n";
        gt.resize(test_number);
        for (int q = 0; q < test_number; ++q)
        {
            gt[q] = exact_search(full_base, test_query + q * vecdim, base_number, vecdim, k);
        }
    }

    // 广播元数据
    MPI_Bcast(&base_number, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&vecdim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&test_number, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter base 数据
    int local_base_number = base_number / size;
    float *local_base = new float[local_base_number * vecdim];

    MPI_Scatter(full_base, local_base_number * vecdim, MPI_FLOAT,
                local_base, local_base_number * vecdim, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    // 广播查询数据
    if (rank != 0)
    {
        test_query = new float[test_number * vecdim];
    }
    MPI_Bcast(test_query, test_number * vecdim, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0 && full_base != nullptr)
    {
        delete[] full_base;
        full_base = nullptr;
    }

    // 各进程在本地数据上构建 HNSW
    if (rank == 0)
    {
        std::cerr << "Building local HNSW indexes...\n";
    }

    hnswlib::L2Space space(vecdim);
    hnswlib::HierarchicalNSW<float> index(&space, local_base_number, 16, 200);

    for (int i = 0; i < local_base_number; ++i)
    {
        index.addPoint(local_base + i * vecdim, i);
    }

    if (rank == 0)
    {
        std::cerr << "Local HNSW build done.\n";
    }

    // 评测
    std::ofstream csv_file;
    if (rank == 0)
    {
        mkdir("files", 0777);
        csv_file.open("files/hnsw_mpi_results.csv");
        csv_file << "Method,Threads,EF,Recall,Latency_us,QPS,ImbalanceRatio,MPIOverhead_us\n";

        std::cerr << "\n=== HNSW MPI Evaluation ===\n";

        std::cout << "Threads | EF    | Lat(ms/q) | Recall  |      QPS |  Imbalance" << std::endl;
    }

    std::vector<int> thread_configs = {1, 2, 4, 8};
    std::vector<int> ef_configs = {50, 100, 200, 400};
    const int k = 10;

    for (int ef : ef_configs)
    {
        for (int t : thread_configs)
        {
            run_hnsw_mpi_evaluation(t, ef, index, test_query, gt,
                                    test_number, vecdim, k, csv_file,
                                    "HNSW-MPI", local_base_number, rank, size);
        }
    }

    if (rank == 0)
    {
        csv_file.close();
        std::cerr << "\nResults saved to files/hnsw_mpi_results.csv\n";
    }

    delete[] local_base;
    delete[] test_query;

    MPI_Finalize();
    return 0;
}

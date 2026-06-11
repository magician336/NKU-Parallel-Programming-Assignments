#include <mpi.h>
#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <sys/time.h>
#include <sys/stat.h>
#include <unistd.h>
#include <queue>
#include <ctime>

#include "profiler.h"
#include "ivf_hnsw_index.h"
#include "ivf_hnsw_searcher.h"
#include "thread_pool.h"

using namespace std;

struct DistancePair {
    float dist;
    int id;
};

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

template<typename SearcherType>
void run_mpi_evaluation(int thread_count, int nprobe, SearcherType* searcher,
                        const float* test_query, const int* test_gt,
                        size_t test_number, size_t vecdim, size_t test_gt_d, size_t k,
                        std::ofstream& csv_file, const std::string& method_name,
                        int local_base_number, int rank, int size) {

    tp::set_num_threads(thread_count);
    std::vector<DistancePair> local_results(test_number * k);

    // Rank 0: 并行计算所有查询的粗量化距离和探查目标
    std::vector<Candidate> all_probes(test_number * nprobe);
    if (rank == 0) {
        tp::parallel_region([&](int tid) {
            const size_t chunk = (test_number + static_cast<size_t>(thread_count) - 1) / static_cast<size_t>(thread_count);
            const size_t start_idx = static_cast<size_t>(tid) * chunk;
            const size_t end_idx = std::min(test_number, start_idx + chunk);

            const auto* idx = searcher->get_index();
            int n_lists = idx->n_lists;
            std::vector<float> dists(n_lists);
            std::vector<Candidate> coarse_cands(n_lists);

            for (size_t i = start_idx; i < end_idx; ++i) {
                const float* q = test_query + i * vecdim;
                compute_all_L2_sqr_d96(q, idx->ivf_centroids.data(), n_lists, dists.data());
                for (int c = 0; c < n_lists; ++c) coarse_cands[c] = {dists[c], static_cast<uint32_t>(c)};
                std::partial_sort(coarse_cands.begin(), coarse_cands.begin() + nprobe, coarse_cands.end(),
                                  [](const Candidate& a, const Candidate& b){ return a.dist < b.dist; });
                for (int p = 0; p < nprobe; ++p) {
                    all_probes[i * nprobe + p] = coarse_cands[p];
                }
            }
        });
    }

    // Rank 0 广播所有 Query 的倒排桶探查目标
    MPI_Bcast(all_probes.data(), static_cast<int>(test_number * nprobe * sizeof(Candidate)), MPI_BYTE, 0, MPI_COMM_WORLD);

    // 各进程本地搜索（Pthread 并行）
    struct timeval start_local, end_local;
    gettimeofday(&start_local, NULL);

    tp::parallel_region([&](int tid) {
        const size_t chunk = (test_number + static_cast<size_t>(thread_count) - 1) / static_cast<size_t>(thread_count);
        const size_t start_idx = static_cast<size_t>(tid) * chunk;
        const size_t end_idx = std::min(test_number, start_idx + chunk);

        for (size_t i = start_idx; i < end_idx; ++i) {
            const float* query_vec = test_query + i * vecdim;
            const Candidate* my_probes = &all_probes[i * nprobe];

            std::priority_queue<Candidate> pq_res = searcher->search(query_vec, k, nprobe, my_probes);

            int idx = static_cast<int>(k) - 1;
            while (!pq_res.empty() && idx >= 0) {
                Candidate c = pq_res.top();
                pq_res.pop();
                DistancePair dp;
                dp.dist = c.dist;
                dp.id = static_cast<int>(c.id);
                local_results[i * k + idx] = dp;
                idx--;
            }
            while (idx >= 0) {
                local_results[i * k + idx] = {std::numeric_limits<float>::max(), -1};
                idx--;
            }
        }
    });

    gettimeofday(&end_local, NULL);
    double local_time_us = (end_local.tv_sec - start_local.tv_sec) * 1000000.0 + (end_local.tv_usec - start_local.tv_usec);

    double max_local_time_us = 0.0;
    double sum_local_time_us = 0.0;
    MPI_Reduce(&local_time_us, &max_local_time_us, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_time_us, &sum_local_time_us, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Gather 结果
    struct timeval start_gather, end_gather;
    gettimeofday(&start_gather, NULL);

    std::vector<DistancePair> global_gathered_results;
    if (rank == 0) {
        global_gathered_results.resize(test_number * k * size);
    }

    MPI_Gather(local_results.data(), static_cast<int>(test_number * k * sizeof(DistancePair)), MPI_BYTE,
               global_gathered_results.data(), static_cast<int>(test_number * k * sizeof(DistancePair)), MPI_BYTE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::vector<int> final_global_indices(test_number * k);
        for (size_t q = 0; q < test_number; ++q) {
            std::vector<DistancePair> candidates;
            candidates.reserve(size * k);
            for (int r = 0; r < size; ++r) {
                int offset = r * static_cast<int>(test_number * k) + static_cast<int>(q * k);
                for (size_t j = 0; j < k; ++j) {
                    candidates.push_back(global_gathered_results[offset + j]);
                }
            }
            std::sort(candidates.begin(), candidates.end(), [](const DistancePair& a, const DistancePair& b) {
                return a.dist < b.dist;
            });
            for (size_t j = 0; j < k; ++j) {
                final_global_indices[q * k + j] = candidates[j].id;
            }
        }

        gettimeofday(&end_gather, NULL);
        double gather_merge_time_us = (end_gather.tv_sec - start_gather.tv_sec) * 1000000.0
                                       + (end_gather.tv_usec - start_gather.tv_usec);

        double total_recall = 0.0;
        for (size_t q = 0; q < test_number; ++q) {
            std::set<uint32_t> gtset;
            for (size_t j = 0; j < k; ++j) {
                gtset.insert(test_gt[q * test_gt_d + j]);
            }
            size_t match_count = 0;
            for (size_t j = 0; j < k; ++j) {
                if (gtset.count(final_global_indices[q * k + j])) match_count++;
            }
            total_recall += static_cast<double>(match_count) / k;
        }

        float final_recall = static_cast<float>(total_recall / test_number);
        float local_latency_per_query = static_cast<float>(max_local_time_us / test_number);
        float qps = static_cast<float>(test_number) / (max_local_time_us / 1000000.0f);
        float imbalance_ratio = (sum_local_time_us / size) > 0.0f
                                    ? static_cast<float>(max_local_time_us / (sum_local_time_us / size))
                                    : 1.0f;
        float mpi_overhead_per_query = static_cast<float>(gather_merge_time_us / test_number);

        std::cerr << std::fixed << std::setprecision(4);
        std::cerr << "Method: " << method_name
                  << " | Threads: " << thread_count
                  << " | NProbe: " << nprobe
                  << " | Recall: " << final_recall
                  << " | Latency: " << local_latency_per_query << " us"
                  << " | QPS: " << qps
                  << " | Imbalance: " << imbalance_ratio
                  << " | MPIOverhead: " << mpi_overhead_per_query << " us\n";

        std::cout << std::setw(7) << thread_count << " | "
                  << std::setw(7) << nprobe << " | "
                  << std::setw(9) << std::fixed << std::setprecision(2) << (local_latency_per_query / 1000.0f) << " | "
                  << std::setw(6) << std::fixed << std::setprecision(4) << final_recall
                  << std::setw(10) << std::fixed << std::setprecision(2) << qps
                  << std::setw(12) << std::fixed << std::setprecision(3) << imbalance_ratio
                  << std::endl;

        csv_file << method_name << ","
                 << thread_count << ","
                 << nprobe << ","
                 << final_recall << ","
                 << local_latency_per_query << ","
                 << qps << ","
                 << imbalance_ratio << ","
                 << mpi_overhead_per_query << "\n";
        csv_file.flush();
    }
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string data_path = "/anndata/";

    int n_lists = 64;
    int k = 10;
    int hnsw_m = 16;
    int hnsw_ef_construction = 200;

    int base_number = 0, vecdim = 0;
    int test_number = 0, query_dim = 0;
    int test_gt_d = 0;

    float* full_base = nullptr;
    float* test_query = nullptr;
    int* test_gt = nullptr;

    if (rank == 0) {
        std::cerr << "Rank 0 loading data...\n";
        full_base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
        test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, query_dim);
        test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);

        if (!full_base || !test_query || !test_gt) {
            std::cerr << "Data files missing!\n";
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        test_number = 2000;
        mkdir("files", 0777);
        std::cerr << "Base: " << base_number << " | Queries: " << test_number
                  << " | Dim: " << vecdim << " | MPI_size: " << size << "\n";
    }

    MPI_Bcast(&base_number, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&vecdim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&test_number, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_base_number = base_number / size;
    float* local_base = new float[local_base_number * vecdim];

    MPI_Scatter(full_base, local_base_number * vecdim, MPI_FLOAT,
                local_base, local_base_number * vecdim, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    if (rank != 0) {
        test_query = new float[test_number * vecdim];
    }
    MPI_Bcast(test_query, test_number * vecdim, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0 && full_base != nullptr) {
        delete[] full_base;
        full_base = nullptr;
    }

    IVFHNSWIndex index(vecdim, n_lists, hnsw_m, hnsw_ef_construction);
    index.build(local_base, local_base_number, rank, size);

    IVFHNSWSearcher searcher(&index);

    std::ofstream csv_file;
    if (rank == 0) {
        csv_file.open("files/ivf_hnsw_mpi_results.csv");
        csv_file << "Method,Threads,NProbe,Recall,Latency_us,QPS,ImbalanceRatio,MPIOverhead_us\n";

        std::cerr << "\n=== IVF-HNSW MPI Evaluation ===\n";

        std::cout << "Threads | NProbe | Lat(ms/q) | Recall  |      QPS |  Imbalance" << std::endl;

    }

    std::vector<int> thread_configs = {1, 2, 4, 8};
    std::vector<int> nprobe_configs = {1, 2, 4, 8, 16};

    for (int t : thread_configs) {
        for (int probe : nprobe_configs) {
            run_mpi_evaluation(t, probe, &searcher, test_query, test_gt,
                               test_number, vecdim, test_gt_d, k, csv_file,
                               "IVF-HNSW-MPI", local_base_number, rank, size);
        }
    }

    if (rank == 0) {
        csv_file.close();
        std::cerr << "\nResults saved to files/ivf_hnsw_mpi_results.csv\n";
        delete[] test_gt;
    }

    delete[] local_base;
    delete[] test_query;

    MPI_Finalize();
    return 0;
}

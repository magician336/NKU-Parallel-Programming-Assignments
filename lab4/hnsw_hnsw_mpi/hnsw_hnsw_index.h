#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <limits>
#include <cstdlib>
#include <cstring>
#include <new>
#include <mpi.h>

#include "kmeans.h"
#include "profiler.h"
#include "simd_l2.h"
#include "aligned_alloc.h"
#include "thread_pool.h"
#include "hnswlib/hnswlib.h"

class HNSWHNSWIndex
{
public:
    int d;
    int n_lists;
    int hnsw_m;
    int hnsw_ef_construction;

    AlignedVector<float> centroids;

    hnswlib::L2Space *space = nullptr;
    hnswlib::HierarchicalNSW<float> *top_hnsw = nullptr;
    std::vector<hnswlib::HierarchicalNSW<float> *> bottom_hnsws;

    HNSWHNSWIndex(int dim = 96, int nlist = 8, int m = 16, int ef_construction = 200)
        : d(dim), n_lists(nlist), hnsw_m(m), hnsw_ef_construction(ef_construction)
    {
        centroids.resize(n_lists * d);
        bottom_hnsws.resize(n_lists, nullptr);
        space = new hnswlib::L2Space(d);
    }

    ~HNSWHNSWIndex()
    {
        if (top_hnsw != nullptr)
            delete top_hnsw;
        for (auto *graph : bottom_hnsws)
        {
            if (graph != nullptr)
                delete graph;
        }
        if (space != nullptr)
            delete space;
    }

    void build(const float *local_data, size_t local_n, int rank, int size)
    {
        if (rank == 0)
        {
            std::cerr << "HNSW-on-HNSW Build: d=" << d << ", n_lists=" << n_lists
                      << ", HNSW_M=" << hnsw_m << ", HNSW_ef_c=" << hnsw_ef_construction
                      << ", MPI_size=" << size << "\n";
        }

        // 全局 KMeans 训练分区中心并广播
        {
            MicroProfiler::Timer _t("1_Train_Centroids_Bcast");
            int sample_per_node = std::min(static_cast<int>(local_n), 25000);
            std::vector<float> local_sample(static_cast<size_t>(sample_per_node) * d);
            for (int i = 0; i < sample_per_node; ++i)
            {
                size_t idx = (local_n / static_cast<size_t>(sample_per_node)) * static_cast<size_t>(i);
                std::memcpy(&local_sample[static_cast<size_t>(i) * d], &local_data[idx * d], d * sizeof(float));
            }

            std::vector<float> global_sample;
            if (rank == 0)
            {
                global_sample.resize(static_cast<size_t>(sample_per_node) * size * d);
            }
            MPI_Gather(local_sample.data(), sample_per_node * d, MPI_FLOAT,
                       global_sample.data(), sample_per_node * d, MPI_FLOAT,
                       0, MPI_COMM_WORLD);

            if (rank == 0)
            {
                KMeans km(d, n_lists);
                km.train(global_sample.data(), static_cast<size_t>(sample_per_node) * size);
                std::memcpy(centroids.data(), km.centroids.data(), n_lists * d * sizeof(float));
            }
            MPI_Bcast(centroids.data(), n_lists * d, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }

        // 在中心点上建顶层 HNSW
        {
            MicroProfiler::Timer _t("2_Build_Top_HNSW");
            top_hnsw = new hnswlib::HierarchicalNSW<float>(space, n_lists, hnsw_m, hnsw_ef_construction);
            for (int c = 0; c < n_lists; ++c)
            {
                top_hnsw->addPoint(centroids.data() + c * d, static_cast<hnswlib::labeltype>(c));
            }
        }

        // 用顶层 HNSW 将本地数据分配到分区
        std::vector<int> assign(local_n);
        std::vector<size_t> bucket_counts(n_lists, 0);

        {
            MicroProfiler::Timer _t("3_Assign_Buckets");
            int num_threads = tp::get_num_threads();
            std::vector<std::vector<size_t>> thread_local_counts(
                static_cast<size_t>(num_threads), std::vector<size_t>(n_lists, 0));

            top_hnsw->setEf(10);

            tp::parallel_region([&](int tid)
                                {
                const size_t chunk = (local_n + static_cast<size_t>(num_threads) - 1) / static_cast<size_t>(num_threads);
                const size_t i0 = static_cast<size_t>(tid) * chunk;
                const size_t i1 = std::min(local_n, i0 + chunk);

                for (size_t i = i0; i < i1; ++i) {
                    auto res = top_hnsw->searchKnn(local_data + i * d, 1);
                    int best_c = static_cast<int>(res.top().second);
                    assign[i] = best_c;
                    thread_local_counts[static_cast<size_t>(tid)][best_c]++;
                } });

            for (int t = 0; t < num_threads; ++t)
            {
                for (int c = 0; c < n_lists; ++c)
                {
                    bucket_counts[c] += thread_local_counts[static_cast<size_t>(t)][c];
                }
            }
        }

        // 按目标进程分类本地数据（用于后续 MPI 重分布）
        auto get_owner = [&](int c)
        { return c % size; };
        std::vector<std::vector<float>> send_lists(size);
        std::vector<std::vector<uint32_t>> send_ids(size);

        for (size_t i = 0; i < local_n; ++i)
        {
            int c = assign[i];
            int owner = get_owner(c);
            for (int j = 0; j < d; ++j)
            {
                send_lists[owner].push_back(local_data[i * d + j]);
            }
            uint32_t global_id = static_cast<uint32_t>(rank * static_cast<int>(local_n) + static_cast<int>(i));
            send_ids[owner].push_back(global_id);
        }

        // MPI 数据重分布：将全局属于同一分区的向量汇聚到负责进程
        {
            MicroProfiler::Timer _t("5_Redistribute_Data");

            // 准备发送计数和位移
            std::vector<int> send_counts(size, 0);
            std::vector<int> send_displs(size, 0);
            for (int i = 0; i < size; ++i)
            {
                send_counts[i] = static_cast<int>(send_ids[i].size());
            }
            send_displs[0] = 0;
            for (int i = 1; i < size; ++i)
            {
                send_displs[i] = send_displs[i - 1] + send_counts[i - 1];
            }

            // 合并发送缓冲区
            std::vector<float> send_buffer_vec;
            std::vector<uint32_t> send_buffer_ids;
            for (int i = 0; i < size; ++i)
            {
                send_buffer_vec.insert(send_buffer_vec.end(), send_lists[i].begin(), send_lists[i].end());
                send_buffer_ids.insert(send_buffer_ids.end(), send_ids[i].begin(), send_ids[i].end());
            }

            // 交换数量
            std::vector<int> recv_counts(size);
            std::vector<int> recv_displs(size);
            MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

            recv_displs[0] = 0;
            for (int i = 1; i < size; ++i)
            {
                recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
            }
            int total_recv = recv_displs[size - 1] + recv_counts[size - 1];

            // 交换数据
            std::vector<int> send_counts_d(size), recv_counts_d(size);
            std::vector<int> send_displs_d(size), recv_displs_d(size);
            for (int i = 0; i < size; ++i)
            {
                send_counts_d[i] = send_counts[i] * d;
                recv_counts_d[i] = recv_counts[i] * d;
            }
            send_displs_d[0] = 0;
            recv_displs_d[0] = 0;
            for (int i = 1; i < size; ++i)
            {
                send_displs_d[i] = send_displs_d[i - 1] + send_counts_d[i - 1];
                recv_displs_d[i] = recv_displs_d[i - 1] + recv_counts_d[i - 1];
            }

            std::vector<float> recv_buffer_vec(static_cast<size_t>(total_recv) * d);
            std::vector<uint32_t> recv_buffer_ids(total_recv);

            MPI_Alltoallv(send_buffer_vec.data(), send_counts_d.data(), send_displs_d.data(), MPI_FLOAT,
                          recv_buffer_vec.data(), recv_counts_d.data(), recv_displs_d.data(), MPI_FLOAT,
                          MPI_COMM_WORLD);
            MPI_Alltoallv(send_buffer_ids.data(), send_counts.data(), send_displs.data(), MPI_UNSIGNED,
                          recv_buffer_ids.data(), recv_counts.data(), recv_displs.data(), MPI_UNSIGNED,
                          MPI_COMM_WORLD);

            // 重新计算全局 bucket_counts 并按分区分类
            std::fill(bucket_counts.begin(), bucket_counts.end(), 0);
            std::vector<std::vector<float>> recv_lists(n_lists);
            std::vector<std::vector<uint32_t>> recv_ids(n_lists);

            for (int idx = 0; idx < total_recv; ++idx)
            {
                float *vec = &recv_buffer_vec[idx * d];
                uint32_t global_id = recv_buffer_ids[idx];
                top_hnsw->setEf(10);
                auto res = top_hnsw->searchKnn((const void *)vec, 1);
                int best_c = static_cast<int>(res.top().second);
                if (get_owner(best_c) == rank)
                {
                    bucket_counts[best_c]++;
                    for (int j = 0; j < d; ++j)
                    {
                        recv_lists[best_c].push_back(vec[j]);
                    }
                    recv_ids[best_c].push_back(global_id);
                }
            }

            if (rank == 0)
            {
                std::cerr << "Redistributed data. Received " << total_recv << " vectors.\n";
            }

            // 初始化并构建底层 HNSW（使用全局数据）
            for (int c = 0; c < n_lists; ++c)
            {
                if (get_owner(c) != rank)
                    continue;
                if (bucket_counts[c] == 0)
                    continue;

                bottom_hnsws[c] = new hnswlib::HierarchicalNSW<float>(
                    space, bucket_counts[c], hnsw_m, hnsw_ef_construction);

                for (size_t i = 0; i < bucket_counts[c]; ++i)
                {
                    bottom_hnsws[c]->addPoint(recv_lists[c].data() + i * d, recv_ids[c][i]);
                }
            }
        }
    }
};

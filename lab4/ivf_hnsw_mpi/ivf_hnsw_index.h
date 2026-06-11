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

class IVFHNSWIndex
{
public:
    int d;
    int n_lists;
    int hnsw_m;
    int hnsw_ef_construction;

    AlignedVector<float> ivf_centroids;

    hnswlib::L2Space *space = nullptr;
    // 每个簇一个 HNSW 图，nullptr 表示本进程不负责该簇
    std::vector<hnswlib::HierarchicalNSW<float> *> hnsw_graphs;

    IVFHNSWIndex(int dim = 96, int nlist = 64, int m = 16, int ef_construction = 200)
        : d(dim), n_lists(nlist), hnsw_m(m), hnsw_ef_construction(ef_construction)
    {
        ivf_centroids.resize(n_lists * d);
        hnsw_graphs.resize(n_lists, nullptr);
        space = new hnswlib::L2Space(d);
    }

    ~IVFHNSWIndex()
    {
        for (auto *graph : hnsw_graphs)
        {
            if (graph != nullptr)
            {
                delete graph;
            }
        }
        if (space != nullptr)
        {
            delete space;
        }
    }

    // 计算本进程负责的簇范围
    void get_my_lists(int rank, int size, int &my_start, int &my_end) const
    {
        int lists_per_node = n_lists / size;
        int remainder = n_lists % size;
        if (rank < remainder)
        {
            my_start = rank * (lists_per_node + 1);
            my_end = my_start + lists_per_node + 1;
        }
        else
        {
            my_start = remainder * (lists_per_node + 1) + (rank - remainder) * lists_per_node;
            my_end = my_start + lists_per_node;
        }
    }

    // 计算簇 c 由哪个进程负责
    int get_owner_of_cluster(int c, int size) const
    {
        int lists_per_node = n_lists / size;
        int remainder = n_lists % size;
        int boundary = remainder * (lists_per_node + 1);
        if (c < boundary)
        {
            return c / (lists_per_node + 1);
        }
        else
        {
            return remainder + (c - boundary) / lists_per_node;
        }
    }

    void build(const float *local_data, size_t local_n, int rank, int size)
    {
        if (rank == 0)
        {
            std::cerr << "IVF-HNSW Build: d=" << d << ", n_lists=" << n_lists
                      << ", HNSW_M=" << hnsw_m << ", HNSW_ef_c=" << hnsw_ef_construction
                      << ", MPI_size=" << size << "\n";
        }

        // 全局 IVF 粗聚类中心训练与广播
        {
            MicroProfiler::Timer _t("1_Train_IVF_Bcast");
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
                KMeans ivf_km(d, n_lists);
                ivf_km.train(global_sample.data(), static_cast<size_t>(sample_per_node) * size);
                std::memcpy(ivf_centroids.data(), ivf_km.centroids.data(), n_lists * d * sizeof(float));
            }
            MPI_Bcast(ivf_centroids.data(), n_lists * d, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }

        // 各进程计算本地数据分配
        std::vector<int> assign(local_n);
        {
            MicroProfiler::Timer _t("2_Compute_Assignments");
            tp::parallel_for_static(0, local_n, [&](size_t i)
                                    {
                float min_dist = std::numeric_limits<float>::max();
                int best_c = 0;
                for (int c = 0; c < n_lists; ++c) {
                    float dist = compute_L2_sqr(local_data + i * d, &ivf_centroids[c * d], d);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_c = c;
                    }
                }
                assign[i] = best_c; });
        }

        // MPI 数据重分布：将全局属于本进程负责簇的向量收集到本地
        {
            MicroProfiler::Timer _t("3_Redistribute_Data");
            int my_start, my_end;
            get_my_lists(rank, size, my_start, my_end);

            // 计算每个本地向量的目标进程（负责其簇的进程）
            std::vector<int> send_counts(size, 0);
            std::vector<int> send_displs(size, 0);
            std::vector<int> dest_rank(local_n);

            for (size_t i = 0; i < local_n; ++i)
            {
                int c = assign[i];
                int owner = get_owner_of_cluster(c, size);
                dest_rank[i] = owner;
                send_counts[owner]++;
            }

            // 计算发送位移
            send_displs[0] = 0;
            for (int i = 1; i < size; ++i)
            {
                send_displs[i] = send_displs[i - 1] + send_counts[i - 1];
            }
            int total_send = send_displs[size - 1] + send_counts[size - 1];

            // 准备发送缓冲区（向量数据 + 全局ID）
            std::vector<float> send_buffer_vec(static_cast<size_t>(total_send) * d);
            std::vector<uint32_t> send_buffer_ids(total_send);
            std::vector<int> send_offsets = send_displs;

            for (size_t i = 0; i < local_n; ++i)
            {
                int owner = dest_rank[i];
                int idx = send_offsets[owner];
                uint32_t global_id = static_cast<uint32_t>(rank * static_cast<int>(local_n) + static_cast<int>(i));
                send_buffer_ids[idx] = global_id;
                for (int j = 0; j < d; ++j)
                {
                    send_buffer_vec[idx * d + j] = local_data[i * d + j];
                }
                send_offsets[owner]++;
            }

            // 使用 MPI_Alltoall 交换数量
            std::vector<int> recv_counts(size);
            std::vector<int> recv_displs(size);
            MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

            recv_displs[0] = 0;
            for (int i = 1; i < size; ++i)
            {
                recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
            }
            int total_recv = recv_displs[size - 1] + recv_counts[size - 1];

            // 使用 MPI_Alltoallv 交换向量数据（counts * d）
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

            // 按簇分类接收到的数据（使用全局质心重新计算簇分配）
            std::vector<std::vector<float>> list_data(n_lists);
            std::vector<std::vector<uint32_t>> list_ids(n_lists);

            for (int idx = 0; idx < total_recv; ++idx)
            {
                float *vec = &recv_buffer_vec[idx * d];
                uint32_t global_id = recv_buffer_ids[idx];

                int best_c = 0;
                float min_dist = std::numeric_limits<float>::max();
                for (int c = 0; c < n_lists; ++c)
                {
                    float dist = compute_L2_sqr(vec, &ivf_centroids[c * d], d);
                    if (dist < min_dist)
                    {
                        min_dist = dist;
                        best_c = c;
                    }
                }

                // 只保留本进程负责的簇（防御性检查）
                if (best_c >= my_start && best_c < my_end)
                {
                    list_data[best_c].push_back(vec[0]);
                    for (int j = 1; j < d; ++j)
                    {
                        list_data[best_c].push_back(vec[j]);
                    }
                    list_ids[best_c].push_back(global_id);
                }
            }

            if (rank == 0)
            {
                std::cerr << "Redistributed data. Received " << total_recv << " vectors.\n";
            }

            // 在负责的簇上建 HNSW（使用全局数据）
            for (int c = my_start; c < my_end; ++c)
            {
                size_t num = list_ids[c].size();
                if (num == 0)
                    continue;

                auto *graph = new hnswlib::HierarchicalNSW<float>(space, num, hnsw_m, hnsw_ef_construction);
                for (size_t i = 0; i < num; ++i)
                {
                    graph->addPoint(list_data[c].data() + i * d, list_ids[c][i]);
                }
                hnsw_graphs[c] = graph;

                if (rank == 0 && (c - my_start) % std::max(1, (my_end - my_start) / 4) == 0)
                {
                    std::cerr << "Rank " << rank << " built HNSW for list " << c
                              << " (" << num << " vectors, global)\n";
                }
            }
        }

        if (rank == 0)
        {
            std::cerr << "IVF-HNSW Build: Distributed build done.\n";
        }
    }
};

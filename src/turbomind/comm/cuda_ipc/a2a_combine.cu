// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/comm/cuda_ipc/common.h"
#include "src/turbomind/comm/cuda_ipc/cuda_ipc_comm.h"
#include "src/turbomind/comm/cuda_ipc/semaphore.cuh"
#include "src/turbomind/comm/cuda_ipc/semaphore.h"

#include "src/turbomind/core/data_type.h"

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/core/meta.h"

namespace turbomind::comm {

template<class T, int vec_size, int tokens_per_batch>
__global__ void AllToAllCombine_Simple_Pull_V2(T*                   hidden,
                                               SystemSemaphoreInfo* semaphores,
                                               Array<T*, kMaxRanks> recv_hidden,
                                               int*                 recv_info,
                                               int                  rank,
                                               int                  ranks,
                                               int*                 token_idx_in_rank,
                                               int                  token_num,
                                               int                  dim,
                                               constant<vec_size>,
                                               constant<tokens_per_batch>)
{
    SystemSemaphore sem(semaphores, ranks, blockIdx.x, threadIdx.x);

    sem.Signal(true);
    sem.Wait(true);
    __syncthreads();

    __shared__ int s_send_idx[tokens_per_batch][kMaxRanks];

    int s_rank_offset[kMaxRanks];
    for (int r = 0; r < ranks; ++r) {
        s_rank_offset[r] = (rank == 0) ? 0 : __ldg(&recv_info[(rank - 1) * ranks + r]);
    }

    using Vec = Array<T, vec_size>;
    using namespace ops;

    const int dim_vecs = dim / vec_size;

    const int total_batches   = cdiv(token_num, tokens_per_batch);
    const int batches_per_cta = cdiv(total_batches, (int)gridDim.x);
    const int batch_start     = blockIdx.x * batches_per_cta;
    const int batch_end       = min(batch_start + batches_per_cta, total_batches);

    for (int batch = batch_start; batch < batch_end; ++batch) {
        const int token_base      = batch * tokens_per_batch;
        const int tokens_in_batch = min(tokens_per_batch, token_num - token_base);
        for (int i = threadIdx.x; i < tokens_in_batch * ranks; i += blockDim.x) {
            const int local_token_idx             = i / ranks;
            const int dst_rank                    = i % ranks;
            const int token_idx                   = token_base + local_token_idx;
            const int index                       = dst_rank * (token_num + 2) + token_idx;
            s_send_idx[local_token_idx][dst_rank] = __ldg(&token_idx_in_rank[index]);
        }
        __syncthreads();

        const int work_per_batch = tokens_in_batch * dim_vecs;
        for (int i = threadIdx.x; i < work_per_batch; i += blockDim.x) {
            const int local_token_idx = i / dim_vecs;
            const int vec_idx         = i % dim_vecs;
            const int token_idx       = token_base + local_token_idx;

            T* src_ptrs[kMaxRanks]{};
            PRAGMA_UNROLL
            for (int r = 0; r < kMaxRanks; ++r) {
                if (r < ranks) {
                    int send_idx = s_send_idx[local_token_idx][r];
                    if (send_idx >= 0) {
                        int offset  = (s_rank_offset[r] + send_idx) * dim + vec_idx * vec_size;
                        src_ptrs[r] = recv_hidden[r] + offset;
                    }
                }
            }

            Vec acc{};
            PRAGMA_UNROLL
            for (int r = 0; r < kMaxRanks; ++r) {
                if (src_ptrs[r]) {
                    Vec tmp;
                    Load(tmp, src_ptrs[r]);
                    acc = acc + tmp;
                }
            }
            Store(hidden + token_idx * dim + vec_idx * vec_size, acc);
        }
        __syncthreads();
    }

    sem.Signal(true);
    sem.Wait(true);
    sem.Update(semaphores, ranks, blockIdx.x, threadIdx.x);
}

template<class T,
         int tokens_per_block,
         int threads_per_token,
         int vec_size>
__global__ void AllToAllCombine_Simple_Pull(T*                   hidden,  //
                                            SystemSemaphoreInfo* semaphores,
                                            Array<T*, kMaxRanks> recv_hidden,
                                            int*                 recv_info,
                                            int                  rank,
                                            int                  ranks,
                                            int*                 token_idx_in_rank,
                                            int                  token_num,
                                            int                  dim,
                                            int                  process_tokens_per_block,
                                            constant<tokens_per_block>,
                                            constant<threads_per_token>,
                                            constant<vec_size>)
{
    SystemSemaphore sem(semaphores, ranks, blockIdx.x, threadIdx.x);

    sem.Signal(false);
    sem.Wait(false);
    __syncthreads();

    int rank_send_offset[kMaxRanks];  // token send offset for each rank
    for (int r = 0; r < ranks; ++r) {
        rank_send_offset[r] = (rank == 0) ? 0 : __ldg(&recv_info[(rank - 1) * ranks + r]);
    }

    const int bi              = blockIdx.x;
    const int token_start_idx = bi * process_tokens_per_block + threadIdx.x / threads_per_token;
    const int token_end_idx   = min((bi + 1) * process_tokens_per_block, token_num);
    const int n_iters         = cdiv(dim, threads_per_token * vec_size);
    const int ti              = threadIdx.x % threads_per_token;

    using Vec = Array<T, vec_size>;
    using namespace ops;
    for (int token_idx = token_start_idx; token_idx < token_end_idx; token_idx += tokens_per_block) {
        T* src[kMaxRanks]{};
        for (int r = 0; r < ranks; ++r) {
            int send_idx = token_idx_in_rank[r * (token_num + 2) + token_idx];
            if (send_idx >= 0) {
                int dst_idx = rank_send_offset[r] + send_idx;
                src[r]      = cvta_generic_to_global(recv_hidden[r]) + (dst_idx * dim);
            }
        }

        T* dst = hidden + token_idx * dim;
        for (int iter = 0; iter < n_iters; ++iter) {
            Vec acc{}, tmp;
            int idx = iter * threads_per_token * vec_size + ti * vec_size;
            if (idx < dim) {
                for (int r = 0; r < ranks; ++r) {
                    if (src[r]) {
                        Load(tmp, src[r] + idx);
                        acc = acc + tmp;
                    }
                }
                Store(dst + idx, acc);
            }
        }
    }

    __syncthreads();
    sem.Signal(true);
    sem.Wait(true);
    sem.Update(semaphores, ranks, blockIdx.x, threadIdx.x);
}

void CudaIpcCommImpl::AllToAllCombine(void*        hidden,
                                      int*         recv_info,
                                      void*        recv_hidden,
                                      int*         token_idx_in_rank,
                                      int          token_num,
                                      int          dim,
                                      DataType     type,
                                      int          group,
                                      cudaStream_t stream)
{
    const int n_ranks = this->n_ranks(group);
    const int rank    = this->rank(group);

    auto semaphore = groups_.at(group).semaphore.handle();

    auto invoke = [&](auto t, auto tokens_per_batch) {
        using T                        = decltype(t);
        auto          symm_recv_hidden = get_symmetric_v2((T*)recv_hidden, group);
        constexpr int vec_size         = sizeof(uint4) / sizeof(T);
        constexpr int threads          = 1024;
        const int     max_ctas         = max_ctas_.apply(48);
        AllToAllCombine_Simple_Pull_V2<<<max_ctas, threads, 0, stream>>>((T*)hidden,
                                                                         semaphore,
                                                                         symm_recv_hidden.uc,
                                                                         recv_info,
                                                                         rank,
                                                                         n_ranks,
                                                                         token_idx_in_rank,
                                                                         token_num,
                                                                         dim,
                                                                         constant<vec_size>{},
                                                                         tokens_per_batch);
        sync_check_cuda_error();
    };

    auto dispatch_tokens = [&](auto t) {
        if (token_num <= 16) {
            return invoke(t, constant<1>{});
        }
        if (token_num <= 64) {
            return invoke(t, constant<4>{});
        }
        else if (token_num <= 256) {
            return invoke(t, constant<16>{});
        }
        return invoke(t, constant<32>{});
    };

    TM_DISPATCH_PRIMARY_DTYPES(type, dispatch_tokens);
}

}  // namespace turbomind::comm

// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/comm/cuda_ipc/cuda_ipc_comm.h"
#include "src/turbomind/comm/cuda_ipc/semaphore.cuh"

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/core/meta.h"

#include "src/turbomind/kernels/gemm/moe_utils_v2.h"

namespace turbomind::comm {

__global__ void AllToAllDispatch_Notify(Array<int*, kMaxRanks> recv_info,  //
                                        SystemSemaphoreInfo*   semaphores,
                                        int*                   token_idx_in_rank,
                                        int                    rank,
                                        int                    ranks,
                                        int                    token_num)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    SystemSemaphore sem(semaphores, ranks, blockIdx.x, threadIdx.x);

    sem.Signal(true);
    sem.Wait(true);

    __syncthreads();

    // use two warps to get token num and expert num for each rank
    const int offset_r = warp_id == 0 ? 0 : 1;
    const int offset_w = warp_id == 0 ? 0 : ranks;

    if (lane_id < ranks) {
        int num = token_idx_in_rank[lane_id * (token_num + 2) + token_num + offset_r];
        for (int i = 0; i < ranks; ++i) {
            auto chn                                 = cvta_generic_to_global(recv_info[i]);
            chn[(offset_w + rank) * ranks + lane_id] = num;
        }
    }

    __syncthreads();

    sem.Signal(false);
    sem.Wait(false);

    __syncthreads();

    if (lane_id < ranks) {
        auto chn = cvta_generic_to_global(recv_info[rank]);
        for (int i = 1; i < ranks; ++i) {
            chn[(offset_w + i) * ranks + lane_id] += chn[(offset_w + i - 1) * ranks + lane_id];
        }
    }

    sem.Signal(true);
    sem.Wait(true);
    sem.Update(semaphores, ranks, blockIdx.x, threadIdx.x);
}

template<class T, int vec_size>
__global__ void __launch_bounds__(1024, 1)  //
    AllToAllDispatch_Simple_Push(Array<T*, kMaxRanks>      recv_hidden,
                                 Array<float*, kMaxRanks>  recv_scales,
                                 Array<int8_t*, kMaxRanks> recv_masks,
                                 SystemSemaphoreInfo*      semaphores,
                                 Array<int*, kMaxRanks>    recv_info,
                                 T*                        hidden,
                                 int                       hidden_load_iters,
                                 float*                    topk_scales,
                                 int*                      topk_experts,
                                 int*                      token_idx_in_rank,
                                 int                       rank,
                                 int                       ranks,
                                 int                       token_num,
                                 int                       dim,
                                 int                       topk,
                                 int                       local_expert_num,
                                 constant<vec_size>)
{
    const int bi = blockIdx.x;

    SystemSemaphore sem(semaphores, ranks, blockIdx.x, threadIdx.x);

    __shared__ int rank_send_offset[kMaxRanks];   // token send offset for each rank
    __shared__ int rank_token_count[kMaxRanks];   // token count for each rank
    __shared__ int rank_token_padded[kMaxRanks];  // padded token count

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id == 0 && lane_id < ranks) {
        rank_send_offset[lane_id] = (rank == 0) ? 0 : __ldg(&recv_info[rank][(rank - 1) * ranks + lane_id]);
    }
    if (warp_id == 1 && lane_id < ranks) {
        rank_token_count[lane_id]  = __ldg(&recv_info[rank][(ranks - 1) * ranks + lane_id]);
        rank_token_padded[lane_id] = round_up(rank_token_count[lane_id], kMoeGateVecSize);
    }

    const int num_threads_per_rank = blockDim.x / ranks;
    const int dst_rank             = threadIdx.x / num_threads_per_rank;
    const int num_warps_per_rank   = num_threads_per_rank / WARP_SIZE;
    const int send_warp_id_in_rank = threadIdx.x / WARP_SIZE % num_warps_per_rank;

    const int tok_per_cta     = cdiv(token_num, (int)gridDim.x);
    const int token_start_idx = bi * tok_per_cta;
    const int token_end_idx   = min(token_start_idx + tok_per_cta, token_num);

    __syncthreads();

    for (int token_idx = token_start_idx + send_warp_id_in_rank; token_idx < token_end_idx;
         token_idx += num_warps_per_rank) {
        int send_idx = __ldg(&token_idx_in_rank[dst_rank * (token_num + 2) + token_idx]);
        int dst_idx  = rank_send_offset[dst_rank] + send_idx;
        if (send_idx >= 0) {
            // hidden
            using Vec = Array<T, vec_size>;
            T* src    = (T*)(hidden + token_idx * dim);
            T* dst    = (T*)(recv_hidden[dst_rank] + dst_idx * dim);
            for (int i = lane_id; i < hidden_load_iters; i += WARP_SIZE) {
                Vec tmp;
                Ldg(tmp, src + i * vec_size);
                Stcg(dst + i * vec_size, tmp);
            }
            // scales and masks
            if (lane_id < topk) {
                const int index = token_idx * ranks * topk + dst_rank * topk + lane_id;

                float s = __ldg(&topk_scales[index]);
                int   e = __ldg(&topk_experts[index]);

                auto scales_chn = recv_scales[dst_rank];
                auto masks_chn  = recv_masks[dst_rank];

                scales_chn[lane_id * rank_token_count[dst_rank] + dst_idx] = s;
                if (e >= 0 && e < local_expert_num) {
                    masks_chn[e * rank_token_padded[dst_rank] + dst_idx] = lane_id;
                }
            }
        }
    }

    __syncthreads();

    sem.Signal(true);
    sem.Wait(true);
    sem.Update(semaphores, ranks, blockIdx.x, threadIdx.x);
}

void CudaIpcCommImpl::AllToAllDispatch(int*         recv_info,
                                       void*        recv_hidden,
                                       float*       recv_scales,
                                       int8_t*      recv_masks,
                                       void*        hidden,
                                       float*       topk_scales,
                                       int*         topk_experts,
                                       int*         token_idx_in_rank,
                                       int          token_num,
                                       int          dim,
                                       int          topk,
                                       int          expert_num,
                                       DataType     type,
                                       int          group,
                                       cudaStream_t stream)
{
    const int n_ranks = this->n_ranks(group);
    const int rank    = this->rank(group);

    auto semaphore = groups_.at(group).semaphore.handle();

    auto invoke = [&](auto t) {
        using T               = decltype(t);
        auto symm_recv_info   = get_symmetric_v2(recv_info, group);
        auto symm_recv_hidden = get_symmetric_v2((T*)recv_hidden, group);
        auto symm_recv_scales = get_symmetric_v2(recv_scales, group);
        auto symm_recv_masks  = get_symmetric_v2(recv_masks, group);

        constexpr int vec_size = sizeof(uint4) / sizeof(T);
        constexpr int threads  = 1024;
        const int     max_ctas = max_ctas_.apply(24);

        TM_CHECK(dim % vec_size == 0);

        AllToAllDispatch_Notify<<<1, WARP_SIZE * 2, 0, stream>>>(  //
            symm_recv_info.uc,
            semaphore,
            token_idx_in_rank,
            rank,
            n_ranks,
            token_num);
        sync_check_cuda_error();

        // TODO
        // 1) wait token_num, expert_num
        // 2) clear mask according to token_num

        AllToAllDispatch_Simple_Push<<<max_ctas, threads, 0, stream>>>(  //
            symm_recv_hidden.uc,
            symm_recv_scales.uc,
            symm_recv_masks.uc,
            semaphore,
            symm_recv_info.uc,
            (T*)hidden,
            dim / vec_size,
            topk_scales,
            topk_experts,
            token_idx_in_rank,
            rank,
            n_ranks,
            token_num,
            dim,
            topk,
            expert_num / n_ranks,
            constant<vec_size>{});
        sync_check_cuda_error();
    };

    TM_DISPATCH_PRIMARY_DTYPES(type, invoke);
}

}  // namespace turbomind::comm

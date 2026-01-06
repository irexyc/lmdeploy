// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/comm/cuda_ipc/common.h"
#include "src/turbomind/comm/cuda_ipc/cuda_ipc_comm.h"
#include "src/turbomind/comm/cuda_ipc/semaphore.cuh"
#include "src/turbomind/comm/cuda_ipc/semaphore.h"

#include "src/turbomind/core/data_type.h"

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/meta.h"

namespace turbomind::comm {

template<class T, int vec_size, class Relaxed>
__global__ void ReduceScatter_Simple_Pull(T*                   buf,  //
                                          Array<T*, kMaxRanks> chns,
                                          SystemSemaphoreInfo* semaphores,
                                          int                  rank,
                                          int                  ranks,
                                          int                  slice,
                                          int                  count,
                                          constant<vec_size>,
                                          Relaxed relaxed)
{
    const int block_num  = gridDim.x;
    const int thread_num = blockDim.x * block_num;
    const int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;

    SystemSemaphore sem(semaphores, ranks, blockIdx.x, threadIdx.x);

    sem.Signal(relaxed);
    sem.Wait(relaxed);

    __syncthreads();

    using Vec = Array<T, vec_size>;

    using namespace ops;

    const int first = rank * slice;
    const int last  = min(count, first + slice);

    for (int idx = first + thread_idx; idx < last; idx += thread_num) {
        Vec acc{}, tmp;
        for (int i = 0; i < ranks; ++i) {
            const int p   = rank + i < ranks ? rank + i : rank + i - ranks;
            auto      chn = cvta_generic_to_global(chns[p]);
            Load(tmp, chn + idx * vec_size);
            acc = acc + tmp;
        }
        Store(buf + idx * vec_size, acc);
    }

    __syncthreads();

    sem.Signal(true);
    sem.Wait(true);

    sem.Update(semaphores, ranks, blockIdx.x, threadIdx.x);
}

void CudaIpcCommImpl::ReduceScatter(const void*  sendbuff,
                                    void*        recvbuff,
                                    size_t       recvcount,
                                    size_t       totalcount,
                                    DataType     type,
                                    int          group,
                                    cudaStream_t stream)
{
    const int n_ranks = this->n_ranks(group);
    const int rank    = this->rank(group);

    FT_CHECK((char*)recvbuff == (char*)sendbuff + rank * recvcount * byte_size(type));  // in-place

    void* data = const_cast<void*>(sendbuff);

    auto semaphore = groups_.at(group).semaphore.handle();

    auto invoke = [&](auto t) {
        using T = decltype(t);

        auto symm_ptr = get_symmetric_v2((T*)data, group);

        constexpr int vec_size = sizeof(uint4) / sizeof(T);
        constexpr int threads  = 1024;
        const int     slice    = recvcount / vec_size;
        const int     max_ctas = max_ctas_.apply(48);
        const int     blocks   = std::min(max_ctas, (slice + threads - 1) / threads);
        ReduceScatter_Simple_Pull<<<blocks, threads, 0, stream>>>((T*)data,
                                                                  symm_ptr.uc,
                                                                  semaphore,
                                                                  rank,
                                                                  n_ranks,
                                                                  slice,
                                                                  totalcount / vec_size,
                                                                  constant<vec_size>{},
                                                                  std::false_type{});
    };

    TM_DISPATCH_PRIMARY_DTYPES(type, invoke);
}

}  // namespace turbomind::comm

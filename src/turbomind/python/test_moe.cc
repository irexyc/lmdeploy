#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <cmath>
#include <iostream>
#include <numeric>

#include <pybind11/pybind11.h>

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/comm/host_comm.h"
#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/tensor.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/utils/cuda_utils.h"

#include "src/turbomind/kernels/gemm/moe_a2a_utils.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/python.h>

using namespace pybind11::literals;  // enables _a argument literal

namespace turbomind {

template<typename Func>
void bench(const std::string& name, int rank, cudaStream_t stream, Func&& func, int warmup = 100, int iterations = 5000)
{
    cudaStreamSynchronize(stream);

    // warmup
    for (int i = 0; i < warmup; ++i) {
        func();
    }
    cudaStreamSynchronize(stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector<float> times;
    times.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
        cudaEventRecord(start, stream);
        func();
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        times.push_back(milliseconds);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::sort(times.begin(), times.end());

    float min_time    = times.front();
    float max_time    = times.back();
    float median_time = times[iterations / 2];
    float avg_time    = std::accumulate(times.begin(), times.end(), 0.0f) / iterations;

    float sq_sum = 0.0f;
    for (auto t : times) {
        sq_sum += (t - avg_time) * (t - avg_time);
    }
    float std_dev = std::sqrt(sq_sum / iterations);

    printf("[%d_%s] iters=%d, min=%.3f us, max=%.3f us, median=%.3f us, avg=%.3f us, std=%.3f us\n",
           rank,
           name.c_str(),
           iterations,
           min_time * 1e3f,
           max_time * 1e3f,
           median_time * 1e3f,
           avg_time * 1e3f,
           std_dev * 1e3f);
}

struct MoeBuffer {

    void init(int num_experts, int ep_size, int token, int hidden, int num_topk, Allocator& symm_alloc)
    {

        // preprocess buffer
        topk_scales       = Buffer_<float>(token * ep_size * num_topk, kDEVICE);
        topk_experts      = Buffer_<int>(token * num_experts, kDEVICE);
        token_idx_in_rank = Buffer_<int>(ep_size * (token + 2), kDEVICE);

        // dispatch buffer
        const int max_tokens    = token * ep_size;
        const int local_experts = num_experts / ep_size;
        recv_info               = Buffer_<int>(2 * ep_size * ep_size, symm_alloc);
        recv_hidden             = Buffer_<__nv_bfloat16>(max_tokens * hidden, symm_alloc);
        recv_scales             = Buffer_<float>(num_topk * max_tokens, symm_alloc);
        recv_masks              = Buffer_<int8_t>(local_experts * max_tokens, symm_alloc);
    }

    Buffer_<float> topk_scales;        // (n, ep_size, num_topk)
    Buffer_<int>   topk_experts;       // (n, num_experts)
    Buffer_<int>   token_idx_in_rank;  // (ep_size, n + 2), idx, token_count, expert_count

    Buffer_<int>           recv_info;    // (2, ep_size, ep_size)
    Buffer_<__nv_bfloat16> recv_hidden;  // max_tokens, hidden        (n, H)
    Buffer_<float>         recv_scales;  // num_topk, max_tokens      (e, n)
    Buffer_<int8_t>        recv_masks;   // local_experts, max_tokens (E, n)
};

class MoeTest {
public:
    MoeTest(comm::HostGroupId* group, int rank, int n_ranks)
    {
        CudaDeviceGuard dev_guard(rank);

        auto h_comm = group->CreateCommunicator(n_ranks, rank);
        d_comm_     = comm::CreateDeviceCommunicator("cuda-ipc", n_ranks, rank, h_comm);
        symm_alloc_ = core::SimpleAllocator::Create([this](ssize_t size) { return SymmAlloc(size); },
                                                    [this](void* p, ssize_t size) { return SymmFree(p, size); },
                                                    kDEVICE);
    }

    void* SymmAlloc(size_t size)
    {
        auto ptr = d_comm_->Allocate(size);
        d_comm_->Register(ptr, size);
        return ptr;
    }

    void SymmFree(void* ptr, size_t size)
    {
        if (!ptr) {
            return;
        }
        d_comm_->Deregister(ptr);
        d_comm_->Free(ptr);
    }

    void allreduce_sum(torch::Tensor& x)
    {
        if (x.scalar_type() != torch::kHalf) {
            throw std::runtime_error("only half supported");
        }
        auto stream = at::cuda::getCurrentCUDAStream();

        auto symm_x = core::Tensor(x.sizes().vec(), DataType::kHalf, symm_alloc_);
        cudaMemcpyAsync(symm_x.raw_data(), x.data_ptr(), x.nbytes(), cudaMemcpyDefault, stream);

        d_comm_->AllReduceSum(symm_x.raw_data(),  //
                              symm_x.raw_data(),
                              symm_x.size(),
                              DataType::kHalf,
                              0,
                              stream);

        cudaMemcpyAsync(x.data_ptr(), symm_x.raw_data(), x.nbytes(), cudaMemcpyDefault, stream);
        cudaStreamSynchronize(stream);
    }

    torch::Tensor reduce_scatter(torch::Tensor& x, bool perf = false)
    {
        if (x.scalar_type() != torch::kHalf) {
            throw std::runtime_error("only half supported");
        }

        int rank    = d_comm_->rank(0);
        int n_ranks = d_comm_->n_ranks(0);

        FT_CHECK(x.sizes().size() == 2);
        FT_CHECK(x.sizes()[0] % n_ranks == 0);

        int token = x.sizes()[0];
        int dim   = x.sizes()[1];

        auto stream = at::cuda::getCurrentCUDAStream();

        if (perf) {
            cudaStreamSynchronize(stream);
            const int   iterations = 2000;
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            std::vector<int> tokens = {1, 4, 16, 64, 256, 512, 1024, 4096};
            const int        hidden = 4096;
            for (auto n : tokens) {
                auto               tx = core::Tensor({n * n_ranks, hidden}, DataType::kHalf, symm_alloc_);
                std::vector<float> times;

                for (int i = 0; i < iterations; ++i) {
                    cudaEventRecord(start, stream);
                    d_comm_->ReduceScatter(tx.raw_data(),  //
                                           (char*)tx.raw_data() + rank * n * hidden * sizeof(__half),
                                           n * hidden,
                                           n * n_ranks * hidden,
                                           DataType::kHalf,
                                           0,
                                           stream);
                    cudaEventRecord(stop, stream);
                    cudaEventSynchronize(stop);
                    float milliseconds = 0;
                    cudaEventElapsedTime(&milliseconds, start, stop);
                    times.push_back(milliseconds);
                }
                std::sort(times.begin(), times.end());
                auto        S     = n * hidden * sizeof(__half) * n_ranks;
                auto        t     = times[iterations / 2];
                const float algbw = S / 1e9f / t * 1e3f;
                const float busbw = algbw / n_ranks * (n_ranks - 1);
                if (rank == 0) {
                    printf("token=%4d, time=%9.3f us, algbw=%8.3f GB/s, busbw=%8.3f GB/s\n", n, t * 1e3, algbw, busbw);
                }
            }
        }

        int recvcount  = (token + n_ranks - 1) / n_ranks * dim;
        int totalcount = token * dim;

        auto symm_x = core::Tensor(x.sizes().vec(), DataType::kHalf, symm_alloc_);
        cudaMemcpyAsync(symm_x.raw_data(), x.data_ptr(), x.nbytes(), cudaMemcpyDefault, stream);

        d_comm_->ReduceScatter(symm_x.raw_data(),  //
                               (char*)symm_x.raw_data() + rank * recvcount * sizeof(__half),
                               recvcount,
                               totalcount,
                               DataType::kHalf,
                               0,
                               stream);

        auto out = torch::empty({token / n_ranks, dim}, dtype(torch::kFloat16).device(torch::kCUDA));
        cudaMemcpyAsync(out.data_ptr(),
                        (char*)symm_x.raw_data() + recvcount * rank * sizeof(__half),
                        out.nbytes(),
                        cudaMemcpyDefault,
                        stream);
        cudaStreamSynchronize(stream);
        return out;
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    all2all(torch::Tensor& x, torch::Tensor& scores, int num_topk)
    {
        auto stream = at::cuda::getCurrentCUDAStream();
        stream.synchronize();  // sync torch

        Context            context(stream.device_index());
        core::ContextGuard guard{context.core_stream, context.allocator};

        int   rank          = d_comm_->rank(0);
        int   ep_size       = d_comm_->n_ranks(0);
        int   tokens        = scores.size(0);
        int   hidden        = x.size(1);
        int   experts       = scores.size(1);
        int   local_experts = experts / ep_size;
        bool  softmax       = true;
        bool  norm_topk     = true;
        float routed_scale  = 1.0f;

        auto recv_x = torch::empty_like(x);

        buffer_.init(experts, ep_size, tokens, hidden, num_topk, symm_alloc_);
        core::Context::stream().Sync();  // sync buffer

        // 1. preprocess
        invokeMoeGate_a2a(buffer_.topk_scales.data(),
                          buffer_.topk_experts.data(),
                          buffer_.token_idx_in_rank.data(),
                          scores.data_ptr<float>(),
                          tokens,
                          experts,
                          ep_size,
                          num_topk,
                          softmax,
                          norm_topk,
                          routed_scale,
                          stream);

        // 2. dispatch (info exchange & a2a)
        bench("AllToAllDispatch", rank, stream, [&]() {
            d_comm_->AllToAllDispatch(buffer_.recv_info.data(),
                                      buffer_.recv_hidden.data(),
                                      buffer_.recv_scales.data(),
                                      buffer_.recv_masks.data(),
                                      x.data_ptr(),
                                      buffer_.topk_scales.data(),
                                      buffer_.topk_experts.data(),
                                      buffer_.token_idx_in_rank.data(),
                                      tokens,
                                      hidden,
                                      num_topk,
                                      experts,
                                      DataType::kBfloat16,
                                      0,
                                      stream);
        });

        // 3. get accum, f2n, f2E, en2f

        // 4. combine
        bench("AllToAllCombine", rank, stream, [&]() {
            d_comm_->AllToAllCombine(recv_x.data_ptr(),
                                     buffer_.recv_info.data(),
                                     buffer_.recv_hidden.data(),
                                     buffer_.token_idx_in_rank.data(),
                                     tokens,
                                     hidden,
                                     DataType::kBfloat16,
                                     0,
                                     stream);
        });

        // check preprocess
        auto ref_token_idx_in_rank = torch::empty({ep_size, tokens + 2}, dtype(torch::kInt32).device(torch::kCUDA));
        cudaMemcpyAsync(ref_token_idx_in_rank.data_ptr(),
                        buffer_.token_idx_in_rank.data(),
                        buffer_.token_idx_in_rank.byte_size(),
                        cudaMemcpyDefault,
                        stream);

        {
            auto               cpu_ref_token_idx_in_rank = ref_token_idx_in_rank.to(torch::kCPU);
            std::ostringstream oss;
            for (int i = 0; i < ep_size; ++i) {
                int send_token = cpu_ref_token_idx_in_rank.data_ptr<int>()[i * (tokens + 2) + tokens];
                oss << send_token;
                if (i != ep_size - 1) {
                    oss << ", ";
                }
            }
            printf("rank=%d, send_tokens=[%s]\n", rank, oss.str().c_str());
        }
        ref_token_idx_in_rank = ref_token_idx_in_rank.slice(1, 0, tokens).contiguous();
        stream.synchronize();

        // check dispatch
        auto ref_recv_info   = torch::empty({2, ep_size, ep_size}, dtype(torch::kInt32));
        auto ref_recv_hidden = torch::zeros({tokens * ep_size, hidden}, dtype(torch::kBFloat16).device(torch::kCUDA));
        cudaMemcpyAsync(ref_recv_info.data_ptr(),
                        buffer_.recv_info.data(),
                        buffer_.recv_info.byte_size(),
                        cudaMemcpyDefault,
                        stream);
        cudaMemcpyAsync(ref_recv_hidden.data_ptr(),
                        buffer_.recv_hidden.data(),
                        buffer_.recv_hidden.byte_size(),
                        cudaMemcpyDefault,
                        stream);
        stream.synchronize();

        int recv_info_elem = ref_recv_info.data_ptr<int>()[(ep_size - 1) * ep_size + rank];
        ref_recv_hidden    = ref_recv_hidden.slice(0, 0, recv_info_elem);

        return {ref_token_idx_in_rank, ref_recv_hidden, recv_x};
    }

    comm::DeviceComm d_comm_;
    core::Allocator  symm_alloc_;

    MoeBuffer buffer_;
};

}  // namespace turbomind

using namespace turbomind;

PYBIND11_MODULE(_moe, m)
{
    pybind11::class_<turbomind::comm::HostGroupId>(m, "HostGroupId")
        .def(pybind11::init([](const std::string& backend) { return turbomind::comm::CreateHostGroupId(backend); }),
             "backend"_a = "")
        .def("init", &turbomind::comm::HostGroupId::Initialize)
        .def("export",
             [](turbomind::comm::HostGroupId& self) {
                 std::ostringstream os;
                 self.Export(os);
                 const std::string& str = os.str();
                 return pybind11::bytearray(str.data(), str.size());
             })
        .def("load", [](turbomind::comm::HostGroupId& self, const pybind11::bytearray& ba) {
            std::istringstream is(std::string{ba});
            self.Import(is);
        });

    pybind11::class_<MoeTest>(m, "MoeTest")
        .def(pybind11::init<comm::HostGroupId*, int, int>(), pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("allreduce_sum", &MoeTest::allreduce_sum, pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("reduce_scatter", &MoeTest::reduce_scatter, pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("all2all", &MoeTest::all2all, pybind11::call_guard<pybind11::gil_scoped_release>());
}

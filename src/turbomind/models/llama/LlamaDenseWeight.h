/*
 * Copyright (c) OpenMMLab. All rights reserved.
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Modified from https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/layers/DenseWeight.h

#pragma once

#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/memory_utils.h"
#include <filesystem>

namespace turbomind {

enum class WeightType : int
{
    kFP32,
    kFP16,
    kFP8,  // not supported yet
    kBF16,
    kINT8,
    kINT4
};

inline size_t getBitSize(WeightType type)
{
    switch (type) {
        case WeightType::kFP32:
            return 32;
        case WeightType::kFP16:
            return 16;
        case WeightType::kFP8:
            return 8;
        case WeightType::kBF16:
            return 16;
        case WeightType::kINT8:
            return 8;
        case WeightType::kINT4:
            return 4;
    }
    return 0;
}

enum class LoraPolicy : int
{
    kNull,
    kPlora,
};

inline LoraPolicy getLoraPolicy(const std::string& policy)
{
    if (policy == "plora") {
        return LoraPolicy::kPlora;
    }
    return LoraPolicy::kNull;
}

struct LoraWeight {
    LoraPolicy policy;
    int        r;
    float      scale;
    void*      a;
    void*      b;
};

template<typename T>
struct LlamaDenseWeight {
    size_t     input_dims;
    size_t     output_dims;
    void*      kernel;
    LoraWeight lora;
    WeightType type;
    T*         bias;
    T*         scales;
    T*         zeros;
    T*         scales_zeros;
    int        group_size;

    gemm::MatrixLayout k_desc;
    gemm::MatrixLayout q_desc;
};

template<typename T>
struct LlamaAttentionWeight {
    LlamaDenseWeight<T> qkv;
    LlamaDenseWeight<T> output;
};

template<typename T>
struct LlamaFfnWeight {
    LlamaDenseWeight<T> gating;
    LlamaDenseWeight<T> intermediate;
    LlamaDenseWeight<T> output;
    LlamaDenseWeight<T> fused_gating_intermediate;
    bool                is_fused_silu;
};

template<typename T>
inline void freeWeights(LlamaDenseWeight<T>& weights)
{
    cudaFree(weights.kernel);
    cudaFree(weights.bias);
    cudaFree(weights.scales);
    cudaFree(weights.zeros);

    weights.kernel = nullptr;
    weights.bias   = nullptr;
    weights.scales = nullptr;
    weights.zeros  = nullptr;

    {
        cudaFree(weights.lora.a);
        cudaFree(weights.lora.b);
        weights.lora.a = nullptr;
        weights.lora.b = nullptr;
    }
}

template<typename T>
inline void mallocWeights(LlamaDenseWeight<T>& weights, bool bias)
{
    if (bias) {
        deviceMalloc((T**)&weights.bias, weights.output_dims);
    }
    const size_t bit_size = getBitSize(weights.type);
    if (bit_size >= 16) {  // fp16, fp32
        deviceMalloc((T**)&weights.kernel, weights.input_dims * weights.output_dims);
    }
    else {  // int8, int4
        const int factor = sizeof(float) * 8 / bit_size;
        FT_CHECK(weights.input_dims % factor == 0);
        deviceMalloc((int**)&weights.kernel, weights.input_dims * weights.output_dims / factor);
        deviceMemSetZero((int*)weights.kernel, weights.input_dims * weights.output_dims / factor);
        deviceMalloc((T**)&weights.scales, weights.input_dims / weights.group_size * weights.output_dims);
        deviceMalloc((T**)&weights.zeros, weights.input_dims / weights.group_size * weights.output_dims);
    }

    if (weights.lora.r > 0) {
        // FT_CHECK(bit_size >= 16);
        deviceMalloc((T**)&weights.lora.a, weights.input_dims * weights.lora.r);
        deviceMalloc((T**)&weights.lora.b, weights.lora.r * weights.output_dims);
    }
}

template<typename T>
inline void loadWeights(
    LlamaDenseWeight<T>& w, std::string prefix, int rank, FtCudaDataType model_file_type, size_t tensor_para_size)
{
    auto weight_file  = prefix + "." + std::to_string(tensor_para_size - 1) + ".weight";
    auto qweight_file = prefix + "." + std::to_string(tensor_para_size - 1) + ".qweight";
    if (!std::filesystem::exists(weight_file) && !std::filesystem::exists(qweight_file)) {
        TM_LOG_ERROR("%s and %s does not exist", weight_file.c_str(), qweight_file.c_str());
        FT_CHECK(false);
    }

    prefix += "." + std::to_string(rank);

    size_t     dim0 = w.input_dims;
    size_t     dim1 = w.output_dims;
    const auto type = model_file_type;

    if (w.bias) {
        loadWeightFromBin((T*)w.bias, {1, dim1}, prefix + ".bias", type);
    }
    const size_t bit_size = getBitSize(w.type);
    if (bit_size >= 16) {  // fp16, fp32
        loadWeightFromBin((T*)w.kernel, {dim0, dim1}, prefix + ".weight", type);
    }
    else {  // int8, int4
        const int factor = sizeof(float) * 8 / bit_size;

        FT_CHECK(dim1 % factor == 0);

        std::vector<size_t> w_shape{dim0, dim1 / factor * sizeof(uint32_t)};
        loadWeightFromBin((int8_t*)w.kernel, w_shape, prefix + ".qweight", FtCudaDataType::INT8);

        const size_t group_count = w.group_size > 0 ? dim0 / w.group_size : 1;

        loadWeightFromBin((half*)w.scales, {group_count, dim1}, prefix + ".scales", type);
        loadWeightFromBin((half*)w.zeros, {group_count, dim1}, prefix + ".zeros", type);
    }
}

}  // namespace turbomind

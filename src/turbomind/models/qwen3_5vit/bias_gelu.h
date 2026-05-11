#pragma once

#include "src/turbomind/core/core.h"

#include <cuda_runtime.h>

namespace turbomind {

// In-place Qwen3.5 ViT MLP activation:
// x <- GELU_tanh(x + bias)
void invokeQwen3_5VitBiasGelu(Tensor& x, const Tensor& bias, cudaStream_t stream);

}  // namespace turbomind

// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <cuda_runtime.h>

#include "src/turbomind/core/core.h"

namespace turbomind {

void invokeLayerNorm(Tensor&       out,
                     const Tensor& x,
                     const Tensor& weight,
                     const Tensor& bias,
                     float         eps,
                     cudaStream_t  stream);

}  // namespace turbomind

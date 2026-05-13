// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/core/data_type.h"

#include <cuda_runtime.h>

namespace turbomind {

// Prepare the Qwen3.5 ViT attention inputs after the fused QKV projection.
//
// qkv layout:
//   [token, local_q_heads + 2 * local_kv_heads, head_dim]
// Q is updated in place with bias + RoPE. K/V are written to `kv` as:
//   [local_kv_heads, 2, token, head_dim]
void invokeQwen3_5VitPrepareQKV(void*        qkv,
                                void*        kv,
                                const void*  qkv_bias,
                                const void*  rotary_pos_emb,
                                const int*   mapped_idx,
                                DataType     dtype,
                                int          token_num,
                                int          local_head_num,
                                int          head_dim,
                                cudaStream_t stream);

}  // namespace turbomind

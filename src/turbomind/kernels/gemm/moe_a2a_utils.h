// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

namespace turbomind {

void invokeMoeGate_a2a(float*       topk_scales,
                       int*         topk_experts,
                       int*         token_idx_in_rank,
                       const float* logits,
                       int          tokens,
                       int          experts,
                       int          ep_size,
                       int          experts_per_token,
                       bool         softmax,
                       bool         norm_topk,
                       float        routed_scale,
                       cudaStream_t stream);

}  // namespace turbomind

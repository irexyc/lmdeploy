// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/qwen3_5vit/qwen3_5vit_weight.h"

#include "src/turbomind/core/registry.h"
#include "src/turbomind/models/layer_norm_weight.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/qwen3_5vit/qwen3_5vit.h"
#include "src/turbomind/models/qwen3_5vit/qwen3_5vit_block_weight.h"
#include "src/turbomind/utils/memory_utils.h"

namespace turbomind {

Qwen3_5VitWeight::Qwen3_5VitWeight(const core::Qwen3_5VitWeightConfig& cfg):
    data_type{cfg.data_type},
    hidden_dim{cfg.hidden_dim},
    out_hidden_dim{cfg.out_hidden_dim},
    depth{cfg.depth},
    head_num{cfg.head_num},
    intermediate_size{cfg.intermediate_size},
    patch_in_dim{cfg.patch_in_dim},
    num_position_embeddings{cfg.num_position_embeddings},
    spatial_merge_size{cfg.spatial_merge_size},
    norm_eps{cfg.norm_eps}
{
}

void Qwen3_5VitWeight::prepare()
{
    for_each_child([](const char* /*name*/, Module* child) {
        if (child)
            child->prepare();
    });

    if (pos_embed) {
        EnsureFloatDtype(pos_embed, data_type);
    }
}

bool Qwen3_5VitWeight::verify(std::vector<std::string>& missing)
{
    Module::verify(missing);
    if (!patch_embed) {
        missing.push_back(full_path() + ": missing patch_embed");
    }
    if (!pos_embed) {
        missing.push_back(full_path() + ": missing pos_embed");
    }
    if (!blocks || blocks->size() != depth) {
        missing.push_back(full_path() + ": blocks count mismatch (expected "
                          + std::to_string(depth) + ")");
    }
    if (!merger_fc1 || !merger_fc2 || !merger_norm) {
        missing.push_back(full_path() + ": missing merger");
    }
    return missing.empty();
}

Qwen3_5VitBlockWeight* Qwen3_5VitWeight::block(int i) const
{
    if (!blocks) {
        return nullptr;
    }
    return static_cast<Qwen3_5VitBlockWeight*>(blocks->child(std::to_string(i)));
}

std::unique_ptr<VisualModel>
Qwen3_5VitWeight::make_model(const EngineParam& engine, const Context& ctx, int phases) const
{
    return std::make_unique<Qwen3_5Vit>(engine, ctx, *this, phases);
}

TM_MODULE_REGISTER(Qwen3_5VitWeight, core::Qwen3_5VitWeightConfig);

TM_MODULE_METHODS(Qwen3_5VitWeight, QWEN3_5VIT_WEIGHT_CHILDREN, QWEN3_5VIT_WEIGHT_PARAMS)

}  // namespace turbomind

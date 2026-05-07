// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/qwen3_5vit/qwen3_5vit.h"

#include "src/turbomind/core/logger.h"
#include "src/turbomind/models/qwen3_5vit/qwen3_5vit_weight.h"

namespace turbomind {

struct Qwen3_5Vit::Impl {
    const Qwen3_5VitWeight& weights_;
    int                     phases_;

    Impl(const EngineParam& /*engine*/, const Context& /*ctx*/, const Qwen3_5VitWeight& weights, int phases):
        weights_{weights}, phases_{phases}
    {
    }

    void Run(BatchOp op, int phase, TensorMap& /*env*/)
    {
        // Stubs — actual ViT inference lands in a follow-up.
        switch (op) {
            case BatchOp::kSetup:
                Setup(phase);
                return;
            case BatchOp::kPrepare:
                Prepare(phase);
                return;
            case BatchOp::kForward:
                Forward(phase);
                return;
            default:
                return;
        }
    }

    void Setup(int /*phase*/) {}
    void Prepare(int /*phase*/) {}
    void Forward(int /*phase*/) {}
};

Qwen3_5Vit::Qwen3_5Vit(const EngineParam& engine, const Context& ctx, const Qwen3_5VitWeight& weights, int phases):
    impl_{std::make_unique<Impl>(engine, ctx, weights, phases)}
{
}

Qwen3_5Vit::~Qwen3_5Vit() = default;

void Qwen3_5Vit::Run(BatchOp op, int phase, TensorMap& env)
{
    impl_->Run(op, phase, env);
}

}  // namespace turbomind

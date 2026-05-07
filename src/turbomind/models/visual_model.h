// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/engine/batch.h"

namespace turbomind {

/// Polymorphic peer of ``LanguageModel`` for the visual sub-graph.
///
/// Concrete subclasses (one per VLM family — ``Qwen3_5Vit``,
/// ``InternVit``, …) wire up the per-family C++ runtime. The
/// engine talks to this base via ``Run(BatchOp, phase, env)``,
/// mirroring ``LanguageModel::Run``.
///
/// Lifetime: owned by ``Engine`` as a ``unique_ptr<VisualModel>`` and
/// non-null only when the corresponding ``ModelRoot::visual_model``
/// child was attached during weight loading.
class VisualModel {
public:
    virtual ~VisualModel() = default;

    /// Phase entry point. Called from ``ModelExecutor::Run`` *before*
    /// the language model. Subclasses dispatch on ``op``.
    virtual void Run(BatchOp op, int phase, TensorMap& env) = 0;
};

}  // namespace turbomind

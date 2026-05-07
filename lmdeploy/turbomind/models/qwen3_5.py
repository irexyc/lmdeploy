# Copyright (c) OpenMMLab. All rights reserved.
"""Qwen3.5 TextModel — text path inherited, visual path added.

Loads ``Qwen3_5ForConditionalGeneration`` checkpoints whose top-level
HF config carries a ``vision_config`` block. Reuses ``_Qwen3_5Model``
verbatim for the language model and adds a visual sub-tree rooted at
``ModelRoot.visual_model``.

The patcher and position embedding are replicated across TP ranks. Visual
transformer blocks and merger linears shard with the model TP group.
"""
from __future__ import annotations

import _turbomind as _tm
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig

from ..builders import (
    AttentionBuilder,
    Builder,
    LayerNormBuilder,
    ModuleListBuilder,
    ModuleListConfig,
    SplitSide,
    VisualModelBuilder,
    make_layer_norm_config,
)
from ..linear import Linear, transform_output_dim
from ..weight_format import TrivialFormat
from .base import INPUT_MODELS
from ._qwen3_5 import _Qwen3_5Model


@transform_output_dim
def _split_packed_visual_qkv(qkv):
    """Split HF visual QKV layout [Q | K | V] along output dim."""
    return tuple(x.contiguous() for x in qkv.chunk(3, dim=-1))


@INPUT_MODELS.register_module(name='qwen3_5')
@INPUT_MODELS.register_module(name='qwen3_5-moe')
class Qwen3_5Model(_Qwen3_5Model):
    """Weight model for Qwen3.5 VLM (text + vision)."""

    _vision = True

    def __init__(self, cfg: Qwen3_5Config | Qwen3_5MoeConfig, *, resolver):
        text_cfg = cfg.text_config
        if text_cfg is None:
            raise ValueError(
                'Qwen3_5Model requires a checkpoint with text_config.')

        vision_cfg = cfg.vision_config
        if vision_cfg is None:
            raise ValueError(
                'Qwen3_5Model requires a checkpoint with vision_config; '
                'got none. Set disable_vision_encoder=True for text-only checkpoints.')

        super().__init__(text_cfg, resolver=resolver)

        self._vis_depth = int(vision_cfg.depth)
        self._vis_hidden = int(vision_cfg.hidden_size)
        self._vis_inter = int(vision_cfg.intermediate_size)
        self._vis_heads = int(vision_cfg.num_heads)
        self._vis_out_hidden = int(vision_cfg.out_hidden_size)
        self._vis_in_chans = int(vision_cfg.in_channels)
        self._vis_patch = int(vision_cfg.patch_size)
        self._vis_temporal = int(vision_cfg.temporal_patch_size)
        self._vis_pos_n = int(vision_cfg.num_position_embeddings)
        self._vis_spatial_merge = int(vision_cfg.spatial_merge_size)
        self._vis_norm_eps = 1e-6

        # in_dim of the patcher when the Conv3d is reinterpreted as a
        # Linear over flattened patches: C * T * H * W.
        self._patch_in_dim = (self._vis_in_chans
                              * self._vis_temporal
                              * self._vis_patch
                              * self._vis_patch)

    # ------------------------------------------------------------------
    # model() — extend the parent text build with the visual sub-tree
    # ------------------------------------------------------------------

    def model(self, pfx):
        super().model(pfx)
        self._build_visual_model(pfx + 'model.visual')

    # ------------------------------------------------------------------
    # Visual sub-tree
    # ------------------------------------------------------------------

    def _build_visual_model(self, pfx):
        cfg = self._make_visual_root_cfg()
        root = VisualModelBuilder(
            cfg, self._ctx, root_handles=self._root_handles)
        root.tp = self._model_tp

        root._add_tensor('pos_embed', (pfx + 'pos_embed').pop('weight'))
        root._add_linear('patch_embed', self._patch_embed(pfx + 'patch_embed.proj'))

        root.blocks = self.vit_blocks(pfx + 'blocks')

        root._add_linear('merger_fc1', self._linear(pfx + 'merger.linear_fc1'), SplitSide.OUTPUT)
        root._add_linear('merger_fc2', self._linear(pfx + 'merger.linear_fc2'), SplitSide.INPUT)
        root.merger_norm = self._layer_norm(pfx + 'merger.norm', dim=self._vis_hidden)

        root.build()

    def _make_visual_root_cfg(self):
        cfg = _tm.Qwen3_5VitWeightConfig()
        cfg.data_type = self._resolver.data_type
        cfg.hidden_dim = self._vis_hidden
        cfg.out_hidden_dim = self._vis_out_hidden
        cfg.depth = self._vis_depth
        cfg.head_num = self._vis_heads
        cfg.intermediate_size = self._vis_inter
        cfg.patch_in_dim = self._patch_in_dim
        cfg.num_position_embeddings = self._vis_pos_n
        cfg.spatial_merge_size = self._vis_spatial_merge
        cfg.norm_eps = self._vis_norm_eps
        return cfg

    def _patch_embed(self, pfx):
        weight = pfx.pop('weight')
        if weight.dim() >= 2:
            weight = weight.reshape(weight.shape[0], -1).t().contiguous()
        tensors = {'weight': weight}
        if pfx.has('bias'):
            tensors['bias'] = pfx.pop('bias')
        return Linear(tensors=tensors, weight_format=TrivialFormat())

    def vit_blocks(self, pfx):
        blocks = ModuleListBuilder(ModuleListConfig(), self._ctx)

        for i, p in pfx.slices(0, self._vis_depth):
            blocks[i] = self.vit_block(p)

        return blocks.build()

    def vit_block(self, pfx):
        cfg = _tm.Qwen3_5VitBlockConfig()
        cfg.data_type = self._resolver.data_type
        cfg.hidden_dim = self._vis_hidden
        cfg.head_num = self._vis_heads
        cfg.intermediate_size = self._vis_inter
        cfg.norm_eps = self._vis_norm_eps

        b = Builder(cfg, self._ctx)
        b.tp = self._model_tp

        b.norm1 = self._layer_norm(pfx + 'norm1', dim=self._vis_hidden)
        b.norm2 = self._layer_norm(pfx + 'norm2', dim=self._vis_hidden)

        b.attention = self.vit_attn(pfx + 'attn')
        b._add_linear('mlp_fc1', self._linear(pfx + 'mlp.linear_fc1'), SplitSide.OUTPUT)
        b._add_linear('mlp_fc2', self._linear(pfx + 'mlp.linear_fc2'), SplitSide.INPUT)
        return b.build()

    def _make_visual_attn_cfg(self):
        cfg = _tm.AttentionConfig()
        cfg.data_type = self._resolver.data_type
        cfg.hidden_dim = self._vis_hidden
        cfg.head_dim = self._vis_hidden // self._vis_heads
        cfg.head_num = self._vis_heads
        cfg.kv_head_num = self._vis_heads
        cfg.window_size = 0
        cfg.softmax_scale = 0.0
        return cfg

    def vit_attn(self, pfx):
        cfg = self._make_visual_attn_cfg()
        q, k, v = _split_packed_visual_qkv(self._linear(pfx + 'qkv'))

        # Qwen3.5 ViT applies RoPE before invoking the attention kernel, so
        # Q/K must keep HF order here rather than using text rotary reorder.
        m = AttentionBuilder(cfg, self._ctx, tp=self._model_tp)
        m.add_qkv_proj(q, k, v)
        m.add_o_proj(self._linear(pfx + 'proj'))
        return m.build()

    # ------------------------------------------------------------------
    # Helper: build a LayerNorm child
    # ------------------------------------------------------------------

    def _layer_norm(self, pfx, *, dim: int):
        weight = pfx.pop('weight')
        bias = pfx.pop('bias') if pfx.has('bias') else None
        cfg = make_layer_norm_config(dim=dim,
                                     data_type=self._resolver.data_type,
                                     norm_eps=self._vis_norm_eps)
        m = LayerNormBuilder(cfg, self._ctx)
        m.set_weight(weight, bias=bias)
        return m.build()

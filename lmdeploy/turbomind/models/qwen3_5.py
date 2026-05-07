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
from ..linear import transform_output_dim
from .base import INPUT_MODELS
from ._qwen3_5 import _Qwen3_5Model
from .utils import layer_progress


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

    def model(self):
        super().model()             # builds text_model under ModelRoot
        self._build_visual_model()  # builds visual_model under ModelRoot

    # ------------------------------------------------------------------
    # Visual sub-tree
    # ------------------------------------------------------------------

    def _build_visual_model(self):
        # The patcher weight in HF is a Conv3d (out, in_chans, T, H, W).
        # Flatten it to a 2D linear weight (out, in_chans·T·H·W) so it
        # passes through the standard TrivialFormat normaliser. This is
        # mathematically equivalent because the patcher uses
        # non-overlapping patches (stride == kernel size in T, H, W).
        pe_key = 'model.visual.patch_embed.proj.weight'
        if pe_key in self.params:
            w = self.params[pe_key]
            if w.dim() == 5:
                self.params[pe_key] = w.reshape(
                    self._vis_hidden, -1).contiguous()

        cfg = self._make_visual_root_cfg()
        root = VisualModelBuilder(
            cfg, self._ctx, root_handles=self._root_handles)
        root.tp = self._model_tp

        root._add_tensor('pos_embed', self._get(
            'model.visual.pos_embed.weight'))
        root._add_linear('patch_embed', self._linear(
            'model.visual.patch_embed.proj'))

        root.blocks = self.vit_blocks('model.visual.blocks')

        # Merger
        root._add_linear('merger_fc1', self._linear(
            'model.visual.merger.linear_fc1'), SplitSide.OUTPUT)
        root._add_linear('merger_fc2', self._linear(
            'model.visual.merger.linear_fc2'), SplitSide.INPUT)
        root.merger_norm = self._layer_norm('model.visual.merger.norm',
                                            dim=self._vis_hidden)

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

    def vit_blocks(self, pfx: str):
        blocks = ModuleListBuilder(ModuleListConfig(), self._ctx)

        for i in layer_progress(self._vis_depth):
            blocks[i] = self.vit_block(f'{pfx}.{i}')

        return blocks.build()

    def vit_block(self, pfx: str):
        cfg = _tm.Qwen3_5VitBlockConfig()
        cfg.data_type = self._resolver.data_type
        cfg.hidden_dim = self._vis_hidden
        cfg.head_num = self._vis_heads
        cfg.intermediate_size = self._vis_inter
        cfg.norm_eps = self._vis_norm_eps

        b = Builder(cfg, self._ctx)
        b.tp = self._model_tp

        b.norm1 = self._layer_norm(f'{pfx}.norm1', dim=self._vis_hidden)
        b.norm2 = self._layer_norm(f'{pfx}.norm2', dim=self._vis_hidden)

        b.attention = self.vit_attn(f'{pfx}.attn')
        b._add_linear('mlp_fc1', self._linear(
            f'{pfx}.mlp.linear_fc1'), SplitSide.OUTPUT)
        b._add_linear('mlp_fc2', self._linear(
            f'{pfx}.mlp.linear_fc2'), SplitSide.INPUT)
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

    def vit_attn(self, pfx: str):
        cfg = self._make_visual_attn_cfg()
        q, k, v = _split_packed_visual_qkv(self._linear(f'{pfx}.qkv'))

        # Qwen3.5 ViT applies RoPE before invoking the attention kernel, so
        # Q/K must keep HF order here rather than using text rotary reorder.
        m = AttentionBuilder(cfg, self._ctx, tp=self._model_tp)
        m.add_qkv_proj(q, k, v)
        m.add_o_proj(self._linear(f'{pfx}.proj'))
        return m.build()

    # ------------------------------------------------------------------
    # Helper: build a LayerNorm child
    # ------------------------------------------------------------------

    def _layer_norm(self, pfx: str, *, dim: int):
        weight = self._get(f'{pfx}.weight')
        bias = self._get(f'{pfx}.bias')
        cfg = make_layer_norm_config(dim=dim,
                                     data_type=self._resolver.data_type,
                                     norm_eps=self._vis_norm_eps)
        m = LayerNormBuilder(cfg, self._ctx)
        m.set_weight(weight, bias=bias)
        return m.build()

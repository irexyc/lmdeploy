# Copyright (c) OpenMMLab. All rights reserved.

import re

import torch

from ..loader import create_loader
from .base import INPUT_MODELS
from .llama import LlamaModel, LlamaReader


class InternLM3Reader(LlamaReader):
    """InternLM2 model reader."""

    attn_layer_prefix = 'model.self_decoder.layers'
    attn_layer_patten = r'model.self_decoder.layers.([0-9]+).'
    cross_attn_layer_prefix = 'model.cross_decoder.layers'
    cross_attn_layer_patten = r'model.cross_decoder.layers.([0-9]+).'
    tok_embeddings_key = 'model.tok_embeddings.weight'
    norm_weight_key = 'model.norm.weight'
    output_weight_key = 'output.weight'

    attn_pattern = r'attention'
    ffn_pattern = r'feed_forward'
    cross_kv_pattern = r'(model.cross_decoder).(?!layers.)'

    def _attn(self, i: int, kind: str):
        """Get q, k, v, o kind for layer i."""
        q, k, v = (None, ) * 3
        kv_head_num = self.model_cfg['num_key_value_heads']
        gs = int(self.model_cfg['num_attention_heads'] / kv_head_num)
        if self.attn_layer_patten != self.cross_attn_layer_patten:
            weight_key = 'wqkv'
            cnt = 2
        else:
            weight_key = 'wq'
            cnt = 0
        qkv = self.params.get(
            f'{self.attn_layer_prefix}.{i}.attention.{weight_key}.{kind}')
        qkv = self.transform(qkv, kind)
        if qkv is not None:
            qkv = qkv.view(kv_head_num, gs + cnt, 128, -1)
            hidden_dim = qkv.shape[-1]
            q, k, v = torch.split(qkv, [gs, cnt // 2, cnt // 2], dim=1)
            q = q.reshape(-1, hidden_dim)
            k = k.reshape(-1, hidden_dim)
            v = v.reshape(-1, hidden_dim)
        o = self.params.get(
            f'{self.attn_layer_prefix}.{i}.attention.wo.{kind}')
        o = self.transform(o, kind)
        return (q, k, v, o)

    def attn_norm(self, i: int):
        """Get attn norm for layer i."""
        return self.params[
            f'{self.attn_layer_prefix}.{i}.attention_norm.weight']

    def _ffn(self, i: int, kind: str):
        """Get ffn kind for layer i."""
        result = []
        for key in ['w1', 'w2', 'w3']:
            tensor = self.params[
                f'{self.attn_layer_prefix}.{i}.feed_forward.{key}.{kind}']
            tensor = self.transform(tensor, kind)
            result.append(tensor)
        return (*result, )

    def ffn_norm(self, i: int):
        """Get ffn norm for layer i."""
        return self.params[f'{self.attn_layer_prefix}.{i}.ffn_norm.weight']

    def cross_kv(self, kind: str):
        if not kind:
            return self.filter(r'model.cross_decoder')
        return self._cross_kv(kind)

    def _cross_kv(self, kind: str):
        k = self.params.get(f'model.cross_decoder.wk.{kind}')
        v = self.params.get(f'model.cross_decoder.wv.{kind}')
        k = self.transform(k, kind)
        v = self.transform(v, kind)
        return k, v

    def cross_norm(self):
        return self.params.get('model.cross_decoder.norm.weight')

    def set_cross_keys(self):
        self.attn_layer_prefix = self.cross_attn_layer_prefix
        self.attn_layer_patten = self.cross_attn_layer_patten

        def _replace(old_key):
            matches = re.match(r'^(.*)\.(\d+)\.(.*)$', old_key)
            old_num = int(matches.group(2))
            num_self_decoder = self.model_cfg['num_self_decoder_layers']
            new_num = old_num + num_self_decoder
            new_key = f'{matches.group(1)}.{new_num}.{matches.group(3)}'
            self.params[new_key] = self.params.pop(old_key)

        old_keys = list(self.params.keys())
        for key in old_keys:
            _replace(key)


@INPUT_MODELS.register_module(name='internlm3')
class InternLM3Model(LlamaModel):
    """InternLM3 model in hf format."""

    Reader = InternLM3Reader

    def readers(self):
        # self attn & misc
        loader = create_loader(self.model_path, self.Reader.attn_layer_patten,
                               self.Reader.get_misc_keys())
        for i, param in loader.items():
            reader = self.Reader(param, {},
                                 False,
                                 self.model_config,
                                 policy=self.policy)
            yield i, reader

        # cross kv
        loader = create_loader(self.model_path, self.Reader.cross_kv_pattern,
                               [])
        for i, param in loader.items():
            reader = self.Reader(param, {},
                                 False,
                                 self.model_config,
                                 policy=self.policy)
            yield -1, reader

        # cross attn
        num_self_layer = self.model_config['num_self_decoder_layers']
        loader = create_loader(self.model_path,
                               self.Reader.cross_attn_layer_patten, [])
        for i, param in loader.items():
            reader = self.Reader(param, {},
                                 False,
                                 self.model_config,
                                 policy=self.policy)
            reader.set_cross_keys()
            yield i + num_self_layer, reader

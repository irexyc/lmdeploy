# Copyright (c) OpenMMLab. All rights reserved.
from .base import INPUT_MODELS
from .llama import LlamaModel, LlamaReader


class InternLM3MoeReader(LlamaReader):

    ffn_pattern = r'shared_experts\.'

    def moe_ffn_expert(self, e=None, i=None, kind=None):
        assert kind != 'bias', 'not supported'
        if not kind:
            return self.filter(r'mlp.experts')
        result = []
        for key in ['w1', 'w2', 'w3']:
            tensor = self.params[
                f'{self.attn_layer_prefix}.{i}.mlp.experts.{e}.{key}.{kind}']
            tensor = self.transform(tensor, kind)
            result.append(tensor)

        residual_scale_factor = float(self.model_cfg['residual_scale_factor'])
        result[1] *= residual_scale_factor
        return (*result, )

    def moe_ffn_gate(self, i):
        return self.params.get(f'model.layers.{i}.mlp.gate.weight')

    def _ffn(self, i: int, kind: str):
        """Get ffn kind for layer i."""
        assert (0), 'not supported'

    def moe_ffn_shared_gate(self, i):
        assert (0), 'not supported'


@INPUT_MODELS.register_module(name='internlm3-moe')
class InternLM3MoeModel(LlamaModel):

    Reader = InternLM3MoeReader

    def model_info(self):
        cfg = self.model_config
        info = super().model_info()
        info['attn_bias'] = int(cfg.get('qkv_bias', False))
        info['expert_num'] = cfg['num_experts']
        info['expert_inter_size'] = cfg['intermediate_size']
        info['experts_per_token'] = cfg['num_experts_per_tok']
        info['norm_topk_prob'] = True
        info['moe_shared_gate'] = False
        if cfg.get('num_shared_experts', 0) > 0:
            assert (0), 'not supported'
        else:
            info['inter_size'] = 0

        return info

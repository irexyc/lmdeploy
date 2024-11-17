# Copyright (c) OpenMMLab. All rights reserved.
from .base import INPUT_MODELS
from .internlm2 import InternLM2Reader, LlamaModel


class InternLM3MoeReader(InternLM2Reader):

    ffn_pattern = r'shared_experts\.'

    def moe_ffn_expert(self, e=None, i=None, kind=None):
        if not kind:
            return self.filter(r'feed_forward')
        result = []
        w13_name = f'model.layers.{i}.feed_forward.experts.fused_w1w3'
        w2_name = f'model.layers.{i}.feed_forward.experts.w2'
        w13 = self.params.get(w13_name)
        w2 = self.params.get(w2_name)
        w1, w3 = w13.chunk(2, dim=-1)
        hidden_dim = int(self.model_cfg['hidden_size'])
        inter_size = int(self.model_cfg['intermediate_size'])
        w1e = w1.view(-1, hidden_dim, w1.shape[-1])[e]
        w3e = w3.view(-1, hidden_dim, w3.shape[-1])[e]
        w2e = w2.view(-1, inter_size, w2.shape[-1])[e]
        residual_scale = float(self.model_cfg['residual_scale_factor'])
        for t in [w1e, w2e, w3e]:
            tensor = self.transform(t, kind).t().contiguous()
            if kind == 'bias':
                tensor *= residual_scale
            result.append(tensor)
        return (*result, )

    def moe_ffn_gate(self, i):
        return self.params.get(f'model.layers.{i}.feed_forward.gate.weight')

    def _ffn(self, i: int, kind: str):
        """Get ffn kind for layer i."""
        assert (0), 'not supported'
        if not kind:
            return self.filter(self.ffn_pattern)
        result = []
        for key in ['gate', 'down', 'up']:
            tensor = self.params[
                f'{self.attn_layer_prefix}.{i}.feed_forward.shared_experts.{key}_proj.{kind}']  # noqa: E501
            tensor = self.transform(tensor, kind)
            result.append(tensor)
        return (*result, )

    def moe_ffn_shared_gate(self, i):
        assert (0), 'not supported'


@INPUT_MODELS.register_module(name='internlm3-moe')
class InternLM3MoeModel(LlamaModel):

    Reader = InternLM3MoeReader

    def model_info(self):
        cfg = self.model_config
        info = super().model_info()
        info['expert_num'] = cfg['num_routed_experts']
        info['expert_inter_size'] = cfg['intermediate_size']
        info['experts_per_token'] = cfg['num_experts_per_tok']
        info['moe_norm_topk'] = True
        info['moe_global_scale'] = cfg['residual_scale_factor']
        if cfg.get('num_shared_experts', 0) > 0:
            info['moe_shared_gate'] = True
            info['inter_size'] = info['inter_size'] * cfg.get(
                'num_shared_experts')
        else:
            info['inter_size'] = 0

        return info

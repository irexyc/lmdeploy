# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
import copy
import json
import os.path as osp
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import partial
from itertools import repeat
from queue import Queue
from typing import Dict, List

import numpy as np
import torch
import yaml
from torch.nn.utils.rnn import pad_sequence

import lmdeploy
from lmdeploy.messages import EngineOutput, GenerationConfig, ResponseType, TurbomindEngineConfig
from lmdeploy.utils import get_logger, get_max_batch_size, get_model

from .deploy.config import TurbomindModelConfig
from .supported_models import is_supported
from .utils import ModelSource, get_model_source

# TODO: find another way import _turbomind
lmdeploy_dir = osp.split(lmdeploy.__file__)[0]
sys.path.append(osp.join(lmdeploy_dir, 'lib'))
import _turbomind as _tm  # noqa: E402

logger = get_logger('lmdeploy')

MAX_LOGPROBS = 1024


def _construct_stop_or_bad_words(words: List[int] = None):
    if words is None or len(words) == 0:
        return None
    offsets = list(range(1, len(words) + 1))
    combined = [words, offsets]
    return combined


def _np_dict_to_tm_dict(np_dict: dict):
    """map numpy.ndarray to turbomind's tensor."""
    ret = _tm.TensorMap()
    for k, v in np_dict.items():
        ret[k] = _tm.from_dlpack(v)

    return ret


def _tm_dict_to_torch_dict(tm_dict: _tm.TensorMap):
    """map turbomind's tensor to torch's tensor."""
    ret = dict()
    for k, v in tm_dict.items():
        if v.type == _tm.DataType.TYPE_UINT32:
            v = v.view(_tm.DataType.TYPE_INT32)
        ret[k] = torch.from_dlpack(v)

    return ret


class TurboMind:
    """LMDeploy's inference engine.

    Args:
        model_path (str): the path of turbomind's model
        mode_name (str): the name of the served model
        chat_template_name (str): the name of the chat template, which is
            supposed to be a builtin chat template defined in
            `lmdeploy/model.py`
        engine_config (TurbomindEngineConfig): the config of the inference
            engine
        model_source (int): the source of the model, which is either
            turbomind model, or a transformers model
    """

    def __init__(self,
                 model_path: str,
                 tokenizer: object,
                 model_name: str = None,
                 chat_template_name: str = None,
                 engine_config: TurbomindEngineConfig = None,
                 model_source: ModelSource = ModelSource.WORKSPACE,
                 **kwargs):
        self.model_name = model_name
        self.chat_template_name = chat_template_name

        _engine_config = copy.deepcopy(engine_config)
        if _engine_config is None:
            _engine_config = TurbomindEngineConfig()
        if _engine_config.max_batch_size is None:
            _engine_config.max_batch_size = get_max_batch_size('cuda')
        assert _engine_config.max_batch_size > 0, 'max_batch_size should be' \
            f' greater than 0, but got {_engine_config.max_batch_size}'

        self.gpu_count = _engine_config.tp

        self.tokenizer = tokenizer
        if model_source == ModelSource.WORKSPACE:
            self.model_comm = self._from_workspace(model_path=model_path, engine_config=_engine_config)
        else:
            if not osp.exists(model_path):
                model_path = get_model(model_path, _engine_config.download_dir, _engine_config.revision)
            self.model_comm = self._from_hf(model_source=model_source,
                                            model_path=model_path,
                                            engine_config=_engine_config)

        with ThreadPoolExecutor(max_workers=self.gpu_count) as e:
            ranks = [self.node_id * self.gpu_count + device_id for device_id in range(self.gpu_count)]
            for _ in e.map(self.model_comm.process_weight, range(self.gpu_count), ranks):
                pass
            # implicit synchronization
            for _ in e.map(self.model_comm.create_engine, range(self.gpu_count), ranks, repeat(self.nccl_params)):
                pass

        self.session_len = self.config.session_len

    def _create_weight(self, model_comm):
        """Allocate weight buffer, load params if from_workspace."""

        # TODO: support mpi
        self.node_id = 0
        self.node_num = 1
        self.nccl_params = model_comm.create_nccl_params(self.node_id)
        torch.cuda.synchronize()

        # create weight
        def _create_weight_func(device_id):
            rank = self.node_id * self.gpu_count + device_id
            model_comm.create_shared_weights(device_id, rank)

        with ThreadPoolExecutor(max_workers=self.gpu_count) as executor:
            futures = []
            for device_id in range(self.gpu_count):
                futures.append(executor.submit(_create_weight_func, device_id))
            for future in futures:
                future.result()

    def _get_model_params(self, model_comm, tm_params):
        """Get turbomind model params when loading from hf."""

        def _get_params(device_id, que):
            rank = self.node_id * self.gpu_count + device_id
            out = model_comm.get_params(device_id, rank)
            que.put(out)

        que = Queue()
        with ThreadPoolExecutor(max_workers=self.gpu_count) as executor:
            futures = []
            for device_id in range(self.gpu_count):
                futures.append(executor.submit(_get_params, device_id, que))
            for future in futures:
                future.result()

        for _ in range(self.gpu_count):
            tensor_map = que.get()
            for k, v in tensor_map.items():
                if k not in tm_params:
                    tm_params[k] = []
                tm_params[k].append(v)

    def _postprocess_config(self, tm_config: TurbomindModelConfig, engine_config: TurbomindEngineConfig):
        """postprocess turbomind config by."""
        import copy
        self.config = copy.deepcopy(tm_config)
        # Update the attribute values in `self.config` with the valid values
        # from the corresponding attributes in `engine_config`, such as
        # `session_len`, `quant_policy`, `rope_scaling_factor`, etc.
        self.config.update_from_engine_config(engine_config)

        # update some attributes of `engine_config` which depends on
        # `session_len`
        self.engine_config = engine_config
        if engine_config.max_prefill_token_num is not None \
                and engine_config.num_tokens_per_iter == 0:
            self.engine_config.num_tokens_per_iter = \
                engine_config.max_prefill_token_num
            self.engine_config.max_prefill_iters = (self.config.session_len + engine_config.max_prefill_token_num -
                                                    1) // engine_config.max_prefill_token_num

        # pack `self.config` and `self.engine_config` into a dict
        self.config_dict = self.config.to_dict()
        self.config_dict.update(dict(engine_config=asdict(self.engine_config)))
        logger.info(f'turbomind model config:\n\n'
                    f'{json.dumps(self.config_dict, indent=2)}')

    def _from_hf(self, model_source: ModelSource, model_path: str, engine_config: TurbomindEngineConfig):
        """Load model which is in hf format."""
        assert model_source == ModelSource.HF_MODEL, \
            f'{model_source} is not supported'
        assert is_supported(model_path), (f'turbomind does not support {model_path}. '
                                          'Plz try pytorch engine instead.')

        # convert transformers model into turbomind model
        from .deploy.converter import get_tm_model
        tm_model = get_tm_model(model_path, self.model_name, self.chat_template_name, engine_config)

        self._postprocess_config(tm_model.tm_config, engine_config)

        model_comm = _tm.AbstractTransformerModel.create_llama_model(model_dir='',
                                                                     config=yaml.safe_dump(self.config_dict),
                                                                     tensor_para_size=self.gpu_count,
                                                                     data_type=self.config.model_config.weight_type)

        # create empty weight
        self._create_weight(model_comm)

        # copy hf model weight to turbomind weight
        tm_params = tm_model.tm_params
        self._get_model_params(model_comm, tm_params)
        logger.warning(f'get {len(tm_params)} model params')
        tm_model.export()
        # there should be no left turbomind params.
        if len(tm_params) > 0:
            uninitialized = list(tm_params.keys())
            logger.warning('the model may not be loaded successfully '
                           f'with {len(tm_params)} uninitialized params:\n{uninitialized}')
        return model_comm

    def _from_workspace(self, model_path: str, engine_config: TurbomindEngineConfig):
        """Load model which is converted by `lmdeploy convert`"""
        config_path = osp.join(model_path, 'triton_models', 'weights', 'config.yaml')
        # load TurbomindModelConfig from config file
        with open(config_path, 'r') as f:
            _cfg = yaml.safe_load(f)
        cfg = TurbomindModelConfig.from_dict(_cfg)

        # always use tp in converted model (config.yaml)
        if cfg.tensor_para_size != engine_config.tp:
            logger.warning('tp in engine_config is different from in config.yaml'
                           f'({config_path}), {engine_config.tp} vs '
                           f'{cfg.tensor_para_size}, using tp={cfg.tensor_para_size}')
        self.gpu_count = cfg.tensor_para_size
        engine_config.tp = self.gpu_count

        self._postprocess_config(cfg, engine_config)

        weight_dir = osp.join(model_path, 'triton_models', 'weights')
        model_comm = _tm.AbstractTransformerModel.create_llama_model(model_dir=weight_dir,
                                                                     config=yaml.safe_dump(self.config_dict),
                                                                     tensor_para_size=self.gpu_count,
                                                                     data_type=self.config.weight_type)

        # create weight and load params
        self._create_weight(model_comm)
        return model_comm

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path: str,
                        tokenizer: object,
                        model_name: str = None,
                        chat_template_name: str = None,
                        engine_config: TurbomindEngineConfig = None,
                        **kwargs):
        """LMDeploy's turbomind inference engine.

        Args:
            pretrained_model_name_or_path (str):
                It could be one of the following options:
                    - i) A local directory path of a turbomind model which is
                      converted by `lmdeploy convert` command or download from
                      ii) and iii)
                    - ii) The model_id of a lmdeploy-quantized model hosted
                      inside a model repo on huggingface.co, such as
                      "InternLM/internlm-chat-20b-4bit",
                      "lmdeploy/llama2-chat-70b-4bit", etc.
                    - iii) The model_id of a model hosted inside a model repo
                      on huggingface.co, such as "internlm/internlm-chat-7b",
                      "Qwen/Qwen-7B-Chat ", "baichuan-inc/Baichuan2-7B-Chat"
                      and so on.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update configuration when initialize the engine.
        """
        model_source = get_model_source(pretrained_model_name_or_path)
        logger.info(f'model_source: {model_source}')
        return cls(model_path=pretrained_model_name_or_path,
                   tokenizer=tokenizer,
                   model_name=model_name,
                   chat_template_name=chat_template_name,
                   engine_config=engine_config,
                   model_source=model_source,
                   **kwargs)

    def close(self):
        if self.nccl_params is not None:
            self.model_comm.destroy_nccl_params(self.nccl_params)
            self.nccl_params = None
        if self.model_comm is not None:
            self.model_comm = None

    def create_instance(self, cuda_stream_id=0):
        """Create a turbomind instance.

        Args:
            cuda_stream_id(int): identity of a cuda stream
        Returns:
            TurboMindInstance: an instance of turbomind
        """
        return TurboMindInstance(self, self.config, cuda_stream_id)


def _get_logits(outputs, offset: int):
    logits = outputs['logits']

    def _func(out: EngineOutput, step: int):
        out.logits = logits[:step - offset - 1, :]

    return _func


def _get_last_hidden_state(outputs, offset: int):
    last_hidden_state = outputs['last_hidden_state']

    def _func(out: EngineOutput, step: int):
        out.last_hidden_state = last_hidden_state[:step - offset - 1, :]

    return _func


def _get_logprobs_impl(logprob_vals: torch.Tensor,
                       logprob_idxs: torch.Tensor,
                       logprob_nums: torch.Tensor,
                       output_ids: List[int],
                       logprobs: int,
                       out_logprobs: List[Dict[int, float]] = None):
    length = len(output_ids)
    offset = len(out_logprobs)
    if length == offset:
        return out_logprobs
    for (pos, idx, val, n) in zip(range(offset, length), logprob_idxs[offset:length], logprob_vals[offset:length],
                                  logprob_nums[offset:length]):
        topn = min(n.item(), logprobs)
        tok_res = {idx[i].item(): val[i].item() for i in range(topn)}
        token_id = output_ids[pos]
        if token_id not in tok_res:
            print(token_id, tok_res)
            valid_n = n.item()
            tok_res[token_id] = \
                val[:valid_n][idx[:valid_n] == token_id].item()
        ids = list(tok_res.keys())
        for k in ids:
            if tok_res[k] == float('-inf'):
                tok_res.pop(k)
        out_logprobs.append(tok_res)
    return out_logprobs


def _get_logprobs(outputs, output_logprobs: int):
    logprob_vals = outputs['logprob_vals']
    logprob_idxs = outputs['logprob_indexes']
    logprob_nums = outputs['logprob_nums']

    logprobs = []

    def _func(out: EngineOutput, step: int):
        _get_logprobs_impl(logprob_vals, logprob_idxs, logprob_nums, out.token_ids, output_logprobs, logprobs)
        out.logprobs = logprobs

    return _func


class StreamingSemaphore:

    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.fut = None
        self.val = 0

    async def acquire(self):
        if self.val:
            self.val = 0
            return
        self.fut = self.loop.create_future()
        await self.fut
        self.fut = None
        self.val = 0

    def release(self):
        if not self.val:
            self.val = 1
            if self.fut:
                self.fut.set_result(None)


class TurboMindInstance:
    """Instance of TurboMind.

    Args:
        tm_model (str): turbomind's model path
        cuda_stream_id(int): identity of a cuda stream
    """

    def __init__(self, tm_model: TurboMind, config: TurbomindModelConfig, cuda_stream_id: int = 0):
        self.tm_model = tm_model
        self.cuda_stream_id = cuda_stream_id

        self.node_id = tm_model.node_id
        self.gpu_count = tm_model.gpu_count

        self.session_len = tm_model.session_len

        self.nccl_params = tm_model.nccl_params

        # create model instances
        self.model_inst = self._create_model_instance(0)

        self.config = config
        self.lock = None

    def _create_model_instance(self, device_id):
        model_inst = self.tm_model.model_comm.create_model_instance(device_id)
        return model_inst

    def _get_extra_output_processors(self, outputs: Dict[str, torch.Tensor], gen_config: GenerationConfig,
                                     input_len: int):

        def _get_offset(type):
            return input_len - 1 if type == 'generation' else 0

        fs = []
        if gen_config.output_logits:
            offset = _get_offset(gen_config.output_logits)
            fs.append(_get_logits(outputs, offset))
        if gen_config.output_last_hidden_state:
            offset = _get_offset(gen_config.output_last_hidden_state)
            fs.append(_get_last_hidden_state(outputs, offset))
        if gen_config.logprobs:
            fs.append(_get_logprobs(outputs, gen_config.logprobs))
        return fs

    def prepare_embeddings(self, input_embeddings=None, input_embedding_ranges=None):
        """Convert embeddings."""
        if input_embeddings is None:
            return None, None

        assert len(input_embeddings) == len(input_embedding_ranges)
        if not isinstance(input_embeddings[0], (list, type(None))):
            input_embeddings = [input_embeddings]
            input_embedding_ranges = [input_embedding_ranges]

        if all([isinstance(x, type(None)) for x in input_embeddings]):
            return None, None

        hidden_dim = None
        for embeddings in input_embeddings:
            if embeddings is not None:
                hidden_dim = embeddings[0].squeeze().shape[-1]
                break
        assert hidden_dim is not None

        # construct input_embeddings
        for i in range(len(input_embeddings)):
            item = input_embeddings[i] or []
            # convert to torch.Tensor if input is np.ndarray
            if item and isinstance(item[0], np.ndarray):
                item = [torch.from_numpy(x).squeeze() for x in item]
            # convert to lookup table type
            _MAP = dict(float=torch.float, bfloat16=torch.bfloat16, float16=torch.float16)
            dtype = _MAP.get(self.tm_model.config.weight_type, torch.float16)
            item = [x.to(dtype=dtype) for x in item]
            item = item or [torch.zeros(0, hidden_dim, dtype=dtype)]
            input_embeddings[i] = item
        input_embeddings = [torch.cat(x) for x in input_embeddings]
        input_embeddings = pad_sequence(input_embeddings, batch_first=True)
        input_embeddings = input_embeddings.reshape(input_embeddings.shape[0], -1).view(torch.int8)
        # construct input_embedding_ranges
        for i in range(len(input_embedding_ranges)):
            item = input_embedding_ranges[i] or []
            item = torch.IntTensor(item).reshape(-1, 2)
            input_embedding_ranges[i] = item
        input_embedding_ranges = pad_sequence(input_embedding_ranges, batch_first=True, padding_value=-1)

        return input_embeddings, input_embedding_ranges

    def prepare_inputs(self,
                       input_ids,
                       gen_config: GenerationConfig,
                       input_embeddings=None,
                       input_embedding_ranges=None):
        """Convert inputs format."""
        assert isinstance(input_ids, Sequence)

        input_ids = torch.IntTensor(input_ids)
        input_len = len(input_ids)

        inputs = dict(input_ids=input_ids, )

        input_embeddings, input_embedding_ranges = self.prepare_embeddings(input_embeddings, input_embedding_ranges)
        if input_embeddings is not None:
            inputs['input_embeddings'] = input_embeddings
            inputs['input_embedding_ranges'] = input_embedding_ranges

        return inputs, input_len

    async def async_cancel(self, session_id: int = None):
        self.model_inst.cancel()

    def async_end_cb(self, fut: asyncio.Future, status: int):
        """executing on engine's signaling thread."""
        logger.info(f'[async_end_cb] session ended, status = {status}')
        fut.get_loop().call_soon_threadsafe(fut.set_result, status)

    async def async_end(self, session_id):
        fut = asyncio.get_running_loop().create_future()
        self.model_inst.end(partial(self.async_end_cb, fut), session_id)
        await fut

    def async_signal_cb(self, s: StreamingSemaphore):
        """executing on engine's signaling thread."""
        s.loop.call_soon_threadsafe(s.release)

    async def async_stream_infer(self,
                                 session_id,
                                 input_ids,
                                 input_embeddings=None,
                                 input_embedding_ranges=None,
                                 sequence_start: bool = True,
                                 sequence_end: bool = False,
                                 step=0,
                                 gen_config: GenerationConfig = None,
                                 stream_output=False,
                                 **kwargs):
        """Perform model inference.

        Args:
            session_id (int): the id of a session
            input_ids (numpy.ndarray): the token ids of a prompt
            input_embeddings (List[numpy.ndarray]): embeddings features
            input_embedding_ranges (List[Tuple[int,int]]): the begin/end
              offsets of input_embeddings to input_ids
            sequence_start (bool): indicator for starting a sequence
            sequence_end (bool): indicator for ending a sequence
            step (int): the offset of the k/v cache
            stop (bool): indicator for cancelling the session
            gen_config (GenerationConfig): generation config
            stream_output (bool): indicator for stream output
            kwargs (dict): kwargs for backward compatibility
        """
        logger.info(f'[async_stream_infer] session {session_id} start')
        gen_cfg = self._get_generation_config(gen_config)

        inputs, input_len = self.prepare_inputs(input_ids=input_ids,
                                                input_embeddings=input_embeddings,
                                                input_embedding_ranges=input_embedding_ranges,
                                                gen_config=gen_config)

        session = _tm.SessionParam(id=session_id, step=step, start=sequence_start, end=sequence_end)

        inputs = _np_dict_to_tm_dict(inputs)

        sem = StreamingSemaphore()
        signal_cb = partial(self.async_signal_cb, sem)

        outputs, shared_state = self.model_inst.forward(inputs, session, gen_cfg, stream_output, signal_cb)

        outputs = _tm_dict_to_torch_dict(outputs)

        extra_fs = self._get_extra_output_processors(outputs, gen_config, input_len)

        output_ids_buf = outputs['output_ids']

        finish = False
        state = None

        output_ids = []
        output_len = 0
        prev_len = step + input_len
        try:
            while True:
                await sem.acquire()
                state = shared_state.consume()

                status, seq_len = state.status, state.seq_len

                if status in [7, 8]:  # finish / canceled
                    finish, status = True, 0
                elif status:
                    yield self._get_error_output()
                    break

                if seq_len == prev_len and not finish:
                    continue

                output_ids += output_ids_buf[prev_len:seq_len].tolist()
                output_len += seq_len - prev_len
                status = ResponseType.FINISH if finish else ResponseType.SUCCESS  # noqa
                output = EngineOutput(status, output_ids, output_len)

                for f in extra_fs:
                    f(output, seq_len)

                prev_len = seq_len

                yield output

                if finish:
                    break

        except (GeneratorExit, asyncio.CancelledError) as e:
            logger.info(f'[async_stream_infer] {type(e).__name__}')
            self.model_inst.cancel()
        except Exception as e:
            logger.error(f'[async_stream_infer] {type(e).__name__} {e}')
            self.model_inst.cancel()
            yield self._get_error_output()
        finally:
            # Contract: `cb` won't be called again if status is non-zero
            # wait for status to be set as `finish` or `error`
            while not state or state.status == 0:
                await sem.acquire()
                state = shared_state.consume()
            logger.info(f'[async_stream_infer] session {session_id} done')

    def _get_error_output(self):
        return EngineOutput(status=ResponseType.INTERNAL_ENGINE_ERROR, token_ids=[], num_token=0)

    def _get_generation_config(self, cfg: GenerationConfig):
        c = _tm.GenerationConfig()
        c.max_new_tokens = cfg.max_new_tokens
        c.top_k = cfg.top_k
        c.top_p = cfg.top_p
        c.min_p = cfg.min_p
        c.temperature = cfg.temperature
        if cfg.stop_token_ids:
            c.eos_ids = cfg.stop_token_ids
        if cfg.bad_token_ids:
            c.bad_ids = _construct_stop_or_bad_words(cfg.bad_token_ids)
        if not cfg.ignore_eos and cfg.stop_token_ids:
            c.stop_ids = _construct_stop_or_bad_words(cfg.stop_token_ids)
        c.repetition_penalty = cfg.repetition_penalty
        if cfg.min_new_tokens:
            c.min_new_tokens = cfg.min_new_tokens
        output_type = dict(all=1, generation=2)
        if cfg.output_last_hidden_state:
            c.output_last_hidden_state = output_type[cfg.output_last_hidden_state]
        if cfg.output_logits:
            c.output_logits = output_type[cfg.output_logits]
        if cfg.logprobs:
            if cfg.logprobs > MAX_LOGPROBS:
                cfg.logprobs = MAX_LOGPROBS
                logger.warning(f'logprobs shoudd be in range [1, {MAX_LOGPROBS}]'
                               f'update logprobs={cfg.logprobs}')
            c.output_logprobs = cfg.logprobs
        if cfg.random_seed is not None:
            c.random_seed = cfg.random_seed
        # print (c)
        return c

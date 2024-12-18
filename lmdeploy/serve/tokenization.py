# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union, Dict, List, Any
import tempfile
from dataclasses import dataclass
import zmq
from lmdeploy.tokenizer import Tokenizer, DetokenizeState
import asyncio


@dataclass
class TokenizeInput:
    session_id: int
    prompt: str
    add_bos: bool

@dataclass
class TokenizeOutput:
    session_id: int
    input_ids: List[int]

@dataclass
class DeTokenizeInput:
    session_id: int
    sequence_start: bool
    input_ids: List[int]
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    state: DetokenizeState = None

@dataclass
class DeTokenizeOutput:
    session_id: int
    response: str

@dataclass
class ProcessOutput:
    result: Union[TokenizeOutput, DeTokenizeOutput, None]
    event: asyncio.Event


@dataclass
class ProcessArgs:
    """ipc args."""

    to_tokenize_name: str
    from_tokenize_name: str
    to_detokenize_name: str
    from_detokenize_name: str

    @staticmethod
    def init_new():
        return ProcessArgs(
            to_tokenize_name=tempfile.NamedTemporaryFile(delete=False).name,
            from_tokenize_name=tempfile.NamedTemporaryFile(delete=False).name,
            to_detokenize_name=tempfile.NamedTemporaryFile(delete=False).name,
            from_detokenize_name=tempfile.NamedTemporaryFile(delete=False).name)


def get_zmq_socket(context: zmq.Context, socket_type: zmq.SocketType, endpoint: str):
    socket = context.socket(socket_type)
    if socket_type == zmq.PUSH:
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, int(0.5 * 1024**3))
        socket.connect(f"ipc://{endpoint}")
    elif socket_type == zmq.PULL:
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF, int(0.5 * 1024**3))
        socket.bind(f"ipc://{endpoint}")
    else:
        raise ValueError(f"Unsupported socket type: {socket_type}")
    return socket



class Tokenize:

    def __init__(self, tokenizer, process_args: ProcessArgs):
        self.tokenizer = tokenizer
        context = zmq.Context(2)
        self.recv_from_engine = get_zmq_socket(context, zmq.PULL, process_args.to_tokenize_name)
        self.send_to_engine = get_zmq_socket(context, zmq.PUSH, process_args.from_tokenize_name)

    def event_loop(self):
        while True:
            recv_obj: TokenizeInput = self.recv_from_engine.recv_pyobj()
            input_ids = self.tokenizer.encode(recv_obj.prompt, add_bos=recv_obj.add_bos)
            self.send_to_engine.send_pyobj(TokenizeOutput(session_id=recv_obj.session_id, input_ids=input_ids))


class DeTokenize:

    def __init__(self, tokenizer, process_args: ProcessArgs):
        self.tokenizer = tokenizer
        context = zmq.Context()
        self.recv_from_engine = get_zmq_socket(context, zmq.PULL, process_args.to_detokenize_name)
        self.send_to_engine = get_zmq_socket(context, zmq.PUSH, process_args.from_detokenize_name)
        self.state = {}

    def event_loop(self):
        while True:
            recv_obj: DeTokenizeInput = self.recv_from_engine.recv_pyobj()
            if recv_obj.sequence_start:
                 recv_obj.state = DetokenizeState(len(recv_obj.input_ids))
                 _, recv_obj.state = self.tokenizer.detokenize_incrementally(
                     recv_obj.input_ids, recv_obj.state)
                 self.state[recv_obj.session_id] = recv_obj
                 continue
            obj: DeTokenizeInput = self.state.get(recv_obj.session_id)
            obj.input_ids += recv_obj.input_ids
            response, obj.state = self.tokenizer.detokenize_incrementally(
                obj.input_ids, obj.state)
            self.send_to_engine.send_pyobj(DeTokenizeOutput(session_id=obj.session_id, response=response))


def run_tokenize_process(
    tokenizer: Tokenizer,
    process_args: ProcessArgs,
):
    manager = Tokenize(tokenizer, process_args)
    manager.event_loop()


def run_detokenize_process(
    tokenizer: Tokenizer,
    process_args: ProcessArgs,
):
    manager = DeTokenize(tokenizer, process_args)
    manager.event_loop()


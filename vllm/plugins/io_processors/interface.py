# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mini-vLLM: IOProcessor simplified, pooling not supported

from abc import ABC
from typing import Any

from vllm.config import VllmConfig


class IOProcessor(ABC):
    """Stub for IOProcessor. Pooling/IOProcessor is not supported in mini-vLLM."""

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config

    def pre_process(self, *args, **kwargs):
        raise NotImplementedError("IOProcessor is not supported in mini-vLLM")

    async def pre_process_async(self, *args, **kwargs):
        raise NotImplementedError("IOProcessor is not supported in mini-vLLM")

    def post_process(self, *args, **kwargs):
        raise NotImplementedError("IOProcessor is not supported in mini-vLLM")

    async def post_process_async(self, *args, **kwargs):
        raise NotImplementedError("IOProcessor is not supported in mini-vLLM")

    def parse_request(self, request: Any):
        raise NotImplementedError("IOProcessor is not supported in mini-vLLM")

    def validate_or_generate_params(self, params=None):
        raise NotImplementedError("IOProcessor is not supported in mini-vLLM")

    def output_to_response(self, plugin_output):
        raise NotImplementedError("IOProcessor is not supported in mini-vLLM")

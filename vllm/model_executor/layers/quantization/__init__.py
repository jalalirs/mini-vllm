# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# NOTE: mini-vLLM - reduced to only mxfp4, gptq_marlin, fp8

from typing import Literal, get_args

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

logger = init_logger(__name__)

# mini-vLLM: Only keep quantization methods needed for gpt-oss models
QuantizationMethods = Literal[
    "mxfp4",
    "gptq_marlin",
    "fp8",
]
QUANTIZATION_METHODS: list[str] = list(get_args(QuantizationMethods))

# The customized quantization methods which will be added to this dict.
_CUSTOMIZED_METHOD_TO_QUANT_CONFIG = {}


def register_quantization_config(quantization: str):
    """Register a customized vllm quantization config.

    When a quantization method is not supported by vllm, you can register a customized
    quantization config to support it.

    Args:
        quantization (str): The quantization method name.

    Examples:
        >>> from vllm.model_executor.layers.quantization import (
        ...     register_quantization_config,
        ... )
        >>> from vllm.model_executor.layers.quantization import get_quantization_config
        >>> from vllm.model_executor.layers.quantization.base_config import (
        ...     QuantizationConfig,
        ... )
        >>>
        >>> @register_quantization_config("my_quant")
        ... class MyQuantConfig(QuantizationConfig):
        ...     pass
        >>>
        >>> get_quantization_config("my_quant")
        <class 'MyQuantConfig'>
    """  # noqa: E501

    def _wrapper(quant_config_cls):
        if quantization in QUANTIZATION_METHODS:
            logger.warning(
                "The quantization method '%s' already exists and will be "
                "overwritten by the quantization config %s.",
                quantization,
                quant_config_cls,
            )
        else:
            QUANTIZATION_METHODS.append(quantization)

        if not issubclass(quant_config_cls, QuantizationConfig):
            raise ValueError(
                "The quantization config must be a subclass of `QuantizationConfig`."
            )
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization] = quant_config_cls
        return quant_config_cls

    return _wrapper


def get_quantization_config(quantization: str) -> type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")

    # mini-vLLM: Only import quantization methods we actually need
    from .fp8 import Fp8Config
    from .gptq_marlin import GPTQMarlinConfig
    from .mxfp4 import Mxfp4Config

    method_to_config: dict[str, type[QuantizationConfig]] = {
        "mxfp4": Mxfp4Config,
        "gptq_marlin": GPTQMarlinConfig,
        "fp8": Fp8Config,
    }
    # Update the `method_to_config` with customized quantization methods.
    method_to_config.update(_CUSTOMIZED_METHOD_TO_QUANT_CONFIG)

    return method_to_config[quantization]


__all__ = [
    "QuantizationConfig",
    "QuantizationMethods",
    "get_quantization_config",
    "QUANTIZATION_METHODS",
]

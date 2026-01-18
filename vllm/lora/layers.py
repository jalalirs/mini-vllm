# SPDX-License-Identifier: Apache-2.0
# mini-vLLM: LoRA support removed, minimal stub only

from enum import IntEnum
from typing import NamedTuple


class LoRAMappingType(IntEnum):
    """Stub for LoRA mapping type."""
    DEFAULT = 0


class LoRAMapping(NamedTuple):
    """Stub for LoRA mapping. LoRA is not supported in mini-vLLM."""
    index_mapping: tuple = ()
    prompt_mapping: tuple = ()
    is_prefill: bool = False

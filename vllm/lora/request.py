# SPDX-License-Identifier: Apache-2.0
# mini-vLLM: LoRA support removed - stub for compatibility
from dataclasses import dataclass
from typing import Any


@dataclass
class LoRARequest:
    """Stub LoRARequest for compatibility - LoRA support removed in mini-vLLM."""

    lora_name: str = ""
    lora_int_id: int = 0
    lora_path: str = ""
    long_lora_max_len: int | None = None
    base_model_name: str | None = None
    __hash__ = object.__hash__

    def __post_init__(self):
        pass

    @property
    def lora_local_path(self) -> str:
        return self.lora_path

    @property
    def is_empty(self) -> bool:
        return True

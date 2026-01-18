# SPDX-License-Identifier: Apache-2.0
# mini-vLLM: LoRA support removed, minimal stub only

import msgspec


class LoRARequest(
    msgspec.Struct,
    omit_defaults=True,
    array_like=True,
):
    """Stub for LoRA request. LoRA is not supported in mini-vLLM."""

    lora_name: str
    lora_int_id: int
    lora_path: str = ""
    base_model_name: str | None = msgspec.field(default=None)
    tensorizer_config_dict: dict | None = None

    def __post_init__(self):
        raise NotImplementedError("LoRA is not supported in mini-vLLM")

    @property
    def adapter_id(self):
        return self.lora_int_id

    @property
    def name(self):
        return self.lora_name

    @property
    def path(self):
        return self.lora_path

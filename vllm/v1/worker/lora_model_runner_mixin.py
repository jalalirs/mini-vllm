# SPDX-License-Identifier: Apache-2.0
# mini-vLLM: LoRA support removed - stub for compatibility
from contextlib import nullcontext


class LoRAModelRunnerMixin:
    """Stub LoRAModelRunnerMixin - LoRA support removed."""

    def init_lora_manager(self, *args, **kwargs):
        """No-op - LoRA removed."""
        pass

    def set_lora_adapter(self, *args, **kwargs):
        """No-op - LoRA removed."""
        pass

    def add_lora_adapter(self, *args, **kwargs):
        """No-op - LoRA removed."""
        pass

    def remove_lora_adapter(self, *args, **kwargs):
        """No-op - LoRA removed."""
        pass

    def maybe_dummy_run_with_lora(self, *args, **kwargs):
        """No-op context manager - LoRA removed."""
        return nullcontext()

    def _create_lora_mapping(self, *args, **kwargs):
        """No-op - LoRA removed."""
        return None

    def maybe_remove_all_loras(self, *args, **kwargs):
        """No-op - LoRA removed."""
        pass

    def list_loras(self, *args, **kwargs):
        """No-op - LoRA removed."""
        return []

    def add_lora(self, *args, **kwargs):
        """No-op - LoRA removed."""
        return True

    def remove_lora(self, *args, **kwargs):
        """No-op - LoRA removed."""
        return True

    def _load_lora(self, *args, **kwargs):
        """No-op - LoRA removed."""
        return None

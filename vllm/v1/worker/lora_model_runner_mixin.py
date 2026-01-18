# SPDX-License-Identifier: Apache-2.0
# mini-vLLM: LoRA support removed, minimal stub only

from contextlib import contextmanager


class LoRAModelRunnerMixin:
    """Stub for LoRA model runner mixin. LoRA is not supported in mini-vLLM."""

    lora_manager = None
    supports_lora = False

    def set_active_loras(self, *args, **kwargs):
        pass

    def add_lora(self, *args, **kwargs):
        raise NotImplementedError("LoRA is not supported in mini-vLLM")

    def remove_lora(self, *args, **kwargs):
        raise NotImplementedError("LoRA is not supported in mini-vLLM")

    @contextmanager
    def maybe_dummy_run_with_lora(self, *args, **kwargs):
        """No-op context manager for LoRA dummy run."""
        yield

    @contextmanager
    def maybe_setup_dummy_loras(self, *args, **kwargs):
        """No-op context manager for LoRA setup."""
        yield

    @contextmanager
    def maybe_select_dummy_loras(self, *args, **kwargs):
        """No-op context manager for LoRA selection."""
        yield

    def maybe_remove_all_loras(self, *args, **kwargs):
        """No-op for LoRA removal."""
        pass

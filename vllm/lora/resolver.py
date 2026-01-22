# SPDX-License-Identifier: Apache-2.0
# mini-vLLM: LoRA support removed - stub for compatibility


class LoRAResolver:
    """Stub LoRAResolver - LoRA support removed."""
    pass


class LoRAResolverRegistry:
    """Stub LoRAResolverRegistry - LoRA support removed."""

    @classmethod
    def get_supported_resolvers(cls):
        """No resolvers - LoRA removed."""
        return []

    @classmethod
    def get_resolver(cls, name):
        """No resolvers - LoRA removed."""
        return None

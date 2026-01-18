# SPDX-License-Identifier: Apache-2.0
# mini-vLLM: LoRA support removed, minimal stub only


class LoRAResolver:
    """Stub for LoRA resolver. LoRA is not supported in mini-vLLM."""
    pass


class LoRAResolverRegistry:
    """Stub for LoRA resolver registry. LoRA is not supported in mini-vLLM."""

    resolvers = {}

    @classmethod
    def get_supported_resolvers(cls):
        """Return empty set of resolvers."""
        return set()

    @classmethod
    def register_resolver(cls, resolver_name, resolver):
        """No-op for registering resolvers."""
        pass

    @classmethod
    def get_resolver(cls, resolver_name):
        """Return None for any resolver."""
        return None

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention backend registry"""

from collections.abc import Callable
from enum import Enum, EnumMeta
from typing import TYPE_CHECKING, cast

from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

logger = init_logger(__name__)


class _AttentionBackendEnumMeta(EnumMeta):
    """Metaclass for AttentionBackendEnum to provide better error messages."""

    def __getitem__(cls, name: str):
        """Get backend by name with helpful error messages."""
        try:
            return super().__getitem__(name)
        except KeyError:
            members = cast("dict[str, Enum]", cls.__members__).keys()
            valid_backends = ", ".join(members)
            raise ValueError(
                f"Unknown attention backend: '{name}'. "
                f"Valid options are: {valid_backends}"
            ) from None


class AttentionBackendEnum(Enum, metaclass=_AttentionBackendEnumMeta):
    """Enumeration of all supported attention backends.

    mini-vLLM: Only FLASH_ATTN supported for CUDA/H100.
    """

    FLASH_ATTN = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    # mini-vLLM: Removed backends kept as stubs pointing to FLASH_ATTN
    TRITON_ATTN = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    ROCM_ATTN = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    ROCM_AITER_MLA = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    ROCM_AITER_TRITON_MLA = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    ROCM_AITER_FA = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    ROCM_AITER_MLA_SPARSE = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    TORCH_SDPA = ""  # this tag is only used for ViT
    FLASHINFER = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    FLASHINFER_MLA = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    TRITON_MLA = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    CUTLASS_MLA = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    FLASHMLA = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    FLASHMLA_SPARSE = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    FLASH_ATTN_MLA = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    PALLAS = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    IPEX = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    NO_ATTENTION = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    FLEX_ATTENTION = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    TREE_ATTN = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    ROCM_AITER_UNIFIED_ATTN = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    CPU_ATTN = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    CUSTOM = None

    def get_path(self, include_classname: bool = True) -> str:
        """Get the class path for this backend (respects overrides).

        Returns:
            The fully qualified class path string

        Raises:
            ValueError: If Backend.CUSTOM is used without being registered
        """
        path = _ATTN_OVERRIDES.get(self, self.value)
        if not path:
            raise ValueError(
                f"Backend {self.name} must be registered before use. "
                f"Use register_backend(Backend.{self.name}, 'your.module.YourClass')"
            )
        if not include_classname:
            path = path.rsplit(".", 1)[0]
        return path

    def get_class(self) -> "type[AttentionBackend]":
        """Get the backend class (respects overrides).

        Returns:
            The backend class

        Raises:
            ImportError: If the backend class cannot be imported
            ValueError: If Backend.CUSTOM is used without being registered
        """
        return resolve_obj_by_qualname(self.get_path())

    def is_overridden(self) -> bool:
        """Check if this backend has been overridden.

        Returns:
            True if the backend has a registered override
        """
        return self in _ATTN_OVERRIDES

    def clear_override(self) -> None:
        """Clear any override for this backend, reverting to the default."""
        _ATTN_OVERRIDES.pop(self, None)


class MambaAttentionBackendEnum(Enum, metaclass=_AttentionBackendEnumMeta):
    """Enumeration of mamba attention backends.

    mini-vLLM: Mamba backends removed - stubs only.
    """

    # mini-vLLM: Mamba backends removed, kept as stubs
    MAMBA1 = ""
    MAMBA2 = ""
    SHORT_CONV = ""
    LINEAR = ""
    GDN_ATTN = ""
    CUSTOM = None

    def get_path(self, include_classname: bool = True) -> str:
        """Get the class path for this backend (respects overrides).

        Returns:
            The fully qualified class path string

        Raises:
            ValueError: If Backend.CUSTOM is used without being registered
        """
        path = _MAMBA_ATTN_OVERRIDES.get(self, self.value)
        if not path:
            raise ValueError(
                f"Backend {self.name} must be registered before use. "
                f"Use register_backend(Backend.{self.name}, 'your.module.YourClass')"
            )
        if not include_classname:
            path = path.rsplit(".", 1)[0]
        return path

    def get_class(self) -> "type[AttentionBackend]":
        """Get the backend class (respects overrides).

        Returns:
            The backend class

        Raises:
            ImportError: If the backend class cannot be imported
            ValueError: If Backend.CUSTOM is used without being registered
        """
        return resolve_obj_by_qualname(self.get_path())

    def is_overridden(self) -> bool:
        """Check if this backend has been overridden.

        Returns:
            True if the backend has a registered override
        """
        return self in _MAMBA_ATTN_OVERRIDES

    def clear_override(self) -> None:
        """Clear any override for this backend, reverting to the default."""
        _MAMBA_ATTN_OVERRIDES.pop(self, None)


MAMBA_TYPE_TO_BACKEND_MAP = {
    "mamba1": MambaAttentionBackendEnum.MAMBA1.name,
    "mamba2": MambaAttentionBackendEnum.MAMBA2.name,
    "short_conv": MambaAttentionBackendEnum.SHORT_CONV.name,
    "linear_attention": MambaAttentionBackendEnum.LINEAR.name,
    "gdn_attention": MambaAttentionBackendEnum.GDN_ATTN.name,
    "custom": MambaAttentionBackendEnum.CUSTOM.name,
}


_ATTN_OVERRIDES: dict[AttentionBackendEnum, str] = {}
_MAMBA_ATTN_OVERRIDES: dict[MambaAttentionBackendEnum, str] = {}


def register_backend(
    backend: AttentionBackendEnum | MambaAttentionBackendEnum,
    class_path: str | None = None,
    is_mamba: bool = False,
) -> Callable[[type], type]:
    """Register or override a backend implementation.

    Args:
        backend: The AttentionBackendEnum member to register
        class_path: Optional class path. If not provided and used as
            decorator, will be auto-generated from the class.

    Returns:
        Decorator function if class_path is None, otherwise a no-op

    Examples:
        # Override an existing attention backend
        @register_backend(AttentionBackendEnum.FLASH_ATTN)
        class MyCustomFlashAttn:
            ...

        # Override an existing mamba attention backend
        @register_backend(MambaAttentionBackendEnum.LINEAR, is_mamba=True)
        class MyCustomMambaAttn:
            ...

        # Register a custom third-party attention backend
        @register_backend(AttentionBackendEnum.CUSTOM)
        class MyCustomBackend:
            ...

        # Direct registration
        register_backend(
            AttentionBackendEnum.CUSTOM,
            "my.module.MyCustomBackend"
        )
    """

    def decorator(cls: type) -> type:
        if is_mamba:
            _MAMBA_ATTN_OVERRIDES[backend] = f"{cls.__module__}.{cls.__qualname__}"  # type: ignore[index]
        else:
            _ATTN_OVERRIDES[backend] = f"{cls.__module__}.{cls.__qualname__}"  # type: ignore[index]
        return cls

    if class_path is not None:
        if is_mamba:
            _MAMBA_ATTN_OVERRIDES[backend] = class_path  # type: ignore[index]
        else:
            _ATTN_OVERRIDES[backend] = class_path  # type: ignore[index]
        return lambda x: x

    return decorator

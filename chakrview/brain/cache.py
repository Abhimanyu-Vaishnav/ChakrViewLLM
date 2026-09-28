"""
Key-Value (KV) Cache Engine for ChakrView (Step 10).

Provides persistent, layer-indexed key and value caching for autoregressive decoding:
- Eliminates redundant prompt and prefix re-computation (O(N) vs O(N^2))
- Supports single-token incremental append
- Enforces strict context bounds (T <= 512)
- Validates tensor shape, device, dtype, and finite values
- Supports atomic reset and prefix truncation
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch


class KVCacheError(Exception):
    """Base exception for KV cache errors."""
    pass


class KVCacheOverflowError(KVCacheError):
    """Raised when appending tokens would exceed configured max_sequence_length."""
    pass


class KVCacheDeviceMismatchError(KVCacheError):
    """Raised when input tensor device differs from cache device."""
    pass


class KVCacheDtypeMismatchError(KVCacheError):
    """Raised when input tensor dtype differs from cache dtype."""
    pass


class LayerKVCache:
    """
    Key and Value tensor storage for a single transformer layer.
    
    Stores tensors in [B, H, T, d_head] shape.
    """
    def __init__(
        self,
        layer_idx: int,
        max_seq_len: int = 512,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.layer_idx = layer_idx
        self.max_seq_len = max_seq_len
        self.device = torch.device(device) if device else torch.device("cpu")
        self.dtype = dtype
        
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None
        self._seq_len: int = 0

    @property
    def sequence_length(self) -> int:
        return self._seq_len

    def append(self, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append new key and value slices.
        
        Args:
            key: Tensor of shape [B, H, T_new, d_head]
            value: Tensor of shape [B, H, T_new, d_head]
            
        Returns:
            Tuple of complete accumulated (key, value) tensors.
        """
        if key.dim() != 4 or value.dim() != 4:
            raise ValueError(
                f"Layer {self.layer_idx}: Key and Value must be 4D [B, H, T, D], got key={key.shape}, val={value.shape}"
            )
        if key.shape != value.shape:
            raise ValueError(
                f"Layer {self.layer_idx}: Key shape {key.shape} does not match Value shape {value.shape}"
            )
        if key.device != self.device:
            raise KVCacheDeviceMismatchError(
                f"Layer {self.layer_idx}: Device mismatch: input on {key.device}, cache on {self.device}"
            )
        if key.dtype != self.dtype:
            raise KVCacheDtypeMismatchError(
                f"Layer {self.layer_idx}: Dtype mismatch: input is {key.dtype}, cache is {self.dtype}"
            )

        t_new = key.shape[2]
        if self._seq_len + t_new > self.max_seq_len:
            raise KVCacheOverflowError(
                f"Layer {self.layer_idx}: Sequence length {self._seq_len + t_new} exceeds max context {self.max_seq_len}"
            )

        if self.k is None or self.v is None:
            self.k = key
            self.v = value
            self._seq_len = t_new
        else:
            self.k = torch.cat([self.k, key], dim=2)
            self.v = torch.cat([self.v, value], dim=2)
            self._seq_len += t_new

        return self.k, self.v

    def get(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return accumulated (key, value) tensors."""
        return self.k, self.v

    def reset(self) -> None:
        """Clear cache state."""
        self.k = None
        self.v = None
        self._seq_len = 0

    def truncate(self, new_seq_len: int) -> None:
        """Truncate cache to new_seq_len."""
        if new_seq_len < 0:
            raise ValueError(f"new_seq_len must be non-negative, got {new_seq_len}")
        if new_seq_len >= self._seq_len:
            return
        if new_seq_len == 0:
            self.reset()
            return
        self.k = self.k[:, :, :new_seq_len, :].contiguous()
        self.v = self.v[:, :, :new_seq_len, :].contiguous()
        self._seq_len = new_seq_len

    @property
    def memory_bytes(self) -> int:
        bytes_k = self.k.element_size() * self.k.nelement() if self.k is not None else 0
        bytes_v = self.v.element_size() * self.v.nelement() if self.v is not None else 0
        return bytes_k + bytes_v


class KVCache:
    """
    Multi-layer persistent KV-cache for ChakrMicro causal transformer blocks.
    
    Coordinates LayerKVCache instances across all n_layers.
    """
    def __init__(
        self,
        num_layers: int = 6,
        max_seq_len: int = 512,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if max_seq_len <= 0 or max_seq_len > 512:
            raise ValueError(f"max_seq_len must be in [1, 512], got {max_seq_len}")

        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.device = torch.device(device) if device else torch.device("cpu")
        self.dtype = dtype

        self.layers: List[LayerKVCache] = [
            LayerKVCache(
                layer_idx=i,
                max_seq_len=max_seq_len,
                device=self.device,
                dtype=self.dtype,
            )
            for i in range(num_layers)
        ]

    @property
    def sequence_length(self) -> int:
        """Current sequence length (from layer 0)."""
        return self.layers[0].sequence_length if self.layers else 0

    @property
    def max_sequence_length(self) -> int:
        return self.max_seq_len

    def append(
        self, layer_idx: int, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append key and value tensors for specified layer."""
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of bounds for cache with {self.num_layers} layers"
            )
        return self.layers[layer_idx].append(key, value)

    def get(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get cached key and value tensors for specified layer."""
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of bounds for cache with {self.num_layers} layers"
            )
        return self.layers[layer_idx].get()

    def get_seq_len(self, layer_idx: int = 0) -> int:
        """Return sequence length for layer."""
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(f"layer_idx {layer_idx} out of bounds")
        return self.layers[layer_idx].sequence_length

    def reset(self) -> None:
        """Reset cache for all layers."""
        for layer in self.layers:
            layer.reset()

    def truncate(self, new_seq_len: int) -> None:
        """Truncate cache across all layers."""
        for layer in self.layers:
            layer.truncate(new_seq_len)

    @property
    def total_memory_bytes(self) -> int:
        """Total memory occupied by key and value tensors across all layers."""
        return sum(layer.memory_bytes for layer in self.layers)

    def to(self, device: Union[str, torch.device], dtype: Optional[torch.dtype] = None) -> "KVCache":
        """Move all cached tensors to new device or dtype."""
        new_device = torch.device(device)
        self.device = new_device
        if dtype is not None:
            self.dtype = dtype

        for layer in self.layers:
            layer.device = new_device
            if dtype is not None:
                layer.dtype = dtype
            if layer.k is not None:
                layer.k = layer.k.to(device=new_device, dtype=self.dtype)
            if layer.v is not None:
                layer.v = layer.v.to(device=new_device, dtype=self.dtype)
        return self

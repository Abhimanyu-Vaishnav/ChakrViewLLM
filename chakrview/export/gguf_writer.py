"""
chakrview/export/gguf_writer.py

Clean, dependency-free binary GGUF v3 writer for ChakrView (ChakrMicro).
Converts PyTorch state dictionaries into compliant GGUF format according to
the official GGUF specification (Little-Endian, 32-byte alignment).
"""

import struct
import io
from pathlib import Path
from typing import Dict, Any, List, Union
import numpy as np
import torch

# GGUF Value Types (GGUF v3 specification)
GGUF_TYPE_UINT8   = 0
GGUF_TYPE_INT8    = 1
GGUF_TYPE_UINT16  = 2
GGUF_TYPE_INT16   = 3
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9
GGUF_TYPE_UINT64  = 10
GGUF_TYPE_INT64   = 11
GGUF_TYPE_FLOAT64 = 12

# GGML Tensor Types
GGML_TYPE_F32  = 0
GGML_TYPE_F16  = 1
GGML_TYPE_Q4_0 = 2


class GGUFWriter:
    """
    Standard GGUF v3 binary writer writing directly to a binary stream.
    Adheres strictly to the GGUF specification with 32-byte tensor alignment.
    """

    def __init__(self, f, alignment: int = 32):
        self.f = f
        self.alignment = alignment
        self.kv_pairs: List[tuple] = []
        self.tensors: List[tuple] = []

    def add_uint32(self, key: str, val: int):
        self.kv_pairs.append((key, GGUF_TYPE_UINT32, val))

    def add_int32(self, key: str, val: int):
        self.kv_pairs.append((key, GGUF_TYPE_INT32, val))

    def add_float32(self, key: str, val: float):
        self.kv_pairs.append((key, GGUF_TYPE_FLOAT32, val))

    def add_string(self, key: str, val: str):
        self.kv_pairs.append((key, GGUF_TYPE_STRING, val))

    def add_bool(self, key: str, val: bool):
        self.kv_pairs.append((key, GGUF_TYPE_BOOL, val))

    def add_array_string(self, key: str, val: List[str]):
        self.kv_pairs.append((key, GGUF_TYPE_ARRAY, (GGUF_TYPE_STRING, val)))

    def add_tensor(self, name: str, tensor: torch.Tensor, raw_dtype=GGML_TYPE_F32):
        # Flatten tensor data as float32 bytes
        arr = tensor.detach().cpu().to(torch.float32).numpy()
        self.tensors.append((name, arr, raw_dtype))

    def _write_string(self, s: str):
        encoded = s.encode("utf-8")
        self.f.write(struct.pack("<Q", len(encoded)))
        self.f.write(encoded)

    def _write_val(self, vtype: int, val: Any):
        if vtype == GGUF_TYPE_UINT32:
            self.f.write(struct.pack("<I", val))
        elif vtype == GGUF_TYPE_INT32:
            self.f.write(struct.pack("<i", val))
        elif vtype == GGUF_TYPE_FLOAT32:
            self.f.write(struct.pack("<f", val))
        elif vtype == GGUF_TYPE_BOOL:
            self.f.write(struct.pack("<?", val))
        elif vtype == GGUF_TYPE_STRING:
            self._write_string(val)
        elif vtype == GGUF_TYPE_ARRAY:
            subtype, items = val
            self.f.write(struct.pack("<I", subtype))
            self.f.write(struct.pack("<Q", len(items)))
            for item in items:
                self._write_val(subtype, item)
        else:
            raise NotImplementedError(f"GGUF type {vtype} not implemented")

    def write_header_and_data(self):
        # 1. Magic bytes & Version
        self.f.write(b"GGUF")
        self.f.write(struct.pack("<I", 3))  # version 3
        self.f.write(struct.pack("<Q", len(self.tensors)))
        self.f.write(struct.pack("<Q", len(self.kv_pairs)))

        # 2. Key-Value metadata
        for key, vtype, val in self.kv_pairs:
            self._write_string(key)
            self.f.write(struct.pack("<I", vtype))
            self._write_val(vtype, val)

        # 3. Tensor infos: name, n_dims, shape, type, offset
        # Precompute offsets
        tensor_data_offsets = []
        curr_offset = 0
        for name, arr, raw_dtype in self.tensors:
            # Align offset to self.alignment
            padding = (self.alignment - (curr_offset % self.alignment)) % self.alignment
            curr_offset += padding
            tensor_data_offsets.append(curr_offset)
            curr_offset += arr.nbytes

        for idx, (name, arr, raw_dtype) in enumerate(self.tensors):
            self._write_string(name)
            shape = list(arr.shape)[::-1]  # GGUF stores dimensions in reverse order
            self.f.write(struct.pack("<I", len(shape)))
            for dim in shape:
                self.f.write(struct.pack("<Q", dim))
            self.f.write(struct.pack("<I", raw_dtype))
            self.f.write(struct.pack("<Q", tensor_data_offsets[idx]))

        # 4. Align start of tensor data binary block
        curr_pos = self.f.tell()
        padding = (self.alignment - (curr_pos % self.alignment)) % self.alignment
        if padding > 0:
            self.f.write(b"\x00" * padding)

        # 5. Write raw tensor data
        base_data_pos = self.f.tell()
        for idx, (name, arr, raw_dtype) in enumerate(self.tensors):
            target_pos = base_data_pos + tensor_data_offsets[idx]
            pad_needed = target_pos - self.f.tell()
            if pad_needed > 0:
                self.f.write(b"\x00" * pad_needed)
            self.f.write(arr.tobytes())

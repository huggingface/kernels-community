"""The part of qwix (google/qwix) that Tokamax uses, without flax."""

from .qarray import QArray, dequantize, quantize_api as quantize

__all__ = ["QArray", "dequantize", "quantize"]

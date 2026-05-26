"""
Model implementations for GenRec.

This module provides implementations of various recommendation models:

Baseline Models:
    - SASRec: Self-attention with softmax for sequential recommendation
    - HSTU: SiLU attention + update gate + temporal bias

Generative Models:
    - RQVAE: Vector quantized VAE for semantic ID generation
    - TIGER: Generative retrieval with trie-based constrained decoding
    - LCRec: LLM with collaborative semantics via codebook tokens
    - COBRA: Sparse/Dense hybrid with cascaded representations
    - NoteLLM: Qwen2-based LLM for note recommendation
"""

from importlib import import_module


_LAZY_IMPORTS = {
    "RqVae": "genrec.models.rqvae",
    "QuantizeForwardMode": "genrec.models.rqvae",
    "Tiger": "genrec.models.tiger",
    "SASRec": "genrec.models.sasrec",
    "HSTU": "genrec.models.hstu",
    "LCRec": "genrec.models.lcrec",
    "Cobra": "genrec.models.cobra",
}


def __getattr__(name):
    """Lazily import model symbols to avoid importing optional dependencies."""
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'genrec.models' has no attribute '{name}'")

    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value

__all__ = [
    "RqVae",
    "QuantizeForwardMode",
    "Tiger",
    "SASRec",
    "HSTU",
    "LCRec",
    "Cobra",
]

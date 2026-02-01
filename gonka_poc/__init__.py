"""Gonka PoC module for vLLM V1 engine.

This module provides PoC V2 support for vLLM 0.14.0+ (V1 engine).
Ported from vLLM 0.9.1 (V0 engine) to work with V1's collective_rpc.

Key changes from V0 to V1:
- V1 requires input_ids as positional argument
- Uses EmbeddingInjectionHook to inject custom embeddings
- Compatible with vLLM 0.14.0+ V1 engine

Supports both GPU architectures:
- Blackwell B300/B200 (SM 10.x)
- Hopper H100/H200 (SM 9.0)
- Ampere A100 (SM 8.0)
"""

from vllm.gonka_poc.config import PoCState, PoCConfig
from vllm.gonka_poc.manager import PoCManagerV1

__all__ = ["PoCState", "PoCConfig", "PoCManagerV1"]

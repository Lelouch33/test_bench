"""PoC Manager for vLLM V1 engine.

This is a minimal, stateless manager that uses V1's collective_rpc.
All state (generation loop, nonce counter, stats) is managed in the API layer.
"""
from typing import TYPE_CHECKING

from vllm.gonka_poc.data import Artifact, encode_vector

if TYPE_CHECKING:
    from vllm.v1.engine.client import EngineClient


class PoCManagerV1:
    """Manages PoC artifact generation for V1 engine (stateless)."""

    def __init__(
        self,
        engine_client: "EngineClient",  # V1 EngineClient
        vllm_config,
    ):
        self.engine_client = engine_client
        self.vllm_config = vllm_config

    @property
    def model_config(self):
        """Get model_config from vllm_config."""
        if self.vllm_config:
            return self.vllm_config.model_config
        return None

    @property
    def hidden_size(self):
        """Get hidden_size from model_config."""
        config = self.model_config
        if config:
            # Try different methods for vLLM 0.14.0 compatibility
            if hasattr(config, 'get_hidden_size'):
                return config.get_hidden_size()
            if hasattr(config, 'hidden_size'):
                return config.hidden_size
            if hasattr(config, 'huggingface_config'):
                hf_config = config.huggingface_config
                if hasattr(hf_config, 'hidden_size'):
                    return hf_config.hidden_size
        # Default fallback for common models
        return 5120

    async def generate_artifacts(
        self,
        nonces: list[int],
        block_hash: str,
        public_key: str,
        seq_len: int,
        k_dim: int,
    ) -> list[Artifact]:
        """Generate artifacts for specific nonces.

        This is the only public API. The caller provides nonces explicitly;
        nonce progression logic lives in the API layer.

        Uses V1's collective_rpc to execute on all workers.
        NOTE: collective_rpc in AsyncLLM returns a coroutine that must be awaited.
        """
        from vllm.gonka_poc.poc_model_runner import execute_poc_forward_v1

        # Get hidden_size
        hidden_size = self.hidden_size if self.hidden_size else 5120

        # V1's collective_rpc signature (AsyncLLM returns coroutine):
        # collective_rpc(method, args=(), kwargs=None, timeout=None, non_block=False)
        results = await self.engine_client.collective_rpc(
            execute_poc_forward_v1,
            args=(
                block_hash,
                public_key,
                nonces,
                seq_len,
                hidden_size,
                k_dim,
            ),
        )

        # Only the last PP rank returns a result
        result = next((r for r in results if r is not None), None)

        if result is None:
            return []

        vectors = result["vectors"]  # FP16 numpy array
        artifacts = []
        for i, nonce in enumerate(result["nonces"]):
            vector_b64 = encode_vector(vectors[i])
            artifacts.append(Artifact(nonce=nonce, vector_b64=vector_b64))

        return artifacts

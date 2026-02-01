"""PoC model runner for vLLM V1 engine.

CRITICAL FIX: V1 requires input_ids as positional argument.
V0 allowed: model(input_ids=None, inputs_embeds=...)
V1 requires: model(input_ids, positions, ...) where input_ids is mandatory

Solution: Use EmbeddingInjectionHook to intercept embed_tokens and inject
our custom embeddings while still passing valid input_ids to satisfy V1.
"""
import torch
import torch.distributed as dist
from typing import List, Optional, Dict, Any

# V1 imports - different paths from V0
try:
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID
except ImportError:
    try:
        from vllm.attention.backends.utils import PAD_SLOT_ID
    except ImportError:
        PAD_SLOT_ID = -1

from vllm.distributed import get_pp_group, get_tp_group
from vllm.distributed.communication_op import broadcast_tensor_dict
from vllm.forward_context import set_forward_context
from vllm.sequence import IntermediateTensors

from .gpu_random import (
    generate_inputs,
    random_pick_indices,
    apply_haar_rotation,
)
from .layer_hooks import LayerHouseholderHook, poc_forward_context

DEFAULT_K_DIM = 12


class EmbeddingInjectionHook:
    """Hook to inject custom embeddings into V1 model.
    
    V1 models require input_ids as positional argument and don't accept
    inputs_embeds directly. This hook intercepts embed_tokens and replaces
    its output with our generated embeddings.
    
    Usage:
        hook = EmbeddingInjectionHook(model, custom_embeddings)
        output = model(dummy_input_ids, positions, ...)  # Will use custom_embeddings
        hook.remove()
    """
    
    def __init__(self, model: torch.nn.Module, embeddings: torch.Tensor):
        self.embeddings = embeddings
        self.handle = None
        self._install(model)
    
    def _find_embed_tokens(self, model: torch.nn.Module) -> Optional[torch.nn.Module]:
        """Find embedding layer in model."""
        # Try common patterns
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            return model.model.embed_tokens
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
            return model.transformer.wte
        if hasattr(model, 'embed_tokens'):
            return model.embed_tokens
        
        # Search recursively for nn.Embedding
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Embedding) and 'embed' in name.lower():
                return module
        return None
    
    def _install(self, model: torch.nn.Module):
        """Install forward hook on embedding layer."""
        embed_layer = self._find_embed_tokens(model)
        if embed_layer is None:
            raise RuntimeError("Could not find embedding layer in model")
        
        def hook_fn(module, input, output):
            # Return our custom embeddings instead of model's embeddings
            return self.embeddings
        
        self.handle = embed_layer.register_forward_hook(hook_fn)
    
    def remove(self):
        """Remove the hook."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def _get_model_and_device(worker) -> tuple:
    """Get model, device, and model_runner from V1 worker."""
    # V1 worker structure
    model_runner = getattr(worker, 'model_runner', None)
    if model_runner is None:
        raise RuntimeError("Worker missing model_runner attribute")
    
    model = getattr(model_runner, 'model', None)
    if model is None:
        raise RuntimeError("ModelRunner missing model attribute")
    
    device = getattr(model_runner, 'device', None)
    if device is None:
        device = getattr(worker, 'device', None)
    if device is None:
        # Fallback: get device from model parameters
        device = next(model.parameters()).device
    
    return model, device, model_runner


def _ensure_layer_hooks(worker, block_hash: str, hidden_size: int) -> None:
    """Ensure layer hooks are installed for the given block_hash."""
    model, device, _ = _get_model_and_device(worker)
    
    existing_hook = getattr(worker, '_poc_layer_hooks', None)
    
    if existing_hook is not None:
        if existing_hook.block_hash == block_hash:
            return
        existing_hook.detach()
    
    hook = LayerHouseholderHook(model, block_hash, device, hidden_size)
    hook._setup(model, block_hash, device, hidden_size)
    worker._poc_layer_hooks = hook


def _create_prefill_attn_metadata(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    attn_backend,
):
    """Create prefill attention metadata for V1 backends."""
    num_tokens = batch_size * seq_len
    seq_lens = [seq_len] * batch_size
    
    seq_start_loc = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    seq_start_loc[1:] = torch.cumsum(
        torch.tensor(seq_lens, dtype=torch.int32, device=device), dim=0
    )
    
    # Get backend name
    if hasattr(attn_backend, 'get_name'):
        backend_name = attn_backend.get_name()
    elif hasattr(attn_backend, '__class__'):
        backend_name = attn_backend.__class__.__name__
    else:
        backend_name = str(type(attn_backend))
    
    slot_mapping = torch.full((num_tokens,), PAD_SLOT_ID, dtype=torch.long, device=device)
    
    # Try V1-style metadata first, fall back to V0-style
    if "FLASHINFER" in backend_name.upper() or "FlashInfer" in backend_name:
        try:
            from vllm.v1.attention.backends.flashinfer import FlashInferMetadata
            return FlashInferMetadata(
                num_actual_tokens=num_tokens,
                max_seq_len=seq_len,
                seq_start_loc=seq_start_loc,
                block_table=torch.empty((batch_size, 0), dtype=torch.int32, device=device),
                slot_mapping=slot_mapping,
            )
        except (ImportError, TypeError):
            pass
        try:
            from vllm.attention.backends.flashinfer import FlashInferMetadata
            return FlashInferMetadata(
                num_prefills=batch_size,
                num_prefill_tokens=num_tokens,
                num_decode_tokens=0,
                slot_mapping=slot_mapping,
                max_prefill_seq_len=seq_len,
                seq_start_loc=seq_start_loc,
                use_cuda_graph=False,
            )
        except (ImportError, TypeError):
            pass
    
    if "XFORMERS" in backend_name.upper() or "XFormers" in backend_name:
        try:
            from vllm.attention.backends.xformers import XFormersMetadata
            return XFormersMetadata(
                num_prefills=batch_size,
                num_prefill_tokens=num_tokens,
                num_decode_tokens=0,
                slot_mapping=slot_mapping,
                seq_lens=seq_lens,
                seq_lens_tensor=torch.tensor(seq_lens, dtype=torch.int, device=device),
                max_prefill_seq_len=seq_len,
                max_decode_seq_len=0,
                query_start_loc=seq_start_loc.clone(),
                seq_start_loc=seq_start_loc,
                context_lens_tensor=torch.zeros(batch_size, dtype=torch.int, device=device),
                block_tables=torch.empty((batch_size, 0), dtype=torch.int, device=device),
                use_cuda_graph=False,
            )
        except (ImportError, TypeError):
            pass
    
    # Default: FlashAttention
    try:
        from vllm.attention.backends.flash_attn import FlashAttentionMetadata
        return FlashAttentionMetadata(
            num_prefills=batch_size,
            num_prefill_tokens=num_tokens,
            num_decode_tokens=0,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
            seq_lens_tensor=torch.tensor(seq_lens, dtype=torch.int, device=device),
            max_prefill_seq_len=seq_len,
            max_decode_seq_len=0,
            query_start_loc=seq_start_loc.clone(),
            seq_start_loc=seq_start_loc,
            context_lens_tensor=torch.zeros(batch_size, dtype=torch.int, device=device),
            block_tables=torch.empty((batch_size, 0), dtype=torch.int, device=device),
            use_cuda_graph=False,
        )
    except (ImportError, TypeError):
        # Minimal fallback
        return None


@torch.inference_mode()
def execute_poc_forward_v1(
    worker,
    block_hash: str,
    public_key: str,
    nonces: List[int],
    seq_len: int,
    hidden_size: int,
    k_dim: int = DEFAULT_K_DIM,
) -> Optional[Dict[str, Any]]:
    """Execute PoC forward pass on a V1 worker.
    
    CRITICAL FIX: V1 requires input_ids as positional argument.
    We use EmbeddingInjectionHook to inject custom embeddings while
    still satisfying V1's input_ids requirement.
    """
    model, device, model_runner = _get_model_and_device(worker)
    
    # Get config and dtype
    model_config = getattr(model_runner, 'model_config', None)
    if model_config is None:
        model_config = getattr(worker, 'model_config', None)
    
    dtype = torch.float16
    if model_config is not None:
        dtype = getattr(model_config, 'dtype', torch.float16)
    
    # Get vllm_config
    vllm_config = getattr(worker, 'vllm_config', None)
    if vllm_config is None:
        vllm_config = getattr(model_runner, 'vllm_config', None)
    
    tp_group = get_tp_group()
    is_tp_driver = tp_group.rank_in_group == 0
    
    # =========================================================================
    # TP SYNC: Rendezvous + broadcast
    # =========================================================================
    if tp_group.world_size > 1:
        dist.barrier(group=tp_group.cpu_group)
        
        if is_tp_driver:
            broadcast_tensor_dict({
                "poc_go": True,
                "seq_len": seq_len,
                "hidden_size": hidden_size,
                "nonces": nonces,
                "k_dim": k_dim,
            }, src=0)
        else:
            broadcast_data = broadcast_tensor_dict(src=0)
            seq_len = int(broadcast_data["seq_len"])
            hidden_size = int(broadcast_data["hidden_size"])
            nonces = list(broadcast_data["nonces"])
            k_dim = int(broadcast_data["k_dim"])
    
    batch_size = len(nonces)
    num_tokens = batch_size * seq_len
    
    # Generate embeddings on first PP rank
    intermediate_tensors = None
    inputs_embeds = None
    
    pp_group = get_pp_group()
    
    if pp_group.is_first_rank:
        inputs_embeds = generate_inputs(
            block_hash, public_key, nonces,
            dim=hidden_size, seq_len=seq_len,
            device=device, dtype=dtype,
        )
        # Flatten for model input: [batch_size, seq_len, hidden] -> [num_tokens, hidden]
        inputs_embeds_flat = inputs_embeds.view(-1, hidden_size)
    else:
        intermediate_tensors = IntermediateTensors(
            pp_group.recv_tensor_dict(all_gather_group=get_tp_group())
        )
    
    # Create attention metadata and positions
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    positions_flat = positions.flatten()  # [num_tokens]
    
    # Get attention backend
    attn_backend = getattr(model_runner, 'attn_backend', None)
    attn_metadata = None
    if attn_backend is not None:
        attn_metadata = _create_prefill_attn_metadata(batch_size, seq_len, device, attn_backend)
    
    # =========================================================================
    # TP SYNC: Pre-forward rendezvous
    # =========================================================================
    if tp_group.world_size > 1:
        dist.barrier(group=tp_group.cpu_group)
    
    torch.cuda.synchronize()
    
    # Ensure layer hooks are installed
    _ensure_layer_hooks(worker, block_hash, hidden_size)
    
    # =========================================================================
    # FORWARD PASS with V1 compatibility fix
    # =========================================================================
    with set_forward_context(attn_metadata, vllm_config):
        with poc_forward_context():
            if pp_group.is_first_rank:
                # Create dummy input_ids (required by V1)
                dummy_input_ids = torch.zeros(num_tokens, dtype=torch.long, device=device)
                
                # Install embedding injection hook
                embed_hook = EmbeddingInjectionHook(model, inputs_embeds_flat)
                
                try:
                    # V1 forward: input_ids is positional, but embeddings come from hook
                    hidden_states = model(
                        dummy_input_ids,
                        positions_flat,
                        intermediate_tensors=intermediate_tensors,
                    )
                finally:
                    embed_hook.remove()
            else:
                # Non-first PP rank: no embedding injection needed
                hidden_states = model(
                    input_ids=None,
                    positions=positions_flat,
                    intermediate_tensors=intermediate_tensors,
                )
    
    # PP: send to next rank if not last
    if not pp_group.is_last_rank:
        if isinstance(hidden_states, IntermediateTensors):
            pp_group.send_tensor_dict(
                hidden_states.tensors, all_gather_group=get_tp_group()
            )
        return None
    
    # Extract last token hidden state
    hidden_states = hidden_states.view(batch_size, seq_len, -1)
    last_hidden = hidden_states[:, -1, :].float()
    
    # Normalize to unit sphere
    last_hidden = last_hidden / (last_hidden.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Per-nonce k-dim pick + Haar rotation
    indices = random_pick_indices(block_hash, public_key, nonces, hidden_size, k_dim, device)
    xk = torch.gather(last_hidden, 1, indices)
    yk = apply_haar_rotation(block_hash, public_key, nonces, xk, device)
    
    # Normalize output vectors
    yk = yk / (yk.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Convert to FP16 for encoding
    vectors_f16 = yk.half().cpu().numpy()
    
    return {
        "nonces": nonces,
        "vectors": vectors_f16,
    }


# Alias for compatibility
execute_poc_forward = execute_poc_forward_v1

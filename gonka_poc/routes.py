"""PoC API routes for vLLM V1 engine.

Unlike V0 which uses MQLLMEngine with ZMQ RPC, V1 uses direct calls
through the executor's collective_rpc mechanism.
"""
import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, ConfigDict

from vllm.logger import init_logger
from vllm.gonka_poc.config import PoCState
from vllm.gonka_poc.data import Artifact, DEFAULT_DIST_THRESHOLD, DEFAULT_P_MISMATCH, DEFAULT_FRAUD_THRESHOLD
from vllm.gonka_poc.validation import run_validation

logger = init_logger(__name__)

router = APIRouter(prefix="/api/v1/pow", tags=["PoC V1"])


# ---------------------------------------------------------------------------
# Suppress noisy uvicorn access logs for health/status endpoints
# ---------------------------------------------------------------------------
class _RateLimitHealthFilter(logging.Filter):
    """Rate-limit GET /health and GET /api/v1/pow/status to once per minute."""
    _QUIET_PATHS = ("GET /health", "GET /api/v1/pow/status")
    _INTERVAL = 60  # seconds

    def __init__(self):
        super().__init__()
        self._last_logged: dict[str, float] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for path in self._QUIET_PATHS:
            if path in msg:
                now = time.monotonic()
                last = self._last_logged.get(path, 0.0)
                if now - last >= self._INTERVAL:
                    self._last_logged[path] = now
                    return True
                return False
        return True


for _name in ("uvicorn.access", "uvicorn"):
    _logger = logging.getLogger(_name)
    _logger.addFilter(_RateLimitHealthFilter())

POC_GENERATE_CHUNK_TIMEOUT_SEC = float(os.environ.get("POC_GENERATE_CHUNK_TIMEOUT_SEC", "60"))
POC_CHAT_BUSY_BACKOFF_SEC = 0.05
POC_BATCH_SIZE_DEFAULT = int(os.environ.get("POC_BATCH_SIZE_DEFAULT", "32"))

# Track state per-app (FastAPI app instance)
_poc_tasks: Dict[int, Dict[str, Any]] = {}


# =============================================================================
# Request/Response Models
# =============================================================================

class PoCParamsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    seq_len: int
    k_dim: int = 12


class PoCInitGenerateRequest(BaseModel):
    block_hash: str
    block_height: int
    public_key: str
    node_id: int
    node_count: int
    group_id: int = 0
    n_groups: int = 1
    batch_size: int = POC_BATCH_SIZE_DEFAULT
    params: PoCParamsModel
    url: Optional[str] = None


@dataclass
class NonceIterator:
    """Iterator for nonces with multi-node and multi-group support."""
    node_id: int
    n_nodes: int
    group_id: int
    n_groups: int
    _current_x: int = 0

    def __iter__(self):
        return self

    def __next__(self) -> int:
        offset = self.node_id + self.group_id * self.n_nodes
        step = self.n_groups * self.n_nodes
        value = offset + self._current_x * step
        self._current_x += 1
        return value

    def take(self, n: int) -> List[int]:
        """Take the next n nonces."""
        return [next(self) for _ in range(n)]


class ArtifactModel(BaseModel):
    nonce: int
    vector_b64: str


class ValidationModel(BaseModel):
    artifacts: List[ArtifactModel]


class StatTestModel(BaseModel):
    dist_threshold: float = DEFAULT_DIST_THRESHOLD
    p_mismatch: float = DEFAULT_P_MISMATCH
    fraud_threshold: float = DEFAULT_FRAUD_THRESHOLD


class PoCGenerateRequest(BaseModel):
    block_hash: str
    block_height: int
    public_key: str
    node_id: int
    node_count: int
    nonces: List[int]
    params: PoCParamsModel
    batch_size: int = POC_BATCH_SIZE_DEFAULT
    wait: bool = False
    url: Optional[str] = None
    validation: Optional[ValidationModel] = None
    stat_test: Optional[StatTestModel] = None


# =============================================================================
# Helpers
# =============================================================================

def check_params_match(request: Request, params: PoCParamsModel):
    """Check params match deployed config. Raises 409 on mismatch."""
    serving_models = getattr(request.app.state, 'openai_serving_models', None)
    if serving_models and hasattr(serving_models, 'base_model_paths'):
        base_paths = serving_models.base_model_paths
        if base_paths:
            model_path = base_paths[0].model_path
            served_names = [p.name for p in base_paths]
            valid_models = {model_path} | set(served_names)
            if params.model not in valid_models:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "params mismatch",
                        "requested": {"model": params.model, "seq_len": params.seq_len, "k_dim": params.k_dim},
                        "deployed": {"model": list(valid_models), "seq_len": None, "k_dim": None},
                    }
                )

    deployed = getattr(request.app.state, 'poc_deployed', None)
    if deployed:
        mismatches = []
        if deployed.get("model") and params.model != deployed["model"]:
            mismatches.append("model")
        if deployed.get("seq_len") and params.seq_len != deployed["seq_len"]:
            mismatches.append("seq_len")
        if deployed.get("k_dim") and params.k_dim != deployed["k_dim"]:
            mismatches.append("k_dim")

        if mismatches:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "params mismatch",
                    "fields": mismatches,
                    "requested": {"model": params.model, "seq_len": params.seq_len, "k_dim": params.k_dim},
                    "deployed": deployed,
                }
            )


def get_poc_manager(request: Request):
    """Get or create PoCManager for the V1 engine."""
    # In V1, we use engine_client instead of engine
    engine_client = getattr(request.app.state, 'engine_client', None)
    if engine_client is None:
        raise HTTPException(status_code=503, detail="EngineClient not available")

    # Get vllm_config from app.state
    vllm_config = getattr(request.app.state, 'vllm_config', None)

    # Create PoCManager on first use (cached per-app)
    app_id = id(request.app)

    # Initialize app entry if not exists
    if app_id not in _poc_tasks:
        _poc_tasks[app_id] = {}

    if '_poc_manager' not in _poc_tasks[app_id]:
        from vllm.gonka_poc.manager import PoCManagerV1
        _poc_tasks[app_id]['_poc_manager'] = PoCManagerV1(
            engine_client=engine_client,
            vllm_config=vllm_config,
        )

    return _poc_tasks[app_id]['_poc_manager']


def _is_generation_active(app_id: int) -> bool:
    tasks = _poc_tasks.get(app_id)
    if not tasks:
        return False
    gen_task = tasks.get("gen_task")
    return gen_task is not None and not gen_task.done()


def _get_api_status(app_id: int) -> dict:
    tasks = _poc_tasks.get(app_id)

    if not tasks or not _is_generation_active(app_id):
        return {"status": PoCState.IDLE.value, "config": None, "stats": None}

    config = tasks.get("config", {})
    stats = tasks.get("stats", {})
    start_time = stats.get("start_time", 0)
    total_processed = stats.get("total_processed", 0)
    elapsed = time.time() - start_time if start_time > 0 else 0
    nonces_per_second = total_processed / elapsed if elapsed > 0 else 0

    return {
        "status": PoCState.GENERATING.value,
        "config": {
            "block_hash": config.get("block_hash"),
            "block_height": config.get("block_height"),
            "public_key": config.get("public_key"),
            "node_id": config.get("node_id"),
            "node_count": config.get("node_count"),
            "group_id": config.get("group_id"),
            "n_groups": config.get("n_groups"),
            "seq_len": config.get("seq_len"),
            "k_dim": config.get("k_dim"),
        },
        "stats": {
            "total_processed": total_processed,
            "nonces_per_second": nonces_per_second,
        },
    }


async def _cancel_poc_tasks(app_id: int):
    tasks = _poc_tasks.pop(app_id, None)
    if tasks:
        if tasks.get("stop_event"):
            tasks["stop_event"].set()
        if tasks.get("gen_task"):
            tasks["gen_task"].cancel()
            try:
                await tasks["gen_task"]
            except asyncio.CancelledError:
                pass


async def _compute_artifacts_chunk(
    poc_manager,
    nonces: List[int],
    block_hash: str,
    public_key: str,
    seq_len: int,
    k_dim: int,
    timeout_sec: float = POC_GENERATE_CHUNK_TIMEOUT_SEC,
) -> List[Dict]:
    """Compute artifacts for a chunk directly via PoCManager."""
    # V1 doesn't have the "skipped" mechanism like V0
    # We just call generate_artifacts directly (async)
    artifacts = await poc_manager.generate_artifacts(
        nonces=nonces,
        block_hash=block_hash,
        public_key=public_key,
        seq_len=seq_len,
        k_dim=k_dim,
    )

    return [{"nonce": a.nonce, "vector_b64": a.vector_b64} for a in artifacts]


# =============================================================================
# Generation Loop
# =============================================================================

async def _generation_loop(
    poc_manager,
    stop_event: asyncio.Event,
    config: dict,
    stats: dict,
):
    nonce_iter = NonceIterator(
        node_id=config["node_id"],
        n_nodes=config["node_count"],
        group_id=config["group_id"],
        n_groups=config["n_groups"],
    )
    batch_size = config["batch_size"]

    start_time = time.time()
    stats["start_time"] = start_time
    stats["total_processed"] = 0
    last_report_time = start_time

    logger.info(f"PoC V1 generation started (node {config['node_id']}/{config['node_count']}, group {config['group_id']}/{config['n_groups']})")

    try:
        while not stop_event.is_set():
            nonces = nonce_iter.take(batch_size)

            try:
                artifacts = await poc_manager.generate_artifacts(
                    nonces=nonces,
                    block_hash=config["block_hash"],
                    public_key=config["public_key"],
                    seq_len=config["seq_len"],
                    k_dim=config["k_dim"],
                )
            except Exception as e:
                logger.error(f"PoC V1 generation error: {e}")
                await asyncio.sleep(POC_CHAT_BUSY_BACKOFF_SEC)
                continue

            stats["total_processed"] += len(nonces)

            current_time = time.time()
            if current_time - last_report_time >= 5.0:
                elapsed_min = (current_time - start_time) / 60
                rate = stats["total_processed"] / elapsed_min if elapsed_min > 0 else 0
                logger.info(f"Generated: {stats['total_processed']} nonces ({rate:.0f}/min)")
                last_report_time = current_time

    except asyncio.CancelledError:
        elapsed_min = (time.time() - start_time) / 60
        logger.info(f"PoC V1 stopped: {stats['total_processed']} nonces in {elapsed_min:.2f}min")
    except Exception as e:
        logger.error(f"PoC V1 generation crashed: {e}", exc_info=True)
        raise


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/init/generate")
async def init_generate(request: Request, body: PoCInitGenerateRequest) -> dict:
    logger.info(f"PoC V1 /init/generate: {body.block_hash}, {body.block_height}, {body.public_key}, {body.node_id}, {body.node_count}, {body.group_id}, {body.n_groups}, {body.batch_size}, {body.params}, {body.url}")
    check_params_match(request, body.params)

    poc_manager = get_poc_manager(request)
    app_id = id(request.app)

    if _is_generation_active(app_id):
        raise HTTPException(status_code=409, detail="Already generating")

    await _cancel_poc_tasks(app_id)

    config = {
        "block_hash": body.block_hash,
        "block_height": body.block_height,
        "public_key": body.public_key,
        "node_id": body.node_id,
        "node_count": body.node_count,
        "group_id": body.group_id,
        "n_groups": body.n_groups,
        "batch_size": body.batch_size,
        "seq_len": body.params.seq_len,
        "k_dim": body.params.k_dim,
    }

    stats = {"start_time": 0, "total_processed": 0}
    stop_event = asyncio.Event()

    gen_task = asyncio.create_task(
        _generation_loop(poc_manager, stop_event, config, stats)
    )

    _poc_tasks[app_id] = {
        "gen_task": gen_task,
        "stop_event": stop_event,
        "config": config,
        "stats": stats,
    }

    return {"status": "OK", "pow_status": {"status": "GENERATING"}}


@router.post("/generate")
async def generate(request: Request, body: PoCGenerateRequest) -> dict:
    logger.info(f"PoC V1 /generate: {body.block_hash}, {body.block_height}, {body.public_key}, {body.node_id}, {body.node_count}, {body.nonces}, {body.params}, {body.batch_size}, {body.wait}, {body.validation}, {body.stat_test}")
    check_params_match(request, body.params)

    poc_manager = get_poc_manager(request)
    app_id = id(request.app)

    if body.validation:
        validation_nonces = set(a.nonce for a in body.validation.artifacts)
        if validation_nonces != set(body.nonces):
            raise HTTPException(status_code=400, detail="validation.artifacts nonces must match nonces field")

    validation_map = {a.nonce: a.vector_b64 for a in body.validation.artifacts} if body.validation else None
    stat_test = body.stat_test or StatTestModel()

    if not body.wait:
        # For non-wait mode, we just queue and return immediately
        # In V1, we don't have a separate queue system like V0
        # For simplicity, we just do synchronous generation
        pass

    total_nonces = len(body.nonces)
    n_chunks = (total_nonces + body.batch_size - 1) // body.batch_size
    logger.info(f"PoC V1 /generate: {total_nonces} nonces, batch_size={body.batch_size}, chunks={n_chunks}")

    start_time = time.time()
    computed_artifacts = []

    for i in range(0, total_nonces, body.batch_size):
        chunk = body.nonces[i:i + body.batch_size]
        chunk_idx = i // body.batch_size

        try:
            artifacts = await _compute_artifacts_chunk(
                poc_manager, chunk, body.block_hash, body.public_key,
                body.params.seq_len, body.params.k_dim,
                POC_GENERATE_CHUNK_TIMEOUT_SEC
            )
            computed_artifacts.extend(artifacts)
            logger.debug(f"PoC V1 /generate: chunk {chunk_idx+1}/{n_chunks} done ({len(chunk)} nonces)")
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

    elapsed = time.time() - start_time
    rate = total_nonces / elapsed if elapsed > 0 else 0
    logger.info(f"PoC V1 /generate completed: {total_nonces} nonces in {elapsed:.2f}s ({rate:.0f}/s)")

    if not body.validation:
        return {
            "status": "completed",
            "request_id": str(uuid.uuid4()),
            "artifacts": computed_artifacts,
            "encoding": {"dtype": "f16", "k_dim": body.params.k_dim, "endian": "le"},
        }

    validation_result = run_validation(
        computed_artifacts,
        validation_map,
        len(body.nonces),
        stat_test.dist_threshold,
        stat_test.p_mismatch,
        stat_test.fraud_threshold,
    )

    return {
        "status": "completed",
        "request_id": str(uuid.uuid4()),
        **validation_result,
    }


@router.get("/status")
async def get_status(request: Request) -> dict:
    return _get_api_status(id(request.app))


@router.post("/stop")
async def stop_round(request: Request) -> dict:
    app_id = id(request.app)

    await _cancel_poc_tasks(app_id)

    return {"status": "OK", "pow_status": {"status": "STOPPED"}}

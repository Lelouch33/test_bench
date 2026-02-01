#!/usr/bin/env python3
"""
Gonka PoC V2 Benchmark
Бенчмарк для тестирования производительности GPU через vLLM API (cPoC)

В отличие от V1, где вычисления производятся локально через модуль pow.compute,
V2 использует реальный vLLM сервер с gonka_poc модулем.
Бенчмарк отправляет запрос /api/v1/pow/init/generate и опрашивает /api/v1/pow/status
для подсчёта скорости генерации артефактов.

ФОРМУЛА РАСЧЁТА ВЕСА:
  poc_weight = total_nonces × WeightScaleFactor (0.262)
"""

import os
import sys
import time
import json
import argparse
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# ============ КОНФИГУРАЦИЯ ============
DEFAULT_DURATION_MIN = 5
WEIGHT_SCALE_FACTOR = 0.262
DEFAULT_VLLM_PORT = 5000
DEFAULT_BATCH_SIZE = 32
DEFAULT_SEQ_LEN = 1024
DEFAULT_K_DIM = 12
DEFAULT_MODEL_ID = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"

# Fallback значения
DEFAULT_R_TARGET = 1.398077

# Genesis ноды для получения параметров
GENESIS_NODES = [
    "http://node2.gonka.ai:8000",
    "http://node1.gonka.ai:8000",
    "http://node3.gonka.ai:8000",
    "https://node4.gonka.ai",
    "http://47.236.26.199:8000",
    "http://47.236.19.22:18000",
    "http://185.216.21.98:8000",
    "http://36.189.234.197:18026",
    "http://36.189.234.237:17241",
    "http://gonka.spv.re:8000",
]

POLL_INTERVAL = 5  # Секунды между опросами /status
REPORT_INTERVAL = 30  # Секунды между выводом прогресса


# ============ ЦВЕТА ДЛЯ ВЫВОДА ============
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def log_info(msg: str):
    print(f"{Colors.BLUE}[INFO]{Colors.END} {msg}")


def log_success(msg: str):
    print(f"{Colors.GREEN}[\u2713]{Colors.END} {msg}")


def log_warning(msg: str):
    print(f"{Colors.YELLOW}[\u26a0]{Colors.END} {msg}")


def log_error(msg: str):
    print(f"{Colors.RED}[\u2717]{Colors.END} {msg}")


# ============ HELPER: Парсинг значений с экспонентой ============
def parse_exp_value(obj, default=None):
    if obj is None:
        return default
    if isinstance(obj, (int, float)):
        return float(obj)
    if isinstance(obj, dict):
        value = obj.get("value")
        exponent = obj.get("exponent", 0)
        if value is not None:
            try:
                return float(value) * (10 ** int(exponent))
            except (ValueError, TypeError):
                return default
    if isinstance(obj, str):
        try:
            return float(obj)
        except ValueError:
            return default
    return default


# ============ ПОЛУЧЕНИЕ ПАРАМЕТРОВ V2 ИЗ СЕТИ ============
def fetch_v2_params_from_network(timeout: float = 5.0) -> Optional[Dict[str, Any]]:
    """Получает параметры cPoC V2 из сети Gonka."""
    try:
        import httpx
    except ImportError:
        log_warning("httpx не установлен, используем fallback параметры")
        return None

    for node_url in GENESIS_NODES:
        try:
            url = f"{node_url}/chain-api/productscience/inference/inference/params"
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                response = client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    params = data.get("params", {})
                    poc_params = params.get("poc_params", {})

                    model_id = poc_params.get("model_id", DEFAULT_MODEL_ID)
                    seq_len = int(poc_params.get("seq_len", DEFAULT_SEQ_LEN))
                    model_params = poc_params.get("model_params", {})
                    r_target = parse_exp_value(model_params.get("r_target"), DEFAULT_R_TARGET)

                    # stat_test параметры
                    stat_test = poc_params.get("stat_test", {})
                    dist_threshold = parse_exp_value(stat_test.get("dist_threshold"), 0.2)
                    p_mismatch = parse_exp_value(stat_test.get("p_mismatch"), 0.1)
                    p_value_threshold = parse_exp_value(stat_test.get("p_value_threshold"), 0.05)

                    # weight параметры
                    weight_scale_factor = parse_exp_value(poc_params.get("weight_scale_factor"), WEIGHT_SCALE_FACTOR)

                    # confirmation_poc_params
                    cpoc_params = params.get("confirmation_poc_params", {})
                    expected_confirmations = int(cpoc_params.get("expected_confirmations_per_epoch", 4))

                    # epoch_params для длительности
                    epoch_params = params.get("epoch_params", {})
                    poc_stage_duration = int(epoch_params.get("poc_stage_duration", 60))
                    result = {
                        "model_id": model_id,
                        "seq_len": seq_len,
                        "k_dim": DEFAULT_K_DIM,
                        "r_target": r_target,
                        "weight_scale_factor": weight_scale_factor,
                        "confirmation_poc_v2_enabled": poc_params.get("confirmation_poc_v2_enabled", False),
                        "poc_v2_enabled": poc_params.get("poc_v2_enabled", False),
                        "stat_test": {
                            "dist_threshold": dist_threshold,
                            "p_mismatch": p_mismatch,
                            "p_value_threshold": p_value_threshold,
                        },
                        "expected_confirmations_per_epoch": expected_confirmations,
                        "poc_stage_duration_blocks": poc_stage_duration,
                        "source_node": node_url,
                    }

                    log_success(f"V2 параметры получены из сети: {node_url}")
                    return result

        except Exception:
            continue

    log_warning("Не удалось получить V2 параметры из сети")
    return None


def fetch_poc_duration(timeout: float = 10.0) -> Optional[float]:
    """Получает длительность PoC фазы в минутах."""
    try:
        import httpx
    except ImportError:
        return None

    for node_url in GENESIS_NODES:
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                params_url = f"{node_url}/chain-api/productscience/inference/inference/params"
                params_resp = client.get(params_url)
                if params_resp.status_code != 200:
                    continue

                params_data = params_resp.json()
                epoch_params = params_data.get("params", {}).get("epoch_params", {})
                epoch_length = int(epoch_params.get("epoch_length", 15391))
                poc_stage_duration = int(epoch_params.get("poc_stage_duration", 60))
                epoch_shift = int(epoch_params.get("epoch_shift", 16980))

                block_resp = client.get(f"{node_url}/chain-rpc/block")
                if block_resp.status_code != 200:
                    continue

                current_block = int(block_resp.json()["result"]["block"]["header"]["height"])
                adjusted = current_block - epoch_shift
                if adjusted < 0:
                    adjusted = 0
                current_epoch = adjusted // epoch_length
                epoch_start = epoch_shift + (current_epoch * epoch_length)
                poc_start = epoch_start
                poc_end = epoch_start + poc_stage_duration

                if current_block < poc_end and current_epoch > 0:
                    prev_start = epoch_shift + ((current_epoch - 1) * epoch_length)
                    poc_start = prev_start
                    poc_end = prev_start + poc_stage_duration

                start_resp = client.get(f"{node_url}/chain-rpc/block?height={poc_start}")
                end_resp = client.get(f"{node_url}/chain-rpc/block?height={poc_end}")
                if start_resp.status_code != 200 or end_resp.status_code != 200:
                    continue

                def parse_time(ts):
                    if "." in ts:
                        base, frac = ts.split(".")
                        frac = frac.rstrip("Z")[:6]
                        ts = f"{base}.{frac}+00:00"
                    else:
                        ts = ts.replace("Z", "+00:00")
                    return datetime.fromisoformat(ts)

                st = parse_time(start_resp.json()["result"]["block"]["header"]["time"])
                et = parse_time(end_resp.json()["result"]["block"]["header"]["time"])
                duration_sec = (et - st).total_seconds()
                return round(duration_sec / 60, 2)

        except Exception:
            continue

    return None


def get_gpu_stats() -> Dict[str, Any]:
    """Получает статистику GPU через nvidia-smi."""
    stats = {}
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
            stats[i] = {
                'vram_percent': round(mem.used / mem.total * 100, 1),
                'gpu_util': util.gpu,
                'power_watts': round(power, 1),
            }
        pynvml.nvmlShutdown()
    except Exception:
        pass
    return stats


def get_gpu_info() -> Dict[str, Any]:
    """Получает информацию о GPU."""
    info = {}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            line = result.stdout.strip().split("\n")[0]
            parts = [p.strip() for p in line.split(",")]
            info["name"] = parts[0] if len(parts) > 0 else "Unknown"
            info["total_memory_gb"] = parts[1] if len(parts) > 1 else "N/A"
            info["driver_version"] = parts[2] if len(parts) > 2 else "N/A"

        count_result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10
        )
        if count_result.returncode == 0:
            info["num_gpus"] = len(count_result.stdout.strip().split("\n"))
    except Exception:
        info["name"] = "Unknown"
    return info


# ============ V2 БЕНЧМАРК ============
class GonkaBenchmarkV2:
    """Бенчмарк PoC V2 через vLLM API."""

    def __init__(
        self,
        vllm_url: str = None,
        vllm_urls: list = None,
        duration_min: float = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        seq_len: int = DEFAULT_SEQ_LEN,
        k_dim: int = DEFAULT_K_DIM,
        model_id: str = DEFAULT_MODEL_ID,
        use_network_params: bool = True,
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.k_dim = k_dim
        self.model_id = model_id
        self.r_target = DEFAULT_R_TARGET
        self.weight_scale_factor = WEIGHT_SCALE_FACTOR

        # Получаем параметры из сети
        self.v2_params = None
        poc_duration = None
        if use_network_params:
            log_info("Получение V2 параметров из сети...")
            self.v2_params = fetch_v2_params_from_network()
            if self.v2_params:
                self.model_id = self.v2_params["model_id"]
                self.seq_len = self.v2_params["seq_len"]
                self.k_dim = self.v2_params["k_dim"]
                self.r_target = self.v2_params["r_target"]
                self.weight_scale_factor = self.v2_params.get("weight_scale_factor", WEIGHT_SCALE_FACTOR)

                self._print_v2_params(self.v2_params)

            log_info("Получение длительности PoC фазы...")
            poc_duration = fetch_poc_duration()
            if poc_duration:
                log_success(f"PoC длительность: {poc_duration} минут")

        # Длительность: аргумент > из сети > default
        if duration_min is not None:
            self.duration_sec = duration_min * 60
            log_info(f"Используем указанную длительность: {duration_min} мин")
        elif poc_duration is not None:
            self.duration_sec = poc_duration * 60
            log_success(f"Используем длительность из PoC фазы: {poc_duration} мин")
        else:
            self.duration_sec = DEFAULT_DURATION_MIN * 60
            log_info(f"Используем default длительность: {DEFAULT_DURATION_MIN} мин")

        # URL vLLM серверов (multi-instance)
        if vllm_urls and len(vllm_urls) > 0:
            self.vllm_urls = [u.rstrip("/") for u in vllm_urls]
        elif vllm_url:
            self.vllm_urls = [vllm_url.rstrip("/")]
        else:
            self.vllm_urls = [f"http://localhost:{DEFAULT_VLLM_PORT}"]

        # Обратная совместимость
        self.vllm_url = self.vllm_urls[0]
        self.n_instances = len(self.vllm_urls)

        if self.n_instances > 1:
            log_info(f"Multi-instance: {self.n_instances} бэкендов")
            for i, url in enumerate(self.vllm_urls):
                log_info(f"  instance {i}: {url}")

        # Тестовые данные
        self.block_hash = hashlib.sha256(b"gonka_benchmark_v2").hexdigest()
        self.block_height = 1
        self.public_key = "benchmark_v2"

    def _print_v2_params(self, params: Dict):
        W = 62
        print()
        print(f"{Colors.BOLD}{Colors.CYAN}\u2554{'═' * W}\u2557{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}\u2551{'PoC V2 Parameters (from network)':^{W}}\u2551{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}\u2560{'═' * W}\u2563{Colors.END}")
        print(f"{Colors.CYAN}\u2551{Colors.END}  {Colors.GREEN}cPoC Parameters:{Colors.END}{' ' * (W - 19)}{Colors.CYAN}\u2551{Colors.END}")
        print(f"{Colors.CYAN}\u2551{Colors.END}    model_id:            {str(params['model_id']):<{W - 26}}{Colors.CYAN}\u2551{Colors.END}")
        print(f"{Colors.CYAN}\u2551{Colors.END}    seq_len:             {str(params['seq_len']):<{W - 26}}{Colors.CYAN}\u2551{Colors.END}")
        print(f"{Colors.CYAN}\u2551{Colors.END}    k_dim:               {str(params['k_dim']):<{W - 26}}{Colors.CYAN}\u2551{Colors.END}")
        print(f"{Colors.CYAN}\u2551{Colors.END}    r_target:            {str(params['r_target']):<{W - 26}}{Colors.CYAN}\u2551{Colors.END}")
        print(f"{Colors.CYAN}\u2560{'═' * W}\u2563{Colors.END}")
        wsf = str(params.get('weight_scale_factor', WEIGHT_SCALE_FACTOR))
        print(f"{Colors.CYAN}\u2551{Colors.END}  {Colors.GREEN}Weight:{Colors.END}{' ' * (W - 10)}{Colors.CYAN}\u2551{Colors.END}")
        print(f"{Colors.CYAN}\u2551{Colors.END}    WeightScaleFactor:   {wsf:<{W - 26}}{Colors.CYAN}\u2551{Colors.END}")
        print(f"{Colors.CYAN}\u2560{'═' * W}\u2563{Colors.END}")
        print(f"{Colors.CYAN}\u2551{Colors.END}  {Colors.GREEN}Flags:{Colors.END}{' ' * (W - 9)}{Colors.CYAN}\u2551{Colors.END}")
        cpoc_str = str(params.get('confirmation_poc_v2_enabled', False))
        poc_str = str(params.get('poc_v2_enabled', False))
        print(f"{Colors.CYAN}\u2551{Colors.END}    confirmation_poc_v2: {cpoc_str:<{W - 26}}{Colors.CYAN}\u2551{Colors.END}")
        print(f"{Colors.CYAN}\u2551{Colors.END}    poc_v2_enabled:      {poc_str:<{W - 26}}{Colors.CYAN}\u2551{Colors.END}")
        print(f"{Colors.CYAN}\u2560{'═' * W}\u2563{Colors.END}")
        st = params.get('stat_test', {})
        print(f"{Colors.CYAN}\u2551{Colors.END}  {Colors.GREEN}Stat Test:{Colors.END}{' ' * (W - 13)}{Colors.CYAN}\u2551{Colors.END}")
        print(f"{Colors.CYAN}\u2551{Colors.END}    dist_threshold:      {str(st.get('dist_threshold', 'N/A')):<{W - 26}}{Colors.CYAN}\u2551{Colors.END}")
        print(f"{Colors.CYAN}\u2551{Colors.END}    p_mismatch:          {str(st.get('p_mismatch', 'N/A')):<{W - 26}}{Colors.CYAN}\u2551{Colors.END}")
        print(f"{Colors.CYAN}\u2551{Colors.END}    p_value_threshold:   {str(st.get('p_value_threshold', 'N/A')):<{W - 26}}{Colors.CYAN}\u2551{Colors.END}")
        print(f"{Colors.CYAN}\u2560{'═' * W}\u2563{Colors.END}")
        source = str(params.get('source_node', 'unknown'))[:W - 11]
        print(f"{Colors.CYAN}\u2551{Colors.END}  Source: {source:<{W - 11}}{Colors.CYAN}\u2551{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}\u255a{'═' * W}\u255d{Colors.END}")
        print()

    def print_header(self):
        gpu_info = get_gpu_info()
        gpu_name = gpu_info.get("name", "Unknown")
        num_gpus = gpu_info.get("num_gpus", "?")
        if len(gpu_name) > 38:
            gpu_name = gpu_name[:35] + "..."

        print()
        print(f"{Colors.BOLD}{Colors.CYAN}", end="")
        print("\u2554" + "═" * 60 + "\u2557")
        print("\u2551" + " " * 12 + "Gonka PoC V2 Benchmark v1.0" + " " * 21 + "\u2551")
        print("\u2560" + "═" * 60 + "\u2563")

        gpu_line = f"\u2551  GPU: {num_gpus}x {gpu_name}"
        print(gpu_line + " " * (63 - 2 - len(gpu_line)) + "\u2551")

        if self.n_instances > 1:
            vllm_line = f"\u2551  vLLM: {self.n_instances} instances ({self.vllm_urls[0]}...)"
        else:
            vllm_line = f"\u2551  vLLM: {self.vllm_url}"
        print(vllm_line + " " * (63 - 2 - len(vllm_line)) + "\u2551")

        model_line = f"\u2551  Model: {self.model_id}"
        if len(model_line) > 61:
            model_line = model_line[:58] + "..."
        print(model_line + " " * (63 - 2 - len(model_line)) + "\u2551")

        cfg_line = f"\u2551  seq_len={self.seq_len} k_dim={self.k_dim} batch={self.batch_size}"
        print(cfg_line + " " * (63 - 2 - len(cfg_line)) + "\u2551")

        dur_line = f"\u2551  Duration: {self.duration_sec / 60:.2f} minutes"
        print(dur_line + " " * (63 - 2 - len(dur_line)) + "\u2551")

        print("\u255a" + "═" * 60 + "\u255d")
        print(f"{Colors.END}")

    def _wait_for_vllm(self, timeout: int = 300) -> bool:
        """Ждёт пока все vLLM инстансы станут доступны."""
        import httpx
        log_info(f"Ожидание {self.n_instances} vLLM инстанс(ов)...")
        start = time.time()
        ready = set()
        while time.time() - start < timeout:
            for i, url in enumerate(self.vllm_urls):
                if i in ready:
                    continue
                try:
                    with httpx.Client(timeout=5) as client:
                        resp = client.get(f"{url}/health")
                        if resp.status_code == 200:
                            ready.add(i)
                            log_success(f"vLLM инстанс {i} доступен: {url}")
                except Exception:
                    pass
            if len(ready) == self.n_instances:
                log_success(f"Все {self.n_instances} инстанс(ов) доступны!")
                return True
            time.sleep(5)
            elapsed = int(time.time() - start)
            print(f"  ... ожидание {elapsed}s / {timeout}s ({len(ready)}/{self.n_instances} ready)", end="\r")

        log_error(f"Не все vLLM инстансы стали доступны за {timeout}s ({len(ready)}/{self.n_instances})")
        return False

    def _start_generation(self) -> bool:
        """Отправляет запрос на начало генерации на все инстансы."""
        import httpx

        ok_count = 0
        for i, url in enumerate(self.vllm_urls):
            payload = {
                "block_hash": self.block_hash,
                "block_height": self.block_height,
                "public_key": self.public_key,
                "node_id": 0,
                "node_count": 1,
                "group_id": i,
                "n_groups": self.n_instances,
                "batch_size": self.batch_size,
                "params": {
                    "model": self.model_id,
                    "seq_len": self.seq_len,
                    "k_dim": self.k_dim,
                },
            }

            try:
                with httpx.Client(timeout=30) as client:
                    resp = client.post(f"{url}/api/v1/pow/init/generate", json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        log_success(f"Генерация запущена на инстансе {i}: {data}")
                        ok_count += 1
                    else:
                        log_error(f"Ошибка запуска на инстансе {i}: HTTP {resp.status_code} - {resp.text}")
            except Exception as e:
                log_error(f"Ошибка подключения к инстансу {i} ({url}): {e}")

        if ok_count == 0:
            log_error("Не удалось запустить генерацию ни на одном инстансе")
            return False
        if ok_count < self.n_instances:
            log_warning(f"Генерация запущена на {ok_count}/{self.n_instances} инстансах")
        return True

    def _poll_status(self) -> Optional[Dict]:
        """Опрашивает /api/v1/pow/status со всех инстансов и агрегирует."""
        import httpx
        total_processed = 0
        total_nonces_per_sec = 0.0
        any_generating = False
        any_response = False

        for url in self.vllm_urls:
            try:
                with httpx.Client(timeout=10) as client:
                    resp = client.get(f"{url}/api/v1/pow/status")
                    if resp.status_code == 200:
                        data = resp.json()
                        any_response = True
                        if data.get("status") == "GENERATING":
                            any_generating = True
                        stats = data.get("stats", {})
                        total_processed += stats.get("total_processed", 0)
                        total_nonces_per_sec += stats.get("nonces_per_second", 0)
            except Exception:
                pass

        if not any_response:
            return None

        return {
            "status": "GENERATING" if any_generating else "IDLE",
            "stats": {
                "total_processed": total_processed,
                "nonces_per_second": total_nonces_per_sec,
            },
        }

    def _stop_generation(self):
        """Останавливает генерацию на всех инстансах."""
        import httpx
        for i, url in enumerate(self.vllm_urls):
            try:
                with httpx.Client(timeout=10) as client:
                    resp = client.post(f"{url}/api/v1/pow/stop")
                    if resp.status_code == 200:
                        if self.n_instances > 1:
                            log_success(f"Генерация остановлена на инстансе {i}")
                        else:
                            log_success("Генерация остановлена")
                    else:
                        log_warning(f"Остановка инстанса {i}: HTTP {resp.status_code}")
            except Exception as e:
                log_warning(f"Ошибка при остановке инстанса {i}: {e}")

    def run(self) -> Optional[Dict]:
        """Запускает V2 бенчмарк."""
        self.print_header()

        # Проверяем доступность vLLM
        if not self._wait_for_vllm():
            return None

        # Запускаем генерацию
        log_info("Запуск генерации артефактов...")
        if not self._start_generation():
            return None

        # Ждём первый batch — аналогично V1 где таймер стартует после загрузки модели
        log_info("Ожидание первого batch (генерация начнётся после готовности модели)...")
        warmup_timeout = 300  # 5 минут на прогрев
        warmup_start = time.time()
        while time.time() - warmup_start < warmup_timeout:
            time.sleep(2)
            status = self._poll_status()
            if status and status.get("status") == "GENERATING":
                stats = status.get("stats", {})
                if stats.get("total_processed", 0) > 0:
                    log_success(f"Генерация активна! Первые nonces получены.")
                    break
            elapsed_w = int(time.time() - warmup_start)
            print(f"  ... прогрев {elapsed_w}s / {warmup_timeout}s", end="\r")
        else:
            log_error(f"Генерация не началась за {warmup_timeout}s")
            self._stop_generation()
            return None
        print()

        # Сбрасываем счётчик — останавливаем и перезапускаем чтобы таймер был чистый
        self._stop_generation()
        time.sleep(1)
        if not self._start_generation():
            return None
        time.sleep(2)

        # Основной цикл: опрос статуса
        start_time = time.time()
        last_report_time = start_time
        last_nonces = 0
        peak_rate = 0.0

        log_info(f"Бенчмарк запущен на {self.duration_sec / 60:.1f} минут...")
        print()

        try:
            while time.time() - start_time < self.duration_sec:
                time.sleep(POLL_INTERVAL)

                status = self._poll_status()
                if not status:
                    continue

                if status.get("status") != "GENERATING":
                    # Генерация могла остановиться
                    state = status.get("status", "unknown")
                    if state == "IDLE":
                        log_warning("Генерация не активна (IDLE), перезапускаем...")
                        self._start_generation()
                        time.sleep(3)
                    continue

                stats = status.get("stats", {})
                total_nonces = stats.get("total_processed", 0)
                nonces_per_sec = stats.get("nonces_per_second", 0)
                nonces_per_min = nonces_per_sec * 60

                if nonces_per_min > peak_rate:
                    peak_rate = nonces_per_min

                # Прогресс каждые REPORT_INTERVAL секунд
                current_time = time.time()
                if current_time - last_report_time >= REPORT_INTERVAL:
                    elapsed = current_time - start_time
                    elapsed_min = elapsed / 60
                    poc_weight = int(total_nonces * self.weight_scale_factor)

                    # Мгновенная скорость (за последний интервал)
                    delta_nonces = total_nonces - last_nonces
                    delta_time = current_time - last_report_time
                    instant_rate = (delta_nonces / delta_time * 60) if delta_time > 0 else 0

                    print(f"[{int(elapsed//60):02d}:{int(elapsed%60):02d}] "
                          f"nonces: {total_nonces} | poc_weight: {poc_weight} | "
                          f"nonces/min: {nonces_per_min:.1f} | instant: {instant_rate:.1f}/min")

                    # GPU статистика
                    gpu_stats = get_gpu_stats()
                    for gpu_id, gs in gpu_stats.items():
                        print(f"                     GPU{gpu_id}: "
                              f"{gs['vram_percent']}% VRAM {gs['gpu_util']}% GPU {gs['power_watts']}W")

                    print(f"                     {datetime.now().strftime('%H:%M:%S')}")

                    last_report_time = current_time
                    last_nonces = total_nonces

        except KeyboardInterrupt:
            log_warning("\nБенчмарк прерван пользователем")

        # Финальный опрос статуса
        final_status = self._poll_status()
        total_nonces = 0
        nonces_per_sec = 0
        if final_status and final_status.get("stats"):
            total_nonces = final_status["stats"].get("total_processed", 0)
            nonces_per_sec = final_status["stats"].get("nonces_per_second", 0)

        # Останавливаем генерацию
        self._stop_generation()

        # Результаты
        elapsed_min = (time.time() - start_time) / 60
        nonces_per_min = total_nonces / elapsed_min if elapsed_min > 0 else 0
        poc_weight = int(total_nonces * self.weight_scale_factor)

        # Вывод результатов
        print(f"\n{Colors.BOLD}{Colors.GREEN}", end="")
        print("\u2554" + "═" * 60 + "\u2557")
        print("\u2551" + " " * 17 + "V2 BENCHMARK RESULTS" + " " * 23 + "\u2551")
        print("\u2560" + "═" * 60 + "\u2563")
        print(f"\u2551  {Colors.CYAN}total_nonces:{Colors.END}    {total_nonces:<41}\u2551")
        print(f"\u2551  {Colors.GREEN}poc_weight:{Colors.END}      {poc_weight:<41}\u2551")
        print("\u2560" + "═" * 60 + "\u2563")
        print(f"\u2551  {Colors.CYAN}nonces/min:{Colors.END}      {nonces_per_min:<41.2f}\u2551")
        print(f"\u2551  {Colors.CYAN}nonces/sec:{Colors.END}      {nonces_per_sec:<41.2f}\u2551")
        print(f"\u2551  {Colors.CYAN}peak_rate/min:{Colors.END}   {peak_rate:<41.2f}\u2551")
        print("\u2560" + "═" * 60 + "\u2563")
        print(f"\u2551  {Colors.CYAN}duration_min:{Colors.END}    {elapsed_min:<41.2f}\u2551")
        print(f"\u2551  {Colors.CYAN}vllm_url:{Colors.END}        {self.vllm_url:<41}\u2551")
        print("\u255a" + "═" * 60 + "\u255d")
        print(f"{Colors.END}")
        print(f"{Colors.YELLOW}Формула:{Colors.END} poc_weight = total_nonces \u00d7 {self.weight_scale_factor}")
        print(f"         (WeightScaleFactor из сети)\n")

        # Собираем результат
        gpu_info = get_gpu_info()
        results = {
            "mode": "v2",
            "total_nonces": total_nonces,
            "valid_nonces": total_nonces,
            "poc_weight": poc_weight,
            "nonces_per_min": round(nonces_per_min, 2),
            "nonces_per_sec": round(nonces_per_sec, 2),
            "valid_per_min": round(nonces_per_min, 2),
            "peak_rate_per_min": round(peak_rate, 2),
            "duration_min": round(elapsed_min, 2),
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "k_dim": self.k_dim,
            "model_id": self.model_id,
            "r_target": self.r_target,
            "vllm_url": self.vllm_url,
            "vllm_urls": self.vllm_urls,
            "n_instances": self.n_instances,
            "weight_scale_factor": self.weight_scale_factor,
            "timestamp": datetime.now().isoformat(),
            "gpu": gpu_info,
            "v2_params": self.v2_params,
        }

        return results

    def save_results(self, results: Dict, output_dir: str = "results") -> Optional[Path]:
        if results is None:
            return None

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        gpu_name = results.get("gpu", {}).get("name", "Unknown").replace(" ", "_").replace("/", "_")
        mode_upper = results.get("mode", "v2").upper()
        filename = f"{timestamp}_{mode_upper}_{gpu_name}_{results['poc_weight']}.json"
        filepath = output_path / filename

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        log_success(f"Результат сохранён: {filepath}")
        return filepath


# ============ ГЛАВНАЯ ФУНКЦИЯ ============
def main():
    parser = argparse.ArgumentParser(
        description="Gonka PoC V2 Benchmark - тест через vLLM API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python3 gonka_benchmark_v2.py                           # Default: localhost:5000
  python3 gonka_benchmark_v2.py --vllm-url http://localhost:5000
  python3 gonka_benchmark_v2.py --duration 3
  python3 gonka_benchmark_v2.py --batch-size 64 --seq-len 1024

Формула: poc_weight = total_nonces × 0.262
        """
    )

    parser.add_argument("--vllm-url", type=str, default=None,
                        help=f"URL vLLM сервера (по умолчанию: http://localhost:{DEFAULT_VLLM_PORT})")
    parser.add_argument("--vllm-port", type=int, default=None,
                        help=f"Порт vLLM сервера (по умолчанию: {DEFAULT_VLLM_PORT})")
    parser.add_argument("--vllm-ports", type=str, default=None,
                        help="Порты нескольких vLLM инстансов через запятую (например: 5001,5002)")
    parser.add_argument("--duration", "-d", type=float, default=None,
                        help=f"Время теста в минутах (по умолчанию: из сети или {DEFAULT_DURATION_MIN})")
    parser.add_argument("--batch-size", "-b", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size (по умолчанию: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--seq-len", type=int, default=None,
                        help=f"Sequence length (по умолчанию: из сети или {DEFAULT_SEQ_LEN})")
    parser.add_argument("--k-dim", type=int, default=None,
                        help=f"K dimensions (по умолчанию: {DEFAULT_K_DIM})")
    parser.add_argument("--model", type=str, default=None,
                        help=f"Model ID (по умолчанию: из сети или {DEFAULT_MODEL_ID})")
    parser.add_argument("--output", "-o", type=str, default="results",
                        help="Директория для результатов (по умолчанию: results)")
    parser.add_argument("--no-save", action="store_true",
                        help="Не сохранять результаты")
    parser.add_argument("--offline", action="store_true",
                        help="Не запрашивать параметры из сети")
    parser.add_argument("--mode-label", type=str, default="v2",
                        help="Метка режима для результатов (по умолчанию: v2)")

    args = parser.parse_args()

    # Определяем URL(s)
    vllm_url = args.vllm_url
    vllm_urls = None

    if args.vllm_ports:
        # Multi-instance: --vllm-ports 5001,5002,5003
        ports = [int(p.strip()) for p in args.vllm_ports.split(",")]
        vllm_urls = [f"http://localhost:{p}" for p in ports]
    elif vllm_url is None and args.vllm_port is not None:
        vllm_url = f"http://localhost:{args.vllm_port}"

    benchmark = GonkaBenchmarkV2(
        vllm_url=vllm_url,
        vllm_urls=vllm_urls,
        duration_min=args.duration,
        batch_size=args.batch_size,
        seq_len=args.seq_len or DEFAULT_SEQ_LEN,
        k_dim=args.k_dim or DEFAULT_K_DIM,
        model_id=args.model or DEFAULT_MODEL_ID,
        use_network_params=not args.offline,
    )

    results = benchmark.run()

    if results:
        results["mode"] = args.mode_label
        if not args.no_save:
            benchmark.save_results(results, args.output)
        log_success("Бенчмарк V2 завершён!")
        return 0
    else:
        log_error("Бенчмарк V2 завершился с ошибкой")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Gonka PoW Benchmark
Автономный бенчмарк для тестирования производительности GPU

ФОРМУЛА РАСЧЁТА ВЕСА (из исходного кода gonka):
  poc_weight = valid_nonces × WeightScaleFactor

Где:
- valid_nonces = количество валидных nonce, найденных вашим GPU
- WeightScaleFactor = 2.5 (из inference-chain/x/inference/module/chainvalidation.go)

Источник формулы:
  weight = mathsdk.LegacyNewDec(weight).Mul(WeightScaleFactor).TruncateInt64()
  var WeightScaleFactor = mathsdk.LegacyNewDecWithPrec(25, 1) // 2.5
"""

import os
import sys
import time
import multiprocessing as mp
from datetime import datetime
import json
import argparse
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# Настройка окружения для cudnn
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch

# ============ КОНФИГУРАЦИЯ ============
# Время теста по умолчанию (минуты) - как реальная PoC фаза в gonka
DEFAULT_DURATION_MIN = 5

# Путь к коду gonka (относительно текущей директории)
DEFAULT_GONKA_PATH = "./gonka"

# WeightScaleFactor из исходного кода gonka
# Источник: inference-chain/x/inference/module/chainvalidation.go
# var WeightScaleFactor = mathsdk.LegacyNewDecWithPrec(25, 1) // 2.5
WEIGHT_SCALE_FACTOR = 2.5

# Fallback RTarget если не удалось получить из сети
DEFAULT_R_TARGET = 1.398077

# PoC параметры v2 (из inference-chain/x/inference/types/params.go DefaultPoCModelParams)
# Используются как fallback если не удалось получить из сети
DEFAULT_POC_PARAMS = {
    "dim": 1792,
    "n_layers": 64,
    "n_heads": 64,
    "n_kv_heads": 64,
    "vocab_size": 8196,
    "ffn_dim_multiplier": 10.0,
    "multiple_of": 8192,  # Явно указываем 8192 для читаемости
    "norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "use_scaled_rope": False,
    "seq_len": 256,
}

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
    print(f"{Colors.GREEN}[✓]{Colors.END} {msg}")


def log_warning(msg: str):
    print(f"{Colors.YELLOW}[⚠]{Colors.END} {msg}")


def log_error(msg: str):
    print(f"{Colors.RED}[✗]{Colors.END} {msg}")


# ============ ПОЛУЧЕНИЕ ПАРАМЕТРОВ ИЗ СЕТИ ============
def fetch_poc_params_from_network(timeout: float = 5.0) -> Optional[Dict[str, Any]]:
    """
    Получает актуальные PoC параметры из сети Gonka.
    Пробует все genesis ноды пока не получит ответ.
    
    Returns:
        Dict с параметрами или None если все ноды недоступны
    """
    try:
        import httpx
    except ImportError:
        log_warning("httpx не установлен, используем fallback параметры")
        return None
    
    for node_url in GENESIS_NODES:
        try:
            # Эндпоинт для получения параметров inference модуля
            url = f"{node_url}/chain-api/productscience/inference/inference/params"
            
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                response = client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    
                    # Извлекаем PoC параметры
                    params = data.get("params", {})
                    poc_params = params.get("poc_model_params", {})
                    
                    if not poc_params:
                        continue
                    
                    result = {
                        "r_target": float(poc_params.get("r_target", DEFAULT_R_TARGET)),
                        "dim": int(poc_params.get("dim", DEFAULT_POC_PARAMS["dim"])),
                        "n_layers": int(poc_params.get("n_layers", DEFAULT_POC_PARAMS["n_layers"])),
                        "n_heads": int(poc_params.get("n_heads", DEFAULT_POC_PARAMS["n_heads"])),
                        "n_kv_heads": int(poc_params.get("n_kv_heads", DEFAULT_POC_PARAMS["n_kv_heads"])),
                        "vocab_size": int(poc_params.get("vocab_size", DEFAULT_POC_PARAMS["vocab_size"])),
                        "ffn_dim_multiplier": float(poc_params.get("ffn_dim_multiplier", DEFAULT_POC_PARAMS["ffn_dim_multiplier"])),
                        "multiple_of": int(poc_params.get("multiple_of", DEFAULT_POC_PARAMS["multiple_of"])),
                        "norm_eps": float(poc_params.get("norm_eps", DEFAULT_POC_PARAMS["norm_eps"])),
                        "rope_theta": float(poc_params.get("rope_theta", DEFAULT_POC_PARAMS["rope_theta"])),
                        "use_scaled_rope": bool(poc_params.get("use_scaled_rope", DEFAULT_POC_PARAMS["use_scaled_rope"])),
                        "seq_len": int(poc_params.get("seq_len", DEFAULT_POC_PARAMS["seq_len"])),
                        "source_node": node_url,
                    }
                    
                    log_success(f"Параметры получены из сети: {node_url}")
                    log_info(f"  r_target: {result['r_target']}")
                    
                    return result
                    
        except Exception as e:
            # Тихо пробуем следующую ноду
            continue
    
    log_warning("Не удалось получить параметры из сети, используем fallback")
    return None


def get_poc_params(use_network: bool = True) -> Tuple[Dict[str, Any], float]:
    """
    Получает PoC параметры (из сети или fallback).
    
    Args:
        use_network: Пытаться ли получить из сети
        
    Returns:
        Tuple (poc_params dict без r_target, r_target float)
    """
    if use_network:
        network_params = fetch_poc_params_from_network()
        if network_params:
            r_target = network_params.pop("r_target")
            network_params.pop("source_node", None)
            return network_params, r_target
    
    # Fallback
    log_info(f"Используем fallback параметры (r_target={DEFAULT_R_TARGET})")
    return DEFAULT_POC_PARAMS.copy(), DEFAULT_R_TARGET


# ============ ДОБАВЛЕНИЕ ПУТЕЙ К GONKA ============
def setup_gonka_path(gonka_path: str) -> bool:
    """Добавляет пути к gonka модулям в sys.path"""
    gonka_path = Path(gonka_path).resolve()

    if not gonka_path.exists():
        log_error(f"Путь к gonka не найден: {gonka_path}")
        log_info("Скачайте код:")
        log_info("  git clone --depth 1 --filter=blob:none --sparse https://github.com/gonka-ai/gonka.git")
        log_info("  cd gonka && git sparse-checkout set mlnode/packages/pow common")
        return False

    # Добавляем необходимые пути
    paths_to_add = [
        gonka_path / "mlnode" / "packages" / "pow" / "src",
        gonka_path / "mlnode" / "packages" / "common" / "src",
    ]

    for path in paths_to_add:
        if path.exists():
            sys.path.insert(0, str(path))
            log_info(f"Добавлен путь: {path}")
        else:
            log_warning(f"Путь не найден (может быть не обязательным): {path}")

    return True


# ============ ЗАГРУЗКА МОДУЛЕЙ GONKA ============
def import_gonka_modules():
    """Импортирует необходимые модули из gonka"""
    try:
        from pow.compute.compute import Compute
        from pow.models.utils import Params
        from pow.random import get_target
        from pow.compute.autobs_v2 import get_batch_size_for_gpu_group, GpuGroup
        return Compute, Params, get_target, get_batch_size_for_gpu_group, GpuGroup
    except ImportError as e:
        log_error(f"Не удалось импортировать модули gonka: {e}")
        log_info("Убедитесь что скачали код gonka с нужными модулями")
        return None, None, None, None, None


# ============ NONCE ITERATOR (как в gonka) ============
class NonceIterator:
    """Итератор для разделения nonce между GPU (как в gonka)"""
    def __init__(self, node_id: int, n_nodes: int, group_id: int, n_groups: int):
        self.node_id = node_id
        self.n_nodes = n_nodes
        self.group_id = group_id
        self.n_groups = n_groups
        self._current_x = 0

    def __iter__(self):
        return self

    def __next__(self):
        # Формула из gonka: offset = node_id + group_id * n_nodes
        # step = n_groups * n_nodes
        # value = offset + current_x * step
        offset = self.node_id + self.group_id * self.n_nodes
        step = self.n_groups * self.n_nodes
        value = offset + self._current_x * step
        self._current_x += 1
        return value


# ============ WORKER PROCESS (как в gonka) ============
class WorkerProcess(mp.Process):
    """Worker процесс для одного GPU (как в gonka)"""
    def __init__(
        self,
        worker_id: int,
        device_ids: List[int],
        params,
        block_hash: str,
        block_height: int,
        public_key: str,
        batch_size: int,
        r_target: float,
        duration_sec: float,
        nonce_iterator: NonceIterator,
        result_queue: mp.Queue,
        stop_event: mp.Event,
    ):
        super().__init__()
        self.worker_id = worker_id
        self.device_ids = device_ids
        self.params = params
        self.block_hash = block_hash
        self.block_height = block_height
        self.public_key = public_key
        self.batch_size = batch_size
        self.r_target = r_target
        self.duration_sec = duration_sec
        self.nonce_iterator = nonce_iterator
        self.result_queue = result_queue
        self.stop_event = stop_event

        self.total_valid = 0
        self.total_checked = 0

    def _suppress_model_logs(self):
        """Подавляет логи модели (прогресс-бары)"""
        import logging
        import os
        import sys

        # Подавляем logging
        logging.getLogger('pow').setLevel(logging.WARNING)
        logging.getLogger('pow.compute').setLevel(logging.WARNING)
        logging.getLogger('pow.compute.model_init').setLevel(logging.WARNING)

        # Подавляем tqdm progress bars
        os.environ['TQDM_DISABLE'] = '1'

    def _get_gpu_stats(self):
        """Получает статистику GPU для этого worker'а"""
        stats = {}
        try:
            import pynvml
            pynvml.nvmlInit()

            for device_id in self.device_ids:
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000

                stats[device_id] = {
                    'vram_percent': round(mem_info.used / mem_info.total * 100, 1),
                    'gpu_util': util.gpu,
                    'power_watts': round(power, 1)
                }

            pynvml.nvmlShutdown()
        except Exception:
            pass
        return stats

    def run(self):
        """Основной цикл worker'а"""
        try:
            # Подавляем логи модели
            self._suppress_model_logs()

            # Импортируем внутри процесса
            from pow.compute.compute import Compute

            devices_str = [f"cuda:{i}" for i in self.device_ids]

            self.compute = Compute(
                params=self.params,
                block_hash=self.block_hash,
                block_height=self.block_height,
                public_key=self.public_key,
                r_target=self.r_target,
                devices=devices_str,
                node_id=0,
            )

            start_time = time.time()
            end_time = start_time + self.duration_sec

            # Предварительно получаем batch nonce
            next_nonces = [next(self.nonce_iterator) for _ in range(self.batch_size)]

            while time.time() < end_time and not self.stop_event.is_set():
                nonces = next_nonces
                next_nonces = [next(self.nonce_iterator) for _ in range(self.batch_size)]

                # Выполняем вычисления (возвращает Future)
                future = self.compute(
                    nonces=nonces,
                    public_key=self.public_key,
                    target=self.compute.target,
                    next_nonces=next_nonces,
                )
                proof_batch = future.result()  # Получаем результат из Future

                # Фильтруем по r_target
                valid_batch = proof_batch.sub_batch(self.r_target)

                # Считаем статистику
                self.total_checked += len(nonces)
                self.total_valid += len(valid_batch.nonces)

                # Получаем GPU статистику каждый раз
                gpu_stats = self._get_gpu_stats()

                # Отправляем результат в очередь
                elapsed = time.time() - start_time
                self.result_queue.put({
                    'worker_id': self.worker_id,
                    'total_valid': self.total_valid,
                    'total_checked': self.total_checked,
                    'elapsed': elapsed,
                    'gpu_stats': gpu_stats,
                })

            # Очистка
            del self.compute

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.result_queue.put({'worker_id': self.worker_id, 'error': str(e)})


# ============ КЛАСС БЕНЧМАРКА ============
class GonkaBenchmark:
    """Бенчмарк для Gonka PoW с поддержкой multi-GPU"""

    def __init__(
        self,
        device: str = "cuda:0",
        duration_min: float = DEFAULT_DURATION_MIN,
        batch_size: int = None,
        r_target: float = None,
        num_gpus: int = None,
        poc_params: Dict[str, Any] = None,
        use_network_params: bool = True,
    ):
        # Получаем параметры из сети или используем fallback
        if poc_params is None or r_target is None:
            network_poc_params, network_r_target = get_poc_params(use_network=use_network_params)
            if poc_params is None:
                poc_params = network_poc_params
            if r_target is None:
                r_target = network_r_target
        
        self.poc_params = poc_params
        self.r_target = r_target
        
        # Поддержка нескольких GPU
        if num_gpus is None:
            num_gpus = torch.cuda.device_count()

        self.num_gpus = min(num_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 0
        self.device_ids = list(range(self.num_gpus)) if self.num_gpus > 0 else [0]
        self.device = torch.device(f"cuda:{self.device_ids[0]}") if self.num_gpus > 0 else torch.device("cpu")

        self.duration_sec = duration_min * 60
        self.batch_size = batch_size

        # Параметры для теста
        self.block_hash = hashlib.sha256(b"gonka_benchmark").hexdigest()
        self.block_height = 1
        self.public_key = "benchmark"

        # Статистика
        self.total_valid = 0
        self.total_checked = 0
        self.start_time = None

        # Сбор valid nonce для последующей дедупликации
        self.all_valid_nonces = []
        self.save_nonces = False

        # Compute объект
        self.compute = None
        self.target = None

    def print_header(self):
        """Выводит заголовок бенчмарка"""
        cuda_version = torch.version.cuda or "N/A"

        # Собираем имена всех GPU
        if self.num_gpus > 1:
            gpu_name = f"{self.num_gpus}x {torch.cuda.get_device_name(0)}"
        else:
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

        # Обрезаем длинные имена GPU
        if len(gpu_name) > 38:
            gpu_name = gpu_name[:35] + "..."

        print()  # Отступ перед рамкой
        print(f"{Colors.BOLD}{Colors.CYAN}", end="")
        print("╔" + "═" * 60 + "╗")
        print("║" + " " * 15 + "Gonka PoW Benchmark v1.2" + " " * 21 + "║")
        print("╠" + "═" * 60 + "╣")

        # Ручное выравнивание (+3 пробела)
        gpu_line = "║  GPU: " + gpu_name
        print(gpu_line + " " * (63 - 2 - len(gpu_line)) + "║")

        cuda_line = "║  CUDA: " + str(cuda_version)
        print(cuda_line + " " * (63 - 2 - len(cuda_line)) + "║")

        dur_line = f"║  Test duration: {int(self.duration_sec // 60)} minutes"
        print(dur_line + " " * (63 - 2 - len(dur_line)) + "║")

        rtarget_line = f"║  RTarget: {self.r_target}"
        print(rtarget_line + " " * (63 - 2 - len(rtarget_line)) + "║")

        print("╚" + "═" * 60 + "╝")
        print(f"{Colors.END}")

    def print_results(self, worker_elapsed_min=None):
        """Выводит финальные результаты с дедупликацией

        Args:
            worker_elapsed_min: Время работы worker (как в реальной ноде).
                                 Если None, используется (current_time - start_time).
        """
        if worker_elapsed_min is None:
            duration_min = (time.time() - self.start_time) / 60
        else:
            duration_min = worker_elapsed_min

        valid_per_min = self.total_valid / duration_min if duration_min > 0 else 0
        raw_per_min = self.total_checked / duration_min if duration_min > 0 else 0
        one_in_n = self.total_checked / self.total_valid if self.total_valid > 0 else 0

        # poc_weight по формуле из chainvalidation.go
        poc_weight = int(self.total_valid * WEIGHT_SCALE_FACTOR)

        print(f"\n{Colors.BOLD}{Colors.GREEN}", end="")
        print("╔" + "═" * 60 + "╗")
        print("║" + " " * 20 + "BENCHMARK RESULTS" + " " * 23 + "║")
        print("╠" + "═" * 60 + "╣")
        print(f"║  {Colors.CYAN}valid_nonces:{Colors.END}    {self.total_valid:<41}║")
        print(f"║  {Colors.GREEN}poc_weight:{Colors.END}      {poc_weight:<41}║")
        print("╠" + "═" * 60 + "╣")
        print(f"║  {Colors.CYAN}valid/min:{Colors.END}       {valid_per_min:<41.2f}║")
        print(f"║  {Colors.CYAN}raw/min:{Colors.END}         {raw_per_min:<41.2f}║")
        print("╠" + "═" * 60 + "╣")
        print(f"║  {Colors.CYAN}total_checked:{Colors.END}   {self.total_checked:<41}║")
        print(f"║  {Colors.CYAN}duration_min:{Colors.END}    {duration_min:<41.2f}║")
        print("╚" + "═" * 60 + "╝")
        print(f"{Colors.END}")

        print(f"{Colors.YELLOW}Формула:{Colors.END} poc_weight = valid_nonces × {WEIGHT_SCALE_FACTOR}")
        print(f"         (из chainvalidation.go WeightScaleFactor)\n")

        # Дедупликация и показ уникальных
        unique_valid = self.total_valid
        unique_poc_weight = poc_weight
        duplicates = 0

        if self.save_nonces and len(self.all_valid_nonces) > 0:
            unique_nonces = set(self.all_valid_nonces)
            unique_valid = len(unique_nonces)
            duplicates = len(self.all_valid_nonces) - unique_valid
            unique_poc_weight = int(unique_valid * WEIGHT_SCALE_FACTOR)

            if duplicates > 0:
                print(f"{Colors.YELLOW}⚠ Дедупликация:{Colors.END}")
                print(f"   Всего найдено:    {len(self.all_valid_nonces)}")
                print(f"   Уникальных:       {unique_valid}")
                print(f"   Дубликаты:        {duplicates} ({duplicates/len(self.all_valid_nonces)*100:.1f}%)")
                print(f"   {Colors.GREEN}poc_weight (уникальные): {unique_poc_weight}{Colors.END}\n")
            else:
                print(f"{Colors.GREEN}✓ Дубликатов не найдено{Colors.END}\n")

        return {
            "valid_nonces": self.total_valid,
            "poc_weight": poc_weight,
            "unique_valid_nonces": unique_valid,
            "unique_poc_weight": unique_poc_weight,
            "duplicates": duplicates,
            "valid_per_min": valid_per_min,
            "raw_per_min": raw_per_min,
            "one_in_n": one_in_n,
            "duration_min": duration_min,
            "total_checked": self.total_checked,
        }

    def run(self) -> dict:
        """Запускает бенчмарк с multiprocessing (как в gonka)"""
        self.print_header()

        # Импорт модулей gonka
        Compute, Params, get_target, get_batch_size_for_gpu_group, GpuGroup = import_gonka_modules()
        if Compute is None:
            return None

        # Создаём параметры модели
        params = Params(**self.poc_params)
        params_str = f"Params(dim={params.dim}, n_layers={params.n_layers}, n_heads={params.n_heads}, n_kv_heads={params.n_kv_heads}, vocab_size={params.vocab_size}, ffn_dim_multiplier={params.ffn_dim_multiplier}, multiple_of={params.multiple_of}, norm_eps={params.norm_eps}, rope_theta={params.rope_theta}, use_scaled_rope={params.use_scaled_rope}, seq_len={params.seq_len})"
        log_info(f"params={params_str}")

        # Определяем batch_size для ОДНОГО GPU (так как каждый worker загружает свою модель)
        # При multiprocessing каждый worker имеет свою копию модели на своём GPU
        single_gpu_group = GpuGroup(devices=[self.device_ids[0]])  # Первый GPU
        single_gpu_batch = get_batch_size_for_gpu_group(single_gpu_group, params)

        # Для multi-GPU каждый worker использует этот batch size
        self.batch_size = single_gpu_batch
        log_info(f"Using batch size: {self.batch_size} per GPU (total: {self.batch_size * self.num_gpus} for {self.num_gpus}xGPU)")

        log_info(f"Запуск бенчмарка на {self.duration_sec / 60:.1f} минут с {self.num_gpus}xGPU...")
        print()

        # ВАЖНО: подавляем tqdm и логи ДО создания workers (унаследуется subprocess)
        os.environ['TQDM_DISABLE'] = '1'
        import logging
        logging.getLogger('pow').setLevel(logging.WARNING)
        logging.getLogger('pow.compute').setLevel(logging.WARNING)
        logging.getLogger('pow.compute.model_init').setLevel(logging.WARNING)

        # Multiprocessing setup
        mp.set_start_method('spawn', force=True)
        result_queue = mp.Queue()
        stop_event = mp.Event()

        # Создаём workers (по одному на GPU)
        workers = []
        for i in range(self.num_gpus):
            # Каждый worker получает свой уникальный nonce range
            nonce_iter = NonceIterator(
                node_id=0,
                n_nodes=1,
                group_id=i,
                n_groups=self.num_gpus
            )

            worker = WorkerProcess(
                worker_id=i,
                device_ids=[self.device_ids[i]],  # Каждый worker использует 1 GPU
                params=params,
                block_hash=self.block_hash,
                block_height=self.block_height,
                public_key=self.public_key,
                batch_size=self.batch_size,
                r_target=self.r_target,
                duration_sec=self.duration_sec,
                nonce_iterator=nonce_iter,
                result_queue=result_queue,
                stop_event=stop_event,
            )
            workers.append(worker)
            log_info(f"Worker {i} создан для GPU {self.device_ids[i]} (nonce offset: {i})")

        # Запускаем всех workers
        for worker in workers:
            worker.start()

        # Собираем результаты и показываем прогресс
        worker_stats = {i: {'total_valid': 0, 'total_checked': 0, 'gpu_stats': {}} for i in range(self.num_gpus)}
        report_interval = 30

        # Ждём первый результат (модель загружена)
        log_info("Ожидание загрузки модели...")

        # Получаем первый результат от любого worker
        first_result = result_queue.get(timeout=120)  # 2 минуты таймаут
        if 'error' in first_result:
            log_error(f"Worker {first_result['worker_id']}: {first_result['error']}")
            stop_event.set()
            return None

        worker_id = first_result['worker_id']
        worker_stats[worker_id]['total_valid'] = first_result['total_valid']
        worker_stats[worker_id]['total_checked'] = first_result['total_checked']
        worker_stats[worker_id]['elapsed'] = first_result.get('elapsed', 0)
        if first_result.get('gpu_stats'):
            worker_stats[worker_id]['gpu_stats'] = first_result['gpu_stats']

        # Начинаем отсчёт времени после загрузки модели
        start_time = time.time()
        last_report_time = start_time

        log_success(f"Модель загружена! Worker {worker_id} готов. Начинаем бенчмарк...")
        print()

        try:
            while time.time() - start_time < self.duration_sec:
                # Получаем результаты из очереди с небольшим таймаутом
                try:
                    result = result_queue.get(timeout=1)

                    if 'error' in result:
                        log_error(f"Worker {result['worker_id']}: {result['error']}")
                        stop_event.set()
                        break

                    worker_id = result['worker_id']
                    worker_stats[worker_id]['total_valid'] = result['total_valid']
                    worker_stats[worker_id]['total_checked'] = result['total_checked']
                    worker_stats[worker_id]['elapsed'] = result.get('elapsed', 0)
                    # Обновляем GPU статистику если она есть
                    if result.get('gpu_stats'):
                        worker_stats[worker_id]['gpu_stats'] = result['gpu_stats']

                except:
                    pass  # Таймаут - нормально, продолжаем

                # Прогресс каждые report_interval секунд
                current_time = time.time()
                if current_time - last_report_time >= report_interval:
                    # Используем worker elapsed (как в реальной ноде)
                    worker_elapseds = [s.get('elapsed', 0) for s in worker_stats.values() if s.get('elapsed', 0) > 0]
                    worker_elapsed = max(worker_elapseds) if worker_elapseds else (current_time - start_time)
                    worker_elapsed_min = worker_elapsed / 60

                    total_valid = sum(s['total_valid'] for s in worker_stats.values())
                    total_checked = sum(s['total_checked'] for s in worker_stats.values())

                    valid_rate = total_valid / worker_elapsed_min if worker_elapsed_min > 0 else 0
                    raw_rate = total_checked / worker_elapsed_min if worker_elapsed_min > 0 else 0
                    poc_w = int(total_valid * WEIGHT_SCALE_FACTOR)

                    print(f"[{int(worker_elapsed//60):02d}:{int(worker_elapsed%60):02d}] "
                          f"valid: {total_valid} | poc_weight: {poc_w} | "
                          f"valid/min: {valid_rate:.1f} | raw/min: {raw_rate:.1f}")

                    # Показываем статистику по каждому worker
                    for i in range(self.num_gpus):
                        wv = worker_stats[i]['total_valid']
                        wc = worker_stats[i]['total_checked']
                        wel = worker_stats[i].get('elapsed', worker_elapsed)
                        wel_min = wel / 60 if wel > 0 else 0

                        w_valid_rate = wv / wel_min if wel_min > 0 else 0
                        w_raw_rate = wc / wel_min if wel_min > 0 else 0
                        w_poc = int(wv * WEIGHT_SCALE_FACTOR)

                        gs = worker_stats[i].get('gpu_stats', {})
                        if gs and self.device_ids[i] in gs:
                            g = gs[self.device_ids[i]]
                            print(f"                     W{i}:{wv} ({w_valid_rate:.1f}/min {w_raw_rate:.0f}/min {w_poc}w) | "
                                  f"{g['vram_percent']}% VRAM {g['gpu_util']}% GPU {g['power_watts']}W")
                        else:
                            print(f"                     W{i}:{wv} ({w_valid_rate:.1f}/min {w_raw_rate:.0f}/min {w_poc}w)")

                    print(f"                     {datetime.now().strftime('%H:%M:%S')}")

                    last_report_time = current_time

        except KeyboardInterrupt:
            log_warning("\nБенчмарк прерван пользователем")

        # Останавливаем workers
        stop_event.set()
        for worker in workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()

        # Финальные результаты
        total_valid = sum(s['total_valid'] for s in worker_stats.values())
        total_checked = sum(s['total_checked'] for s in worker_stats.values())
        self.total_valid = total_valid
        self.total_checked = total_checked
        self.start_time = start_time

        # Используем worker elapsed для расчёта ставок (как в реальной ноде)
        worker_elapseds = [s.get('elapsed', 0) for s in worker_stats.values() if s.get('elapsed', 0) > 0]
        worker_elapsed_min = max(worker_elapseds) / 60 if worker_elapseds else None

        results = self.print_results(worker_elapsed_min=worker_elapsed_min)

        # Собираем информацию о GPU
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            gpu_info = {
                "name": torch.cuda.get_device_name(0),
                "total_memory_gb": round(gpu_props.total_memory / (1024**3), 2),
                "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
                "multi_processor_count": gpu_props.multi_processor_count,
                "num_gpus": self.num_gpus,
            }
            try:
                import subprocess
                driver_version = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    stderr=subprocess.DEVNULL
                ).decode().strip()
                gpu_info["driver_version"] = driver_version
            except:
                pass

        # Добавляем полную информацию о результатах
        results.update({
            "timestamp": datetime.now().isoformat(),
            "gpu": gpu_info,
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
            "batch_size": self.batch_size,
            "device": str(self.device),
            "poc_params": self.poc_params,
            "r_target": self.r_target,
            "weight_scale_factor": WEIGHT_SCALE_FACTOR,
        })

        return results

    def save_results(self, results: dict, output_dir: str = "results"):
        """Сохраняет результаты в файл"""
        if results is None:
            return None

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Имя файла с датой и poc_weight
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        gpu_name = results.get("gpu", {}).get("name", "Unknown").replace(" ", "_").replace("/", "_")
        filename = f"{timestamp}_{gpu_name}_{results['poc_weight']}.json"
        filepath = output_path / filename

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        log_success(f"Результат сохранён: {filepath}")
        return filepath


# ============ ГЛАВНАЯ ФУНКЦИЯ ============
def main():
    parser = argparse.ArgumentParser(
        description="Gonka PoW Benchmark - тест производительности GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python3 gonka_benchmark.py              # 5 минут (по умолчанию)
  python3 gonka_benchmark.py --duration 3 # 3 минуты
  python3 gonka_benchmark.py --device cuda:1
  python3 gonka_benchmark.py --batch-size 256
  python3 gonka_benchmark.py --offline    # Без запроса параметров из сети

Формула: poc_weight = valid_nonces × 2.5
        """
    )

    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=DEFAULT_DURATION_MIN,
        help=f"Время теста в минутах (по умолчанию: {DEFAULT_DURATION_MIN})"
    )

    parser.add_argument(
        "--gonka-path", "-g",
        type=str,
        default=DEFAULT_GONKA_PATH,
        help=f"Путь к коду gonka (по умолчанию: {DEFAULT_GONKA_PATH})"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="GPU устройство (по умолчанию: cuda:0, игнорируется если указан --num-gpus)"
    )

    parser.add_argument(
        "--num-gpus", "-n",
        type=int,
        default=None,
        help="Количество GPU для использования (по умолчанию: все доступные)"
    )

    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=None,
        help="Batch size (по умолчанию: автоподбор)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results",
        help="Директория для результатов (по умолчанию: results)"
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Не сохранять результаты в файл"
    )

    parser.add_argument(
        "--save-nonces",
        action="store_true",
        help="Сохранять все найденные valid nonce для дедупликации"
    )
    
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Не запрашивать параметры из сети (использовать fallback)"
    )
    
    parser.add_argument(
        "--r-target",
        type=float,
        default=None,
        help=f"Переопределить RTarget (по умолчанию: из сети или {DEFAULT_R_TARGET})"
    )

    args = parser.parse_args()

    # Настройка путей к gonka
    if not setup_gonka_path(args.gonka_path):
        sys.exit(1)

    # Проверка GPU
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        log_error("CUDA недоступна, используя CPU")
        args.device = "cpu"

    # Создаём и запускаем бенчмарк
    benchmark = GonkaBenchmark(
        device=args.device,
        duration_min=args.duration,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
        r_target=args.r_target,
        use_network_params=not args.offline,
    )
    benchmark.save_nonces = args.save_nonces

    results = benchmark.run()

    if results:
        if not args.no_save:
            filepath = benchmark.save_results(results, args.output)

            # Дополнительно сохраняем уникальные nonce если включено
            if args.save_nonces and len(benchmark.all_valid_nonces) > 0:
                import csv
                nonce_file = filepath.parent / filepath.name.replace(".json", "_nonces.csv")
                unique_nonces = set(benchmark.all_valid_nonces)
                with open(nonce_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["nonce"])
                    for nonce in sorted(unique_nonces):
                        writer.writerow([nonce])
                log_success(f"Уникальные nonce сохранены: {nonce_file} ({len(unique_nonces)} шт)")

        log_success("Бенчмарк завершён!")
        return 0
    else:
        log_error("Бенчмарк завершился с ошибкой")
        return 1


if __name__ == "__main__":
    sys.exit(main())

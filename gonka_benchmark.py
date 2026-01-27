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
from typing import List, Tuple, Dict, Any

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

# PoC параметры v2 (из inference-chain/x/inference/types/params.go DefaultPoCModelParams)
POC_PARAMS = {
    "dim": 1792,
    "n_layers": 64,
    "n_heads": 64,
    "n_kv_heads": 64,
    "vocab_size": 8196,
    "ffn_dim_multiplier": 10.0,
    "multiple_of": 4 * 2048,
    "norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "use_scaled_rope": False,
    "seq_len": 256,
}

# RTarget из chain params (inference-chain/x/inference/types/params.go DefaultPoCModelParams)
R_TARGET = 1.398077

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

    def run(self):
        """Основной цикл worker'а"""
        try:
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

                # Отправляем результат в очередь
                self.result_queue.put({
                    'worker_id': self.worker_id,
                    'total_valid': self.total_valid,
                    'total_checked': self.total_checked,
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
        r_target: float = R_TARGET,
        num_gpus: int = None,
    ):
        # Поддержка нескольких GPU
        if num_gpus is None:
            num_gpus = torch.cuda.device_count()

        self.num_gpus = min(num_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 0
        self.device_ids = list(range(self.num_gpus)) if self.num_gpus > 0 else [0]
        self.device = torch.device(f"cuda:{self.device_ids[0]}") if self.num_gpus > 0 else torch.device("cpu")

        self.duration_sec = duration_min * 60
        self.batch_size = batch_size
        self.r_target = r_target

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
        print("║" + " " * 15 + "Gonka PoW Benchmark v1.1" + " " * 21 + "║")
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

    def print_results(self):
        """Выводит финальные результаты с дедупликацией"""
        duration_min = (time.time() - self.start_time) / 60
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
        print(f"║  {Colors.CYAN}1 in N:{Colors.END}          {one_in_n:<41.0f}║")
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
        params = Params(**POC_PARAMS)
        params_str = f"Params(dim={params.dim}, n_layers={params.n_layers}, n_heads={params.n_heads}, n_kv_heads={params.n_kv_heads}, vocab_size={params.vocab_size}, ffn_dim_multiplier={params.ffn_dim_multiplier}, multiple_of={params.multiple_of}, norm_eps={params.norm_eps}, rope_theta={params.rope_theta}, use_scaled_rope={params.use_scaled_rope}, seq_len={params.seq_len})"
        log_info(f"params={params_str}")

        # Определяем batch_size
        gpu_group = GpuGroup(devices=self.device_ids)
        self.batch_size = get_batch_size_for_gpu_group(gpu_group, params)
        log_info(f"Using batch size: {self.batch_size} for {self.num_gpus}xGPU group {self.device_ids}")

        log_info(f"Запуск бенчмарка на {self.duration_sec / 60:.1f} минут с {self.num_gpus}xGPU...")
        print()

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
        start_time = time.time()
        for worker in workers:
            worker.start()

        # Собираем результаты и показываем прогресс
        worker_stats = {i: {'total_valid': 0, 'total_checked': 0} for i in range(self.num_gpus)}
        last_report_time = start_time
        report_interval = 30

        try:
            while time.time() - start_time < self.duration_sec:
                # Получаем результаты из очереди с таймаутом
                try:
                    result = result_queue.get(timeout=report_interval)

                    if 'error' in result:
                        log_error(f"Worker {result['worker_id']}: {result['error']}")
                        stop_event.set()
                        break

                    worker_id = result['worker_id']
                    worker_stats[worker_id]['total_valid'] = result['total_valid']
                    worker_stats[worker_id]['total_checked'] = result['total_checked']

                except:
                    pass  # Таймаут - нормально, продолжаем

                # Прогресс каждые report_interval секунд
                current_time = time.time()
                if current_time - last_report_time >= report_interval:
                    elapsed = current_time - start_time
                    elapsed_min = elapsed / 60

                    total_valid = sum(s['total_valid'] for s in worker_stats.values())
                    total_checked = sum(s['total_checked'] for s in worker_stats.values())

                    valid_rate = total_valid / elapsed_min if elapsed_min > 0 else 0
                    raw_rate = total_checked / elapsed_min if elapsed_min > 0 else 0
                    one_in = total_checked / total_valid if total_valid > 0 else 0
                    poc_w = int(total_valid * WEIGHT_SCALE_FACTOR)

                    print(f"[{int(elapsed//60):02d}:{int(elapsed%60):02d}] "
                          f"valid: {total_valid} | poc_weight: {poc_w} | "
                          f"1 in {one_in:.0f} | valid/min: {valid_rate:.1f} | raw/min: {raw_rate:.1f}")

                    # Показываем статистику по каждому worker
                    worker_str = " | ".join([f"W{i}:{worker_stats[i]['total_valid']}" for i in range(self.num_gpus)])
                    print(f"                     {worker_str} | {datetime.now().strftime('%H:%M:%S')}")

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

        results = self.print_results()

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
            "poc_params": POC_PARAMS,
            "r_target": R_TARGET,
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

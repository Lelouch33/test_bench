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
from datetime import datetime
import json
import argparse
import hashlib
from pathlib import Path
from typing import List, Tuple

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
        """Запускает бенчмарк"""
        self.print_header()

        # Импорт модулей gonka
        Compute, Params, get_target, get_batch_size_for_gpu_group, GpuGroup = import_gonka_modules()
        if Compute is None:
            return None

        # Создаём параметры модели
        params = Params(**POC_PARAMS)
        params_str = f"Params(dim={params.dim}, n_layers={params.n_layers}, n_heads={params.n_heads}, n_kv_heads={params.n_kv_heads}, vocab_size={params.vocab_size}, ffn_dim_multiplier={params.ffn_dim_multiplier}, multiple_of={params.multiple_of}, norm_eps={params.norm_eps}, rope_theta={params.rope_theta}, use_scaled_rope={params.use_scaled_rope}, seq_len={params.seq_len})"
        log_info(f"params={params_str}")

        # Сначала определяем batch_size (ДО загрузки модели, как в реальной ноде)
        gpu_group = GpuGroup(devices=self.device_ids)
        self.batch_size = get_batch_size_for_gpu_group(gpu_group, params)
        log_info(f"Using batch size: {self.batch_size} for {self.num_gpus}xGPU group {self.device_ids}")

        try:
            # Инициализируем Compute ПОСЛЕ определения batch_size
            log_info("Инициализация модели...")
            model_start = time.time()

            # Формируем список устройств для multi-GPU
            devices_list = [f"cuda:{i}" for i in self.device_ids]

            self.compute = Compute(
                params=params,
                block_hash=self.block_hash,
                block_height=self.block_height,
                public_key=self.public_key,
                r_target=self.r_target,
                devices=devices_list,
                node_id=0,
            )
            self.target = self.compute.target

            model_load_time = time.time() - model_start
            log_success(f"Модель загружена за {model_load_time:.1f}s")

        except Exception as e:
            log_error(f"Ошибка инициализации модели: {e}")
            import traceback
            traceback.print_exc()
            return None

        # Запуск бенчмарка
        log_info(f"Запуск бенчмарка на {self.duration_sec / 60:.1f} минут...")

        self.start_time = time.time()
        end_time = self.start_time + self.duration_sec
        next_nonce = 0
        last_report_time = self.start_time
        report_interval = 30  # Отчёт каждые 30 секунд

        print()

        # Функция для получения GPU статистики через pynvml (multi-GPU)
        def get_gpu_stats():
            if not torch.cuda.is_available():
                return None
            try:
                import pynvml
                pynvml.nvmlInit()

                total_mem_used = 0
                total_mem_total = 0
                total_gpu_util = 0
                total_power = 0

                for device_id in self.device_ids:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

                    # Память в байтах
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    total_mem_used += mem_info.used
                    total_mem_total += mem_info.total

                    # GPU утилизация
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    total_gpu_util += util.gpu

                    # Потребление энергии
                    total_power += pynvml.nvmlDeviceGetPowerUsage(handle) / 1000

                pynvml.nvmlShutdown()

                mem_used_gb = total_mem_used // (1024 ** 3)
                mem_total_gb = total_mem_total // (1024 ** 3)
                mem_percent = (total_mem_used / total_mem_total) * 100
                avg_gpu_util = total_gpu_util / len(self.device_ids)

                return {
                    'mem_used_gb': mem_used_gb,
                    'mem_total_gb': mem_total_gb,
                    'mem_percent': mem_percent,
                    'gpu_util': avg_gpu_util,
                    'power_w': total_power
                }
            except Exception as e:
                # Fallback на nvidia-smi если pynvml не сработал
                try:
                    import subprocess
                    result = subprocess.run([
                        'nvidia-smi',
                        '--query-gpu=memory.used,memory.total,utilization.gpu,power.draw',
                        '--format=csv,noheader,nounits',
                    ], capture_output=True, text=True, timeout=2)
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        if len(lines) >= 1:
                            values = lines[0].split(', ')
                            if len(values) >= 4:
                                mem_used = int(values[0])
                                mem_total = int(values[1])
                                gpu_util = int(values[2])
                                power = float(values[3])
                                mem_percent = (mem_used / mem_total) * 100
                                return {
                                    'mem_used_mb': mem_used // 1024,
                                    'mem_total_mb': mem_total // 1024,
                                    'mem_percent': mem_percent,
                                    'gpu_util': gpu_util,
                                    'power_w': power
                                }
                except:
                    pass
            return None

        try:
            with torch.no_grad():
                while time.time() < end_time:
                    # Генерируем batch nonce
                    nonces = list(range(next_nonce, next_nonce + self.batch_size))
                    next_nonce += self.batch_size

                    # Выполняем вычисления
                    future = self.compute(
                        nonces=nonces,
                        public_key=self.public_key,
                        target=self.target,
                    )

                    # Получаем результат
                    proof_batch = future.result()

                    # Фильтруем по r_target
                    valid_batch = proof_batch.sub_batch(self.r_target)

                    # Обновляем статистику
                    self.total_checked += len(proof_batch.nonces)
                    self.total_valid += len(valid_batch.nonces)

                    # Сохраняем valid nonce для дедупликации
                    if self.save_nonces and len(valid_batch.nonces) > 0:
                        self.all_valid_nonces.extend(valid_batch.nonces)

                    # Прогресс-бар каждые 30 секунд с GPU статистикой
                    current_time = time.time()
                    if current_time - last_report_time >= report_interval:
                        elapsed = current_time - self.start_time
                        elapsed_min = elapsed / 60
                        valid_rate = self.total_valid / elapsed_min if elapsed_min > 0 else 0
                        raw_rate = self.total_checked / elapsed_min if elapsed_min > 0 else 0
                        one_in = self.total_checked / self.total_valid if self.total_valid > 0 else 0
                        poc_w = int(self.total_valid * WEIGHT_SCALE_FACTOR)

                        gpu_stats = get_gpu_stats()
                        if gpu_stats:
                            gpu_line = (f"VRAM: {gpu_stats['mem_used_mb']}GB/{gpu_stats['mem_total_mb']}GB "
                                         f"({gpu_stats['mem_percent']:.1f}%) | "
                                         f"GPU: {gpu_stats['gpu_util']}% | "
                                         f"PWR: {gpu_stats['power_w']:.0f}W")
                        else:
                            gpu_line = "GPU: N/A"

                        print(f"[{int(elapsed//60):02d}:{int(elapsed%60):02d}] "
                              f"valid: {self.total_valid} | poc_weight: {poc_w} | "
                              f"1 in {one_in:.0f} | valid/min: {valid_rate:.1f} | raw/min: {raw_rate:.1f}")
                        print(f"                     {gpu_line} | {datetime.now().strftime('%H:%M:%S')}")

                        last_report_time = current_time

        except KeyboardInterrupt:
            log_warning("\nБенчмарк прерван пользователем")
        except Exception as e:
            log_error(f"Ошибка во время бенчмарка: {e}")
            import traceback
            traceback.print_exc()
            return None

        # Финальные результаты
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
            }
            # NVIDIA Driver version
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
            "model_load_time_s": round(model_load_time, 2),
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

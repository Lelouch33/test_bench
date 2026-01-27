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
import json
import argparse
import hashlib
from datetime import datetime
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
        gonka_path / "common" / "src",
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
        return Compute, Params, get_target
    except ImportError as e:
        log_error(f"Не удалось импортировать модули gonka: {e}")
        log_info("Убедитесь что скачали код gonka с нужными модулями")
        return None, None, None


# ============ КЛАСС БЕНЧМАРКА ============
class GonkaBenchmark:
    """Бенчмарк для Gonka PoW"""

    def __init__(
        self,
        device: str = "cuda:0",
        duration_min: float = DEFAULT_DURATION_MIN,
        batch_size: int = None,
        r_target: float = R_TARGET,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
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
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        cuda_version = torch.version.cuda or "N/A"

        print(f"\n{Colors.BOLD}{Colors.CYAN}", end="")
        print("╔" + "═" * 60 + "╗")
        print("║" + " " * 15 + "Gonka PoW Benchmark v1.1" + " " * 19 + "║")
        print("╠" + "═" * 60 + "╣")
        print(f"║  GPU: {gpu_name:<52}║")
        print(f"║  CUDA: {cuda_version:<51}║")
        print(f"║  Test duration: {int(self.duration_sec // 60)} minutes{'':<34}║")
        print(f"║  RTarget: {self.r_target:<48}║")
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

    def auto_detect_batch_size(self) -> int:
        """Автоматически определяет оптимальный batch_size"""
        log_info("Автоподбор batch_size...")

        # Получаем информацию о GPU
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            log_info(f"GPU память: {gpu_memory_gb:.1f} GB")

            # Эвристика: ~0.5GB на batch для PoC V2 модели
            # Но с учётом overhead модели (~30-40GB)
            estimated_batch = int((gpu_memory_gb - 40) * 2)
            estimated_batch = max(32, min(estimated_batch, 512))

            # Округляем до "красивых" значений
            for nice in [64, 128, 192, 256, 320, 384, 448, 512]:
                if estimated_batch <= nice:
                    estimated_batch = nice
                    break

            log_info(f"Расчётный batch_size: {estimated_batch}")
            return estimated_batch
        else:
            return 32  # Для CPU маленький batch

    def run(self) -> dict:
        """Запускает бенчмарк"""
        self.print_header()

        # Импорт модулей gonka
        Compute, Params, get_target = import_gonka_modules()
        if Compute is None:
            return None

        # Определяем batch_size
        if self.batch_size is None:
            self.batch_size = self.auto_detect_batch_size()

        # Создаём параметры модели
        params = Params(**POC_PARAMS)
        log_info(f"Параметры PoC V2: dim={params.dim}, layers={params.n_layers}, seq_len={params.seq_len}")
        log_info(f"Используемый batch_size: {self.batch_size}")

        try:
            # Инициализируем Compute
            log_info("Инициализация модели...")
            model_start = time.time()

            self.compute = Compute(
                params=params,
                block_hash=self.block_hash,
                block_height=self.block_height,
                public_key=self.public_key,
                r_target=self.r_target,
                devices=[str(self.device)],
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

                    # Прогресс-бар каждые 30 секунд
                    current_time = time.time()
                    if current_time - last_report_time >= report_interval:
                        elapsed = current_time - self.start_time
                        elapsed_min = elapsed / 60
                        valid_rate = self.total_valid / elapsed_min if elapsed_min > 0 else 0
                        raw_rate = self.total_checked / elapsed_min if elapsed_min > 0 else 0
                        one_in = self.total_checked / self.total_valid if self.total_valid > 0 else 0
                        poc_w = int(self.total_valid * WEIGHT_SCALE_FACTOR)

                        print(f"[{int(elapsed//60):02d}:{int(elapsed%60):02d}] "
                              f"valid: {self.total_valid} | poc_weight: {poc_w} | "
                              f"1 in {one_in:.0f} | valid/min: {valid_rate:.1f} | raw/min: {raw_rate:.1f}")

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
        help="GPU устройство (по умолчанию: cuda:0)"
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
    )
    benchmark.save_nonces = args.save_nonces

    results = benchmark.run()

    if results:
        if not args.no_save:
            filepath = benchmark.save_results(results, args.output)

            # Дополнительно сохраняем уникальные nonce если включено
            if args.save_nononces and len(benchmark.all_valid_nonces) > 0:
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

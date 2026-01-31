#!/usr/bin/env python3
"""
Анализ результатов бенчмарка Gonka PoW

Показывает результаты в удобном формате и может сравнивать запуски

Формула: poc_weight = valid_nonces × 2.5
(из inference-chain/x/inference/module/chainvalidation.go)
"""

import json
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Коэффициент из исходного кода gonka
WEIGHT_SCALE_FACTOR = 2.5


# Цвета для вывода
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def load_result(filepath: str) -> Dict[str, Any]:
    """Загружает результат из JSON файла"""
    with open(filepath, 'r') as f:
        return json.load(f)


def find_results(results_dir: str = "results") -> List[Path]:
    """Ищет все JSON файлы с результатами"""
    path = Path(results_dir)
    if not path.exists():
        return []
    return sorted(path.glob("*.json"))


def format_number(num: float, decimals: int = 2) -> str:
    """Форматирует число с разделителями"""
    return f"{num:,.{decimals}f}".replace(",", " ")


def get_poc_weight(result: Dict[str, Any]) -> int:
    """Извлекает или вычисляет poc_weight"""
    # Новый формат
    if "poc_weight" in result:
        return result["poc_weight"]
    # Старый формат с estimated_poc_weight
    if "estimated_poc_weight" in result:
        return result["estimated_poc_weight"]
    # Вычисляем из valid_nonces
    valid_nonces = result.get("valid_nonces", result.get("comp_power", 0))
    return int(valid_nonces * WEIGHT_SCALE_FACTOR)


def get_mode(result: Dict[str, Any]) -> str:
    """Определяет режим бенчмарка (v1 или v2)"""
    return result.get("mode", "v1")


def get_mode_label(result: Dict[str, Any]) -> str:
    """Возвращает цветной лейбл режима"""
    mode = get_mode(result)
    if mode == "v2":
        return f"{Colors.YELLOW}[V2]{Colors.END}"
    return f"{Colors.BLUE}[V1]{Colors.END}"


def print_result(result: Dict[str, Any], index: int = None):
    """Выводит информацию о результате"""
    prefix = f"{Colors.BOLD}[{index}]{Colors.END} " if index is not None else ""
    mode_label = get_mode_label(result)
    mode = get_mode(result)

    # Основные метрики
    valid_nonces = result.get("valid_nonces", result.get("total_nonces", result.get("comp_power", 0)))
    poc_weight = get_poc_weight(result)
    valid_per_min = result.get("valid_per_min", result.get("nonces_per_min", 0))
    raw_per_min = result.get("raw_per_min", 0)
    one_in_n = result.get("one_in_n", 0)
    duration = result.get("duration_min", 0)

    # GPU инфо
    gpu = result.get("gpu", {})
    gpu_name = gpu.get("name", "Unknown")
    gpu_memory = gpu.get("total_memory_gb", "N/A")
    driver = gpu.get("driver_version", "N/A")

    # Время
    timestamp = result.get("timestamp", "")
    try:
        dt = datetime.fromisoformat(timestamp)
        date_str = dt.strftime("%Y-%m-%d %H:%M")
    except:
        date_str = timestamp[:16] if timestamp else "Unknown"

    print(f"{prefix}{mode_label} {Colors.CYAN}{date_str}{Colors.END}")
    print(f"  GPU: {gpu_name} ({gpu_memory}GB)")
    print(f"  Driver: {driver} | CUDA: {result.get('cuda_version', 'N/A')}")
    print(f"  Batch size: {result.get('batch_size', 'N/A')}")

    if mode == "v2":
        print(f"  Model: {result.get('model_id', 'N/A')}")
        print(f"  seq_len: {result.get('seq_len', 'N/A')} | k_dim: {result.get('k_dim', 'N/A')}")

    # Основная таблица
    print(f"  \u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510")
    print(f"  \u2502 {Colors.CYAN}valid_nonces:{Colors.END}     {valid_nonces:<10}                    \u2502")
    print(f"  \u2502 {Colors.GREEN}poc_weight:{Colors.END}       {poc_weight:<10}                    \u2502")
    print(f"  \u2502 {Colors.CYAN}valid/min:{Colors.END}        {format_number(valid_per_min):<10}                    \u2502")
    if mode == "v1":
        print(f"  \u2502 {Colors.CYAN}raw/min:{Colors.END}          {format_number(raw_per_min):<10}                    \u2502")
        print(f"  \u2502 {Colors.CYAN}1 in N:{Colors.END}           {one_in_n:<10.0f}                    \u2502")
    else:
        peak = result.get("peak_rate_per_min", 0)
        print(f"  \u2502 {Colors.CYAN}peak/min:{Colors.END}         {format_number(peak):<10}                    \u2502")
    print(f"  \u2502 {Colors.CYAN}duration:{Colors.END}         {duration:.2f} min                       \u2502")
    print(f"  \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518")


def print_comparison_table(results: List[Dict[str, Any]]):
    """Выводит таблицу сравнения результатов"""
    if len(results) < 2:
        return

    print(f"\n{Colors.BOLD}Сравнение запусков:{Colors.END}")
    print(f"{'Mode':<6} {'Дата':<16} {'GPU':<20} {'valid':<8} {'poc_w':<8} {'valid/min':<12} {'raw/min':<12}")
    print("\u2500" * 88)

    for r in results:
        mode = get_mode(r).upper()
        timestamp = r.get("timestamp", "")[:16]
        gpu = r.get("gpu", {}).get("name", "Unknown")[:20]
        vn = r.get("valid_nonces", r.get("total_nonces", r.get("comp_power", 0)))
        pw = get_poc_weight(r)
        vpm = r.get("valid_per_min", r.get("nonces_per_min", 0))
        rpm = r.get("raw_per_min", r.get("peak_rate_per_min", 0))

        print(f"{mode:<6} {timestamp:<16} {gpu:<20} {vn:<8} {pw:<8} {format_number(vpm):<12} {format_number(rpm):<12}")

    # Лучший по valid_per_min
    best = max(results, key=lambda x: x.get("valid_per_min", x.get("nonces_per_min", 0)))
    print(f"\n{Colors.GREEN}Лучший результат:{Colors.END}")
    print_result(best)


def print_full_info(result: Dict[str, Any]):
    """Выводит полную информацию о результате"""
    print(f"\n{Colors.BOLD}Полная информация:{Colors.END}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(
        description="Анализ результатов Gonka PoW Benchmark",
        epilog="""
Примеры:
  python3 analyze_results.py              # Все результаты из results/
  python3 analyze_results.py --latest     # Только последний
  python3 analyze_results.py --full       # С полной информацией
  python3 analyze_results.py results/*.json  # Конкретные файлы

Формула: poc_weight = valid_nonces × 2.5
        """
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Файлы с результатами (по умолчанию: все из results/)"
    )
    parser.add_argument(
        "--dir", "-d",
        default="results",
        help="Директория с результатами (по умолчанию: results)"
    )
    parser.add_argument(
        "--latest", "-l",
        action="store_true",
        help="Показать только последний результат"
    )
    parser.add_argument(
        "--full", "-f",
        action="store_true",
        help="Показать полную информацию (JSON)"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["v1", "v2", "all"],
        default="all",
        help="Фильтр по режиму: v1, v2 или all (по умолчанию: all)"
    )

    args = parser.parse_args()

    # Загружаем результаты
    results = []
    result_files = []

    if args.files:
        for f in args.files:
            try:
                result_files.append(Path(f))
            except:
                pass
    else:
        result_files = find_results(args.dir)

    if not result_files:
        print(f"{Colors.YELLOW}Нет файлов с результатами в {args.dir}/{Colors.END}")
        return 1

    for f in result_files:
        try:
            results.append(load_result(f))
        except Exception as e:
            print(f"{Colors.RED}Ошибка чтения {f}: {e}{Colors.END}")

    if not results:
        print(f"{Colors.RED}Не удалось загрузить результаты{Colors.END}")
        return 1

    # Фильтруем по режиму
    if args.mode != "all":
        results = [r for r in results if get_mode(r) == args.mode]

    if not results:
        print(f"{Colors.YELLOW}Нет результатов для режима {args.mode}{Colors.END}")
        return 1

    # Сортируем по времени
    results.sort(key=lambda x: x.get("timestamp", ""))

    print(f"\n{Colors.BOLD}{Colors.CYAN}", end="")
    print("╔" + "═" * 60 + "╗")
    print("║" + " " * 15 + "Gonka Benchmark Results" + " " * 22 + "║")
    print("╚" + "═" * 60 + "╝")
    print(f"{Colors.END}")

    # Только последний
    if args.latest:
        print_result(results[-1])
        if args.full:
            print_full_info(results[-1])
        return 0

    # Все результаты
    print(f"\n{Colors.BOLD}Найдено результатов: {len(results)}{Colors.END}\n")

    for i, r in enumerate(results, 1):
        print_result(r, i)
        print()

    # Сравнение
    if len(results) > 1:
        print_comparison_table(results)

    # Полная информация для последнего
    if args.full:
        print_full_info(results[-1])

    print(f"\n{Colors.YELLOW}Формула:{Colors.END} poc_weight = valid_nonces × {WEIGHT_SCALE_FACTOR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

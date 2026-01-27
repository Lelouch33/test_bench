#!/usr/bin/env python3
"""
Дедупликация и объединение nonce из нескольких результатов

Использование:
  python3 deduplicate_nonces.py results/*.json
  python3 deduplicate_nonces.py results/*.json --output merged_results.json
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Set, List, Dict, Any

# Коэффициент из исходного кода gonka
WEIGHT_SCALE_FACTOR = 2.5


def load_results(files: List[Path]) -> List[Dict[str, Any]]:
    """Загружает результаты из JSON файлов"""
    results = []
    for f in files:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                data['_source_file'] = str(f)
                results.append(data)
        except Exception as e:
            print(f"⚠ Ошибка чтения {f}: {e}")
    return results


def merge_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Объединяет результаты и удаляет дубликаты"""

    all_nonces: Set[int] = set()
    total_raw_checked = 0
    sources = []

    for r in results:
        # Если есть сохранённые nonce - используем их
        if 'all_valid_nonces' in r:
            all_nonces.update(r['all_valid_nonces'])
        # Иначе пробуем загрузить из CSV файла
        else:
            nonce_file = Path(r['_source_file']).parent / Path(r['_source_file']).stem.replace("_nonces", "") + "_nonces.csv"
            if nonce_file.exists():
                import csv
                with open(nonce_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        all_nonces.add(int(row['nonce']))

        total_raw_checked += r.get('total_checked', 0)
        sources.append({
            'file': Path(r['_source_file']).name,
            'valid_nonces': r.get('valid_nonces', 0),
            'poc_weight': r.get('poc_weight', 0),
            'timestamp': r.get('timestamp', ''),
        })

    unique_count = len(all_nonces)
    duplicates = sum(r.get('valid_nonces', 0) for r in results) - unique_count

    # Вычисляем aggregated poc_weight
    unique_poc_weight = int(unique_count * WEIGHT_SCALE_FACTOR)

    merged = {
        'merged_from': len(results),
        'sources': sources,
        'unique_valid_nonces': unique_count,
        'unique_poc_weight': unique_poc_weight,
        'total_raw_checked': total_raw_checked,
        'duplicates_removed': duplicates,
    }

    return merged


def print_merged(merged: Dict[str, Any]):
    """Выводит результаты объединения"""

    print("\n╔════════════════════════════════════════════════════════════════╗")
    print("║                    ДЕДУПЛИКАЦИЯ NONCE                         ║")
    print("╠════════════════════════════════════════════════════════════════╣")

    print(f"║  Объединено файлов:    {merged['merged_from']:<38}║")
    print(f"║  Уникальных nonce:     {merged['unique_valid_nonces']:<38}║")
    print(f"║  Дубликатов удалено:   {merged['duplicates_removed']:<38}║")
    print(f"║                                                           ║")
    print(f"║  {GREEN}unique_poc_weight:     {merged['unique_poc_weight']:<38}║{END}")
    print("╚════════════════════════════════════════════════════════════════╝")

    print("\n📊 Источники:")
    for s in merged['sources']:
        print(f"  • {s['file']}: valid={s['valid_nonces']}, weight={s['poc_weight']}")


GREEN = '\033[92m'
END = '\033[0m'


def main():
    parser = argparse.ArgumentParser(
        description="Дедупликация и объединение nonce из нескольких результатов"
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="JSON файлы с результатами"
    )
    parser.add_argument(
        "--output", "-o",
        help="Сохранить результат в файл"
    )

    args = parser.parse_args()

    files = [Path(f) for f in args.files]
    results = load_results(files)

    if not results:
        print("❌ Не удалось загрузить результаты")
        return 1

    merged = merge_results(results)
    print_merged(merged)

    print(f"\n📐 Формула: poc_weight = unique_valid_nonces × {WEIGHT_SCALE_FACTOR}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(merged, f, indent=2)
        print(f"\n✓ Результат сохранён: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

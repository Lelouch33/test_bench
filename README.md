# PoW Benchmark

Автономный бенчмарк для тестирования производительности GPU для Proof-of-Compute.

## Быстрый старт

```bash
# Клонирование и запуск (всё автоматически)
git clone https://github.com/Lelouch33/test_bench.git
cd test_bench
bash run.sh
```

## Формула расчёта веса

```
poc_weight = valid_nonces × 2.5
```

Источник: `inference-chain/x/inference/module/chainvalidation.go`

```go
var WeightScaleFactor = mathsdk.LegacyNewDecWithPrec(25, 1) // 2.5
weight = mathsdk.LegacyNewDec(weight).Mul(WeightScaleFactor).TruncateInt64()
```

## Быстрый старт (один скрипт)

```bash
# Скачать и запустить — всё автоматически
bash run.sh
```

Скрипт `run.sh` автоматически:
1. Проверяет GPU и определяет версию CUDA
2. Устанавливает Python 3.12 (если нет)
3. Устанавливает uv
4. Создаёт venv и ставит PyTorch под вашу CUDA
5. Скачивает код PoW модуля
6. Запускает бенчмарк

### С параметрами

```bash
bash run.sh --duration 3      # 3 минуты
bash run.sh --duration 10     # 10 минут  
bash run.sh --device cuda:1   # Другой GPU
bash run.sh --batch-size 256  # Ручной batch size
```

## Ручная установка

Если хотите установить вручную:

```bash
# 1. Установка зависимостей
bash setup.sh

# 2. Активация окружения
source .venv/bin/activate

# 3. Запуск
python3 gonka_benchmark.py
```

## Метрики

| Метрика | Описание |
|---------|----------|
| `valid_nonces` | Количество валидных nonce за тест |
| `poc_weight` | Вес в сети = valid_nonces × 2.5 |
| `valid/min` | Валидных nonce в минуту |
| `raw/min` | Всего проверок в минуту |
| `1 in N` | Соотношение (всего / валидные) |

## Параметры

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--duration, -d` | `5` | Время теста в минутах |
| `--device` | `cuda:0` | GPU устройство |
| `--batch-size, -b` | `auto` | Batch size |
| `--save-nonces` | `false` | Сохранять valid nonce для дедупликации |
| `--no-save` | `false` | Не сохранять результаты |

## Пример вывода

```
╔══════════════════════════════════════════════════════════════════╗
║            PoW Benchmark - All-in-One                            ║
╚══════════════════════════════════════════════════════════════════╝

═══ 1/6 Проверка GPU ═══
index, name, memory.total, driver_version
0, NVIDIA H200, 141120 MiB, 550.54
[✓] Найдено GPU: 1
[INFO] CUDA версия: 12.6

═══ 2/6 Python 3.12 ═══
[✓] Python 3.12: Python 3.12.8

═══ 3/6 Менеджер пакетов uv ═══
[✓] uv: uv 0.5.14

═══ 4/6 Виртуальное окружение и PyTorch ═══
[INFO] PyTorch: cu124
[✓] Зависимости установлены
[INFO] PyTorch версия: 2.5.1+cu124
[✓] GPU: NVIDIA H200

═══ 5/6 Код PoW модуля ═══
[✓] PoW модуль скачан

═══ 6/6 Запуск бенчмарка ═══

╔════════════════════════════════════════════════════════════════╗
║               PoW Benchmark v1.1                                ║
╠════════════════════════════════════════════════════════════════╣
║  GPU: NVIDIA H200                                              ║
║  CUDA: 12.6                                                    ║
║  Test duration: 5 minutes                                      ║
║  RTarget: 1.398077                                             ║
╚════════════════════════════════════════════════════════════════╝

[00:30] valid: 28 | poc_weight: 70 | 1 in 48 | valid/min: 56.0
[01:00] valid: 58 | poc_weight: 145 | 1 in 48 | valid/min: 58.0
...
[05:00] valid: 295 | poc_weight: 738 | 1 in 48 | valid/min: 59.0

╔════════════════════════════════════════════════════════════════╗
║                    BENCHMARK RESULTS                           ║
╠════════════════════════════════════════════════════════════════╣
║  valid_nonces:    295                                          ║
║  poc_weight:      738                                          ║
╠════════════════════════════════════════════════════════════════╣
║  valid/min:       59.00                                        ║
║  raw/min:         2832.00                                      ║
║  1 in N:          48                                           ║
╚════════════════════════════════════════════════════════════════╝

Формула: poc_weight = valid_nonces × 2.5

[✓] Готово! Результаты в results/
```

## Автоопределение PyTorch

Скрипт автоматически выбирает версию PyTorch по вашей CUDA:

| CUDA | PyTorch |
|------|---------|
| 13.0+ | cu130 |
| 12.8+ | cu128 |
| 12.4-12.7 | cu124 |
| 12.0-12.3 | cu121 |
| 11.x | cu118 |
| нет GPU | cpu |

## Структура проекта

```
pow-benchmark/
├── run.sh                 # Главный скрипт (всё в одном)
├── gonka_benchmark.py     # Бенчмарк
├── analyze_results.py     # Анализ результатов
├── deduplicate_nonces.py  # Дедупликация и объединение
├── setup.sh               # Только установка (без запуска)
├── README.md              # Документация
└── results/               # Результаты тестов (JSON)
```

## Анализ результатов

```bash
python3 analyze_results.py           # Все результаты
python3 analyze_results.py --latest  # Последний
python3 analyze_results.py --full    # С полным JSON
```

## Дедупликация nonce

При запуске нескольких узлов или перезапусках могут появляться дубликаты nonce.

**Сохранение nonce при бенчмарке:**
```bash
python3 gonka_benchmark.py --save-nonces
```

После завершения покажет:
```
⚠ Дедупликация:
   Всего найдено:    458
   Уникальных:       455
   Дубликаты:        3 (0.7%)
   poc_weight (уникальные): 1137
```

**Объединение нескольких результатов:**
```bash
python3 deduplicate_nonces.py results/*.json
python3 deduplicate_nonces.py results/*.json --output merged.json
```

## Troubleshooting

### OOM (Out of Memory)

```bash
bash run.sh --batch-size 128
```

### Проверка GPU

```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"
```

## Лицензия

MIT License

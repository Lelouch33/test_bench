# Gonka PoW Benchmark

Автономный бенчмарк для тестирования производительности GPU для Proof-of-Compute в сети Gonka.

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

## Возможности v1.2

- ✅ **Динамическое получение параметров** — r_target и другие параметры автоматически загружаются из сети Gonka
- ✅ **Отказоустойчивость** — пробует все genesis ноды пока не получит ответ
- ✅ **Информация о PoC фазе** — получение реальной длительности PoC из блокчейна
- ✅ **Multi-GPU поддержка** — автоматическое использование всех доступных GPU

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
bash run.sh --offline         # Без запроса параметров из сети
```

## Утилиты

### Получение параметров PoC из сети

```bash
# Показать текущие параметры (r_target, dim, n_layers и т.д.)
python3 fetch_poc_duration.py --params

# Показать информацию о PoC фазе (длительность, блоки)
python3 fetch_poc_duration.py

# JSON вывод
python3 fetch_poc_duration.py --json
```

Пример вывода:
```
╔══════════════════════════════════════════════════════════════╗
║                    PoC Parameters                            ║
╠══════════════════════════════════════════════════════════════╣
║  Model Parameters:                                           ║
║    r_target:            1.398077                             ║
║    dim:                 1792                                 ║
║    n_layers:            64                                   ║
...
╠══════════════════════════════════════════════════════════════╣
║  PoC Settings:                                               ║
║    weight_scale_factor: 2.5                                  ║
╚══════════════════════════════════════════════════════════════╝
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
| `--num-gpus, -n` | `all` | Количество GPU |
| `--offline` | `false` | Не запрашивать параметры из сети |
| `--r-target` | `auto` | Переопределить RTarget |
| `--save-nonces` | `false` | Сохранять valid nonce для дедупликации |
| `--no-save` | `false` | Не сохранять результаты |

## Пример вывода

```
╔══════════════════════════════════════════════════════════════════╗
║            Gonka PoW Benchmark - All-in-One v1.2                 ║
╚══════════════════════════════════════════════════════════════════╝

═══ 1/6 Проверка GPU ═══
index, name, memory.total, driver_version
0, NVIDIA H200, 141120 MiB, 550.54
[✓] Найдено GPU: 1
[INFO] CUDA версия: 12.6

...

[✓] Параметры получены из сети: http://node2.gonka.ai:8000
[INFO]   r_target: 1.398077

╔════════════════════════════════════════════════════════════════╗
║               Gonka PoW Benchmark v1.2                         ║
╠════════════════════════════════════════════════════════════════╣
║  GPU: NVIDIA H200                                              ║
║  CUDA: 12.6                                                    ║
║  Test duration: 5 minutes                                      ║
║  RTarget: 1.398077                                             ║
╚════════════════════════════════════════════════════════════════╝

[00:30] valid: 28 | poc_weight: 70 | valid/min: 56.0 | raw/min: 2688.0
[01:00] valid: 58 | poc_weight: 145 | valid/min: 58.0 | raw/min: 2784.0
...

╔════════════════════════════════════════════════════════════════╗
║                    BENCHMARK RESULTS                           ║
╠════════════════════════════════════════════════════════════════╣
║  valid_nonces:    295                                          ║
║  poc_weight:      738                                          ║
╠════════════════════════════════════════════════════════════════╣
║  valid/min:       59.00                                        ║
║  raw/min:         2832.00                                      ║
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

## Genesis Nodes

Скрипт использует следующие ноды для получения параметров:

- `http://node1.gonka.ai:8000`
- `http://node2.gonka.ai:8000`
- `http://node3.gonka.ai:8000`
- `https://node4.gonka.ai`
- `http://47.236.26.199:8000`
- `http://47.236.19.22:18000`
- `http://185.216.21.98:8000`
- `http://36.189.234.197:18026`
- `http://36.189.234.237:17241`
- `http://gonka.spv.re:8000`

## Структура проекта

```
pow-benchmark/
├── run.sh                    # Главный скрипт (всё в одном)
├── gonka_benchmark.py        # Бенчмарк
├── fetch_poc_duration.py     # Получение параметров из сети
├── benchmark_visualizer.py   # Визуализация результатов
├── analyze_results.py        # Анализ результатов
├── deduplicate_nonces.py     # Дедупликация и объединение
├── setup.sh                  # Только установка (без запуска)
├── README.md                 # Документация
└── results/                  # Результаты тестов (JSON)
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

### Проверка подключения к сети Gonka

```bash
python3 fetch_poc_duration.py --params
```

## Лицензия

MIT License

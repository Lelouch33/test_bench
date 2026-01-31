# Gonka PoW Benchmark

Автономный бенчмарк для тестирования производительности GPU для Proof-of-Compute в сети Gonka.

Поддерживает два режима:
- **V1** — локальные GPU вычисления через модуль `pow.compute`
- **V2** — cPoC через vLLM API (docker-контейнер с моделью Qwen3-235B)

## Быстрый старт

```bash
git clone https://github.com/Lelouch33/test_bench.git
cd test_bench
bash run.sh
```

При запуске появится интерактивное меню выбора режима:

```
╔══════════════════════════════════════════════════════════════════╗
║              Gonka Benchmark - Выбор режима                      ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║   1) PoC V1 — локальные GPU вычисления (pow.compute)            ║
║   2) PoC V2 — через vLLM API (cPoC, docker)                    ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

Выберите режим [1/2]:
```

Можно также указать режим через флаг: `bash run.sh --mode v1` или `bash run.sh --mode v2`.

## Формула расчёта веса

```
poc_weight = valid_nonces × 2.5
```

Источник: `inference-chain/x/inference/module/chainvalidation.go`

```go
var WeightScaleFactor = mathsdk.LegacyNewDecWithPrec(25, 1) // 2.5
weight = mathsdk.LegacyNewDec(weight).Mul(WeightScaleFactor).TruncateInt64()
```

## Режимы работы

### PoC V1 — локальные вычисления

Использует модуль `pow.compute` из gonka для расчёта PoW напрямую на GPU.

- Скачивает код gonka (sparse checkout)
- Загружает модель на каждый GPU
- Автоподбор batch size
- Multi-GPU через multiprocessing

```bash
bash run.sh --mode v1                # или просто выбрать 1 в меню
bash run.sh --mode v1 --duration 3   # 3 минуты
bash run.sh --mode v1 --num-gpus 2   # 2 GPU
```

### PoC V2 — cPoC через vLLM API

Использует docker-контейнер с кастомным vLLM (`gonka-ai/vllm`) и моделью `Qwen3-235B-A22B-Instruct-2507-FP8`. Бенчмарк отправляет запросы через HTTP API и измеряет скорость генерации артефактов.

**Требования:** Docker с поддержкой NVIDIA GPU (`nvidia-container-toolkit`)

```bash
bash run.sh --mode v2                # или выбрать 2 в меню
bash run.sh --mode v2 --duration 5   # 5 минут
```

Что делает V2:
1. Пуллит docker-образ `ghcr.io/gonka-ai/vllm:v0.9.1-poc-v2-post1-blackwell`
2. Запускает контейнер с `--gpus all`, `--enforce-eager`, `--tensor-parallel-size`
3. Ждёт health check (загрузка модели)
4. Прогревает — ждёт первый batch nonces
5. Запускает бенчмарк: `POST /api/v1/pow/init/generate`
6. Опрашивает `GET /api/v1/pow/status` каждые 5 секунд
7. По окончании останавливает генерацию и контейнер

V2 параметры (из сети Gonka):
- `model`: Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
- `seq_len`: 1024
- `k_dim`: 12
- `batch_size`: 32

## Возможности

- ✅ **Два режима бенчмарка** — V1 (локальный) и V2 (vLLM API)
- ✅ **Интерактивное меню** — выбор режима при запуске
- ✅ **Динамическое получение параметров** — r_target, model_id, seq_len автоматически из сети
- ✅ **Отказоустойчивость** — пробует все genesis ноды пока не получит ответ
- ✅ **Информация о PoC фазе** — реальная длительность из блокчейна
- ✅ **Multi-GPU поддержка** — V1: multiprocessing, V2: tensor parallelism
- ✅ **Автоочистка** — V2 автоматически останавливает docker-контейнер

## Быстрый старт (один скрипт)

```bash
bash run.sh
```

Скрипт `run.sh` автоматически:
1. Предлагает выбрать режим (V1/V2)
2. Проверяет GPU и определяет версию CUDA
3. Устанавливает Python 3.12 (если нет)
4. Устанавливает uv и создаёт venv с PyTorch
5. **V1:** скачивает код PoW модуля / **V2:** запускает docker-контейнер vLLM
6. Запускает бенчмарк

### С параметрами

```bash
# V1
bash run.sh --mode v1 --duration 3      # 3 минуты
bash run.sh --mode v1 --device cuda:1   # Другой GPU
bash run.sh --mode v1 --batch-size 256  # Ручной batch size

# V2
bash run.sh --mode v2 --duration 5      # 5 минут
bash run.sh --mode v2 --batch-size 64   # Другой batch size
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

| Метрика | V1 | V2 | Описание |
|---------|----|----|----------|
| `valid_nonces` | ✅ | ✅ | Количество nonces за тест |
| `poc_weight` | ✅ | ✅ | Вес в сети = nonces × 2.5 |
| `valid/min` | ✅ | ✅ | Nonces в минуту |
| `raw/min` | ✅ | - | Всего проверок в минуту |
| `peak/min` | - | ✅ | Пиковая скорость за тест |
| `1 in N` | ✅ | - | Соотношение (всего / валидные) |

## Параметры V1

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--duration, -d` | из сети / `5` | Время теста в минутах |
| `--device` | `cuda:0` | GPU устройство |
| `--batch-size, -b` | `auto` | Batch size |
| `--num-gpus, -n` | `all` | Количество GPU |
| `--offline` | `false` | Не запрашивать параметры из сети |
| `--r-target` | `auto` | Переопределить RTarget |
| `--save-nonces` | `false` | Сохранять valid nonce для дедупликации |
| `--no-save` | `false` | Не сохранять результаты |

## Параметры V2

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--duration, -d` | из сети / `5` | Время теста в минутах |
| `--vllm-url` | `http://localhost:5000` | URL vLLM сервера |
| `--vllm-port` | `5000` | Порт vLLM сервера |
| `--batch-size, -b` | `32` | Batch size |
| `--seq-len` | из сети / `1024` | Sequence length |
| `--k-dim` | `12` | K dimensions |
| `--model` | из сети | Model ID |
| `--offline` | `false` | Не запрашивать параметры из сети |
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
test_bench/
├── run.sh                    # Главный скрипт (V1/V2, интерактивное меню)
├── gonka_benchmark.py        # V1 бенчмарк (локальные GPU вычисления)
├── gonka_benchmark_v2.py     # V2 бенчмарк (cPoC через vLLM API)
├── fetch_poc_duration.py     # Получение параметров из сети
├── benchmark_visualizer.py   # Визуализация результатов
├── analyze_results.py        # Анализ результатов (V1/V2)
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

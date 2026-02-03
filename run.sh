#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# Gonka PoW Benchmark - All-in-One Script
#
# Один скрипт для всего:
# 1. Проверяет/устанавливает Python 3.12
# 2. Устанавливает uv
# 3. Создаёт venv и ставит зависимости (PyTorch под нужную CUDA)
# 4. Скачивает код gonka (V1), запускает docker (V2), или нативный vLLM (V3)
# 5. Запускает бенчмарк
#
# Использование:
#   bash run.sh                    # V1 бенчмарк (по умолчанию)
#   bash run.sh --mode v2          # V2 бенчмарк (cPoC через vLLM Docker)
#   bash run.sh --mode v3          # V3 бенчмарк (cPoC через нативный vLLM 0.14.0)
#   bash run.sh --duration 3       # 3 минуты
#   bash run.sh --help             # Справка
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════════
# Конфигурация
# ═══════════════════════════════════════════════════════════════════════════
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
GONKA_PATH="$SCRIPT_DIR/gonka"
RESULTS_DIR="$SCRIPT_DIR/results"

# V2 Docker конфигурация
VLLM_IMAGE_DEFAULT="ghcr.io/gonka-ai/vllm:v0.9.1-poc-v2-post1"
VLLM_IMAGE_BLACKWELL="ghcr.io/gonka-ai/vllm:v0.9.1-poc-v2-post1-blackwell"
VLLM_IMAGE=""  # определяется после детекции GPU
VLLM_CONTAINER_NAME="gonka-benchmark-vllm"
VLLM_PORT=5000
VLLM_MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"

# Парсим --mode из аргументов (если передан)
BENCH_MODE=""
BENCHMARK_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--mode" ]]; then
        _NEXT_IS_MODE=1
        continue
    fi
    if [[ "${_NEXT_IS_MODE:-}" == "1" ]]; then
        BENCH_MODE="$arg"
        _NEXT_IS_MODE=0
        continue
    fi
    BENCHMARK_ARGS+=("$arg")
done

# ═══════════════════════════════════════════════════════════════════════════
# Цвета
# ═══════════════════════════════════════════════════════════════════════════
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[⚠]${NC} $1"; }
log_error()   { echo -e "${RED}[✗]${NC} $1"; }
log_step()    { echo -e "\n${CYAN}${BOLD}═══ $1 ═══${NC}\n"; }

# ═══════════════════════════════════════════════════════════════════════════
# Интерактивное меню выбора режима (если --mode не передан)
# ═══════════════════════════════════════════════════════════════════════════
if [[ -z "$BENCH_MODE" ]]; then
    echo ""
    echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}${BOLD}║              Gonka Benchmark - Выбор режима                      ║${NC}"
    echo -e "${CYAN}${BOLD}╠══════════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║${NC}                                                                  ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${GREEN}1)${NC} PoC V1 — локальные GPU вычисления (pow.compute)             ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}      Использует модуль gonka для расчёта на GPU напрямую         ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}                                                                  ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${YELLOW}2)${NC} PoC V2 — через vLLM API (cPoC, docker)                      ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}      Запускает vLLM контейнер с моделью и тестирует через API    ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}                                                                  ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${BLUE}3)${NC} PoC V3 — нативный vLLM 0.14.0 + gonka_poc                   ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}      Устанавливает vLLM нативно (без Docker), тот же API        ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}                                                                  ${CYAN}║${NC}"
    echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    while true; do
        read -p "$(echo -e "${BOLD}Выберите режим [1/2/3]: ${NC}")" choice
        case "$choice" in
            1) BENCH_MODE="v1"; break ;;
            2) BENCH_MODE="v2"; break ;;
            3) BENCH_MODE="v3"; break ;;
            *) echo -e "${RED}Введите 1, 2 или 3${NC}" ;;
        esac
    done
    echo ""
fi

# ═══════════════════════════════════════════════════════════════════════════
# Заголовок
# ═══════════════════════════════════════════════════════════════════════════
echo ""
if [[ "$BENCH_MODE" == "v3" ]]; then
    echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}${BOLD}║         Gonka PoC V3 Benchmark - Native vLLM 0.14.0              ║${NC}"
    echo -e "${CYAN}${BOLD}║         Mode: V3 (cPoC via native vLLM + gonka_poc)              ║${NC}"
    echo -e "${CYAN}${BOLD}║  Формула: poc_weight = total_nonces × WeightScaleFactor           ║${NC}"
    echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════════════════════╝${NC}"
elif [[ "$BENCH_MODE" == "v2" ]]; then
    echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}${BOLD}║         Gonka PoC V2 Benchmark - All-in-One v1.0                 ║${NC}"
    echo -e "${CYAN}${BOLD}║         Mode: V2 (cPoC via vLLM API)                             ║${NC}"
    echo -e "${CYAN}${BOLD}║  Формула: poc_weight = total_nonces × WeightScaleFactor           ║${NC}"
    echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════════════════════╝${NC}"
else
    echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}${BOLD}║            Gonka PoW Benchmark - All-in-One v1.2                 ║${NC}"
    echo -e "${CYAN}${BOLD}║            Mode: V1 (local GPU compute)                          ║${NC}"
    echo -e "${CYAN}${BOLD}║  Формула: poc_weight = valid_nonces × 2.5                        ║${NC}"
    echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════════════════════╝${NC}"
fi
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# 1. Проверка GPU и CUDA
# ═══════════════════════════════════════════════════════════════════════════
log_step "1/6 Проверка GPU"

CUDA_VERSION=""
CUDA_MAJOR=""

if command -v nvidia-smi &> /dev/null; then
    echo ""
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
    echo ""
    
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    log_success "Найдено GPU: $GPU_COUNT"
    
    CUDA_FROM_SMI=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+") || true
    
    if [[ -n "$CUDA_FROM_SMI" ]]; then
        CUDA_VERSION="$CUDA_FROM_SMI"
        CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d'.' -f1)
        log_info "CUDA версия: $CUDA_VERSION"
    fi
else
    log_warning "nvidia-smi не найден — будет использован CPU (очень медленно!)"
fi

# ═══════════════════════════════════════════════════════════════════════════
# 2-4. Python 3.12 + uv + venv (пропускаем для V3 — setup скрипт всё ставит)
# ═══════════════════════════════════════════════════════════════════════════
if [[ "$BENCH_MODE" == "v3" ]]; then
    log_step "2-4/6 Пропуск (V3 использует системный Python + vLLM)"
    log_info "V3: Python/uv/venv управляются через setup_v3_*.sh"
else

log_step "2/6 Python 3.12"

install_python312() {
    log_info "Установка Python 3.12..."
    if [[ -f /etc/debian_version ]]; then
        sudo apt-get update -qq
        sudo apt-get install -y -qq software-properties-common
        sudo add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null || true
        sudo apt-get update -qq
        sudo apt-get install -y -qq python3.12 python3.12-venv python3.12-dev
    elif [[ -f /etc/redhat-release ]]; then
        sudo dnf install -y -q python3.12 python3.12-devel 2>/dev/null || \
        sudo yum install -y -q python3.12 python3.12-devel
    else
        log_error "Неподдерживаемая ОС. Установите Python 3.12 вручную."
        exit 1
    fi
}

if command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
    log_success "Python 3.12: $($PYTHON_CMD --version)"
else
    install_python312
    PYTHON_CMD="python3.12"
    log_success "Python 3.12 установлен: $($PYTHON_CMD --version)"
fi

# ═══════════════════════════════════════════════════════════════════════════
# 3. uv
# ═══════════════════════════════════════════════════════════════════════════
log_step "3/6 Менеджер пакетов uv"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! command -v uv &> /dev/null; then
    log_info "Установка uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if command -v uv &> /dev/null; then
    log_success "uv: $(uv --version)"
else
    log_error "Не удалось установить uv"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════
# 4. Virtual Environment + PyTorch
# ═══════════════════════════════════════════════════════════════════════════
log_step "4/6 Виртуальное окружение и PyTorch"

# Определяем PyTorch index по CUDA версии
TORCH_INDEX=""
TORCH_SUFFIX=""

if [[ -z "$CUDA_MAJOR" ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    TORCH_SUFFIX="cpu"
elif [[ "$CUDA_MAJOR" -ge 13 ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu130"
    TORCH_SUFFIX="cu130"
elif [[ "$CUDA_MAJOR" -eq 12 ]]; then
    CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d'.' -f2)
    if [[ "$CUDA_MINOR" -ge 8 ]]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu128"
        TORCH_SUFFIX="cu128"
    elif [[ "$CUDA_MINOR" -ge 4 ]]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu124"
        TORCH_SUFFIX="cu124"
    else
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        TORCH_SUFFIX="cu121"
    fi
elif [[ "$CUDA_MAJOR" -eq 11 ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    TORCH_SUFFIX="cu118"
else
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
    TORCH_SUFFIX="cu124"
fi

log_info "PyTorch: $TORCH_SUFFIX"

# Создаём venv если нет
if [[ ! -d "$VENV_DIR" ]]; then
    log_info "Создание виртуального окружения..."
    uv venv --python 3.12 "$VENV_DIR"
    
    # Активируем
    source "$VENV_DIR/bin/activate"
    
    # Устанавливаем PyTorch
    log_info "Установка PyTorch ($TORCH_SUFFIX)..."
    uv pip install torch torchvision --index-url "$TORCH_INDEX"
    
    # Остальные зависимости
    log_info "Установка зависимостей..."
    uv pip install numpy scipy accelerate transformers tqdm fire toml nvidia-ml-py tiktoken aiohttp httpx
    
    log_success "Зависимости установлены"
else
    log_success "Виртуальное окружение уже существует"
    source "$VENV_DIR/bin/activate"
    
    # Проверяем наличие httpx
    if ! python3 -c "import httpx" 2>/dev/null; then
        log_info "Установка httpx..."
        uv pip install httpx
    fi
fi

# Проверка PyTorch
TORCH_VERSION=$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo "не установлен")
CUDA_AVAILABLE=$(python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo "False")

log_info "PyTorch версия: $TORCH_VERSION"
log_info "CUDA доступна: $CUDA_AVAILABLE"

if [[ "$CUDA_AVAILABLE" == "True" ]]; then
    GPU_NAME=$(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')
    log_success "GPU: $GPU_NAME"
fi

fi  # end of V3 skip block

# ═══════════════════════════════════════════════════════════════════════════
# 5. Подготовка (V1: код Gonka, V2: Docker, V3: Native vLLM)
# ═══════════════════════════════════════════════════════════════════════════

# Создаём директорию для результатов
mkdir -p "$RESULTS_DIR"

if [[ "$BENCH_MODE" == "v3" ]]; then
    # ===================== V3: Native vLLM 0.14.0 =====================
    log_step "5/6 Native vLLM 0.14.0 (V3)"

    # Определяем GPU
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo "1")
    # NOTE: || true needed because pipefail + head can cause SIGPIPE (exit 141) with multiple GPUs
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | xargs || true)
    GPU_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs || true)
    GPU_VRAM_GB=$((GPU_VRAM_MB / 1024))
    TOTAL_VRAM_GB=$((GPU_VRAM_GB * GPU_COUNT))

    MODEL_VRAM_REQUIRED=310

    IS_BLACKWELL=0
    if echo "$GPU_NAME" | grep -qiE '(B200|B100|GB200|B300)'; then
        IS_BLACKWELL=1
        log_info "GPU: ${GPU_COUNT}× ${GPU_NAME} — Blackwell"
    else
        log_info "GPU: ${GPU_COUNT}× ${GPU_NAME}"
    fi

    log_info "GPU: ${GPU_COUNT}× ${GPU_NAME} (${GPU_VRAM_GB}GB каждая, ${TOTAL_VRAM_GB}GB всего)"

    if [[ $TOTAL_VRAM_GB -lt $MODEL_VRAM_REQUIRED ]]; then
        log_error "Недостаточно VRAM! Нужно ~${MODEL_VRAM_REQUIRED}GB, доступно ${TOTAL_VRAM_GB}GB"
        log_error "Модель Qwen3-235B-FP8 не поместится на ${GPU_COUNT}× ${GPU_NAME}"
        exit 1
    fi

    # Автоподбор TP: минимальное количество GPU, чтобы суммарный VRAM >= MODEL_VRAM_REQUIRED
    TP_SIZE=1
    for tp in 1 2 4 8; do
        if [[ $tp -le $GPU_COUNT ]] && [[ $((GPU_VRAM_GB * tp)) -ge $MODEL_VRAM_REQUIRED ]]; then
            TP_SIZE=$tp
            break
        fi
    done

    if [[ $((GPU_VRAM_GB * TP_SIZE)) -lt $MODEL_VRAM_REQUIRED ]]; then
        log_error "Не удалось подобрать TP (степень 2) для ${GPU_COUNT}× ${GPU_NAME} (${GPU_VRAM_GB}GB)"
        exit 1
    fi

    log_info "Tensor Parallel: TP=${TP_SIZE} (${GPU_VRAM_GB}GB × ${TP_SIZE} = $((GPU_VRAM_GB * TP_SIZE))GB >= ${MODEL_VRAM_REQUIRED}GB)"

    # Проверяем, установлен ли уже vLLM + gonka_poc
    V3_INSTALLED=0
    if python3.12 -c "import vllm; from vllm.gonka_poc import PoCManagerV1" 2>/dev/null; then
        VLLM_VER=$(python3.12 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
        if [[ "$VLLM_VER" == "0.14.0" ]]; then
            V3_INSTALLED=1
            log_success "vLLM 0.14.0 + gonka_poc уже установлены"
        fi
    fi

    if [[ $V3_INSTALLED -eq 0 ]]; then
        log_info "Запуск установки vLLM 0.14.0 + gonka_poc..."
        if [[ $IS_BLACKWELL -eq 1 ]]; then
            log_info "Используем setup_v3_blackwell.sh"
            sudo bash "$SCRIPT_DIR/setup_v3_blackwell.sh"
        else
            log_info "Используем setup_v3_universal.sh"
            sudo bash "$SCRIPT_DIR/setup_v3_universal.sh"
        fi
        log_success "Установка завершена"
    fi

    # Очищаем процессы на GPU перед запуском
    log_info "Очистка процессов на GPU..."
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    pkill -9 -f "python.*vllm" 2>/dev/null || true
    sleep 3

    # Multi-instance: сколько инстансов можно запустить
    INSTANCE_COUNT=$((GPU_COUNT / TP_SIZE))
    if [[ $INSTANCE_COUNT -lt 1 ]]; then
        INSTANCE_COUNT=1
    fi

    log_info "Multi-instance: ${INSTANCE_COUNT} инстанс(ов) (${GPU_COUNT} GPU / TP=${TP_SIZE})"

    # Запуск vLLM инстансов
    VLLM_PIDS=()
    VLLM_PORTS_LIST=""
    for ((i=0; i<INSTANCE_COUNT; i++)); do
        INST_PORT=$((VLLM_PORT + i + 1))
        START_GPU=$((i * TP_SIZE))
        GPU_IDS=""
        for ((g=START_GPU; g<START_GPU+TP_SIZE; g++)); do
            if [[ -n "$GPU_IDS" ]]; then GPU_IDS="${GPU_IDS},"; fi
            GPU_IDS="${GPU_IDS}${g}"
        done

        # Задержка между запусками инстансов (кроме первого)
        if [[ $i -gt 0 ]]; then
            SLEEP_TIME=$((5 * i))
            log_info "Задержка ${SLEEP_TIME}s перед запуском инстанса ${i}..."
            sleep "$SLEEP_TIME"
        fi

        log_info "Запуск инстанса ${i}: порт ${INST_PORT}, GPU [${GPU_IDS}]"
        CUDA_VISIBLE_DEVICES="$GPU_IDS" \
        VLLM_USE_V1=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
        python3.12 -m vllm.entrypoints.openai.api_server \
            --model "$VLLM_MODEL" --host 0.0.0.0 --port "$INST_PORT" \
            --tensor-parallel-size "$TP_SIZE" \
            --dtype auto --max-model-len 240000 \
            --max-num-batched-tokens 32768 \
            --gpu-memory-utilization 0.92 \
            --kv-cache-dtype fp8_e5m2 &
        VLLM_PIDS+=($!)

        if [[ -n "$VLLM_PORTS_LIST" ]]; then VLLM_PORTS_LIST="${VLLM_PORTS_LIST},"; fi
        VLLM_PORTS_LIST="${VLLM_PORTS_LIST}${INST_PORT}"
    done

    log_success "${INSTANCE_COUNT} vLLM инстанс(ов) запущено (порты: ${VLLM_PORTS_LIST})"

    # Health check — ждём все инстансы
    log_info "Ожидание готовности ${INSTANCE_COUNT} инстанс(ов) (таймаут 20 мин)..."
    VLLM_TIMEOUT=1200
    elapsed=0
    READY_COUNT=0
    declare -A READY_MAP
    while [[ $elapsed -lt $VLLM_TIMEOUT ]]; do
        READY_COUNT=0
        for ((i=0; i<INSTANCE_COUNT; i++)); do
            INST_PORT=$((VLLM_PORT + i + 1))
            if [[ "${READY_MAP[$i]:-}" == "1" ]]; then
                READY_COUNT=$((READY_COUNT + 1))
                continue
            fi
            if curl -s -f -m 5 "http://127.0.0.1:${INST_PORT}/health" > /dev/null 2>&1; then
                READY_MAP[$i]=1
                READY_COUNT=$((READY_COUNT + 1))
                log_success "Инстанс ${i} (порт ${INST_PORT}) готов!"
            fi
        done

        if [[ $READY_COUNT -ge $INSTANCE_COUNT ]]; then
            log_success "Все ${INSTANCE_COUNT} инстанс(ов) готовы!"
            break
        fi

        # Проверяем что процессы живы
        ALL_ALIVE=1
        for ((i=0; i<INSTANCE_COUNT; i++)); do
            if ! kill -0 "${VLLM_PIDS[$i]}" 2>/dev/null; then
                if [[ "${READY_MAP[$i]:-}" != "1" ]]; then
                    log_error "vLLM инстанс ${i} (PID: ${VLLM_PIDS[$i]}) завершился!"
                    ALL_ALIVE=0
                fi
            fi
        done
        if [[ $ALL_ALIVE -eq 0 ]] && [[ $READY_COUNT -eq 0 ]]; then
            log_error "Все инстансы завершились без готовности"
            exit 1
        fi

        echo -ne "  ... ожидание ${elapsed}s / ${VLLM_TIMEOUT}s (${READY_COUNT}/${INSTANCE_COUNT} ready)\r"
        sleep 10
        elapsed=$((elapsed + 10))
    done
    echo ""

    if [[ $READY_COUNT -lt $INSTANCE_COUNT ]]; then
        log_error "Не все инстансы стали доступны за ${VLLM_TIMEOUT}s (${READY_COUNT}/${INSTANCE_COUNT})"
        for pid in "${VLLM_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
        exit 1
    fi

    # Функция очистки при выходе
    cleanup_v3() {
        echo ""
        log_info "Останавливаем ${#VLLM_PIDS[@]} vLLM инстанс(ов)..."
        for pid in "${VLLM_PIDS[@]}"; do
            kill "$pid" 2>/dev/null || true
        done
        for pid in "${VLLM_PIDS[@]}"; do
            wait "$pid" 2>/dev/null || true
        done
        log_success "Все vLLM инстансы остановлены"
    }
    trap cleanup_v3 EXIT

elif [[ "$BENCH_MODE" == "v2" ]]; then
    # ===================== V2: Docker vLLM =====================
    log_step "5/6 Docker vLLM (V2)"

    if ! command -v docker &> /dev/null; then
        log_error "Docker не установлен! Установите docker для V2 бенчмарка."
        exit 1
    fi
    log_success "Docker: $(docker --version | head -1)"

    # Проверяем/останавливаем старые контейнеры
    for old_c in $(docker ps -a --format '{{.Names}}' | grep "^${VLLM_CONTAINER_NAME}"); do
        log_info "Останавливаем старый контейнер: $old_c"
        docker stop "$old_c" 2>/dev/null || true
        docker rm "$old_c" 2>/dev/null || true
    done
    sleep 2

    # Определяем количество GPU и VRAM для автоподбора tensor-parallel-size
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo "1")
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)
    GPU_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | xargs)
    GPU_VRAM_GB=$((GPU_VRAM_MB / 1024))
    TOTAL_VRAM_GB=$((GPU_VRAM_GB * GPU_COUNT))

    # Qwen3-235B-FP8 требует ~310GB VRAM
    MODEL_VRAM_REQUIRED=310

    # Выбор образа: Blackwell (B200, B100, GB200) или стандартный
    if echo "$GPU_NAME" | grep -qiE '(B200|B100|GB200|B300)'; then
        VLLM_IMAGE="$VLLM_IMAGE_BLACKWELL"
        log_info "GPU: ${GPU_COUNT}× ${GPU_NAME} — Blackwell, образ: blackwell"
    else
        VLLM_IMAGE="$VLLM_IMAGE_DEFAULT"
        log_info "GPU: ${GPU_COUNT}× ${GPU_NAME} — образ: default"
    fi

    log_info "GPU: ${GPU_COUNT}× ${GPU_NAME} (${GPU_VRAM_GB}GB каждая, ${TOTAL_VRAM_GB}GB всего)"

    if [[ $TOTAL_VRAM_GB -lt $MODEL_VRAM_REQUIRED ]]; then
        log_error "Недостаточно VRAM! Нужно ~${MODEL_VRAM_REQUIRED}GB, доступно ${TOTAL_VRAM_GB}GB"
        log_error "Модель Qwen3-235B-FP8 не поместится на ${GPU_COUNT}× ${GPU_NAME}"
        exit 1
    fi

    # Автоподбор TP: минимальное количество GPU, чтобы суммарный VRAM >= MODEL_VRAM_REQUIRED
    # TP должен быть степенью 2 (1, 2, 4, 8)
    TP_SIZE=1
    for tp in 1 2 4 8; do
        if [[ $tp -le $GPU_COUNT ]] && [[ $((GPU_VRAM_GB * tp)) -ge $MODEL_VRAM_REQUIRED ]]; then
            TP_SIZE=$tp
            break
        fi
    done

    # Если ни одна степень 2 не подошла — ошибка
    # vLLM требует TP = степень 2 (1, 2, 4, 8)
    if [[ $((GPU_VRAM_GB * TP_SIZE)) -lt $MODEL_VRAM_REQUIRED ]]; then
        log_error "Не удалось подобрать TP (степень 2) для ${GPU_COUNT}× ${GPU_NAME} (${GPU_VRAM_GB}GB)"
        log_error "vLLM поддерживает tensor-parallel-size = 1, 2, 4, 8"
        log_error "Нужно ~${MODEL_VRAM_REQUIRED}GB, но ближайшая степень 2 GPU не покрывает"
        exit 1
    fi

    log_info "Tensor Parallel: TP=${TP_SIZE} (${GPU_VRAM_GB}GB × ${TP_SIZE} = $((GPU_VRAM_GB * TP_SIZE))GB >= ${MODEL_VRAM_REQUIRED}GB)"

    # Проверяем наличие образа
    if ! docker image inspect "$VLLM_IMAGE" &> /dev/null; then
        log_info "Скачивание образа vLLM (это может занять время)..."
        docker pull "$VLLM_IMAGE"
    fi
    log_success "Образ vLLM готов"

    # Multi-instance: сколько инстансов можно запустить
    INSTANCE_COUNT=$((GPU_COUNT / TP_SIZE))
    if [[ $INSTANCE_COUNT -lt 1 ]]; then
        INSTANCE_COUNT=1
    fi

    log_info "Multi-instance: ${INSTANCE_COUNT} контейнер(ов) (${GPU_COUNT} GPU / TP=${TP_SIZE})"

    # Запускаем контейнеры
    # VLLM_USE_V1=0 — V0 engine, как в mlnode образе
    V2_CONTAINER_NAMES=()
    VLLM_PORTS_LIST=""
    for ((i=0; i<INSTANCE_COUNT; i++)); do
        INST_PORT=$((VLLM_PORT + i + 1))
        CONT_NAME="${VLLM_CONTAINER_NAME}-${i}"
        START_GPU=$((i * TP_SIZE))
        GPU_IDS=""
        for ((g=START_GPU; g<START_GPU+TP_SIZE; g++)); do
            if [[ -n "$GPU_IDS" ]]; then GPU_IDS="${GPU_IDS},"; fi
            GPU_IDS="${GPU_IDS}${g}"
        done

        log_info "Запуск контейнера ${i}: порт ${INST_PORT}, GPU [${GPU_IDS}]"
        docker run -d \
            --name "$CONT_NAME" \
            --gpus "\"device=${GPU_IDS}\"" \
            --shm-size=32g \
            -p "${INST_PORT}:${INST_PORT}" \
            -v "$HOME/.cache:/root/.cache" \
            -e VLLM_USE_V1=0 \
            "$VLLM_IMAGE" \
            --model "$VLLM_MODEL" \
            --host 0.0.0.0 \
            --port "$INST_PORT" \
            --tensor-parallel-size "$TP_SIZE" \
            --dtype auto \
            --max-model-len 240000

        V2_CONTAINER_NAMES+=("$CONT_NAME")
        if [[ -n "$VLLM_PORTS_LIST" ]]; then VLLM_PORTS_LIST="${VLLM_PORTS_LIST},"; fi
        VLLM_PORTS_LIST="${VLLM_PORTS_LIST}${INST_PORT}"
    done

    log_success "${INSTANCE_COUNT} контейнер(ов) запущено (порты: ${VLLM_PORTS_LIST})"

    # Ожидаем health check — все инстансы
    log_info "Ожидание готовности ${INSTANCE_COUNT} контейнер(ов) (таймаут 20 мин)..."
    VLLM_TIMEOUT=1200
    elapsed=0
    READY_COUNT=0
    declare -A V2_READY_MAP
    while [[ $elapsed -lt $VLLM_TIMEOUT ]]; do
        READY_COUNT=0
        for ((i=0; i<INSTANCE_COUNT; i++)); do
            INST_PORT=$((VLLM_PORT + i + 1))
            if [[ "${V2_READY_MAP[$i]:-}" == "1" ]]; then
                READY_COUNT=$((READY_COUNT + 1))
                continue
            fi
            if curl -s -f -m 5 "http://127.0.0.1:${INST_PORT}/health" > /dev/null 2>&1; then
                V2_READY_MAP[$i]=1
                READY_COUNT=$((READY_COUNT + 1))
                log_success "Контейнер ${i} (порт ${INST_PORT}) готов!"
            fi
        done

        if [[ $READY_COUNT -ge $INSTANCE_COUNT ]]; then
            log_success "Все ${INSTANCE_COUNT} контейнер(ов) готовы!"
            break
        fi

        # Проверяем что контейнеры живы
        for ((i=0; i<INSTANCE_COUNT; i++)); do
            CONT_NAME="${V2_CONTAINER_NAMES[$i]}"
            if [[ "${V2_READY_MAP[$i]:-}" != "1" ]]; then
                if ! docker ps --format '{{.Names}}' | grep -q "^${CONT_NAME}$"; then
                    log_error "Контейнер ${CONT_NAME} остановился! Логи:"
                    docker logs --tail 30 "$CONT_NAME" 2>&1 || true
                fi
            fi
        done

        echo -ne "  ... ожидание ${elapsed}s / ${VLLM_TIMEOUT}s (${READY_COUNT}/${INSTANCE_COUNT} ready)\r"
        sleep 10
        elapsed=$((elapsed + 10))
    done
    echo ""

    if [[ $READY_COUNT -lt $INSTANCE_COUNT ]]; then
        log_error "Не все контейнеры стали доступны за ${VLLM_TIMEOUT}s (${READY_COUNT}/${INSTANCE_COUNT})"
        for cn in "${V2_CONTAINER_NAMES[@]}"; do
            docker stop "$cn" 2>/dev/null || true
            docker rm "$cn" 2>/dev/null || true
        done
        exit 1
    fi

    # Функция очистки при выходе
    cleanup_v2() {
        echo ""
        log_info "Останавливаем ${#V2_CONTAINER_NAMES[@]} vLLM контейнер(ов)..."
        for cn in "${V2_CONTAINER_NAMES[@]}"; do
            docker stop "$cn" 2>/dev/null || true
            docker rm "$cn" 2>/dev/null || true
        done
        log_success "Все контейнеры остановлены"
    }
    trap cleanup_v2 EXIT

else
    # ===================== V1: Код Gonka =====================
    log_step "5/6 Код Gonka"

    if [[ ! -d "$GONKA_PATH" ]]; then
        log_info "Скачивание gonka (sparse checkout)..."

        git clone --depth 1 --filter=blob:none --sparse \
            https://github.com/gonka-ai/gonka.git "$GONKA_PATH" 2>/dev/null

        cd "$GONKA_PATH"
        git sparse-checkout set mlnode/packages/pow mlnode/packages/common common 2>/dev/null
        cd "$SCRIPT_DIR"

        log_success "Gonka скачан"
    else
        log_success "Gonka уже есть"
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════
# 6. Запуск бенчмарка
# ═══════════════════════════════════════════════════════════════════════════
log_step "6/6 Запуск бенчмарка"

if [[ "$BENCH_MODE" == "v3" ]]; then
    BENCHMARK_SCRIPT="$SCRIPT_DIR/gonka_benchmark_v2.py"

    if [[ ! -f "$BENCHMARK_SCRIPT" ]]; then
        log_error "Не найден gonka_benchmark_v2.py"
        exit 1
    fi

    log_info "Запуск V3: python3.12 gonka_benchmark_v2.py --mode-label v3 --vllm-ports $VLLM_PORTS_LIST ${BENCHMARK_ARGS[*]:-}"
    echo ""

    if [[ ${#BENCHMARK_ARGS[@]} -gt 0 ]]; then
        python3.12 "$BENCHMARK_SCRIPT" --mode-label v3 --vllm-ports "$VLLM_PORTS_LIST" --output "$RESULTS_DIR" "${BENCHMARK_ARGS[@]}"
    else
        python3.12 "$BENCHMARK_SCRIPT" --mode-label v3 --vllm-ports "$VLLM_PORTS_LIST" --output "$RESULTS_DIR"
    fi

elif [[ "$BENCH_MODE" == "v2" ]]; then
    BENCHMARK_SCRIPT="$SCRIPT_DIR/gonka_benchmark_v2.py"

    if [[ ! -f "$BENCHMARK_SCRIPT" ]]; then
        log_error "Не найден gonka_benchmark_v2.py"
        exit 1
    fi

    log_info "Запуск V2: python3 gonka_benchmark_v2.py --vllm-ports $VLLM_PORTS_LIST ${BENCHMARK_ARGS[*]:-}"
    echo ""

    if [[ ${#BENCHMARK_ARGS[@]} -gt 0 ]]; then
        python3 "$BENCHMARK_SCRIPT" --vllm-ports "$VLLM_PORTS_LIST" --output "$RESULTS_DIR" "${BENCHMARK_ARGS[@]}"
    else
        python3 "$BENCHMARK_SCRIPT" --vllm-ports "$VLLM_PORTS_LIST" --output "$RESULTS_DIR"
    fi
else
    BENCHMARK_SCRIPT="$SCRIPT_DIR/gonka_benchmark.py"

    if [[ ! -f "$BENCHMARK_SCRIPT" ]]; then
        log_error "Не найден gonka_benchmark.py"
        log_info "Убедитесь что скрипт находится в той же директории"
        exit 1
    fi

    log_info "Запуск V1: python3 gonka_benchmark.py ${BENCHMARK_ARGS[*]:-}"
    echo ""

    if [[ ${#BENCHMARK_ARGS[@]} -gt 0 ]]; then
        python3 "$BENCHMARK_SCRIPT" --gonka-path "$GONKA_PATH" --output "$RESULTS_DIR" "${BENCHMARK_ARGS[@]}"
    else
        python3 "$BENCHMARK_SCRIPT" --gonka-path "$GONKA_PATH" --output "$RESULTS_DIR"
    fi
fi

exit_code=$?

echo ""
if [[ $exit_code -eq 0 ]]; then
    log_success "Готово! Результаты в $RESULTS_DIR/"
else
    log_error "Бенчмарк завершился с ошибкой (код $exit_code)"
fi

exit $exit_code

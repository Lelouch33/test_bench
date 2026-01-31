#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# Gonka PoW Benchmark - All-in-One Script
#
# Один скрипт для всего:
# 1. Проверяет/устанавливает Python 3.12
# 2. Устанавливает uv
# 3. Создаёт venv и ставит зависимости (PyTorch под нужную CUDA)
# 4. Скачивает код gonka (V1) или запускает docker (V2)
# 5. Запускает бенчмарк
#
# Использование:
#   bash run.sh                    # V1 бенчмарк (по умолчанию)
#   bash run.sh --mode v2          # V2 бенчмарк (cPoC через vLLM)
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
VLLM_IMAGE="ghcr.io/gonka-ai/vllm:v0.9.1-poc-v2-post1-blackwell"
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
    echo -e "${CYAN}║${NC}   ${GREEN}1)${NC} PoC V1 — локальные GPU вычисления (pow.compute)            ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}      Использует модуль gonka для расчёта на GPU напрямую         ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}                                                                  ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${YELLOW}2)${NC} PoC V2 — через vLLM API (cPoC, docker)                    ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}      Запускает vLLM контейнер с моделью и тестирует через API    ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}                                                                  ${CYAN}║${NC}"
    echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    while true; do
        read -p "$(echo -e "${BOLD}Выберите режим [1/2]: ${NC}")" choice
        case "$choice" in
            1) BENCH_MODE="v1"; break ;;
            2) BENCH_MODE="v2"; break ;;
            *) echo -e "${RED}Введите 1 или 2${NC}" ;;
        esac
    done
    echo ""
fi

# ═══════════════════════════════════════════════════════════════════════════
# Заголовок
# ═══════════════════════════════════════════════════════════════════════════
echo ""
if [[ "$BENCH_MODE" == "v2" ]]; then
    echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}${BOLD}║         Gonka PoC V2 Benchmark - All-in-One v1.0                 ║${NC}"
    echo -e "${CYAN}${BOLD}║         Mode: V2 (cPoC via vLLM API)                             ║${NC}"
    echo -e "${CYAN}${BOLD}║  Формула: poc_weight = total_nonces × 2.5                        ║${NC}"
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
# 2. Python 3.12
# ═══════════════════════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════════════════════
# 5. Подготовка (V1: код Gonka, V2: Docker)
# ═══════════════════════════════════════════════════════════════════════════

# Создаём директорию для результатов
mkdir -p "$RESULTS_DIR"

if [[ "$BENCH_MODE" == "v2" ]]; then
    # ===================== V2: Docker vLLM =====================
    log_step "5/6 Docker vLLM (V2)"

    if ! command -v docker &> /dev/null; then
        log_error "Docker не установлен! Установите docker для V2 бенчмарка."
        exit 1
    fi
    log_success "Docker: $(docker --version | head -1)"

    # Проверяем/останавливаем старый контейнер
    if docker ps -a --format '{{.Names}}' | grep -q "^${VLLM_CONTAINER_NAME}$"; then
        log_info "Останавливаем старый контейнер..."
        docker stop "$VLLM_CONTAINER_NAME" 2>/dev/null || true
        docker rm "$VLLM_CONTAINER_NAME" 2>/dev/null || true
        sleep 2
    fi

    # Определяем количество GPU и VRAM для автоподбора tensor-parallel-size
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo "1")
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)
    GPU_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | xargs)
    GPU_VRAM_GB=$((GPU_VRAM_MB / 1024))
    TOTAL_VRAM_GB=$((GPU_VRAM_GB * GPU_COUNT))

    # Qwen3-235B-FP8 требует ~320GB VRAM
    MODEL_VRAM_REQUIRED=320

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

    # Запускаем контейнер
    log_info "Запуск vLLM контейнера..."
    docker run -d \
        --name "$VLLM_CONTAINER_NAME" \
        --gpus all \
        --shm-size=32g \
        -p "${VLLM_PORT}:${VLLM_PORT}" \
        -v "$HOME/.cache:/root/.cache" \
        -e VLLM_USE_V1=1 \
        -e VLLM_USE_CUDA_GRAPHS=0 \
        -e VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
        "$VLLM_IMAGE" \
        --model "$VLLM_MODEL" \
        --host 0.0.0.0 \
        --port "$VLLM_PORT" \
        --enforce-eager \
        --tensor-parallel-size "$TP_SIZE" \
        --dtype float16 \
        --max-model-len 240000

    log_success "Контейнер запущен: $VLLM_CONTAINER_NAME"

    # Ожидаем health check
    log_info "Ожидание готовности vLLM (загрузка модели)..."
    VLLM_TIMEOUT=600
    elapsed=0
    while [[ $elapsed -lt $VLLM_TIMEOUT ]]; do
        if curl -s -f -m 5 "http://127.0.0.1:${VLLM_PORT}/health" > /dev/null 2>&1; then
            log_success "vLLM готов!"
            break
        fi

        # Проверяем что контейнер ещё жив
        if ! docker ps --format '{{.Names}}' | grep -q "^${VLLM_CONTAINER_NAME}$"; then
            log_error "Контейнер остановился! Логи:"
            docker logs --tail 50 "$VLLM_CONTAINER_NAME" 2>&1 || true
            exit 1
        fi

        echo -ne "  ... ожидание ${elapsed}s / ${VLLM_TIMEOUT}s\r"
        sleep 10
        elapsed=$((elapsed + 10))
    done
    echo ""

    if [[ $elapsed -ge $VLLM_TIMEOUT ]]; then
        log_error "vLLM не стал доступен за ${VLLM_TIMEOUT}s"
        log_info "Логи контейнера:"
        docker logs --tail 30 "$VLLM_CONTAINER_NAME" 2>&1 || true
        docker stop "$VLLM_CONTAINER_NAME" 2>/dev/null || true
        docker rm "$VLLM_CONTAINER_NAME" 2>/dev/null || true
        exit 1
    fi

    # Функция очистки при выходе
    cleanup_v2() {
        echo ""
        log_info "Останавливаем vLLM контейнер..."
        docker stop "$VLLM_CONTAINER_NAME" 2>/dev/null || true
        docker rm "$VLLM_CONTAINER_NAME" 2>/dev/null || true
        log_success "Контейнер остановлен"
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

if [[ "$BENCH_MODE" == "v2" ]]; then
    BENCHMARK_SCRIPT="$SCRIPT_DIR/gonka_benchmark_v2.py"

    if [[ ! -f "$BENCHMARK_SCRIPT" ]]; then
        log_error "Не найден gonka_benchmark_v2.py"
        exit 1
    fi

    log_info "Запуск V2: python3 gonka_benchmark_v2.py --vllm-port $VLLM_PORT ${BENCHMARK_ARGS[*]:-}"
    echo ""

    if [[ ${#BENCHMARK_ARGS[@]} -gt 0 ]]; then
        python3 "$BENCHMARK_SCRIPT" --vllm-port "$VLLM_PORT" --output "$RESULTS_DIR" "${BENCHMARK_ARGS[@]}"
    else
        python3 "$BENCHMARK_SCRIPT" --vllm-port "$VLLM_PORT" --output "$RESULTS_DIR"
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

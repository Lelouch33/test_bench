#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# Gonka PoW Benchmark - All-in-One Script
# 
# Один скрипт для всего:
# 1. Проверяет/устанавливает Python 3.12
# 2. Устанавливает uv
# 3. Создаёт venv и ставит зависимости (PyTorch под нужную CUDA)
# 4. Скачивает код gonka
# 5. Запускает бенчмарк
#
# Использование:
#   bash run.sh              # 6 минут (по умолчанию)
#   bash run.sh --duration 3 # 3 минуты
#   bash run.sh --help       # Справка
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════════
# Конфигурация
# ═══════════════════════════════════════════════════════════════════════════
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
GONKA_PATH="$SCRIPT_DIR/gonka"
RESULTS_DIR="$SCRIPT_DIR/results"

# Аргументы для бенчмарка (передаются как есть)
BENCHMARK_ARGS=("$@")

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
# Заголовок
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}${BOLD}║            Gonka PoW Benchmark - All-in-One                      ║${NC}"
echo -e "${CYAN}${BOLD}║                                                                  ║${NC}"
echo -e "${CYAN}${BOLD}║  Формула: poc_weight = valid_nonces × 2.5                        ║${NC}"
echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════════════════════╝${NC}"
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
    uv pip install numpy scipy accelerate transformers tqdm
    
    log_success "Зависимости установлены"
else
    log_success "Виртуальное окружение уже существует"
    source "$VENV_DIR/bin/activate"
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
# 5. Код Gonka
# ═══════════════════════════════════════════════════════════════════════════
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

# Создаём директорию для результатов
mkdir -p "$RESULTS_DIR"

# ═══════════════════════════════════════════════════════════════════════════
# 6. Запуск бенчмарка
# ═══════════════════════════════════════════════════════════════════════════
log_step "6/6 Запуск бенчмарка"

# Проверяем что benchmark скрипт существует
BENCHMARK_SCRIPT="$SCRIPT_DIR/gonka_benchmark.py"

if [[ ! -f "$BENCHMARK_SCRIPT" ]]; then
    log_error "Не найден gonka_benchmark.py"
    log_info "Убедитесь что скрипт находится в той же директории"
    exit 1
fi

# Запускаем бенчмарк с переданными аргументами
log_info "Запуск: python3 gonka_benchmark.py ${BENCHMARK_ARGS[*]}"
echo ""

if [[ ${#BENCHMARK_ARGS[@]} -gt 0 ]]; then
    python3 "$BENCHMARK_SCRIPT" --gonka-path "$GONKA_PATH" --output "$RESULTS_DIR" "${BENCHMARK_ARGS[@]}"
else
    python3 "$BENCHMARK_SCRIPT" --gonka-path "$GONKA_PATH" --output "$RESULTS_DIR"
fi

exit_code=$?

echo ""
if [[ $exit_code -eq 0 ]]; then
    log_success "Готово! Результаты в $RESULTS_DIR/"
else
    log_error "Бенчмарк завершился с ошибкой (код $exit_code)"
fi

exit $exit_code

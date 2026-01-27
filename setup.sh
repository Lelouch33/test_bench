#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# Gonka PoW Benchmark - Setup Script
# 
# - Python 3.12
# - uv для управления пакетами
# - Автоопределение CUDA и установка соответствующего PyTorch
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

# Цвета
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[⚠]${NC} $1"; }
log_error()   { echo -e "${RED}[✗]${NC} $1"; }

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║          Gonka PoW Benchmark - Setup v1.2                        ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

#################################
# 1. Проверка GPU и CUDA версии
#################################
log_info "Проверка GPU..."

CUDA_VERSION=""
CUDA_MAJOR=""

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    log_success "Найдено $GPU_COUNT GPU"
    
    # Получаем версию CUDA из nvidia-smi
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    CUDA_FROM_SMI=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+") || true
    
    if [[ -n "$CUDA_FROM_SMI" ]]; then
        CUDA_VERSION="$CUDA_FROM_SMI"
        CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d'.' -f1)
        log_info "CUDA версия: $CUDA_VERSION (major: $CUDA_MAJOR)"
    fi
else
    log_warning "nvidia-smi не найден. Бенчмарк будет работать на CPU (очень медленно!)"
fi

#################################
# 2. Установка Python 3.12
#################################
log_info "Проверка Python 3.12..."

install_python312() {
    log_info "Установка Python 3.12..."
    if [[ -f /etc/debian_version ]]; then
        sudo apt-get update
        sudo apt-get install -y software-properties-common
        sudo add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null || true
        sudo apt-get update
        sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
    elif [[ -f /etc/redhat-release ]]; then
        sudo dnf install -y python3.12 python3.12-devel || \
        sudo yum install -y python3.12 python3.12-devel
    else
        log_error "Неподдерживаемая ОС. Установите Python 3.12 вручную."
        exit 1
    fi
}

if command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
    log_success "Python 3.12 найден: $($PYTHON_CMD --version)"
else
    install_python312
    PYTHON_CMD="python3.12"
    log_success "Python 3.12 установлен: $($PYTHON_CMD --version)"
fi

#################################
# 3. Установка uv
#################################
log_info "Проверка uv..."

if ! command -v uv &> /dev/null; then
    log_info "Установка uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Добавляем в PATH для текущей сессии
    export PATH="$HOME/.local/bin:$PATH"
    export PATH="$HOME/.cargo/bin:$PATH"
fi

if command -v uv &> /dev/null; then
    log_success "uv установлен: $(uv --version)"
else
    log_error "Не удалось установить uv"
    exit 1
fi

#################################
# 4. Создание виртуального окружения
#################################
VENV_DIR=".venv"

log_info "Создание виртуального окружения с Python 3.12..."

if [[ -d "$VENV_DIR" ]]; then
    log_warning "Виртуальное окружение уже существует"
    read -p "Пересоздать? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
        uv venv --python 3.12 "$VENV_DIR"
    fi
else
    uv venv --python 3.12 "$VENV_DIR"
fi

log_success "Виртуальное окружение создано: $VENV_DIR"

# Активируем для установки пакетов
source "$VENV_DIR/bin/activate"

#################################
# 5. Определение PyTorch версии
#################################
log_info "Определение версии PyTorch для установки..."

# Определяем какой PyTorch ставить на основе CUDA версии
TORCH_INDEX=""

if [[ -z "$CUDA_MAJOR" ]]; then
    log_warning "CUDA не найдена, устанавливаю CPU версию PyTorch"
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    TORCH_SUFFIX="+cpu"
elif [[ "$CUDA_MAJOR" -ge 13 ]]; then
    # CUDA 13.0+
    log_info "CUDA $CUDA_VERSION -> PyTorch с cu130"
    TORCH_INDEX="https://download.pytorch.org/whl/cu130"
    TORCH_SUFFIX="+cu130"
elif [[ "$CUDA_MAJOR" -eq 12 ]]; then
    # CUDA 12.x
    CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d'.' -f2)
    if [[ "$CUDA_MINOR" -ge 8 ]]; then
        log_info "CUDA $CUDA_VERSION -> PyTorch с cu128"
        TORCH_INDEX="https://download.pytorch.org/whl/cu128"
        TORCH_SUFFIX="+cu128"
    elif [[ "$CUDA_MINOR" -ge 4 ]]; then
        log_info "CUDA $CUDA_VERSION -> PyTorch с cu124"
        TORCH_INDEX="https://download.pytorch.org/whl/cu124"
        TORCH_SUFFIX="+cu124"
    else
        log_info "CUDA $CUDA_VERSION -> PyTorch с cu121"
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        TORCH_SUFFIX="+cu121"
    fi
elif [[ "$CUDA_MAJOR" -eq 11 ]]; then
    # CUDA 11.x
    log_info "CUDA $CUDA_VERSION -> PyTorch с cu118"
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    TORCH_SUFFIX="+cu118"
else
    log_warning "Неизвестная CUDA версия $CUDA_VERSION, использую cu124"
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
    TORCH_SUFFIX="+cu124"
fi

#################################
# 6. Установка PyTorch
#################################
log_info "Установка PyTorch${TORCH_SUFFIX}..."

uv pip install torch torchvision --index-url "$TORCH_INDEX"

log_success "PyTorch установлен"

#################################
# 7. Установка остальных зависимостей
#################################
log_info "Установка зависимостей..."

uv pip install \
    numpy \
    scipy \
    accelerate \
    transformers \
    tqdm \
    fire \
    toml \
    nvidia-ml-py \
    tiktoken \
    aiohttp \
    httpx

log_success "Зависимости установлены"

#################################
# 8. Скачивание кода gonka
#################################
GONKA_PATH="./gonka"

if [[ -d "$GONKA_PATH" ]]; then
    log_warning "Директория $GONKA_PATH уже существует"
    read -p "Перескачать? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$GONKA_PATH"
    fi
fi

if [[ ! -d "$GONKA_PATH" ]]; then
    log_info "Скачивание кода gonka (sparse checkout)..."
    
    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/gonka-ai/gonka.git "$GONKA_PATH"
    
    cd "$GONKA_PATH"
    git sparse-checkout set mlnode/packages/pow mlnode/packages/common common
    cd ..
    
    log_success "Код gonka скачан в $GONKA_PATH"
else
    log_info "Используем существующий код в $GONKA_PATH"
fi

#################################
# 9. Создание директории для результатов
#################################
mkdir -p results
log_success "Директория results создана"

#################################
# 10. Проверка установки
#################################
echo ""
log_info "Проверка установки..."
echo ""

echo -e "Python:         $($PYTHON_CMD --version)"
echo -e "uv:             $(uv --version)"
echo -e "PyTorch:        $($PYTHON_CMD -c 'import torch; print(torch.__version__)')"
echo -e "CUDA available: $($PYTHON_CMD -c 'import torch; print(torch.cuda.is_available())')"
echo -e "httpx:          $($PYTHON_CMD -c 'import httpx; print(httpx.__version__)')"

if $PYTHON_CMD -c 'import torch; exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    echo -e "CUDA version:   $($PYTHON_CMD -c 'import torch; print(torch.version.cuda)')"
    echo -e "GPU:            $($PYTHON_CMD -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    УСТАНОВКА ЗАВЕРШЕНА                           ║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║                                                                  ║${NC}"
echo -e "${GREEN}║  Активация окружения:                                            ║${NC}"
echo -e "${GREEN}║    source .venv/bin/activate                                     ║${NC}"
echo -e "${GREEN}║                                                                  ║${NC}"
echo -e "${GREEN}║  Запуск бенчмарка:                                               ║${NC}"
echo -e "${GREEN}║    python3 gonka_benchmark.py                                    ║${NC}"
echo -e "${GREEN}║                                                                  ║${NC}"
echo -e "${GREEN}║  Получить параметры из сети:                                     ║${NC}"
echo -e "${GREEN}║    python3 fetch_poc_duration.py --params                        ║${NC}"
echo -e "${GREEN}║                                                                  ║${NC}"
echo -e "${GREEN}║  Или с кастомным временем:                                       ║${NC}"
echo -e "${GREEN}║    python3 gonka_benchmark.py --duration 3                       ║${NC}"
echo -e "${GREEN}║                                                                  ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

log_info "Для активации окружения выполните: source .venv/bin/activate"

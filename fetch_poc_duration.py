#!/usr/bin/env python3
"""
Gonka PoC Phase Duration Fetcher

Получает информацию о длительности PoC фазы из блокчейна Gonka.
Анализирует блоки чтобы найти начало и конец последней PoC фазы.

Использование:
  python3 fetch_poc_duration.py
  python3 fetch_poc_duration.py --verbose
  python3 fetch_poc_duration.py --json
  python3 fetch_poc_duration.py --params
"""

import argparse
import sys
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

try:
    import httpx
except ImportError:
    print("Ошибка: httpx не установлен")
    print("Установите: pip install httpx --break-system-packages")
    sys.exit(1)


# ============ КОНФИГУРАЦИЯ ============
GENESIS_NODES = [
    "http://node2.gonka.ai:8000",
    "http://node1.gonka.ai:8000",
    "http://node3.gonka.ai:8000",
    "https://node4.gonka.ai",
    "http://47.236.26.199:8000",
    "http://47.236.19.22:18000",
    "http://185.216.21.98:8000",
    "http://36.189.234.197:18026",
    "http://36.189.234.237:17241",
    "http://gonka.spv.re:8000",
]

# Таймауты
REQUEST_TIMEOUT = 10.0


# ============ ЦВЕТА ============
class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def log_info(msg: str):
    print(f"[INFO] {msg}")


def log_success(msg: str):
    print(f"{Colors.GREEN}[✓]{Colors.END} {msg}")


def log_warning(msg: str):
    print(f"{Colors.YELLOW}[⚠]{Colors.END} {msg}")


def log_error(msg: str):
    print(f"{Colors.RED}[✗]{Colors.END} {msg}")


# ============ HELPER: Парсинг значений с экспонентой ============
def parse_exp_value(obj: Any) -> Optional[float]:
    """
    Парсит значение из формата {"value": "1398077", "exponent": -6}
    Возвращает float или None
    """
    if obj is None:
        return None
    
    if isinstance(obj, (int, float)):
        return float(obj)
    
    if isinstance(obj, dict):
        value = obj.get("value")
        exponent = obj.get("exponent", 0)
        if value is not None:
            try:
                return float(value) * (10 ** int(exponent))
            except (ValueError, TypeError):
                return None
    
    if isinstance(obj, str):
        try:
            return float(obj)
        except ValueError:
            return None
    
    return None


# ============ API ФУНКЦИИ ============
def try_request(method: str, path: str, timeout: float = REQUEST_TIMEOUT) -> Optional[Dict]:
    """
    Пробует выполнить запрос к нодам по очереди.
    """
    for node_url in GENESIS_NODES:
        try:
            url = f"{node_url}{path}"
            
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                if method.upper() == "GET":
                    response = client.get(url)
                else:
                    response = client.post(url)
                    
                if response.status_code == 200:
                    return {
                        "data": response.json(),
                        "source_node": node_url
                    }
                    
        except Exception as e:
            continue
            
    return None


def fetch_chain_params() -> Optional[Dict[str, Any]]:
    """Получает все параметры из chain API"""
    result = try_request("GET", "/chain-api/productscience/inference/inference/params")
    if result:
        return {
            "params": result["data"].get("params", {}),
            "source_node": result["source_node"]
        }
    return None


def fetch_poc_params() -> Optional[Dict[str, Any]]:
    """
    Получает параметры PoC из сети.
    Реальный путь: params.poc_params.model_params
    """
    chain_data = fetch_chain_params()
    if not chain_data:
        return None
    
    params = chain_data["params"]
    poc_params = params.get("poc_params", {})
    model_params = poc_params.get("model_params", {})
    epoch_params = params.get("epoch_params", {})
    
    # Парсим r_target из формата {"value": "1398077", "exponent": -6}
    r_target = parse_exp_value(model_params.get("r_target"))
    ffn_dim_multiplier = parse_exp_value(model_params.get("ffn_dim_multiplier"))
    norm_eps = parse_exp_value(model_params.get("norm_eps"))
    weight_scale_factor = parse_exp_value(poc_params.get("weight_scale_factor"))
    
    return {
        # Model params
        "r_target": r_target,
        "dim": model_params.get("dim"),
        "n_layers": model_params.get("n_layers"),
        "n_heads": model_params.get("n_heads"),
        "n_kv_heads": model_params.get("n_kv_heads"),
        "vocab_size": model_params.get("vocab_size"),
        "ffn_dim_multiplier": ffn_dim_multiplier,
        "multiple_of": model_params.get("multiple_of"),
        "norm_eps": norm_eps,
        "rope_theta": model_params.get("rope_theta"),
        "use_scaled_rope": model_params.get("use_scaled_rope"),
        "seq_len": model_params.get("seq_len"),
        
        # PoC params
        "weight_scale_factor": weight_scale_factor,
        "default_difficulty": poc_params.get("default_difficulty"),
        "validation_sample_size": poc_params.get("validation_sample_size"),
        
        # Epoch params (для расчёта длительности PoC)
        "epoch_length": int(epoch_params.get("epoch_length", 0)),
        "poc_stage_duration": int(epoch_params.get("poc_stage_duration", 0)),
        "poc_exchange_duration": int(epoch_params.get("poc_exchange_duration", 0)),
        "poc_validation_duration": int(epoch_params.get("poc_validation_duration", 0)),
        
        "source_node": chain_data["source_node"],
    }


def fetch_latest_block() -> Optional[Dict[str, Any]]:
    """Получает информацию о последнем блоке"""
    result = try_request("GET", "/chain-rpc/block")
    if result:
        return result["data"]
    return None


def fetch_block_info(height: int) -> Optional[Dict[str, Any]]:
    """Получает информацию о блоке по высоте"""
    result = try_request("GET", f"/chain-rpc/block?height={height}")
    if result:
        return result["data"]
    return None


def parse_block_time(time_str: str) -> datetime:
    """Парсит timestamp блока в datetime"""
    # Формат: 2025-01-27T12:34:56.123456789Z
    if "." in time_str:
        base, frac = time_str.split(".")
        frac = frac.rstrip("Z")[:6]
        time_str = f"{base}.{frac}+00:00"
    else:
        time_str = time_str.replace("Z", "+00:00")
    
    return datetime.fromisoformat(time_str)


def calculate_poc_phase_blocks(params: Dict[str, Any], current_block: int) -> Dict[str, Any]:
    """
    Рассчитывает блоки начала и конца PoC фазы на основе параметров эпохи.
    
    PoC фаза происходит в начале каждой эпохи.
    epoch_length = длина эпохи в блоках
    poc_stage_duration = длительность PoC стадии в блоках
    """
    epoch_length = params.get("epoch_length", 15391)
    poc_stage_duration = params.get("poc_stage_duration", 60)
    epoch_shift = 16980  # Из параметров
    
    if epoch_length == 0:
        return {}
    
    # Определяем текущую эпоху
    # epoch_index = (current_block - epoch_shift) // epoch_length
    adjusted_block = current_block - epoch_shift
    if adjusted_block < 0:
        adjusted_block = 0
    
    current_epoch = adjusted_block // epoch_length
    
    # Начало текущей эпохи
    epoch_start_block = epoch_shift + (current_epoch * epoch_length)
    
    # PoC фаза в начале эпохи
    poc_start_block = epoch_start_block
    poc_end_block = epoch_start_block + poc_stage_duration
    
    # Если мы уже прошли PoC фазу текущей эпохи, показываем её
    # Если нет - показываем предыдущую
    if current_block < poc_end_block and current_epoch > 0:
        # Мы в PoC фазе или до неё - показываем предыдущую завершённую
        prev_epoch_start = epoch_shift + ((current_epoch - 1) * epoch_length)
        poc_start_block = prev_epoch_start
        poc_end_block = prev_epoch_start + poc_stage_duration
        current_epoch -= 1
    
    return {
        "current_block": current_block,
        "epoch_index": current_epoch,
        "epoch_length": epoch_length,
        "epoch_start_block": epoch_start_block,
        "poc_start_block": poc_start_block,
        "poc_end_block": poc_end_block,
        "poc_stage_duration_blocks": poc_stage_duration,
    }


def fetch_poc_phase_info() -> Optional[Dict[str, Any]]:
    """
    Получает информацию о PoC фазе.
    Использует параметры chain для расчёта блоков.
    """
    log_info("Получение параметров из сети...")
    
    params = fetch_poc_params()
    if not params:
        log_error("Не удалось получить параметры")
        return None
    
    source_node = params.get("source_node", "unknown")
    log_success(f"Подключено к: {source_node}")
    
    # Получаем текущий блок
    log_info("Получение текущего блока...")
    latest_block_data = fetch_latest_block()
    if not latest_block_data:
        log_error("Не удалось получить текущий блок")
        return None
    
    try:
        current_block = int(latest_block_data["result"]["block"]["header"]["height"])
        log_info(f"Текущий блок: {current_block}")
    except (KeyError, TypeError, ValueError) as e:
        log_error(f"Ошибка парсинга блока: {e}")
        return None
    
    # Рассчитываем блоки PoC фазы
    poc_blocks = calculate_poc_phase_blocks(params, current_block)
    
    if not poc_blocks:
        log_error("Не удалось рассчитать блоки PoC фазы")
        return None
    
    log_info(f"Эпоха: {poc_blocks['epoch_index']}")
    log_info(f"PoC фаза: блоки {poc_blocks['poc_start_block']} - {poc_blocks['poc_end_block']}")
    
    result = {
        **poc_blocks,
        "poc_params": params,
        "source_node": source_node,
    }
    
    # Получаем timestamps блоков для расчёта реальной длительности
    log_info("Получение timestamps блоков...")
    
    start_block_info = fetch_block_info(poc_blocks['poc_start_block'])
    end_block_info = fetch_block_info(poc_blocks['poc_end_block'])
    
    if start_block_info and end_block_info:
        try:
            start_time_str = start_block_info["result"]["block"]["header"]["time"]
            end_time_str = end_block_info["result"]["block"]["header"]["time"]
            
            start_time = parse_block_time(start_time_str)
            end_time = parse_block_time(end_time_str)
            
            duration_seconds = (end_time - start_time).total_seconds()
            
            result["poc_start_time"] = start_time.isoformat()
            result["poc_end_time"] = end_time.isoformat()
            result["poc_duration_seconds"] = duration_seconds
            result["poc_duration_minutes"] = round(duration_seconds / 60, 2)
            
            # Средняя длительность блока
            blocks_count = poc_blocks['poc_end_block'] - poc_blocks['poc_start_block']
            if blocks_count > 0:
                result["avg_block_time_sec"] = round(duration_seconds / blocks_count, 2)
            
            log_success(f"PoC длительность: {result['poc_duration_minutes']} минут ({int(duration_seconds)} сек)")
            
        except Exception as e:
            log_warning(f"Ошибка парсинга времени блоков: {e}")
    
    return result


def print_params(params: Dict[str, Any]):
    """Красиво выводит параметры PoC"""
    W = 62  # Ширина таблицы (внутренняя)
    
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}╔{'═' * W}╗{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}║{'PoC Parameters':^{W}}║{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}╠{'═' * W}╣{Colors.END}")
    
    # Model params
    print(f"{Colors.CYAN}║{Colors.END}  {Colors.GREEN}Model Parameters:{Colors.END}{' ' * (W - 20)}{Colors.CYAN}║{Colors.END}")
    print(f"{Colors.CYAN}║{Colors.END}    r_target:            {str(params.get('r_target')):<{W - 26}}{Colors.CYAN}║{Colors.END}")
    print(f"{Colors.CYAN}║{Colors.END}    dim:                 {str(params.get('dim')):<{W - 26}}{Colors.CYAN}║{Colors.END}")
    print(f"{Colors.CYAN}║{Colors.END}    n_layers:            {str(params.get('n_layers')):<{W - 26}}{Colors.CYAN}║{Colors.END}")
    print(f"{Colors.CYAN}║{Colors.END}    n_heads:             {str(params.get('n_heads')):<{W - 26}}{Colors.CYAN}║{Colors.END}")
    print(f"{Colors.CYAN}║{Colors.END}    n_kv_heads:          {str(params.get('n_kv_heads')):<{W - 26}}{Colors.CYAN}║{Colors.END}")
    print(f"{Colors.CYAN}║{Colors.END}    vocab_size:          {str(params.get('vocab_size')):<{W - 26}}{Colors.CYAN}║{Colors.END}")
    print(f"{Colors.CYAN}║{Colors.END}    ffn_dim_multiplier:  {str(params.get('ffn_dim_multiplier')):<{W - 26}}{Colors.CYAN}║{Colors.END}")
    print(f"{Colors.CYAN}║{Colors.END}    multiple_of:         {str(params.get('multiple_of')):<{W - 26}}{Colors.CYAN}║{Colors.END}")
    print(f"{Colors.CYAN}║{Colors.END}    norm_eps:            {str(params.get('norm_eps')):<{W - 26}}{Colors.CYAN}║{Colors.END}")
    print(f"{Colors.CYAN}║{Colors.END}    rope_theta:          {str(params.get('rope_theta')):<{W - 26}}{Colors.CYAN}║{Colors.END}")
    print(f"{Colors.CYAN}║{Colors.END}    seq_len:             {str(params.get('seq_len')):<{W - 26}}{Colors.CYAN}║{Colors.END}")
    
    print(f"{Colors.CYAN}╠{'═' * W}╣{Colors.END}")
    
    # PoC params
    print(f"{Colors.CYAN}║{Colors.END}  {Colors.GREEN}PoC Settings:{Colors.END}{' ' * (W - 16)}{Colors.CYAN}║{Colors.END}")
    print(f"{Colors.CYAN}║{Colors.END}    weight_scale_factor: {str(params.get('weight_scale_factor')):<{W - 26}}{Colors.CYAN}║{Colors.END}")
    print(f"{Colors.CYAN}║{Colors.END}    default_difficulty:  {str(params.get('default_difficulty')):<{W - 26}}{Colors.CYAN}║{Colors.END}")
    
    print(f"{Colors.CYAN}╠{'═' * W}╣{Colors.END}")
    
    # Epoch params
    print(f"{Colors.CYAN}║{Colors.END}  {Colors.GREEN}Epoch Settings:{Colors.END}{' ' * (W - 18)}{Colors.CYAN}║{Colors.END}")
    print(f"{Colors.CYAN}║{Colors.END}    epoch_length:        {str(params.get('epoch_length')):<{W - 26}}{Colors.CYAN}║{Colors.END}")
    print(f"{Colors.CYAN}║{Colors.END}    poc_stage_duration:  {str(params.get('poc_stage_duration')):<{W - 26}}{Colors.CYAN}║{Colors.END}")
    
    print(f"{Colors.CYAN}╠{'═' * W}╣{Colors.END}")
    source = params.get('source_node', 'unknown')[:W - 12]
    print(f"{Colors.CYAN}║{Colors.END}  Source: {source:<{W - 11}}{Colors.CYAN}║{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}╚{'═' * W}╝{Colors.END}")
    print()


def print_poc_info(info: Dict[str, Any], verbose: bool = False):
    """Красиво выводит информацию о PoC фазе"""
    W = 62  # Ширина таблицы (внутренняя)
    
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}╔{'═' * W}╗{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}║{'PoC Phase Information':^{W}}║{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}╠{'═' * W}╣{Colors.END}")
    
    print(f"{Colors.CYAN}║{Colors.END}  Current Block:       {str(info.get('current_block', 'N/A')):<{W - 25}}{Colors.CYAN}║{Colors.END}")
    print(f"{Colors.CYAN}║{Colors.END}  Epoch Index:         {str(info.get('epoch_index', 'N/A')):<{W - 25}}{Colors.CYAN}║{Colors.END}")
    print(f"{Colors.CYAN}║{Colors.END}  Epoch Length:        {str(info.get('epoch_length', 'N/A')):<{W - 25}}{Colors.CYAN}║{Colors.END}")
    
    print(f"{Colors.CYAN}╠{'═' * W}╣{Colors.END}")
    
    print(f"{Colors.CYAN}║{Colors.END}  PoC Start Block:     {str(info.get('poc_start_block', 'N/A')):<{W - 25}}{Colors.CYAN}║{Colors.END}")
    print(f"{Colors.CYAN}║{Colors.END}  PoC End Block:       {str(info.get('poc_end_block', 'N/A')):<{W - 25}}{Colors.CYAN}║{Colors.END}")
    print(f"{Colors.CYAN}║{Colors.END}  PoC Blocks:          {str(info.get('poc_stage_duration_blocks', 'N/A')):<{W - 25}}{Colors.CYAN}║{Colors.END}")
    
    if info.get("poc_duration_minutes"):
        print(f"{Colors.CYAN}╠{'═' * W}╣{Colors.END}")
        duration_str = f"{info['poc_duration_minutes']} min ({int(info['poc_duration_seconds'])} sec)"
        print(f"{Colors.CYAN}║{Colors.END}  {Colors.GREEN}PoC Duration:{Colors.END}        {duration_str:<{W - 24}}{Colors.CYAN}║{Colors.END}")
        
        if info.get("avg_block_time_sec"):
            avg_str = f"{info['avg_block_time_sec']} sec"
            print(f"{Colors.CYAN}║{Colors.END}  Avg Block Time:      {avg_str:<{W - 25}}{Colors.CYAN}║{Colors.END}")
    
    if info.get("poc_start_time") and info.get("poc_end_time"):
        print(f"{Colors.CYAN}║{Colors.END}  Start Time:          {info['poc_start_time'][:19]:<{W - 25}}{Colors.CYAN}║{Colors.END}")
        print(f"{Colors.CYAN}║{Colors.END}  End Time:            {info['poc_end_time'][:19]:<{W - 25}}{Colors.CYAN}║{Colors.END}")
    
    print(f"{Colors.CYAN}╠{'═' * W}╣{Colors.END}")
    source = info.get('source_node', 'unknown')[:W - 25]
    print(f"{Colors.CYAN}║{Colors.END}  Source Node:         {source:<{W - 25}}{Colors.CYAN}║{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}╚{'═' * W}╝{Colors.END}")
    
    # Рекомендация
    if info.get("poc_duration_minutes"):
        print()
        print(f"{Colors.YELLOW}Рекомендуемая длительность бенчмарка:{Colors.END}")
        print(f"  python3 gonka_benchmark.py --duration {info['poc_duration_minutes']}")
    
    # r_target из параметров
    poc_params = info.get("poc_params", {})
    if poc_params.get("r_target"):
        print()
        print(f"{Colors.YELLOW}Текущий r_target:{Colors.END} {poc_params['r_target']}")
    
    print()
    
    if verbose:
        print(f"\n{Colors.BOLD}Полные параметры PoC:{Colors.END}")
        import json
        poc_params = info.get("poc_params", {})
        # Убираем source_node для чистоты
        output_params = {k: v for k, v in poc_params.items() if k != "source_node"}
        print(json.dumps(output_params, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(
        description="Получение информации о длительности PoC фазы из блокчейна Gonka",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python3 fetch_poc_duration.py              # Информация о PoC фазе
  python3 fetch_poc_duration.py --params     # Показать параметры PoC
  python3 fetch_poc_duration.py --verbose    # С дополнительными данными
  python3 fetch_poc_duration.py --json       # Вывод в JSON
        """
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Показать дополнительную информацию"
    )
    
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Вывод в формате JSON"
    )
    
    parser.add_argument(
        "--params", "-p",
        action="store_true",
        help="Показать параметры PoC (r_target и т.д.)"
    )
    
    args = parser.parse_args()
    
    # Если запрошены параметры
    if args.params:
        log_info("Получение параметров PoC...")
        params = fetch_poc_params()
        
        if params:
            if args.json:
                import json
                print(json.dumps(params, indent=2, default=str))
            else:
                log_success(f"Подключено к: {params.get('source_node')}")
                print_params(params)
        else:
            log_error("Не удалось получить параметры")
            return 1
        
        return 0
    
    # Основная функция - получение информации о PoC фазе
    info = fetch_poc_phase_info()
    
    if not info:
        log_error("Не удалось получить информацию о PoC фазе")
        return 1
    
    if args.json:
        import json
        # Убираем вложенные данные для чистого вывода
        output = {k: v for k, v in info.items() if k != "poc_params"}
        output["r_target"] = info.get("poc_params", {}).get("r_target")
        print(json.dumps(output, indent=2, default=str))
    else:
        print_poc_info(info, verbose=args.verbose)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

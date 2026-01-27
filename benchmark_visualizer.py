#!/usr/bin/env python3
"""
Gonka Benchmark Visualizer

Модуль для посекундной записи метрик и генерации визуализаций.

Использование:
  # В gonka_benchmark.py:
  from benchmark_visualizer import BenchmarkLogger
  
  logger = BenchmarkLogger()
  logger.start()
  # ... в цикле бенчмарка:
  logger.log(valid_nonces, total_checked, gpu_stats)
  # ... после бенчмарка:
  logger.save("benchmark_data.json")
  logger.generate_chart_html("benchmark_chart.html")
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


class BenchmarkLogger:
    """
    Логгер для посекундной записи метрик бенчмарка.
    
    Записывает:
    - valid_nonces (накопительно)
    - total_checked (накопительно) 
    - valid/sec и raw/sec (моментальные)
    - GPU статистику (VRAM, утилизация, мощность)
    """
    
    def __init__(self, output_dir: str = "results"):
        self.start_time: Optional[datetime] = None
        self.data_points: List[Dict[str, Any]] = []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Для расчёта моментальных скоростей
        self._last_valid = 0
        self._last_checked = 0
        self._last_timestamp = None
        
    def start(self):
        """Начинает запись"""
        self.start_time = datetime.now()
        self.data_points = []
        self._last_valid = 0
        self._last_checked = 0
        self._last_timestamp = self.start_time
        
    def log(
        self,
        valid_nonces: int,
        total_checked: int,
        gpu_stats: Optional[Dict[int, Dict[str, Any]]] = None,
        worker_stats: Optional[Dict[int, Dict[str, Any]]] = None,
    ):
        """
        Записывает точку данных.
        
        Args:
            valid_nonces: Общее количество valid nonces (накопительно)
            total_checked: Общее количество проверенных (накопительно)
            gpu_stats: Статистика GPU {device_id: {vram_percent, gpu_util, power_watts}}
            worker_stats: Статистика по workers {worker_id: {total_valid, total_checked}}
        """
        if not self.start_time:
            self.start()
            
        now = datetime.now()
        elapsed_sec = (now - self.start_time).total_seconds()
        
        # Моментальные скорости (за последний интервал)
        time_delta = (now - self._last_timestamp).total_seconds()
        if time_delta > 0:
            instant_valid_per_sec = (valid_nonces - self._last_valid) / time_delta
            instant_raw_per_sec = (total_checked - self._last_checked) / time_delta
        else:
            instant_valid_per_sec = 0
            instant_raw_per_sec = 0
        
        # Средние скорости (за всё время)
        avg_valid_per_sec = valid_nonces / elapsed_sec if elapsed_sec > 0 else 0
        avg_raw_per_sec = total_checked / elapsed_sec if elapsed_sec > 0 else 0
        
        # Формируем точку данных
        point = {
            "timestamp": now.isoformat(),
            "elapsed_sec": round(elapsed_sec, 1),
            "valid_nonces": valid_nonces,
            "total_checked": total_checked,
            
            # Моментальные скорости
            "instant_valid_per_sec": round(instant_valid_per_sec, 2),
            "instant_raw_per_sec": round(instant_raw_per_sec, 2),
            "instant_valid_per_min": round(instant_valid_per_sec * 60, 2),
            "instant_raw_per_min": round(instant_raw_per_sec * 60, 2),
            
            # Средние скорости
            "avg_valid_per_sec": round(avg_valid_per_sec, 2),
            "avg_raw_per_sec": round(avg_raw_per_sec, 2),
            "avg_valid_per_min": round(avg_valid_per_sec * 60, 2),
            "avg_raw_per_min": round(avg_raw_per_sec * 60, 2),
        }
        
        # Добавляем GPU статистику если есть
        if gpu_stats:
            point["gpu_stats"] = {}
            for device_id, stats in gpu_stats.items():
                point["gpu_stats"][str(device_id)] = {
                    "vram_percent": stats.get("vram_percent", 0),
                    "gpu_util": stats.get("gpu_util", 0),
                    "power_watts": stats.get("power_watts", 0),
                }
        
        # Добавляем статистику по workers если есть
        if worker_stats:
            point["workers"] = {}
            for worker_id, stats in worker_stats.items():
                point["workers"][str(worker_id)] = {
                    "valid": stats.get("total_valid", 0),
                    "checked": stats.get("total_checked", 0),
                }
        
        self.data_points.append(point)
        
        # Обновляем для следующего расчёта
        self._last_valid = valid_nonces
        self._last_checked = total_checked
        self._last_timestamp = now
        
    def save(self, filename: str = "benchmark_timeseries.json") -> Path:
        """
        Сохраняет все данные в JSON файл.
        
        Returns:
            Path к сохранённому файлу
        """
        filepath = self.output_dir / filename
        
        output = {
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "total_points": len(self.data_points),
            "data": self.data_points,
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
            
        return filepath
    
    def generate_chart_html(self, filename: str = "benchmark_chart.html") -> Path:
        """
        Генерирует HTML файл с интерактивным графиком (Chart.js).
        
        Returns:
            Path к HTML файлу
        """
        if not self.data_points:
            raise ValueError("Нет данных для визуализации")
        
        # Подготавливаем данные для графика
        timestamps = [p["elapsed_sec"] for p in self.data_points]
        valid_rates = [p["instant_valid_per_min"] for p in self.data_points]
        avg_valid_rates = [p["avg_valid_per_min"] for p in self.data_points]
        valid_cumulative = [p["valid_nonces"] for p in self.data_points]
        
        # GPU данные (берём первый GPU если есть)
        gpu_util = []
        gpu_power = []
        gpu_vram = []
        for p in self.data_points:
            gs = p.get("gpu_stats", {})
            if gs:
                first_gpu = list(gs.values())[0]
                gpu_util.append(first_gpu.get("gpu_util", 0))
                gpu_power.append(first_gpu.get("power_watts", 0))
                gpu_vram.append(first_gpu.get("vram_percent", 0))
            else:
                gpu_util.append(0)
                gpu_power.append(0)
                gpu_vram.append(0)
        
        # Финальные метрики
        final_valid = self.data_points[-1]["valid_nonces"]
        final_duration = self.data_points[-1]["elapsed_sec"]
        final_avg_rate = self.data_points[-1]["avg_valid_per_min"]
        poc_weight = int(final_valid * 2.5)
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gonka PoW Benchmark Results</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            color: #00d4ff;
            margin-bottom: 10px;
            font-size: 2.5em;
            text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
        }}
        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 30px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            transition: transform 0.3s, box-shadow 0.3s;
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0, 212, 255, 0.2);
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #00d4ff;
        }}
        .metric-value.green {{
            color: #00ff88;
        }}
        .metric-label {{
            color: #888;
            margin-top: 5px;
            font-size: 0.9em;
        }}
        .chart-container {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .chart-title {{
            color: #00d4ff;
            margin-bottom: 15px;
            font-size: 1.2em;
        }}
        .charts-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        footer {{
            text-align: center;
            margin-top: 30px;
            color: #666;
            font-size: 0.9em;
        }}
        footer a {{
            color: #00d4ff;
            text-decoration: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>⚡ Gonka PoW Benchmark</h1>
        <p class="subtitle">Duration: {final_duration:.0f} seconds | {len(self.data_points)} data points</p>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{final_valid}</div>
                <div class="metric-label">Valid Nonces</div>
            </div>
            <div class="metric-card">
                <div class="metric-value green">{poc_weight}</div>
                <div class="metric-label">PoC Weight</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{final_avg_rate:.1f}</div>
                <div class="metric-label">Valid/min (avg)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{final_duration/60:.1f}</div>
                <div class="metric-label">Minutes</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h3 class="chart-title">📈 Valid Nonces Rate (per minute)</h3>
            <canvas id="rateChart" height="100"></canvas>
        </div>
        
        <div class="charts-row">
            <div class="chart-container">
                <h3 class="chart-title">📊 Cumulative Valid Nonces</h3>
                <canvas id="cumulativeChart" height="150"></canvas>
            </div>
            <div class="chart-container">
                <h3 class="chart-title">🖥️ GPU Utilization</h3>
                <canvas id="gpuChart" height="150"></canvas>
            </div>
        </div>
        
        <footer>
            Generated by <a href="https://github.com/gonka-ai/gonka">Gonka PoW Benchmark</a> | 
            Formula: poc_weight = valid_nonces × 2.5
        </footer>
    </div>
    
    <script>
        const timestamps = {json.dumps(timestamps)};
        const validRates = {json.dumps(valid_rates)};
        const avgValidRates = {json.dumps(avg_valid_rates)};
        const validCumulative = {json.dumps(valid_cumulative)};
        const gpuUtil = {json.dumps(gpu_util)};
        const gpuPower = {json.dumps(gpu_power)};
        
        // Rate Chart
        new Chart(document.getElementById('rateChart').getContext('2d'), {{
            type: 'line',
            data: {{
                labels: timestamps,
                datasets: [{{
                    label: 'Instant Rate',
                    data: validRates,
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 1,
                }}, {{
                    label: 'Average Rate',
                    data: avgValidRates,
                    borderColor: '#00ff88',
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.4,
                    pointRadius: 0,
                }}]
            }},
            options: {{
                responsive: true,
                interaction: {{
                    intersect: false,
                    mode: 'index',
                }},
                scales: {{
                    x: {{ 
                        title: {{ display: true, text: 'Time (seconds)', color: '#888' }},
                        grid: {{ color: 'rgba(255,255,255,0.1)' }},
                        ticks: {{ color: '#888' }}
                    }},
                    y: {{ 
                        title: {{ display: true, text: 'Valid/min', color: '#888' }},
                        grid: {{ color: 'rgba(255,255,255,0.1)' }},
                        ticks: {{ color: '#888' }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        labels: {{ color: '#888' }}
                    }}
                }}
            }}
        }});
        
        // Cumulative Chart
        new Chart(document.getElementById('cumulativeChart').getContext('2d'), {{
            type: 'line',
            data: {{
                labels: timestamps,
                datasets: [{{
                    label: 'Valid Nonces',
                    data: validCumulative,
                    borderColor: '#00ff88',
                    backgroundColor: 'rgba(0, 255, 136, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 1,
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    x: {{ 
                        title: {{ display: true, text: 'Time (seconds)', color: '#888' }},
                        grid: {{ color: 'rgba(255,255,255,0.1)' }},
                        ticks: {{ color: '#888' }}
                    }},
                    y: {{ 
                        title: {{ display: true, text: 'Total Valid', color: '#888' }},
                        grid: {{ color: 'rgba(255,255,255,0.1)' }},
                        ticks: {{ color: '#888' }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        labels: {{ color: '#888' }}
                    }}
                }}
            }}
        }});
        
        // GPU Chart
        new Chart(document.getElementById('gpuChart').getContext('2d'), {{
            type: 'line',
            data: {{
                labels: timestamps,
                datasets: [{{
                    label: 'GPU Util %',
                    data: gpuUtil,
                    borderColor: '#ff6b6b',
                    fill: false,
                    tension: 0.4,
                    pointRadius: 1,
                    yAxisID: 'y',
                }}, {{
                    label: 'Power (W)',
                    data: gpuPower,
                    borderColor: '#ffd93d',
                    fill: false,
                    tension: 0.4,
                    pointRadius: 1,
                    yAxisID: 'y1',
                }}]
            }},
            options: {{
                responsive: true,
                interaction: {{
                    mode: 'index',
                    intersect: false,
                }},
                scales: {{
                    x: {{ 
                        title: {{ display: true, text: 'Time (seconds)', color: '#888' }},
                        grid: {{ color: 'rgba(255,255,255,0.1)' }},
                        ticks: {{ color: '#888' }}
                    }},
                    y: {{
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {{ display: true, text: 'Utilization %', color: '#888' }},
                        grid: {{ color: 'rgba(255,255,255,0.1)' }},
                        ticks: {{ color: '#888' }},
                        max: 100,
                    }},
                    y1: {{
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {{ display: true, text: 'Power (W)', color: '#888' }},
                        ticks: {{ color: '#888' }},
                        grid: {{ drawOnChartArea: false }},
                    }},
                }},
                plugins: {{
                    legend: {{
                        labels: {{ color: '#888' }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
        
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            f.write(html)
            
        return filepath
    
    def get_summary(self) -> Dict[str, Any]:
        """Возвращает сводку по собранным данным"""
        if not self.data_points:
            return {}
            
        last = self.data_points[-1]
        
        # Находим пики
        max_instant_rate = max(p["instant_valid_per_min"] for p in self.data_points)
        min_instant_rate = min(p["instant_valid_per_min"] for p in self.data_points)
        
        return {
            "duration_sec": last["elapsed_sec"],
            "total_valid": last["valid_nonces"],
            "total_checked": last["total_checked"],
            "avg_valid_per_min": last["avg_valid_per_min"],
            "max_instant_rate": max_instant_rate,
            "min_instant_rate": min_instant_rate,
            "data_points_count": len(self.data_points),
            "poc_weight": int(last["valid_nonces"] * 2.5),
        }


def generate_ascii_card(results: Dict[str, Any]) -> str:
    """
    Генерирует ASCII карточку с результатами для Discord/Reddit.
    
    Args:
        results: Словарь с результатами бенчмарка
        
    Returns:
        Строка с ASCII карточкой
    """
    gpu_name = results.get("gpu", {}).get("name", "Unknown GPU")
    if len(gpu_name) > 30:
        gpu_name = gpu_name[:27] + "..."
    
    poc_weight = results.get("poc_weight", 0)
    valid_nonces = results.get("valid_nonces", 0)
    valid_per_min = results.get("valid_per_min", 0)
    duration = results.get("duration_min", 0)
    
    card = f"""
╔═══════════════════════════════════════════╗
║      ⚡ GONKA PoW BENCHMARK RESULTS ⚡      ║
╠═══════════════════════════════════════════╣
║  GPU: {gpu_name:<35} ║
╠═══════════════════════════════════════════╣
║  valid_nonces:  {valid_nonces:<25} ║
║  poc_weight:    {poc_weight:<25} ║
║  valid/min:     {valid_per_min:<25.1f} ║
║  duration:      {duration:<25.1f} min ║
╚═══════════════════════════════════════════╝
  Formula: poc_weight = valid_nonces × 2.5
"""
    return card


# Пример использования
if __name__ == "__main__":
    import random
    import time
    
    print("Демонстрация BenchmarkLogger")
    print("=" * 40)
    
    logger = BenchmarkLogger(output_dir="./demo_results")
    logger.start()
    
    valid = 0
    checked = 0
    
    # Симуляция 30 секунд бенчмарка
    for i in range(30):
        # Симулируем данные
        valid += random.randint(1, 3)
        checked += random.randint(40, 60)
        
        gpu_stats = {
            0: {
                "vram_percent": 85 + random.uniform(-5, 5),
                "gpu_util": 95 + random.uniform(-10, 5),
                "power_watts": 350 + random.uniform(-20, 20),
            }
        }
        
        logger.log(valid, checked, gpu_stats)
        time.sleep(0.1)  # Быстрая демонстрация
    
    # Сохраняем
    json_path = logger.save("demo_data.json")
    html_path = logger.generate_chart_html("demo_chart.html")
    
    print(f"\nСохранено:")
    print(f"  JSON: {json_path}")
    print(f"  HTML: {html_path}")
    
    # Сводка
    summary = logger.get_summary()
    print(f"\nСводка:")
    print(f"  Точек данных: {summary['data_points_count']}")
    print(f"  Valid nonces: {summary['total_valid']}")
    print(f"  PoC Weight: {summary['poc_weight']}")
    
    # ASCII карточка
    demo_results = {
        "gpu": {"name": "NVIDIA H200"},
        "poc_weight": summary['poc_weight'],
        "valid_nonces": summary['total_valid'],
        "valid_per_min": summary['avg_valid_per_min'],
        "duration_min": summary['duration_sec'] / 60,
    }
    print(generate_ascii_card(demo_results))

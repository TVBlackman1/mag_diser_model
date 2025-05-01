import time
import csv
from config.version import EXPERIMENT_FOLDER


# Хранилище времени и статистики
_timings = {}
_start_times = {}


def start(label: str):
    """Начинает отсчет времени для данной метки"""
    _start_times[label] = time.perf_counter()


def stop(label: str):
    """Останавливает отсчет и обновляет статистику по метке"""
    if label not in _start_times:
        raise ValueError(f"Не было вызова start() для '{label}'")

    elapsed = time.perf_counter() - _start_times.pop(label)

    if label not in _timings:
        _timings[label] = {
            "count": 1,
            "total": elapsed,
            "min": elapsed,
            "max": elapsed
        }
    else:
        stats = _timings[label]
        stats["count"] += 1
        stats["total"] += elapsed
        stats["min"] = min(stats["min"], elapsed)
        stats["max"] = max(stats["max"], elapsed)

def export_csv():
    """Экспортирует данные в CSV"""
    filepath = f"{EXPERIMENT_FOLDER}/timer.csv"
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Label", "AvgTime", "MinTime", "MaxTime", "Calls"])
        for label, stats in _timings.items():
            avg = stats["total"] / stats["count"]
            writer.writerow([label, f"{avg:.6f}", f"{stats['min']:.6f}", f"{stats['max']:.6f}", stats["count"]])

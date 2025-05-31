import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from inference.get_last import get_latest_result_path
from utils.save import save_file


def save_analyze_training_chart():
    last_result_path = get_latest_result_path()
    log_path = f"results/{last_result_path}/training_log.csv"
    path_coverage_path = f"results/{last_result_path}/training_log"

    print(log_path)
    if not os.path.exists(log_path):
        print("Training log not found!")
        return

    df = pd.read_csv(log_path)

    required_columns = {"Episode", "PercentCovered"}
    if not required_columns.issubset(df.columns):
        print(f"CSV missing columns: {required_columns}")
        return

    avg_window = 10
    avg_covered = np.convolve(df['PercentCovered'], np.ones(avg_window) / avg_window, mode='same')

    plt.figure(figsize=(10, 6))
    plt.plot(df['Episode'], avg_covered, label='Path Covered %', linewidth=1.5)
    plt.xlabel('Episode')
    plt.ylabel('Percent of Path Covered (%)')
    plt.title('Progress over Training')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_file(plt, path_coverage_path)
    plt.close()

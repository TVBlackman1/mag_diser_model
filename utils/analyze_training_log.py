import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    log_path = "results/training_log.csv"

    if not os.path.exists(log_path):
        print("Training log not found!")
        return

    # Читаем CSV
    df = pd.read_csv(log_path)

    # Проверка, что нужные столбцы есть
    required_columns = {"Episode", "PercentCovered"}
    if not required_columns.issubset(df.columns):
        print(f"CSV missing columns: {required_columns}")
        return

    # Фильтруем эпизоды с пройденным путём >= 0% (или можно 50%, если хочешь)
    df_filtered = df[df['PercentCovered'] >= 0]

    # График 1: Процент пройденного пути по эпизодам
    plt.figure(figsize=(10, 6))
    plt.plot(df_filtered['Episode'], df_filtered['PercentCovered'], label='Path Covered %', linewidth=1.5)
    plt.xlabel('Episode')
    plt.ylabel('Percent of Path Covered (%)')
    plt.title('Progress over Training')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # График 2: Гистограмма пройденного пути
    plt.figure(figsize=(8, 5))
    plt.hist(df_filtered['PercentCovered'], bins=20, edgecolor='black')
    plt.xlabel('Percent of Path Covered')
    plt.ylabel('Number of Episodes')
    plt.title('Distribution of Path Coverage')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    print(f"Analyzed {len(df_filtered)} episodes.")

if __name__ == "__main__":
    main()

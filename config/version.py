from datetime import datetime

EXPERIMENT_TAG = "reward_buffer_ratio"
EXPERIMENT_NOTES = '''
— нормализованная награда по расстоянию
— critic loss логируется
— буфер возвращает процентную выборку в зависимости от mean, std
- warmup
'''

EXPERIMENT_DATE = datetime.now().strftime("%Y_%m_%d_%H%M")
EXPERIMENT_FOLDER = f"results/{EXPERIMENT_DATE}_{EXPERIMENT_TAG}"
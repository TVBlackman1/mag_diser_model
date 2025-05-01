from datetime import datetime

EXPERIMENT_TAG = "reward_buffer_block"
EXPERIMENT_NOTES = '''
— нормализованная награда по расстоянию
— critic loss логируется
— топ-400 приоритетизированных переходов
— буфер имеет фильтр на reward по mean
- warmup
'''

EXPERIMENT_DATE = datetime.now().strftime("%Y_%m_%d_%H%M")
EXPERIMENT_FOLDER = f"results/{EXPERIMENT_DATE}_{EXPERIMENT_TAG}"
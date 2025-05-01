from datetime import datetime

EXPERIMENT_TAG = "actor_rare_update"
EXPERIMENT_NOTES = '''
— нормализованная награда по расстоянию
— critic loss логируется
— буфер возвращает процентную выборку в зависимости от critic loss, td
- смягчение для td через log1p
- warmup
- actor обновляется реже чем critic
'''

EXPERIMENT_DATE = datetime.now().strftime("%Y_%m_%d_%H%M")
RESULTS_DIR = "results"
EXPERIMENT_FOLDER = f"{RESULTS_DIR}/{EXPERIMENT_DATE}_{EXPERIMENT_TAG}"
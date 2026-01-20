from datetime import datetime

EXPERIMENT_TAG = "obs_closing_penalty_v2"
EXPERIMENT_NOTES = '''
— нормализованная награда по расстоянию
— critic loss логируется
— буфер возвращает процентную выборку в зависимости от critic loss, td
- смягчение для td через log1p
- warmup
- actor обновляется реже чем critic
- теперь видит препятствия
- штраф за приближение к препятствию
- штраф за именно приближение к препятствию, а не нахождение рядом с ним
- все возможные направления для передвижения
- бонус за уход от препятствия в сторону, не только назад
- убраны награды за уход от препятствий, есть только штрафы за приближение к ним
- штраф за бездействие постоянно растет
- теперь используется движение по скорости и скорости поворота, а не dx, dy
'''

PROJECT_GLOBAL_STATE = "ANGLE_VELOCITY"


EXPERIMENT_DATE = datetime.now().strftime("%Y_%m_%d_%H%M%S")
RESULTS_DIR = "results"
EXPERIMENT_FOLDER = f"{RESULTS_DIR}/{EXPERIMENT_DATE}_{PROJECT_GLOBAL_STATE}_{EXPERIMENT_TAG}"

# добавить свертку в нейронку актора, добавить инфу о приближении к дрону, добавляющую очки за отъезд от препятствия
# tanh в критике недооценивает недостатки

# 2025_05_31_0023_OBS_obs_closing_penalty_v2
# 2025_05_31_0400_PENALTY_TIMER_obs_closing_penalty_v2
# 2025_05_31_0506_PENALTY_TIMER_obs_closing_penalty_v2
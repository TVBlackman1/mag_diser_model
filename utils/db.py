import duckdb

from config.version import EXPERIMENT_FOLDER

RESULT_SUCCESS = "success"
RESULT_FAILED = "failed"
RESULT_PROGRESS = "progress"

class DBSaver:
    def __init__(self):
        file = f"{EXPERIMENT_FOLDER}/data.db"
        self.con = duckdb.connect(file)
        self.episode = 0
        self.con.execute("CREATE TABLE experiments (episode INTEGER, step INTEGER,"
                         "x FLOAT, y FLOAT, new_x FLOAT, new_y FLOAT,"
                         "speed_ratio FLOAT, angle_ratio FLOAT,"
                         "target_distance FLOAT, new_target_distance FLOAT, reward FLOAT, result VARCHAR(10)"
                         ")")
    def start_new_episode(self, episode: int):
        self.episode = episode
    
    def add_step(self, step: int, x, y, new_x, new_y,
                 speed_ratio,
                 angle_ratio,
                 target_distance, new_target_distance, reward, result):
        self.con.execute("INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [
            self.episode, step, float(x), float(y), float(new_x), float(new_y), float(speed_ratio),
            float(angle_ratio),
            float(target_distance), float(new_target_distance), float(reward), result,
        ])

import duckdb
from pathlib import Path
from config.version import EXPERIMENT_FOLDER

class DBSaver:
    def __init__(self):
        file = f"{EXPERIMENT_FOLDER}/data.db"
        self.con = duckdb.connect(file)
        self.episode = 0
        self.is_train = True

        migration_path = Path("./utils/db/migration.sql")
        sql_content = migration_path.read_text(encoding="utf-8")

        self.con.execute(sql_content)
        print(f"db created: {Path(file).absolute()}")
    def start_new_episode(self, episode: int, is_train: bool):
        self.episode = episode
        self.is_train = is_train
    
    def add_step(self, step: int, x, y, new_x, new_y,
                 speed_ratio,
                 angle_ratio,
                 target_distance, new_target_distance, reward, result):
        self.con.execute("INSERT INTO experiments VALUES (nextval('seq_experiment_id'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [
            self.episode, step, float(x), float(y), float(new_x), float(new_y), float(speed_ratio),
            float(angle_ratio),
            float(target_distance), float(new_target_distance), float(reward), result,
            self.is_train,
        ])

    def add_step2(self, step: int,old_drone_pos, new_drone_pos,
                  obstacles: [],
                  speed_ratio,
                  angle_ratio,
                  target_distance, new_target_distance, reward, result
                  ):
        # Kept for backward compatibility: accept vector positions.
        x, y = float(old_drone_pos[0]), float(old_drone_pos[1])
        new_x, new_y = float(new_drone_pos[0]), float(new_drone_pos[1])
        self.con.execute("INSERT INTO experiments VALUES (nextval('seq_experiment_id'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [
            self.episode, step, float(x), float(y), float(new_x), float(new_y), float(speed_ratio),
            float(angle_ratio),
            float(target_distance), float(new_target_distance), float(reward), result,
            self.is_train,
        ])


CREATE TABLE experiments (
    id BIGINT PRIMARY KEY,
    episode INTEGER, step INTEGER,
    x FLOAT, y FLOAT, new_x FLOAT, new_y FLOAT,
    speed_ratio FLOAT, angle_ratio FLOAT,
    target_distance FLOAT, new_target_distance FLOAT, reward FLOAT, result VARCHAR(10),
    is_train BOOLEAN);

CREATE SEQUENCE seq_experiment_id START 1;
CREATE INDEX experiments_episode ON experiments (episode);

CREATE TABLE objects (
    id BIGINT PRIMARY KEY,
    experiment_id BIGINT, FOREIGN KEY (experiment_id) REFERENCES experiments (id),
    type VARCHAR(10),
    x FLOAT, y FLOAT,
    distance FLOAT, angle FLOAT);

CREATE SEQUENCE seq_object_id START 1;
CREATE INDEX objects_episode ON objects (id);

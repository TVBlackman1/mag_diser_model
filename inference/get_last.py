import os

import utils.save
from config import version

results_dir = version.RESULTS_DIR

def get_latest_model_path():
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Directory '{results_dir}' does not exist.")

    subdirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in '{results_dir}'.")

    latest_dir = sorted(subdirs)[-1]

    checkpoints_dir = os.path.join(results_dir, latest_dir, utils.save.CHECKPOINTS_SUBDIR)

    if not os.path.exists(checkpoints_dir):
        raise FileNotFoundError(f"Directory '{checkpoints_dir}' not found in '{latest_dir}'.")

    checkpoints_paths = [d for d in os.listdir(checkpoints_dir) if not os.path.isdir(os.path.join(checkpoints_dir, d))]
    sorted_checkpoints_paths = sorted(checkpoints_paths)
    last_model = sorted_checkpoints_paths[-1]
    last_model_path = os.path.join(checkpoints_dir, last_model)

    return last_model_path

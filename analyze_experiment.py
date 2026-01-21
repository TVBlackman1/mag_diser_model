"""
Analyze a completed experiment folder produced by this project.

Reads step-level data from DuckDB (results/<EXPERIMENT>/data.db) and produces:
- per-episode summary CSV
- basic plots (return, length, success rate, min distance)

Usage:
  python analyze_experiment.py --latest
  python analyze_experiment.py --path results/<EXPERIMENT_FOLDER_NAME>
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExperimentPaths:
    folder: Path
    db_path: Path
    out_dir: Path


def _find_latest_experiment(results_dir: Path) -> Path:
    if not results_dir.exists():
        raise FileNotFoundError(f"Results dir not found: {results_dir}")

    candidates = [p for p in results_dir.iterdir() if p.is_dir() and (p / "data.db").exists()]
    if not candidates:
        raise FileNotFoundError(f"No experiments with data.db found in: {results_dir}")

    # Prefer latest by modification time (more robust than lexicographic name).
    candidates.sort(key=lambda p: (p.stat().st_mtime, p.name))
    return candidates[-1]


def _resolve_paths(path: str | None, latest: bool) -> ExperimentPaths:
    root = Path(__file__).resolve().parent
    results_dir = root / "results"

    if latest:
        folder = _find_latest_experiment(results_dir)
    else:
        if not path:
            raise ValueError("Provide --path or use --latest")
        folder = (root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()

    db_path = folder / "data.db"
    if not db_path.exists():
        raise FileNotFoundError(f"data.db not found: {db_path}")

    out_dir = folder / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    return ExperimentPaths(folder=folder, db_path=db_path, out_dir=out_dir)


def _load_episode_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Build a per-episode summary from step-level rows in DuckDB.

    Expects a table named `experiments` (see `utils/db/migration.sql`) with one row per env step.

    Returns a DataFrame with one row per episode:
    - `episode`: episode index
    - `is_train`: whether the episode was logged as training or evaluation
    - `last_step`: last step index observed for the episode
    - `episode_return`: sum(reward) over the episode
    - `episode_len`: last_step + 1 (assuming step starts at 0 and increases by 1)
    - `min_target_distance`: min(new_target_distance) seen during the episode
    - `final_result`: the `result` at the maximum step (e.g. success/fail/timeout)

    Assumption: `step` is monotonically increasing within an episode.
    """
    query = """
    WITH per_ep AS (
      SELECT
        episode,
        any_value(is_train) AS is_train,
        max(step) AS last_step,
        sum(reward) AS episode_return,
        max(step) + 1 AS episode_len,
        min(new_target_distance) AS min_target_distance,
        arg_max(result, step) AS final_result
      FROM experiments
      GROUP BY episode
    )
    SELECT *
    FROM per_ep
    ORDER BY episode;
    """
    return con.execute(query).df()


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    s = pd.Series(x)
    return s.rolling(window=window, min_periods=1).mean().to_numpy()


def _save_plots(df: pd.DataFrame, out_dir: Path, window: int) -> None:
    # Local import so matplotlib isn't required for non-plot usage.
    import matplotlib.pyplot as plt

    # Split train/eval if present
    for is_train, label in [(True, "train"), (False, "eval")]:
        sub = df[df["is_train"] == is_train].copy()
        if sub.empty:
            continue

        ep = sub["episode"].to_numpy()
        ret = sub["episode_return"].to_numpy(dtype=float)
        ln = sub["episode_len"].to_numpy(dtype=float)
        min_dist = sub["min_target_distance"].to_numpy(dtype=float)

        # Return
        plt.figure(figsize=(10, 4))
        plt.plot(ep, ret, alpha=0.4, linewidth=1.0, label="episode_return")
        plt.plot(ep, _rolling_mean(ret, window), linewidth=2.0, label=f"rolling_mean({window})")
        plt.grid(True)
        plt.xlabel("Episode")
        plt.ylabel("Return (sum reward)")
        plt.title(f"Episode return ({label})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"episode_return_{label}.png", dpi=150)
        plt.close()

        # Length
        plt.figure(figsize=(10, 4))
        plt.plot(ep, ln, alpha=0.6, linewidth=1.0)
        plt.grid(True)
        plt.xlabel("Episode")
        plt.ylabel("Episode length (steps)")
        plt.title(f"Episode length ({label})")
        plt.tight_layout()
        plt.savefig(out_dir / f"episode_len_{label}.png", dpi=150)
        plt.close()

        # Min distance
        plt.figure(figsize=(10, 4))
        plt.plot(ep, min_dist, alpha=0.6, linewidth=1.0)
        plt.grid(True)
        plt.xlabel("Episode")
        plt.ylabel("Min distance to target")
        plt.title(f"Min target distance ({label})")
        plt.tight_layout()
        plt.savefig(out_dir / f"min_target_distance_{label}.png", dpi=150)
        plt.close()

        # Success rate (rolling)
        success = (sub["final_result"] == "success").to_numpy(dtype=float)
        success_rate = _rolling_mean(success, window)
        plt.figure(figsize=(10, 4))
        plt.plot(ep, success_rate, linewidth=2.0)
        plt.ylim(-0.05, 1.05)
        plt.grid(True)
        plt.xlabel("Episode")
        plt.ylabel("Success rate (rolling)")
        plt.title(f"Success rate ({label}, window={window})")
        plt.tight_layout()
        plt.savefig(out_dir / f"success_rate_{label}.png", dpi=150)
        plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latest", action="store_true", help="Analyze latest experiment in ./results")
    ap.add_argument("--path", type=str, default=None, help="Path to experiment folder (e.g. results/<name>)")
    ap.add_argument("--window", type=int, default=50, help="Rolling window for smoothing / success rate")
    args = ap.parse_args()

    paths = _resolve_paths(args.path, args.latest)
    con = duckdb.connect(str(paths.db_path), read_only=True)

    df = _load_episode_summary(con)
    out_csv = paths.out_dir / "episode_summary.csv"
    df.to_csv(out_csv, index=False)

    # Print a compact console summary
    def _summ(sub: pd.DataFrame, label: str) -> None:
        if sub.empty:
            return
        n = len(sub)
        succ = float((sub["final_result"] == "success").mean())
        fail = float((sub["final_result"] == "fail").mean())
        timeout = float((sub["final_result"] == "timeout").mean())
        print(
            f"{label}: episodes={n}  success={succ:.3f}  fail={fail:.3f}  timeout={timeout:.3f}  "
            f"return_mean={sub['episode_return'].mean():.2f}  len_mean={sub['episode_len'].mean():.1f}"
        )

    _summ(df[df["is_train"] == True], "train")
    _summ(df[df["is_train"] == False], "eval")

    _save_plots(df, paths.out_dir, window=max(1, args.window))
    print(f"Saved: {out_csv}")
    print(f"Plots in: {paths.out_dir}")


if __name__ == "__main__":
    main()


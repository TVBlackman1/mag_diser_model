"""
Run inference for the latest saved SB3 model many times and plot distributions:
- distance covered towards the target (start_distance - min_distance, and start - final)
- episode return (sum of rewards)

Outputs:
  <out_dir>/episode_metrics.csv
  <out_dir>/hist_distance_covered.png
  <out_dir>/hist_distance_covered_fraction.png
  <out_dir>/hist_episode_return.png

Usage:
  python inference_last_distributions.py --episodes 300 --latest
  python inference_last_distributions.py --episodes 300 --model checkpoints/td3_drone_final.zip
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stable_baselines3 import DDPG, TD3

from env.drone_env import DroneEnv
from train_sb3 import create_env


@dataclass(frozen=True)
class EpisodeMetrics:
    episode: int
    final_result: str
    episode_len: int
    episode_return: float
    start_distance: float
    min_distance: float
    final_distance: float
    distance_covered: float
    distance_covered_fraction: float


def _latest_model_path() -> Path:
    root = Path(__file__).resolve().parent
    candidates: list[Path] = []

    for folder in [root / "best_model", root / "checkpoints"]:
        if not folder.exists():
            continue
        candidates.extend([p for p in folder.glob("*.zip") if p.is_file()])

    if not candidates:
        raise FileNotFoundError("No SB3 model .zip found in ./best_model or ./checkpoints")

    candidates.sort(key=lambda p: (p.stat().st_mtime, p.name))
    return candidates[-1]


def _load_model(model_path: Path):
    # Try TD3 first (your training script defaults to TD3), then DDPG.
    try:
        return TD3.load(str(model_path), device="cpu")
    except Exception:
        return DDPG.load(str(model_path), device="cpu")


def _unwrap_to_drone_env(env: Any) -> DroneEnv:
    current = env
    # Monitor / wrappers expose `.env`
    while hasattr(current, "env"):
        if isinstance(current, DroneEnv):
            return current
        current = current.env
    if isinstance(current, DroneEnv):
        return current
    raise RuntimeError("Failed to unwrap to DroneEnv")


def _distance(drone_env: DroneEnv) -> float:
    # Requires env_data to be initialized (after reset()).
    pos = drone_env.env_data.drone.position
    target = drone_env.env_data.target_position
    return float(np.linalg.norm(pos - target))


def run_inference(
    model_path: Path,
    episodes: int,
    seed: int,
    deterministic: bool,
) -> tuple[list[EpisodeMetrics], Path]:
    root = Path(__file__).resolve().parent
    ts = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    out_dir = root / "inference_runs" / f"{ts}_last_model_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model(model_path)

    # Eval env: different episodes generator (see train_sb3.create_env)
    env = create_env(is_eval=True, db_saver=None)
    env.reset(seed=seed)

    metrics: list[EpisodeMetrics] = []

    for ep in range(episodes):
        obs, _info = env.reset(seed=seed + ep)
        drone_env = _unwrap_to_drone_env(env)
        start_dist = _distance(drone_env)
        min_dist = start_dist
        final_dist = start_dist

        done = False
        total_reward = 0.0
        steps = 0
        final_result = "process"

        while not done:
            action, _state = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            total_reward += float(reward)
            steps += 1

            # Track distances
            # Prefer info distances when present; otherwise recompute from env state.
            new_d = info.get("new_distance", None)
            if new_d is None or float(new_d) < 0:
                final_dist = _distance(drone_env)
            else:
                final_dist = float(new_d)
            min_dist = min(min_dist, final_dist)

            final_result = info.get("result", final_result)

        covered = max(0.0, start_dist - min_dist)
        covered_frac = covered / start_dist if start_dist > 1e-9 else 0.0

        metrics.append(
            EpisodeMetrics(
                episode=ep,
                final_result=str(final_result),
                episode_len=int(steps),
                episode_return=float(total_reward),
                start_distance=float(start_dist),
                min_distance=float(min_dist),
                final_distance=float(final_dist),
                distance_covered=float(covered),
                distance_covered_fraction=float(covered_frac),
            )
        )

    env.close()
    return metrics, out_dir


def _plot_hist(values: np.ndarray, title: str, xlabel: str, out_path: Path, bins: int = 40) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 4))
    plt.hist(values, bins=bins, color="skyblue", edgecolor="black", alpha=0.9)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--deterministic", action="store_true", help="Use deterministic policy actions")
    ap.add_argument("--latest", action="store_true", help="Auto-pick latest .zip in best_model/ or checkpoints/")
    ap.add_argument("--model", type=str, default=None, help="Path to SB3 .zip model")
    args = ap.parse_args()

    if args.latest:
        model_path = _latest_model_path()
    else:
        if not args.model:
            raise ValueError("Provide --latest or --model <path>")
        model_path = Path(args.model).resolve()

    metrics, out_dir = run_inference(
        model_path=model_path,
        episodes=max(1, int(args.episodes)),
        seed=int(args.seed),
        deterministic=bool(args.deterministic),
    )

    df = pd.DataFrame([m.__dict__ for m in metrics])
    df.to_csv(out_dir / "episode_metrics.csv", index=False)

    # Console summary
    counts = df["final_result"].value_counts(dropna=False).to_dict()
    print(f"Model: {model_path}")
    print(f"Saved to: {out_dir}")
    print(f"Result counts: {counts}")
    print(df[["episode_return", "distance_covered", "distance_covered_fraction"]].describe(percentiles=[0.1, 0.5, 0.9]).to_string())

    _plot_hist(
        df["distance_covered"].to_numpy(dtype=float),
        title="Distribution: distance covered towards target (start - min_distance)",
        xlabel="Distance covered",
        out_path=out_dir / "hist_distance_covered.png",
    )
    _plot_hist(
        df["distance_covered_fraction"].to_numpy(dtype=float),
        title="Distribution: fraction of initial distance covered",
        xlabel="Covered fraction",
        out_path=out_dir / "hist_distance_covered_fraction.png",
    )
    _plot_hist(
        df["episode_return"].to_numpy(dtype=float),
        title="Distribution: episode return (sum of rewards)",
        xlabel="Episode return",
        out_path=out_dir / "hist_episode_return.png",
    )


if __name__ == "__main__":
    main()


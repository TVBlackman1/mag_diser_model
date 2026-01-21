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

TARGET_STRATEGIES = ("static", "random", "fleeing")


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

    # Search recursively (models are saved under best_model/<strategy>/ and checkpoints/<strategy>/).
    for folder in [root / "best_model", root / "checkpoints"]:
        if not folder.exists():
            continue
        candidates.extend([p for p in folder.rglob("*.zip") if p.is_file()])

    if not candidates:
        raise FileNotFoundError("No SB3 model .zip found in ./best_model or ./checkpoints")

    candidates.sort(key=lambda p: (p.stat().st_mtime, p.name))
    return candidates[-1]


def _latest_model_for_strategy_optional(strategy: str, root: Path | None = None) -> Path | None:
    """
    Find the latest model for a given target strategy.
    Prefers ./best_model/<strategy>/*.zip, then ./checkpoints/<strategy>/*.zip.
    """
    root = root or Path(__file__).resolve().parent

    best_dir = root / "best_model" / strategy
    ckpt_dir = root / "checkpoints" / strategy

    best = [p for p in best_dir.glob("*.zip") if p.is_file()] if best_dir.exists() else []
    ckpt = [p for p in ckpt_dir.glob("*.zip") if p.is_file()] if ckpt_dir.exists() else []

    if best:
        best.sort(key=lambda p: (p.stat().st_mtime, p.name))
        return best[-1]
    if ckpt:
        ckpt.sort(key=lambda p: (p.stat().st_mtime, p.name))
        return ckpt[-1]
    return None


def _latest_model_for_strategy(strategy: str) -> Path:
    """Strict variant used by non-matrix flows."""
    p = _latest_model_for_strategy_optional(strategy)
    if p is None:
        raise FileNotFoundError(f"No model .zip found for strategy '{strategy}' in best_model/ or checkpoints/")
    return p


def _load_model(model_path: Path):
    # Try TD3 first (your training script defaults to TD3), then DDPG.
    try:
        return TD3.load(str(model_path), device="cpu")
    except Exception:
        return DDPG.load(str(model_path), device="cpu")

def _infer_strategy_from_model_path(model_path: Path) -> str:
    parts = {p.lower() for p in model_path.parts}
    for s in TARGET_STRATEGIES:
        if s in parts:
            return s
    return "static"


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
    env_target_strategy: str = "static",
) -> tuple[list[EpisodeMetrics], Path]:
    root = Path(__file__).resolve().parent
    ts = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    out_dir = root / "inference_runs" / f"{ts}_last_model_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model(model_path)

    # Eval env: different episodes generator (see train_sb3.create_env)
    env = create_env(is_eval=True, db_saver=None, target_strategy=env_target_strategy)
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


def run_matrix_inference(
    episodes: int,
    seed: int,
    deterministic: bool,
    root: Path | None = None,
) -> Path:
    """
    Run 9 evaluations:
      model trained on {static,random,fleeing}  x  env target strategy {static,random,fleeing}

    Prints two matrices to console:
      - P97 of path coverage (%) per cell
      - P75 of path coverage (%) per cell

    Coverage% = (start_distance - min_distance) / start_distance * 100
    """
    root = root or Path(__file__).resolve().parent
    ts = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    out_dir = root / "inference_runs" / f"{ts}_matrix_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(TARGET_STRATEGIES)  # trained-on
    cols = list(TARGET_STRATEGIES)  # eval-on

    p97 = np.full((len(rows), len(cols)), np.nan, dtype=float)
    p75 = np.full((len(rows), len(cols)), np.nan, dtype=float)
    success_rate = np.full((len(rows), len(cols)), np.nan, dtype=float)

    all_rows: list[dict[str, Any]] = []

    for i, model_strat in enumerate(rows):
        model_path = _latest_model_for_strategy_optional(model_strat, root=root)
        if model_path is None:
            # No model for this strategy; keep NA row.
            continue
        model = _load_model(model_path)

        for j, env_strat in enumerate(cols):
            env = create_env(is_eval=True, db_saver=None, target_strategy=env_strat)
            env.reset(seed=seed)

            coverages: list[float] = []
            successes = 0

            for ep in range(episodes):
                obs, _ = env.reset(seed=seed + ep)
                drone_env = _unwrap_to_drone_env(env)
                start_dist = _distance(drone_env)
                min_dist = start_dist

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

                    new_d = info.get("new_distance", None)
                    if new_d is None or float(new_d) < 0:
                        cur_d = _distance(drone_env)
                    else:
                        cur_d = float(new_d)
                    min_dist = min(min_dist, cur_d)
                    final_result = str(info.get("result", final_result))

                covered = max(0.0, start_dist - min_dist)
                covered_frac = covered / start_dist if start_dist > 1e-9 else 0.0
                coverage_pct = covered_frac * 100.0
                coverages.append(float(coverage_pct))

                if final_result == "success":
                    successes += 1

                all_rows.append(
                    {
                        "model_strategy": model_strat,
                        "env_strategy": env_strat,
                        "model_path": str(model_path),
                        "episode": ep,
                        "final_result": final_result,
                        "episode_len": steps,
                        "episode_return": total_reward,
                        "coverage_pct": coverage_pct,
                    }
                )

            env.close()

            p97[i, j] = float(np.percentile(np.array(coverages, dtype=float), 97))
            p75[i, j] = float(np.percentile(np.array(coverages, dtype=float), 75))
            success_rate[i, j] = (successes / float(episodes)) * 100.0 if episodes > 0 else float("nan")

    def _print_matrix(name: str, mat: np.ndarray) -> None:
        print("")
        print(name)
        header = "model\\env".ljust(12) + " ".join([c.rjust(10) for c in cols])
        print(header)
        for r_idx, r in enumerate(rows):
            cells = []
            for c_idx in range(len(cols)):
                v = mat[r_idx, c_idx]
                cells.append("       NA" if np.isnan(v) else f"{v:9.2f}%")
            line = r.ljust(12) + " ".join(cells)
            print(line)

    # Report which models are available
    available = {s: _latest_model_for_strategy_optional(s, root=root) for s in rows}
    print("Available models:")
    for s in rows:
        print(f"  - {s}: {available[s] if available[s] is not None else 'NA'}")

    _print_matrix("P97 coverage (%)", p97)
    _print_matrix("P75 coverage (%)", p75)
    _print_matrix("Success rate (%)", success_rate)

    pd.DataFrame(all_rows).to_csv(out_dir / "matrix_episode_metrics.csv", index=False)
    print(f"\nSaved to: {out_dir}")
    return out_dir


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
    ap.add_argument("--matrix", action="store_true", help="Run 3x3 (9) evaluations across target strategies")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--deterministic", action="store_true", help="Use deterministic policy actions")
    ap.add_argument("--latest", action="store_true", help="Auto-pick latest .zip in best_model/ or checkpoints/")
    ap.add_argument("--model", type=str, default=None, help="Path to SB3 .zip model")
    ap.add_argument(
        "--target-strategy",
        type=str,
        default="auto",
        choices=(*TARGET_STRATEGIES, "auto"),
        help="Target strategy used during evaluation (single-model mode). 'auto' infers from model path.",
    )
    args = ap.parse_args()

    if args.matrix:
        run_matrix_inference(
            episodes=max(1, int(args.episodes)),
            seed=int(args.seed),
            deterministic=bool(args.deterministic),
        )
        return

    if args.latest:
        model_path = _latest_model_path()
    else:
        if not args.model:
            raise ValueError("Provide --latest or --model <path>")
        model_path = Path(args.model).resolve()

    env_target_strategy = (
        _infer_strategy_from_model_path(model_path) if args.target_strategy == "auto" else args.target_strategy
    )

    metrics, out_dir = run_inference(
        model_path=model_path,
        episodes=max(1, int(args.episodes)),
        seed=int(args.seed),
        deterministic=bool(args.deterministic),
        env_target_strategy=env_target_strategy,
    )

    df = pd.DataFrame([m.__dict__ for m in metrics])
    df.to_csv(out_dir / "episode_metrics.csv", index=False)

    # Console summary
    counts = df["final_result"].value_counts(dropna=False).to_dict()
    print(f"Model: {model_path}")
    print(f"Eval target strategy: {env_target_strategy}")
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


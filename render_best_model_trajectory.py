"""
Pick the most successful saved SB3 model and render a trajectory to PDF.
Additionally (for now) can render an animated GIF for the random-moving target.

Definition of "most successful":
  1) Highest success rate over N eval episodes (deterministic policy)
  2) Tie-breaker: higher mean episode return

Outputs a single PDF plot with:
  - drone trajectory (world coordinates)
  - obstacles (markers)
  - target trajectory (world coordinates) and current target marker

For `target-strategy=random`, also outputs a GIF animation where the drone and target
trajectories are filled in over time.

Usage:
  poetry run python render_best_model_trajectory.py --target-strategy random --eval-episodes 50 --seed 123
  poetry run python render_best_model_trajectory.py --model checkpoints/random/td3_random_final.zip --target-strategy random
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from stable_baselines3 import DDPG, TD3

from config.env_config import FIELD_SIZE, OBSTACLE_MEDIUM_START_FRAC
from config.train_config import NUM_EPISODES
from env.drone_env import DroneEnv
from env.drone_env_state_wrapper import DroneEnvStateWrapper
from train_sb3 import create_env

TARGET_STRATEGIES = ("static", "random", "fleeing")


@dataclass(frozen=True)
class ModelScore:
    model_path: Path
    success_rate: float
    mean_return: float
    successes: int
    episodes: int


def _iter_model_paths() -> list[Path]:
    root = Path(__file__).resolve().parent
    paths: list[Path] = []
    for folder in [root / "best_model", root / "checkpoints"]:
        if folder.exists():
            paths.extend([p for p in folder.rglob("*.zip") if p.is_file()])
    # Newest last
    paths.sort(key=lambda p: (p.stat().st_mtime, p.name))
    return paths


def _iter_model_paths_for_strategy(strategy: str) -> list[Path]:
    root = Path(__file__).resolve().parent
    paths: list[Path] = []
    for folder in [root / "best_model" / strategy, root / "checkpoints" / strategy]:
        if folder.exists():
            paths.extend([p for p in folder.rglob("*.zip") if p.is_file()])
    paths.sort(key=lambda p: (p.stat().st_mtime, p.name))
    return paths


def _load_model(model_path: Path):
    # Try TD3 first (default), then DDPG.
    try:
        return TD3.load(str(model_path), device="cpu")
    except Exception:
        return DDPG.load(str(model_path), device="cpu")


def _unwrap_to_drone_env(env: Any) -> DroneEnv:
    current = env
    while hasattr(current, "env"):
        if isinstance(current, DroneEnv):
            return current
        current = current.env
    if isinstance(current, DroneEnv):
        return current
    raise RuntimeError("Failed to unwrap to DroneEnv")

def _unwrap_to_state_wrapper(env: Any) -> DroneEnvStateWrapper | None:
    current = env
    while hasattr(current, "env"):
        if isinstance(current, DroneEnvStateWrapper):
            return current
        current = current.env
    return current if isinstance(current, DroneEnvStateWrapper) else None


def _rollout_one_episode(
    env, model, *, seed: int, deterministic: bool
) -> tuple[float, str, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """
    Returns:
      (episode_return, final_result, drone_positions, target_positions, obstacles)
    """
    obs, _ = env.reset(seed=seed)
    base = _unwrap_to_drone_env(env)

    drone_traj: list[np.ndarray] = [base.env_data.drone.position.copy()]
    target_traj: list[np.ndarray] = [base.env_data.target_position.copy()]
    obstacles = [np.array(o, dtype=float).copy() for o in (base.env_data.obstacles or [])]

    total_reward = 0.0
    final_result = "process"
    done = False

    while not done:
        action, _state = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        done = bool(terminated or truncated)
        final_result = str(info.get("result", final_result))
        drone_traj.append(base.env_data.drone.position.copy())
        target_traj.append(base.env_data.target_position.copy())

    return float(total_reward), final_result, drone_traj, target_traj, obstacles


def _score_model(
    model_path: Path,
    *,
    episodes: int,
    seed: int,
    deterministic: bool,
    target_strategy: str,
    start_episode: int,
    difficulty: str,
    with_obstacles: bool,
) -> ModelScore:
    model = _load_model(model_path)
    env = create_env(is_eval=True, db_saver=None, target_strategy=target_strategy)

    returns: list[float] = []
    successes = 0
    used_eps = 0

    state = _unwrap_to_state_wrapper(env)
    if state is not None:
        state.current_episode = int(start_episode)
        state.difficulty_level = str(difficulty)

    max_attempts = episodes * 10 if with_obstacles else episodes
    for i in range(max_attempts):
        ep_ret, result, _drone_traj, _target_traj, obstacles = _rollout_one_episode(
            env, model, seed=seed + i, deterministic=deterministic
        )
        if with_obstacles and not obstacles:
            continue
        used_eps += 1
        returns.append(ep_ret)
        if result == "success":
            successes += 1
        if used_eps >= episodes:
            break

    env.close()
    mean_ret = float(np.mean(returns)) if returns else float("nan")
    success_rate = successes / float(used_eps) if used_eps > 0 else 0.0
    return ModelScore(
        model_path=model_path,
        success_rate=float(success_rate),
        mean_return=mean_ret,
        successes=int(successes),
        episodes=int(used_eps),
    )


def _pick_best_model(model_paths: Iterable[Path], *, episodes: int, seed: int) -> ModelScore:
    best: ModelScore | None = None
    for p in model_paths:
        # Model is evaluated on the same target strategy as its folder selection.
        # Caller is expected to pass the desired target strategy.
        raise RuntimeError("Call _pick_best_model_for_strategy instead")
        if best is None:
            best = score
            continue

        if score.success_rate > best.success_rate:
            best = score
        elif score.success_rate == best.success_rate and score.mean_return > best.mean_return:
            best = score

    if best is None:
        raise FileNotFoundError("No model .zip found in ./best_model or ./checkpoints")
    return best


def _pick_best_model_for_strategy(
    target_strategy: str,
    *,
    episodes: int,
    seed: int,
    start_episode: int,
    difficulty: str,
    with_obstacles: bool,
) -> ModelScore:
    candidates = _iter_model_paths_for_strategy(target_strategy)
    if not candidates:
        raise FileNotFoundError(f"No model .zip found for strategy '{target_strategy}'")

    best: ModelScore | None = None
    for p in candidates:
        score = _score_model(
            p,
            episodes=episodes,
            seed=seed,
            deterministic=True,
            target_strategy=target_strategy,
            start_episode=start_episode,
            difficulty=difficulty,
            with_obstacles=with_obstacles,
        )
        if best is None:
            best = score
            continue
        if score.success_rate > best.success_rate:
            best = score
        elif score.success_rate == best.success_rate and score.mean_return > best.mean_return:
            best = score

    assert best is not None
    return best


def _save_pdf(
    out_path: Path,
    *,
    model_path: Path,
    episode_seed: int,
    episode_return: float,
    final_result: str,
    drone_trajectory: list[np.ndarray],
    target_trajectory: list[np.ndarray],
    obstacles: list[np.ndarray],
) -> None:
    import matplotlib.pyplot as plt

    drone_traj = np.array(drone_trajectory, dtype=float)
    target_traj = np.array(target_trajectory, dtype=float)

    plt.figure(figsize=(8.5, 8.5))
    plt.title(
        f"Trajectory ({final_result}) | return={episode_return:.2f}\n"
        f"model={model_path.name} | seed={episode_seed}"
    )
    plt.grid(True, alpha=0.3)

    # World bounds
    plt.xlim(0, FIELD_SIZE)
    plt.ylim(0, FIELD_SIZE)
    plt.gca().set_aspect("equal", adjustable="box")

    # Plot drone trajectory
    plt.plot(drone_traj[:, 0], drone_traj[:, 1], "-", linewidth=1.5, alpha=0.9, label="Drone path")
    plt.plot(drone_traj[0, 0], drone_traj[0, 1], "bo", label="Drone start")
    plt.plot(drone_traj[-1, 0], drone_traj[-1, 1], "ko", label="Drone end")

    # Target trajectory (moving target) and final marker
    plt.plot(target_traj[:, 0], target_traj[:, 1], "-", linewidth=1.2, alpha=0.8, label="Target path")
    plt.plot(target_traj[0, 0], target_traj[0, 1], "r.", markersize=8, label="Target start")
    plt.plot(target_traj[-1, 0], target_traj[-1, 1], "r*", markersize=14, label="Target end")

    # Obstacles
    if obstacles:
        obs = np.array(obstacles, dtype=float)
        plt.scatter(obs[:, 0], obs[:, 1], marker="x", c="black", s=40, label="Obstacles")

    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()


def _save_gif(
    out_path: Path,
    *,
    model_path: Path,
    episode_seed: int,
    episode_return: float,
    final_result: str,
    drone_trajectory: list[np.ndarray],
    target_trajectory: list[np.ndarray],
    obstacles: list[np.ndarray],
    fps: int = 20,
    frame_stride: int = 1,
) -> None:
    """
    Save an animated GIF where drone/target paths are revealed over time.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    drone_traj = np.array(drone_trajectory, dtype=float)
    target_traj = np.array(target_trajectory, dtype=float)

    # Coverage (%): how much of initial distance to target has been "covered".
    # We use the best-so-far (min) distance up to current step to make it monotonic.
    d0 = float(np.linalg.norm(drone_traj[0] - target_traj[0]))
    dists = np.linalg.norm(drone_traj - target_traj, axis=1).astype(float)
    min_dists = np.minimum.accumulate(dists)

    # Stride frames to control gif size
    idx = np.arange(0, len(drone_traj), max(1, int(frame_stride)))
    if idx[-1] != len(drone_traj) - 1:
        idx = np.append(idx, len(drone_traj) - 1)

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.set_title(
        f"Trajectory ({final_result}) | return={episode_return:.2f}\n"
        f"model={model_path.name} | seed={episode_seed}"
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, FIELD_SIZE)
    ax.set_ylim(0, FIELD_SIZE)
    ax.set_aspect("equal", adjustable="box")

    # Obstacles
    if obstacles:
        obs = np.array(obstacles, dtype=float)
        ax.scatter(obs[:, 0], obs[:, 1], marker="x", c="black", s=40, label="Obstacles")

    # Paths and current positions
    (drone_line,) = ax.plot([], [], "-", linewidth=1.8, alpha=0.95, label="Drone path")
    (target_line,) = ax.plot([], [], "-", linewidth=1.4, alpha=0.85, label="Target path")
    drone_dot = ax.plot([], [], "bo", markersize=6, label="Drone")[0]
    target_dot = ax.plot([], [], "r*", markersize=10, label="Target")[0]

    # Start markers
    ax.plot(drone_traj[0, 0], drone_traj[0, 1], "bo", markersize=8)
    ax.plot(target_traj[0, 0], target_traj[0, 1], "r.", markersize=8)

    # HUD text (top-left)
    hud = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75, edgecolor="none"),
    )

    ax.legend(loc="upper right")

    def init():
        drone_line.set_data([], [])
        target_line.set_data([], [])
        drone_dot.set_data([], [])
        target_dot.set_data([], [])
        hud.set_text("")
        return drone_line, target_line, drone_dot, target_dot

    def update(frame_i: int):
        k = int(idx[frame_i])
        drone_line.set_data(drone_traj[: k + 1, 0], drone_traj[: k + 1, 1])
        target_line.set_data(target_traj[: k + 1, 0], target_traj[: k + 1, 1])
        # Line2D.set_data expects sequences, not scalars
        drone_dot.set_data([drone_traj[k, 0]], [drone_traj[k, 1]])
        target_dot.set_data([target_traj[k, 0]], [target_traj[k, 1]])

        if d0 > 1e-9:
            coverage_pct = max(0.0, (d0 - float(min_dists[k])) / d0) * 100.0
        else:
            coverage_pct = 0.0
        hud.set_text(f"Step: {k}\nCoverage: {coverage_pct:5.1f}%")
        return drone_line, target_line, drone_dot, target_dot

    anim = FuncAnimation(fig, update, frames=len(idx), init_func=init, interval=1000 / fps, blit=True)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=None, help="Specific SB3 model .zip to render")
    ap.add_argument(
        "--target-strategy",
        type=str,
        default="random",
        choices=TARGET_STRATEGIES,
        help="Target movement strategy to render (GIF currently implemented for 'random').",
    )
    ap.add_argument("--eval-episodes", type=int, default=50, help="Episodes per model for scoring")
    ap.add_argument("--seed", type=int, default=123, help="Base seed for evaluation")
    ap.add_argument(
        "--start-episode",
        type=int,
        default=0,
        help="Force episode index for difficulty schedule (e.g. 240 to enter 'medium' in default schedule)",
    )
    ap.add_argument(
        "--difficulty",
        type=str,
        default="",
        help="Force difficulty level for generators (e.g. 'medium'); empty = generator decides",
    )
    ap.add_argument(
        "--with-obstacles",
        action="store_true",
        help="Pick best model/episode only from scenarios with obstacles (defaults to medium difficulty unless overridden).",
    )
    ap.add_argument("--out", type=str, default=None, help="Output PDF path")
    ap.add_argument("--gif-fps", type=int, default=20, help="GIF frames per second (random strategy only)")
    ap.add_argument("--gif-stride", type=int, default=1, help="Take every N-th step as a GIF frame")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent

    # If obstacle-only evaluation is requested, ensure episode/difficulty settings produce obstacles.
    with_obstacles = bool(args.with_obstacles)
    start_episode = int(args.start_episode)
    difficulty = str(args.difficulty)
    if with_obstacles:
        if start_episode == 0:
            start_episode = int(NUM_EPISODES * OBSTACLE_MEDIUM_START_FRAC)
        if difficulty == "":
            difficulty = "medium"

    if args.model:
        best = ModelScore(model_path=Path(args.model).resolve(), success_rate=float("nan"), mean_return=float("nan"), successes=0, episodes=0)
    else:
        best = _pick_best_model_for_strategy(
            str(args.target_strategy),
            episodes=max(1, int(args.eval_episodes)),
            seed=int(args.seed),
            start_episode=start_episode,
            difficulty=difficulty,
            with_obstacles=with_obstacles,
        )
        print(
            f"Best model: {best.model_path} | success_rate={best.success_rate:.3f} "
            f"({best.successes}/{best.episodes}) | mean_return={best.mean_return:.2f}"
        )

    model = _load_model(best.model_path)
    env = create_env(is_eval=True, db_saver=None, target_strategy=str(args.target_strategy))

    # Ensure we can render episodes with obstacles:
    # - either force difficulty explicitly
    # - or start from a higher episode index so schedule picks 'medium'
    state = _unwrap_to_state_wrapper(env)
    if state is not None:
        state.current_episode = int(start_episode)
        state.difficulty_level = str(difficulty)

    # Rollout multiple episode seeds and pick the best successful one; fallback to best return.
    best_ep = None
    best_success_ep = None
    used_eps = 0
    max_requested = max(1, int(args.eval_episodes))
    max_attempts = max_requested * (10 if with_obstacles else 1)
    for i in range(max_attempts):
        ep_seed = int(args.seed) + i
        ep_ret, result, drone_traj, target_traj, obstacles = _rollout_one_episode(
            env, model, seed=ep_seed, deterministic=True
        )

        if with_obstacles and not obstacles:
            continue

        used_eps += 1
        cand = (ep_ret, result, drone_traj, target_traj, obstacles, ep_seed)
        if best_ep is None or ep_ret > best_ep[0]:
            best_ep = cand
        if result == "success" and (best_success_ep is None or ep_ret > best_success_ep[0]):
            best_success_ep = cand
        if used_eps >= max_requested:
            break

    env.close()

    chosen = best_success_ep if best_success_ep is not None else best_ep
    assert chosen is not None
    ep_ret, result, drone_traj, target_traj, obstacles, ep_seed = chosen

    if args.out:
        out_path = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out).resolve()
    else:
        ts = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        out_path = root / "inference_runs" / f"{ts}_best_model_trajectory.pdf"
        out_path.parent.mkdir(parents=True, exist_ok=True)

    _save_pdf(
        out_path,
        model_path=best.model_path,
        episode_seed=ep_seed,
        episode_return=float(ep_ret),
        final_result=str(result),
        drone_trajectory=drone_traj,
        target_trajectory=target_traj,
        obstacles=obstacles,
    )
    print(f"Saved PDF: {out_path}")

    # GIF is currently implemented only for random-moving target (architectural placeholder for others).
    if str(args.target_strategy) == "random":
        gif_path = out_path.with_suffix(".gif")
        _save_gif(
            gif_path,
            model_path=best.model_path,
            episode_seed=ep_seed,
            episode_return=float(ep_ret),
            final_result=str(result),
            drone_trajectory=drone_traj,
            target_trajectory=target_traj,
            obstacles=obstacles,
            fps=int(args.gif_fps),
            frame_stride=int(args.gif_stride),
        )
        print(f"Saved GIF: {gif_path}")


if __name__ == "__main__":
    main()

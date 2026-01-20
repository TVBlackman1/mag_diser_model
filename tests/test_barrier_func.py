import numpy as np

from config.env_config import FIELD_SIZE
from utils.barrier import compute_normalized_force
from utils.generation import generate_environment_categorial


def test_compute_normalized_force_smoke():
    """Basic sanity check: barrier force computation returns finite values."""
    np.random.seed(42)
    env = generate_environment_categorial(FIELD_SIZE, "medium")

    rel_target = np.array(env["target_pos"]) - env["drone_pos"]
    rel_obstacles = [np.array(obs) - env["drone_pos"] for obs in env["obstacles"]]

    direction, magnitude = compute_normalized_force(np.array([0.0, 0.0]), rel_target, rel_obstacles)
    assert direction.shape == (2,)
    assert np.isfinite(direction).all()
    assert np.isfinite(magnitude)


def main() -> None:
    """Manual visualization helper (not run during pytest)."""
    from matplotlib import pyplot as plt

    np.random.seed(42)
    env = generate_environment_categorial(FIELD_SIZE, "medium")

    plt.figure(figsize=(10, 6))
    plt.title("World barrier function (relative to drone)")
    plt.grid(True)

    plt.plot(0, 0, "bo", label="Drone")

    max_dist = 1.0
    rel_target = np.array(env["target_pos"]) - env["drone_pos"]
    plt.plot(rel_target[0], rel_target[1], "ro", label="Target")
    max_dist = max(max_dist, float(np.linalg.norm(rel_target)))

    rel_obstacles = []
    for i, obs in enumerate(env["obstacles"]):
        rel_obs = np.array(obs) - env["drone_pos"]
        max_dist = max(max_dist, float(np.linalg.norm(rel_obs)))
        rel_obstacles.append(rel_obs)
        plt.plot(rel_obs[0], rel_obs[1], "kx", label="Obstacle" if i == 0 else None)

    lim = max_dist * 1.2

    portrait_resolution = 56
    world_grid = np.linspace(-lim, lim, portrait_resolution)
    X, Y = np.meshgrid(world_grid, world_grid)

    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    M = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i, j], Y[i, j]])
            direction, magnitude = compute_normalized_force(point, rel_target, rel_obstacles)
            U[i, j] = direction[0]
            V[i, j] = direction[1]
            M[i, j] = magnitude

    quiv = plt.quiver(X, Y, U, V, M, cmap="cool", scale=90)
    cbar = plt.colorbar(quiv)
    cbar.set_label("Field Strength (|F|)", rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=8)

    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("barrier_func.pdf", format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
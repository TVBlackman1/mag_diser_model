"""
Target movement strategies for the drone environment.
Each strategy defines how the target moves during each step.
"""
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import numpy.typing as npt
from utils.generation import EnvData
from utils.checks import is_collision
from config.env_config import OBSTACLE_COLLISION_MAX_RADIUS, FIELD_SIZE, DELTA_T


def is_target_too_close_to_obstacle(
    target_position: npt.NDArray[np.float64],
    obstacles: list[npt.NDArray[np.float64]],
    min_distance: float
) -> bool:
    """
    Check if target is too close to any obstacle.
    
    Args:
        target_position: Target position.
        obstacles: List of obstacle positions.
        min_distance: Minimum distance to maintain from obstacles.
        
    Returns:
        True if target is too close to any obstacle.
    """
    for obstacle in obstacles:
        distance = np.linalg.norm(target_position - obstacle)
        if distance < min_distance:
            return True
    return False


class TargetStrategy(ABC):
    """Base class for target movement strategies."""
    
    def __init__(self, rng: Optional[np.random.Generator] = None):
        """
        Initialize the target strategy.
        
        Args:
            rng: Random number generator for reproducibility. If None, creates a new one.
        """
        self.rng = rng if rng is not None else np.random.default_rng()
        self.orientation: float = 0.0  # Current orientation of the target (for movement strategies)
    
    @abstractmethod
    def update(
        self, 
        current_position: npt.NDArray[np.float64],
        env_data: EnvData,
        delta_time: float
    ) -> npt.NDArray[np.float64]:
        """
        Update target position based on the strategy.
        
        Args:
            current_position: Current target position.
            env_data: Current environment data (drone, obstacles, etc.).
            delta_time: Time step duration.
            
        Returns:
            New target position.
        """
        pass
    
    def ensure_valid_position(
        self,
        position: npt.NDArray[np.float64],
        obstacles: list[npt.NDArray[np.float64]],
        min_distance_to_obstacle: float
    ) -> npt.NDArray[np.float64]:
        """
        Ensure target position is valid (not too close to obstacles, within bounds).
        If position is invalid, moves it to a valid location.
        
        Args:
            position: Position to validate.
            obstacles: List of obstacle positions.
            min_distance_to_obstacle: Minimum distance to maintain from obstacles.
            
        Returns:
            Valid position (may be the same as input if already valid).
        """
        # Check if position is too close to obstacles
        if obstacles:
            for obstacle in obstacles:
                distance = np.linalg.norm(position - obstacle)
                if distance < min_distance_to_obstacle:
                    # Move away from obstacle
                    direction = position - obstacle
                    if np.linalg.norm(direction) < 1e-6:
                        # If exactly on obstacle, choose random direction
                        angle = self.rng.uniform(0.0, 2 * np.pi)
                        direction = np.array([np.cos(angle), np.sin(angle)])
                    direction = direction / np.linalg.norm(direction)
                    position = obstacle + direction * min_distance_to_obstacle
        
        # Clip to field boundaries
        position = np.clip(position, 0.0, FIELD_SIZE)
        
        return position
    
    def move_like_drone(
        self,
        position: npt.NDArray[np.float64],
        speed_ratio: float,
        angle_ratio: float,
        delta_time: float,
        max_speed: float,
        max_angular_velocity: float
    ) -> tuple[npt.NDArray[np.float64], float]:
        """
        Move position similar to how a drone moves (forward/backward + rotation).
        
        Args:
            position: Current position.
            speed_ratio: Speed ratio (-1 to 1).
            angle_ratio: Angular velocity ratio (-1 to 1).
            delta_time: Time step duration.
            max_speed: Maximum forward speed.
            max_angular_velocity: Maximum angular velocity.
            
        Returns:
            Tuple of (new_position, new_orientation).
        """
        # Update orientation
        self.orientation += angle_ratio * max_angular_velocity * delta_time
        self.orientation = np.arctan2(np.sin(self.orientation), np.cos(self.orientation))
        
        # Move forward/backward
        vector = speed_ratio * max_speed * delta_time
        new_position = position + np.array([
            np.cos(self.orientation) * vector,
            np.sin(self.orientation) * vector
        ])
        
        return new_position, self.orientation


class StaticTargetStrategy(TargetStrategy):
    """Target does not move."""
    
    def update(
        self,
        current_position: npt.NDArray[np.float64],
        env_data: EnvData,
        delta_time: float
    ) -> npt.NDArray[np.float64]:
        """Target remains static."""
        return current_position.copy()


class RandomMovingTargetStrategy(TargetStrategy):
    """Target moves randomly, avoiding obstacles."""
    
    def __init__(
        self,
        max_speed: float,
        max_angular_velocity: float,
        min_distance_to_obstacle: float,
        rng: Optional[np.random.Generator] = None
    ):
        """
        Initialize random moving target strategy.
        
        Args:
            max_speed: Maximum forward speed of the target.
            max_angular_velocity: Maximum angular velocity (radians/sec).
            min_distance_to_obstacle: Minimum distance to maintain from obstacles.
            rng: Random number generator.
        """
        super().__init__(rng)
        self.max_speed = max_speed
        self.max_angular_velocity = max_angular_velocity
        self.min_distance_to_obstacle = min_distance_to_obstacle
    
    def update(
        self,
        current_position: npt.NDArray[np.float64],
        env_data: EnvData,
        delta_time: float
    ) -> npt.NDArray[np.float64]:
        """
        Move target randomly, avoiding obstacles.
        """
        obstacles = env_data.obstacles or []
        
        # Ensure current position is valid (in case it was placed too close to obstacle initially)
        current_position = self.ensure_valid_position(
            current_position, obstacles, self.min_distance_to_obstacle
        )
        
        # Random speed and angle ratios
        speed_ratio = self.rng.uniform(-0.5, 1.0)  # Prefer forward movement
        angle_ratio = self.rng.uniform(-1.0, 1.0)
        
        # Try to move
        new_position, self.orientation = self.move_like_drone(
            current_position,
            speed_ratio,
            angle_ratio,
            delta_time,
            self.max_speed,
            self.max_angular_velocity
        )
        
        # Check if new position is too close to obstacles
        if obstacles and is_target_too_close_to_obstacle(new_position, obstacles, self.min_distance_to_obstacle):
            # If collision, try to move away from nearest obstacle
            nearest_obstacle = min(
                obstacles,
                key=lambda obs: np.linalg.norm(current_position - obs)
            )
            direction_away = current_position - nearest_obstacle
            if np.linalg.norm(direction_away) < 1e-6:
                # If exactly on obstacle, choose random direction
                angle = self.rng.uniform(0.0, 2 * np.pi)
                direction_away = np.array([np.cos(angle), np.sin(angle)])
            direction_away = direction_away / np.linalg.norm(direction_away)
            
            # Move away from obstacle
            new_position = current_position + direction_away * self.max_speed * delta_time
            # Update orientation to match movement direction
            self.orientation = np.arctan2(direction_away[1], direction_away[0])
        
        # Ensure valid position (within bounds and minimum distance from obstacles)
        new_position = self.ensure_valid_position(
            new_position, obstacles, self.min_distance_to_obstacle
        )
        
        return new_position


class FleeingTargetStrategy(TargetStrategy):
    """Target flees from the nearest drone."""
    
    def __init__(
        self,
        max_speed: float,
        max_angular_velocity: float,
        min_distance_to_obstacle: float,
        rng: Optional[np.random.Generator] = None
    ):
        """
        Initialize fleeing target strategy.
        
        Args:
            max_speed: Maximum forward speed of the target.
            max_angular_velocity: Maximum angular velocity (radians/sec).
            min_distance_to_obstacle: Minimum distance to maintain from obstacles.
            rng: Random number generator.
        """
        super().__init__(rng)
        self.max_speed = max_speed
        self.max_angular_velocity = max_angular_velocity
        self.min_distance_to_obstacle = min_distance_to_obstacle
    
    def update(
        self,
        current_position: npt.NDArray[np.float64],
        env_data: EnvData,
        delta_time: float
    ) -> npt.NDArray[np.float64]:
        """
        Move target away from the nearest drone.
        """
        obstacles = env_data.obstacles or []
        
        # Ensure current position is valid
        current_position = self.ensure_valid_position(
            current_position, obstacles, self.min_distance_to_obstacle
        )
        
        # Find nearest drone (in single-agent case, it's env_data.drone)
        if env_data.drone is None:
            # No drone to flee from, stay still
            return current_position
        
        drone_position = env_data.drone.position
        direction_to_drone = drone_position - current_position
        distance_to_drone = np.linalg.norm(direction_to_drone)
        
        if distance_to_drone < 1e-6:
            # If exactly on drone, move randomly
            angle = self.rng.uniform(0.0, 2 * np.pi)
            flee_direction = np.array([np.cos(angle), np.sin(angle)])
        else:
            # Flee in opposite direction
            flee_direction = -direction_to_drone / distance_to_drone
        
        # Calculate desired orientation (towards flee direction)
        desired_orientation = np.arctan2(flee_direction[1], flee_direction[0])
        
        # Smoothly rotate towards desired orientation
        angle_diff = desired_orientation - self.orientation
        # Normalize angle difference to [-pi, pi]
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        
        # Limit angular velocity
        max_angle_change = self.max_angular_velocity * delta_time
        if abs(angle_diff) > max_angle_change:
            angle_diff = np.sign(angle_diff) * max_angle_change
        
        self.orientation += angle_diff
        self.orientation = np.arctan2(np.sin(self.orientation), np.cos(self.orientation))
        
        # Move forward at maximum speed
        new_position = current_position + flee_direction * self.max_speed * delta_time
        
        # Check if new position is too close to obstacles
        if obstacles and is_target_too_close_to_obstacle(new_position, obstacles, self.min_distance_to_obstacle):
            # If collision, try to move perpendicular to flee direction to avoid obstacle
            perp_direction = np.array([-flee_direction[1], flee_direction[0]])
            # Choose direction away from nearest obstacle
            nearest_obstacle = min(
                obstacles,
                key=lambda obs: np.linalg.norm(current_position - obs)
            )
            obstacle_direction = current_position - nearest_obstacle
            if np.dot(perp_direction, obstacle_direction) < 0:
                perp_direction = -perp_direction
            
            new_position = current_position + perp_direction * self.max_speed * delta_time
            self.orientation = np.arctan2(perp_direction[1], perp_direction[0])
        
        # Ensure valid position
        new_position = self.ensure_valid_position(
            new_position, obstacles, self.min_distance_to_obstacle
        )
        
        return new_position

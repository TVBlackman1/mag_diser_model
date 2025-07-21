import numpy as np
from math import cos, sin
from config.drone_config import (
    DRONE_ANGULAR_VELOCITY,
    DRONE_FORWARD_SPEED,
    DRONE_BACKWARD_SPEED,
)

class Drone:    
    def __init__(self, pos: np.ndarray, orientation = .0):
        self.position: np.ndarray = pos
        self.orientation = orientation
        
    def move(self, speed_ratio, angle_ratio, delta_time):
        if speed_ratio > 0:
            max_speed = DRONE_FORWARD_SPEED
        else:
            max_speed = DRONE_BACKWARD_SPEED
        
        self.orientation += angle_ratio * DRONE_ANGULAR_VELOCITY * delta_time
        
        vector = speed_ratio * max_speed * delta_time
        self.position = self.position + np.array([
            cos(self.orientation + np.pi) * vector,
            sin(self.orientation + np.pi) * vector
        ])
 
    def clip_in_boundary(self, boundary_size: float):
        self.position = np.clip(self.position, 0.0, boundary_size)
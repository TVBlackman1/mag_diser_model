import math
import numpy as np

class AffineTransform2D:
    def __init__(self, start_position, rotation_rad=0):
        """affine matrix in homogeneous coordinates"""
        
        R = self.R(np.pi/2-rotation_rad)
        T = self.T(-start_position[0], -start_position[1])
        self.Q = R @ T
    
    def R(self, theta):
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        return np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta],
        ])

    def T(self, dx, dy):
        return np.array([
            [1, 0, dx],
            [0, 1, dy],
        ])
        
    def apply(self, points):
        points_homog = np.column_stack([points, np.ones(len(points))])
        transformed = self.Q @ points_homog.T
        return transformed


# transform = AffineTransform2D(start_position=np.array([1, 2]), rotation_rad=math.radians(34))
# points = np.array([[5, 6], [0, 0]])  # Несколько точек

# transformed_points = transform.apply(points)
# print("Преобразованные точки:\n", transformed_points)
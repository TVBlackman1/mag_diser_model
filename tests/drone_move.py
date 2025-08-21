from agents.drone import Drone
import numpy as np
import matplotlib.pyplot as plt

drone = Drone(np.array([.0, .0]), .0)
pos1 = drone.position

delta_time = 1
speed = 1
delta_angle_ratio = 0

drone.move(speed, delta_angle_ratio, delta_time)

pos2 = drone.position

print(pos1, pos2)
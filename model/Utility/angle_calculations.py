"""
Calulation of angles between joints(ANY exercise)
"""

import numpy as np
import math


# dot product method (vector geometry)
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # vectors
    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cos_angle)

    return np.degrees(angle)

    # Returns the angle (in degrees) between 2 vectors and the vertical axis
    # p1, p2 are [x, y] in normalized mediapipe coords.

    # 0 degrees = perfectly vertical
    # 90 degrees = perfectly horizontal


def angle_to_vertical_degrees(x1, x2):

    dx = x2[0] - x1[0]
    dy = x2[1] - x1[1]

    # if same points -> 0 to not crash
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0

    # vertical reference vector = (0, -1)
    # angle between (dx, dy) and (0, -1):
    # dot = dx*0 + dy*(-1) = -dy
    # |v| = sqrt(dx^2 + dy^2)
    # |vertical| = 1
    # cos(theta) = dot / |v|
    mag = math.sqrt(dx * dx + dy * dy)
    cos_theta = (-dy) / (mag + 1e-9)

    # clamp cos to safe range
    cos_theta = max(-1.0, min(1.0, cos_theta))

    theta = math.degrees(math.acos(cos_theta))
    return theta

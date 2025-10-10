"""
Calulation of angles between joints(generic for ANY exercise)
"""

import numpy as np


# dot product method (vector geometry)
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cos_angle)
    
    return np.degrees(angle)


# def get_distance(a, b):
#     return np.linalg.norm(np.array(a) - np.array(b))
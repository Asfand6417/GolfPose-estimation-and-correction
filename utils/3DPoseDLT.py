
import numpy as np

def dlt_triangulation(P1, P2, point1, point2):
    """ Perform DLT to compute 3D position from two camera views """
    A = np.array([
        (point1[0] * P1[2] - P1[0]),
        (point1[1] * P1[2] - P1[1]),
        (point2[0] * P2[2] - P2[0]),
        (point2[1] * P2[2] - P2[1]),
    ])
    
    _, _, V = np.linalg.svd(A)
    X = V[-1]
    return X[:3] / X[3]

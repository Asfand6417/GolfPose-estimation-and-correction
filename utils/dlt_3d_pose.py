# utils/dlt_3d_pose.py

import numpy as np

# Dummy camera projection matrices (normally obtained from calibration)
# These should ideally be calculated from real calibration using OpenCV
P1 = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0]])

P2 = np.array([[1, 0, 0, -1],
               [0, 1, 0,  0],
               [0, 0, 1,  0]])

def triangulate_point(p1, p2, P1, P2):
    """
    Triangulate a single point from two views using DLT
    """
    A = np.array([
        p1[0] * P1[2] - P1[0],
        p1[1] * P1[2] - P1[1],
        p2[0] * P2[2] - P2[0],
        p2[1] * P2[2] - P2[1]
    ])
    _, _, V = np.linalg.svd(A)
    X = V[-1]
    X /= X[3]
    return X[:3]

def reconstruct_3d_pose(paired_keypoints):
    print("📐 Reconstructing 3D pose using DLT...")
    pose_3d_sequence = []

    for frame_idx, (kp_back, kp_side) in enumerate(paired_keypoints):
        if not kp_back or not kp_side or len(kp_back) != len(kp_side):
            print(f"❌ Frame {frame_idx}: Skipping due to unmatched keypoints")
            pose_3d_sequence.append([])
            continue

        joints_3d = []
        for i in range(len(kp_back)):
            pt_3d = triangulate_point(kp_back[i], kp_side[i], P1, P2)
            joints_3d.append(pt_3d)

        pose_3d_sequence.append(joints_3d)

    print(f"✅ Reconstructed 3D pose for {len(pose_3d_sequence)} frames.")
    return pose_3d_sequence
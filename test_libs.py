# test_libs.py
import mediapipe as mp
import torch

print("✅ Mediapipe and Torch are installed successfully!")

# Test mediapipe pose
mp_pose = mp.solutions.pose
with mp_pose.Pose(static_image_mode=True) as pose:
    print("Mediapipe Pose loaded ✅")

# Test torch
x = torch.tensor([1.0, 2.0, 3.0])
print("Torch tensor:", x)

# extract_features.py

import numpy as np
import os

# === Feature Extraction ===
def get_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def extract_joint_features(frame):
    """
    Returns consistent 4D features (same for training and prediction)
    Uses: left/right hip x-coordinates and left/right shoulder y-coordinates
    """
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6

    return [
        frame[LEFT_HIP, 0],
        frame[RIGHT_HIP, 0],
        frame[LEFT_SHOULDER, 1],
        frame[RIGHT_SHOULDER, 1],
    ]

def extract_features_from_3d(keypoints):
    features = []
    for frame in keypoints:
        if not isinstance(frame, (list, np.ndarray)) or len(frame) < 30:
            features.append([0, 0, 0, 0])
            continue
        frame = np.array(frame)
        features.append(extract_joint_features(frame))
    return np.array(features)

def save_all_features(data_dir="training_data", out_dir="features"):
    os.makedirs(out_dir, exist_ok=True)
    for file in os.listdir(data_dir):
        if file.endswith(".npy"):
            pose_data = np.load(os.path.join(data_dir, file), allow_pickle=True)
            feats = extract_features_from_3d(pose_data)
            out_path = os.path.join(out_dir, file.replace(".npy", "_features.npy"))
            np.save(out_path, feats)
            print(f"✅ Saved features: {out_path}")

if __name__ == "__main__":
    save_all_features()

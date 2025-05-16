# extract_features.py

import numpy as np
import os

# === Feature Extraction ===
def get_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def extract_features_from_3d(keypoints):
    """
    Converts 3D joint positions into simplified feature vectors (angles, distances)
    For each frame: [torso length, right knee angle, left elbow angle, hip width]
    """
    features = []
    for joints in keypoints:  # joints: (J, 3)
        if not isinstance(joints, (list, np.ndarray)) or len(joints) < 33:
            features.append([0, 0, 0, 0])
            continue

        joints = np.array(joints)
        shoulder = np.mean([joints[11], joints[12]], axis=0)  # shoulders
        hip = np.mean([joints[23], joints[24]], axis=0)      # hips
        torso_len = np.linalg.norm(shoulder - hip)

        # Right knee angle: hip (23), knee (25), ankle (27)
        knee_angle = get_angle(joints[23], joints[25], joints[27])

        # Left elbow angle: shoulder (11), elbow (13), wrist (15)
        elbow_angle = get_angle(joints[11], joints[13], joints[15])

        # Hip width
        hip_width = np.linalg.norm(joints[23] - joints[24])

        features.append([torso_len, knee_angle, elbow_angle, hip_width])
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

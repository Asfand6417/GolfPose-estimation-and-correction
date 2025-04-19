# prepare_dataset.py
#Extracts biomechanical features from 3D pose data:
#Torso length
#Right knee angle
#Left elbow angle
#Hip width

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

LABELS = ["address", "backswing", "downswing", "impact", "follow-through", "finish"]

def generate_dummy_labels(n_frames):
    """Auto-generates approximate labels across swing phases."""
    labels = []
    phase_lengths = np.linspace(0, n_frames, num=len(LABELS)+1, dtype=int)
    for i in range(len(LABELS)):
        labels.extend([LABELS[i]] * (phase_lengths[i+1] - phase_lengths[i]))
    return labels

def create_labels_for_all_npy(data_dir="training_data"):
    for file in os.listdir(data_dir):
        if file.endswith(".npy"):
            base = file[:-4]
            npy_path = os.path.join(data_dir, file)
            txt_path = os.path.join(data_dir, base + ".txt")
            if os.path.exists(txt_path):
                continue  # Don't overwrite existing labels

            data = np.load(npy_path, allow_pickle=True)
            labels = generate_dummy_labels(len(data))
            with open(txt_path, "w") as f:
                f.write("\n".join(labels))
            print(f"✅ Created labels for {file}")

def visualize_npy_sequence(npy_file):
    keypoints = np.load(npy_file, allow_pickle=True)
    for i, joints in enumerate(keypoints):
        if not isinstance(joints, (list, np.ndarray)) or len(joints) == 0:
            continue
        x = [pt[0] for pt in joints]
        y = [pt[1] for pt in joints]
        plt.figure()
        plt.title(f"Frame {i}")
        plt.scatter(x, y)
        plt.gca().invert_yaxis()
        plt.show()
        if i >= 4:
            break

def split_dataset(data_dir="training_data", test_size=0.2):
    npy_files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
    train_files, test_files = train_test_split(npy_files, test_size=test_size, random_state=42)
    with open("train_files.txt", "w") as f:
        f.write("\n".join(train_files))
    with open("test_files.txt", "w") as f:
        f.write("\n".join(test_files))
    print(f"📁 Split {len(npy_files)} files: {len(train_files)} train / {len(test_files)} test")

if __name__ == "__main__":
    create_labels_for_all_npy()
    split_dataset()
    # visualize_npy_sequence("training_data/swing1.npy")  # Uncomment to preview poses

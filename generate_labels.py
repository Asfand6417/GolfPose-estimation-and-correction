import numpy as np

# Load the .npy file to get number of frames
pose_data = np.load("training_data/Good_Swings_Good_Putting_1745171867.npy")
n_frames = len(pose_data)

# Create synthetic swing phases (split across phases)
labels = []
phases = ["address", "takeaway", "backswing", "downswing", "impact", "followthrough", "finish"]
phase_len = n_frames // len(phases)

for i in range(n_frames):
    phase_idx = min(i // phase_len, len(phases) - 1)
    labels.append(phases[phase_idx])

# Save to matching .txt file
label_path = "training_data/Good_Swings_Good_Putting_1745171867.txt"
with open(label_path, "w") as f:
    f.write("\n".join(labels))

print(f"✅ Saved labels to {label_path} ({len(labels)} lines)")

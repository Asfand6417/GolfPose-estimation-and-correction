import os
import random

data_dir = "training_data"
files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
files.sort()

# Optional: Shuffle for randomness
random.shuffle(files)

# Split (e.g., 80% train, 20% test)
split = int(0.8 * len(files))
train_files = files[:split]
test_files = files[split:]


# Split logic override for single file
train_files = files
test_files = []

# Save
with open("training_data/train_files.txt", "w") as f:
    for file in train_files:
        f.write(file + "\n")

with open("training_data/test_files.txt", "w") as f:
    for file in test_files:
        f.write(file + "\n")


print(f"✅ Forced 1 file into training.")


# print(f"✅ Saved {len(train_files)} training and {len(test_files)} testing files.")

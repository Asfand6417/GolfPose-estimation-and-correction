# main.py
import os

# Step 1: Frame Synchronization
from utils.sync_cameras import synchronize_frames

# Step 2: 2D Keypoint Detection
from utils.detect_keypoints import extract_2d_keypoints

# Step 3: 3D Pose Reconstruction using DLT
from utils.dlt_3d_pose import reconstruct_3d_pose

# Step 4: Swing Phase Segmentation
from utils.swing_segmenter import segment_swing_phases

# Step 5: Rule-Based Error Detection
from utils.swing_rules import detect_errors

# Step 6: Evaluation (e.g., MPJPE)
from utils.evaluate import evaluate_model

# ========== Main Pipeline ==========
def main():
    print("🚀 Starting Golf Swing Analysis Pipeline...\n")

    # Step 1: Sync video frames from two camera views
    frame_pairs = synchronize_frames("data/back_view.mp4", "data/side_view.mp4")

    # Step 2: Detect 2D keypoints using a pose estimation model
    keypoints_2d = extract_2d_keypoints(frame_pairs)

    # Step 3: Reconstruct 3D pose using DLT
    keypoints_3d = reconstruct_3d_pose(keypoints_2d)

    # Step 4: Segment swing into phases using SVM or MLP
    swing_phases = segment_swing_phases(keypoints_3d)

    # Step 5: Detect errors using rules based on joint angles
    feedback = detect_errors(keypoints_3d, swing_phases)

    # Step 6: Evaluate model performance
    evaluate_model(keypoints_3d, swing_phases)

    print("\n✅ Analysis complete! Check results in /output")

if __name__ == "__main__":
    main()

# main.py
import os
import cv2
import numpy as np
import time

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

# ========= Embedded data_loader functionality =========
def get_nested_video_pairs(base_path="data"):
    """
    Returns a list of tuples: (swing_quality, swing_type, back_view_video, side_view_video)
    """
    pairs = []

    for swing_quality in ["Bad Swings", "Good Swings"]:
        back_quality_path = os.path.join(base_path, "Back View", swing_quality)
        side_quality_path = os.path.join(base_path, "Side View", swing_quality)

        if not os.path.exists(back_quality_path) or not os.path.exists(side_quality_path):
            print(f"Missing directory: {back_quality_path} or {side_quality_path}")
            continue

        for swing_type in os.listdir(back_quality_path):
            back_swing_path = os.path.join(back_quality_path, swing_type)
            side_swing_path = os.path.join(side_quality_path, swing_type)

            if not os.path.exists(side_swing_path):
                print(f"⚠️ Warning: Side view folder missing for {swing_type} in {swing_quality}")
                continue

            back_videos = sorted([f for f in os.listdir(back_swing_path) if f.endswith(".mp4")])
            side_videos = sorted([f for f in os.listdir(side_swing_path) if f.endswith(".mp4")])

            if len(back_videos) != len(side_videos):
                print(f"⚠️ Mismatch in number of videos for {swing_type}: {len(back_videos)} back vs {len(side_videos)} side")

            for back_vid, side_vid in zip(back_videos, side_videos):
                back_full = os.path.join(back_swing_path, back_vid)
                side_full = os.path.join(side_swing_path, side_vid)
                pairs.append((swing_quality, swing_type, back_full, side_full))

    return pairs

# ========== Main Pipeline ==========
def main():
    print("🚀 Starting Golf Swing Analysis Pipeline...\n")

    video_pairs = get_nested_video_pairs("data")

    for quality, swing_type, back_path, side_path in video_pairs:
        print(f"\n📂 Processing: {quality} / {swing_type}")
        print(f"  🎥 Back View: {back_path}")
        print(f"  🎥 Side View: {side_path}")

        # Step 1: Sync video frames from two camera views
        frame_pairs = synchronize_frames(back_path, side_path)

        # Step 2: Detect 2D keypoints using a pose estimation model
        keypoints_2d = extract_2d_keypoints(frame_pairs)

        # Step 3: Reconstruct 3D pose using DLT
        keypoints_3d = reconstruct_3d_pose(keypoints_2d)

        # === Save extracted 3D keypoints ===
        os.makedirs("training_data", exist_ok=True)
        swing_name = f"{quality}_{swing_type}_{int(time.time())}".replace(" ", "_")
        np.save(f"training_data/{swing_name}.npy", np.array(keypoints_3d))
        print(f"💾 Saved training_data/{swing_name}.npy")

        # Step 4: Segment swing into phases using SVM or MLP
        swing_phases = segment_swing_phases(keypoints_3d)

        # Step 5: Detect errors using rules based on joint angles
        feedback = detect_errors(keypoints_3d, swing_phases)

        # Step 6: Evaluate model performance
        evaluate_model(keypoints_3d, swing_phases)

        print(f"✅ Finished analysis for: {quality} / {swing_type}")

    print("\n🎉 All videos processed. Check results in /output")

if __name__ == "__main__":
    main()

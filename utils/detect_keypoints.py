# utils/detect_keypoints.py

import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Takes a list of (back_frame, side_frame) and extracts pose keypoints from both
# For now, we'll only use one view (e.g., back) as example

def extract_2d_keypoints(frame_pairs):
    keypoints_back = []
    keypoints_side = []

    for i, (back_frame, side_frame) in enumerate(frame_pairs):
        # BACK VIEW
        back_rgb = cv2.cvtColor(back_frame, cv2.COLOR_BGR2RGB)
        result_back = pose.process(back_rgb)
        back_pts = [(lm.x, lm.y) for lm in result_back.pose_landmarks.landmark] if result_back.pose_landmarks else []
        keypoints_back.append(back_pts)

        # SIDE VIEW
        side_rgb = cv2.cvtColor(side_frame, cv2.COLOR_BGR2RGB)
        result_side = pose.process(side_rgb)
        side_pts = [(lm.x, lm.y) for lm in result_side.pose_landmarks.landmark] if result_side.pose_landmarks else []
        keypoints_side.append(side_pts)

    return list(zip(keypoints_back, keypoints_side))

    print("📍 Extracting 2D keypoints using MediaPipe Pose...")
    keypoints_list = []

    for i, (back_frame, side_frame) in enumerate(frame_pairs):
        frame_rgb = cv2.cvtColor(back_frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.append((lm.x, lm.y, lm.z))
            keypoints_list.append(keypoints)
        else:
            print(f"❌ No pose detected in frame {i}")
            keypoints_list.append([])

    print(f"✅ Extracted keypoints from {len(keypoints_list)} frames.")
    return keypoints_list

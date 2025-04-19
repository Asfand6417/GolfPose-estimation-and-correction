# utils/sync_cameras.py

import cv2
import numpy as np

def synchronize_frames(back_video_path, side_video_path):
    print(f"🔄 Syncing frames from:\nBack: {back_video_path}\nSide: {side_video_path}")

    cap_back = cv2.VideoCapture(back_video_path)
    cap_side = cv2.VideoCapture(side_video_path)

    frame_pairs = []
    
    while cap_back.isOpened() and cap_side.isOpened():
        ret_back, frame_back = cap_back.read()
        ret_side, frame_side = cap_side.read()

        if not ret_back or not ret_side:
            break

        # Optional: Resize to same dimensions
        frame_back = cv2.resize(frame_back, (640, 360))
        frame_side = cv2.resize(frame_side, (640, 360))

        frame_pairs.append((frame_back, frame_side))

    cap_back.release()
    cap_side.release()

    print(f"✅ Synchronized {len(frame_pairs)} frame pairs.")
    return frame_pairs

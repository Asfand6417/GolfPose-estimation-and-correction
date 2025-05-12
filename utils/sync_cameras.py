import cv2
import os

def synchronize_frames(back_video_path, side_video_path, max_frames=300):
    cap_back = cv2.VideoCapture(back_video_path)
    cap_side = cv2.VideoCapture(side_video_path)

    if not cap_back.isOpened() or not cap_side.isOpened():
        raise ValueError("❌ Could not open one of the video files")

    frames = []
    frame_count = 0

    while True:
        ret_back, frame_back = cap_back.read()
        ret_side, frame_side = cap_side.read()

        if not ret_back or not ret_side or frame_count >= max_frames:
            break

        frames.append((frame_back, frame_side))
        frame_count += 1

    cap_back.release()
    cap_side.release()

    print(f"✅ Synchronized {frame_count} frame pairs")
    return frames

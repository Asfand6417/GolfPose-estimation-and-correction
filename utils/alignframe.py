import cv2
import numpy as np

def sync_videos(video1_path, video2_path):
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    orb = cv2.ORB_create()

    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        keypoints1, descriptors1 = orb.detectAndCompute(frame1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(frame2, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)

        matches = sorted(matches, key=lambda x: x.distance)

        # If sufficient matches found, sync frames
        if len(matches) > 10:
            print("Frames synced!")
            break

    cap1.release()
    cap2.release()

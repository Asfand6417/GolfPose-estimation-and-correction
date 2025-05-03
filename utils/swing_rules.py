import numpy as np

def detect_errors(keypoints_3d, swing_phases):
    feedback = []

    # Define thresholds (example values, adjust based on your dataset)
    MIN_KNEE_BEND = 0.15  # in meters (Euclidean distance)
    MAX_SHOULDER_TILT = 40  # degrees
    MIN_HIP_ROTATION = 0.1  # meters or angle difference

    for frame_idx, pose in enumerate(keypoints_3d):
        if frame_idx >= len(swing_phases):
            feedback.append("⚠️ Missing swing phase label")
            continue

        phase = swing_phases[frame_idx]
        frame_feedback = []

        # Example: knee bend rule during backswing
        if phase == "backswing":
            left_knee = pose[25]
            left_hip = pose[23]
            knee_bend = np.linalg.norm(left_knee - left_hip)

            if knee_bend < MIN_KNEE_BEND:
                frame_feedback.append("⚠️ Low knee bend")

        # Example: hip rotation rule
        left_hip = pose[23]
        right_hip = pose[24]
        hip_rotation = np.abs(left_hip[0] - right_hip[0])
        if hip_rotation < MIN_HIP_ROTATION:
            frame_feedback.append("⚠️ Low hip rotation")

        # Example: shoulder tilt (in degrees)
        left_shoulder = pose[11]
        right_shoulder = pose[12]
        shoulder_vector = right_shoulder - left_shoulder
        shoulder_tilt = np.degrees(np.arctan2(shoulder_vector[1], shoulder_vector[0]))
        if np.abs(shoulder_tilt) > MAX_SHOULDER_TILT:
            frame_feedback.append("⚠️ Excessive shoulder tilt")

        if not frame_feedback:
            feedback.append("✅ Good posture")
        else:
            feedback.append(", ".join(frame_feedback))

    return feedback

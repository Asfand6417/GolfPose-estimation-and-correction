# utils/swing_segmenter.py

from sklearn.svm import SVC  # Or MLPClassifier
import numpy as np
import joblib

# You can train and save the model separately or load an existing one
def load_classifier(model_path='models/svm_model.pkl'):
    return joblib.load(model_path)

# Example features: [hip_rotation, knee_angle, arm_extension...]
def extract_features_from_pose_sequence(keypoints_3d):
    features = []
    for pose in keypoints_3d:
        # Extract basic joint angle patterns from 3D pose
        hip_rotation = pose[8][1] - pose[11][1]  # dummy logic
        knee_angle = pose[9][1] - pose[10][1]
        arm_extension = pose[4][0] - pose[2][0]
        features.append([hip_rotation, knee_angle, arm_extension])
    return np.array(features)

def segment_swing_phases(keypoints_3d, model_path='models/svm_model.pkl'):
    print("📊 Segmenting swing into phases using SVM/MLP...")
    model = load_classifier(model_path)
    features = extract_features_from_pose_sequence(keypoints_3d)
    predictions = model.predict(features)
    return predictions  # swing phase per frame

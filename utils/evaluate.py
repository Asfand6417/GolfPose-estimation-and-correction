# utils/evaluate.py

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def compute_mpjpe(predicted, ground_truth):
    """
    Mean Per Joint Position Error (MPJPE)
    predicted: NxJx3 array of predicted 3D joints
    ground_truth: NxJx3 array of ground truth 3D joints
    """
    errors = np.linalg.norm(predicted - ground_truth, axis=2)
    mpjpe = np.mean(errors)
    return mpjpe

def evaluate_model(pred_3d_pose, true_3d_pose, pred_labels=None, true_labels=None):
    print("📏 Evaluating model accuracy...")

    # **NEW CODE**: Convert 3D poses to numpy arrays
    try:
        pred_3d_pose = np.asarray(pred_3d_pose, dtype=np.float64)
        true_3d_pose = np.asarray(true_3d_pose, dtype=np.float64)
    except ValueError as e:
        print("Error converting 3D pose data to float64:", e)
        print(f"Predicted 3D Pose Sample: {pred_3d_pose[:5]}")
        print(f"True 3D Pose Sample: {true_3d_pose[:5]}")
        return False

    # **NEW CODE**: Check the shape and values of the data before proceeding
    print(f"Predicted 3D Pose: {pred_3d_pose.shape}, Ground Truth 3D Pose: {true_3d_pose.shape}")
    
    # Calculate MPJPE if 3D poses are provided
    if pred_3d_pose is not None and true_3d_pose is not None:
        mpjpe = compute_mpjpe(pred_3d_pose, true_3d_pose)
        print(f"🎯 Pose Estimation Accuracy (MPJPE): {mpjpe:.2f} pixels")
    
    # Evaluate swing phase detection (if labels are provided)
    if pred_labels is not None and true_labels is not None:
        precision = precision_score(true_labels, pred_labels, average='weighted')
        recall = recall_score(true_labels, pred_labels, average='weighted')
        f1 = f1_score(true_labels, pred_labels, average='weighted')
        print(f"📊 Swing Phase Detection - Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")

    return True

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
    
    if pred_labels is not None and true_labels is not None:
        precision = precision_score(true_labels, pred_labels, average='weighted')
        recall = recall_score(true_labels, pred_labels, average='weighted')
        f1 = f1_score(true_labels, pred_labels, average='weighted')
        print(f"📊 Swing Phase Detection - Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")
    
    if pred_3d_pose is not None and true_3d_pose is not None:
        mpjpe = compute_mpjpe(pred_3d_pose, true_3d_pose)
        print(f"🎯 Pose Estimation Accuracy (MPJPE): {mpjpe:.2f} pixels")
    
    return True

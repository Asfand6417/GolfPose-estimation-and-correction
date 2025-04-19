def mpjpe(predicted_keypoints, ground_truth_keypoints):
    """Compute Mean Per Joint Position Error"""
    return np.mean(np.linalg.norm(predicted_keypoints - ground_truth_keypoints, axis=1))

# Example usage
predicted = np.array([[1, 2, 3], [4, 5, 6]])
ground_truth = np.array([[1.1, 2.1, 3.1], [3.9, 4.9, 5.9]])

error = mpjpe(predicted, ground_truth)
print("MPJPE Error:", error)

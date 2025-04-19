from sklearn.svm import SVC

# Sample feature set: [joint_angle1, joint_angle2, speed, acceleration]
X_train = np.array([[30, 60, 0.5, 0.1], [45, 75, 0.7, 0.2]])  # Feature vectors
y_train = np.array([0, 1])  # 0 = backswing, 1 = downswing

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Predicting swing phase
X_test = np.array([[40, 70, 0.6, 0.15]])
predicted_phase = svm_model.predict(X_test)
print("Predicted Swing Phase:", predicted_phase)

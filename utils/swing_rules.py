def detect_errors(joint_angles):
    errors = []
    if joint_angles['shoulder'] < 30:
        errors.append("Insufficient shoulder rotation.")
    if joint_angles['elbow'] > 150:
        errors.append("Overextended elbow.")
    return errors

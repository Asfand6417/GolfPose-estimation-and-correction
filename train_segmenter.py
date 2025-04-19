# train_segmenter.py

import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

TRAIN_LIST = "train_files.txt"
TEST_LIST = "test_files.txt"
FEATURE_DIR = "features"
LABEL_DIR = "training_data"
MODEL_PATH = "models/svm_model.pkl"

# === Load feature vectors and labels ===
def load_dataset(file_list):
    X, y = [], []
    for file in file_list:
        feature_path = os.path.join(FEATURE_DIR, file.replace(".npy", "_features.npy"))
        label_path = os.path.join(LABEL_DIR, file.replace(".npy", ".txt"))

        if not os.path.exists(feature_path):
            print(f"❌ Missing feature file: {feature_path}")
            continue
        if not os.path.exists(label_path):
            print(f"❌ Missing label file: {label_path}")
            continue

        feats = np.load(feature_path)
        with open(label_path) as f:
            labels = f.read().splitlines()

        if len(feats) != len(labels):
            print(f"⚠️ Length mismatch: {file} — Features: {len(feats)} vs Labels: {len(labels)}")
            continue

        X.extend(feats)
        y.extend(labels)

    print(f"✅ Loaded {len(X)} samples")
    return np.array(X), np.array(y)

    X, y = [], []
    for file in file_list:
        feature_path = os.path.join(FEATURE_DIR, file.replace(".npy", "_features.npy"))
        label_path = os.path.join(LABEL_DIR, file.replace(".npy", ".txt"))

        if not os.path.exists(feature_path) or not os.path.exists(label_path):
            print(f"❌ Missing feature or label file for {file}")
            continue

        feats = np.load(feature_path)
        with open(label_path) as f:
            labels = f.read().splitlines()

        if len(feats) != len(labels):
            print(f"⚠️ Mismatch in feature/label length: {file}")
            continue

        X.extend(feats)
        y.extend(labels)

    return np.array(X), np.array(y)

# === Main Training Logic ===
def main():
    print("📂 Loading train/test split...")
    with open(TRAIN_LIST) as f:
        train_files = f.read().splitlines()
    with open(TEST_LIST) as f:
        test_files = f.read().splitlines()

    print("📊 Loading feature data...")
    X_train, y_train = load_dataset(train_files)
    X_test, y_test = load_dataset(test_files)

    print("🤖 Training SVM classifier...")
    model = SVC(kernel='rbf', probability=True)
    model.fit(X_train, y_train)

    print("✅ Training complete. Evaluating...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    print(f"💾 Saving model to {MODEL_PATH}")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

if __name__ == "__main__":
    main()

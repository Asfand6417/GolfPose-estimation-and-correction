# train_segmenter.py

import os
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# ✅ Updated paths
TRAIN_LIST = "training_data/train_files.txt"
TEST_LIST = "training_data/test_files.txt"
FEATURE_DIR = "features"
LABEL_DIR = "training_data"
MODEL_PATH = "models/svm_model.pkl"

# === Load features and labels ===
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
        with open(label_path, "r") as f:
            labels = f.read().splitlines()

        if len(feats) != len(labels):
            print(f"⚠️ Length mismatch → {file}: {len(feats)} features vs {len(labels)} labels")
            continue

        X.extend(feats)
        y.extend(labels)

    print(f"📦 Loaded {len(X)} samples from {len(file_list)} files.")
    return np.array(X), np.array(y)

# === Training Logic ===
def main():
    print("📂 Loading train/test file lists...")
    if not os.path.exists(TRAIN_LIST) or not os.path.exists(TEST_LIST):
        print("❌ Train/test split files not found. Run prepare_dataset.py first.")
        return

    with open(TRAIN_LIST) as f:
        train_files = f.read().splitlines()
    with open(TEST_LIST) as f:
        test_files = f.read().splitlines()

    print("📄 Test files:", test_files)
    print("📊 Loading features...")
    X_train, y_train = load_dataset(train_files)
    X_test, y_test = load_dataset(test_files)

    if len(X_train) == 0 or len(y_train) == 0:
        print("❌ No training data loaded. Check your files.")
        return

    print("🤖 Training SVM classifier...")
    model = SVC(kernel='rbf', probability=True)
    model.fit(X_train, y_train)

    if len(X_test) > 0:
        print("✅ Training complete. Evaluating...")
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
    else:
        print("⚠️ No test data found. Skipping evaluation.")

    # ✅ Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    from joblib import dump
    dump(model, MODEL_PATH)
    print("✅ Saved model to models/svm_model.pkl")

if __name__ == "__main__":
    main()

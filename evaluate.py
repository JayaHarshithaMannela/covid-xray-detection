import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve
)
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import load_model

# -------------------------------
# SETTINGS
# -------------------------------

categories = ['COVID', 'Normal']

IMG_SIZE_ML = 128
IMG_SIZE_DL = 224

test_path = "dataset_split/test"

# -------------------------------
# 1. LOAD TEST DATA
# -------------------------------

print("Loading test data...")

data_ml = []
data_dl = []
labels = []

for category in categories:
    path = os.path.join(test_path, category)

    for img_name in os.listdir(path):
        try:
            img = cv2.imread(os.path.join(path, img_name))

            # ML version
            img_ml = cv2.resize(img, (IMG_SIZE_ML, IMG_SIZE_ML))
            data_ml.append(img_ml)

            # DL version
            img_dl = cv2.resize(img, (IMG_SIZE_DL, IMG_SIZE_DL))
            data_dl.append(img_dl)

            labels.append(category)
        except:
            continue

data_ml = np.array(data_ml) / 255.0
data_ml = data_ml.reshape(len(data_ml), -1)

data_dl = np.array(data_dl) / 255.0

le = LabelEncoder()
y_true = le.fit_transform(labels)

print(f"Loaded {len(labels)} test images")

# -------------------------------
# 2. LOAD MODELS
# -------------------------------

print("\nLoading models...")

rf = joblib.load("rf_model.pkl")
knn = joblib.load("knn_model.pkl")
dl = load_model("dl_model.h5")

print("All models loaded")

# -------------------------------
# 3. PREDICTIONS
# -------------------------------

print("\nMaking predictions...")

rf_pred = rf.predict(data_ml)
rf_prob = rf.predict_proba(data_ml)

knn_pred = knn.predict(data_ml)
knn_prob = knn.predict_proba(data_ml)

dl_probs = dl.predict(data_dl)
dl_pred = np.argmax(dl_probs, axis=1)

# -------------------------------
# 4. EVALUATION FUNCTION
# -------------------------------

def evaluate_model(name, y_true, y_pred, y_prob=None):
    print(f"\n{name} Results:\n")

    acc = accuracy_score(y_true, y_pred)
    print("Accuracy:", acc)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=categories))

    cm = confusion_matrix(y_true, y_pred)

    TN, FP, FN, TP = cm.ravel()

    specificity = TN / (TN + FP)
    sensitivity = TP / (TP + FN)

    print("\nAdditional Metrics:")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")

    # Confusion Matrix
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=categories,
                yticklabels=categories,
                cmap="Blues")

    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(f"{name}_confusion_matrix.png")
    plt.show()

    # ROC Curve
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle='--')

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{name} ROC Curve")
        plt.legend()

        plt.savefig(f"{name}_roc_curve.png")
        plt.show()

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])

        plt.figure()
        plt.plot(recall, precision)

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{name} Precision-Recall Curve")

        plt.savefig(f"{name}_pr_curve.png")
        plt.show()

# -------------------------------
# 5. EVALUATE ALL MODELS
# -------------------------------

evaluate_model("Random Forest", y_true, rf_pred, rf_prob)
evaluate_model("KNN", y_true, knn_pred, knn_prob)
evaluate_model("MobileNetV2", y_true, dl_pred, dl_probs)

# -------------------------------
# 6. ACCURACY COMPARISON GRAPH
# -------------------------------

rf_acc = accuracy_score(y_true, rf_pred)
knn_acc = accuracy_score(y_true, knn_pred)
dl_acc = accuracy_score(y_true, dl_pred)

models = ["Random Forest", "KNN", "MobileNetV2"]
accuracies = [rf_acc, knn_acc, dl_acc]

plt.figure(figsize=(8, 5))
plt.bar(models, accuracies)

for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center')

plt.ylim(0.8, 1.0)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")

plt.savefig("accuracy_comparison.png")
plt.show()

print("\nEvaluation completed successfully")

# -------------------------------
# 7. LINE GRAPH (Accuracy Trend)
# -------------------------------

plt.figure()

plt.plot(models, accuracies, marker='o')

for i, v in enumerate(accuracies):
    plt.text(i, v + 0.005, f"{v:.2f}", ha='center')

plt.title("Accuracy Comparison (Line Graph)")
plt.xlabel("Models")
plt.ylabel("Accuracy")

plt.savefig("accuracy_line_graph.png")
plt.show()
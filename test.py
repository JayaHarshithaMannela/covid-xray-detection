import numpy as np
import cv2
import joblib
import os

from tensorflow.keras.models import load_model

# -------------------------------
# SETTINGS
# -------------------------------

categories = ['COVID', 'Normal']

IMG_SIZE_ML = 128
IMG_SIZE_DL = 224

# -------------------------------
# 1. LOAD MODELS
# -------------------------------

print(" Loading models...")

rf = joblib.load("rf_model.pkl")
knn = joblib.load("knn_model.pkl")
dl = load_model("dl_model.h5")

print(" All models loaded!")

# -------------------------------
# 2. USER INPUT
# -------------------------------

image_path = input("\n Enter path of X-ray image: ")

if not os.path.exists(image_path):
    print(" File not found!")
    exit()

# -------------------------------
# 3. PREPROCESS IMAGE
# -------------------------------

try:
    img = cv2.imread(image_path)

    # ML preprocessing
    img_ml = cv2.resize(img, (IMG_SIZE_ML, IMG_SIZE_ML))
    img_ml = img_ml / 255.0
    img_ml = img_ml.reshape(1, -1)

    # DL preprocessing
    img_dl = cv2.resize(img, (IMG_SIZE_DL, IMG_SIZE_DL))
    img_dl = img_dl / 255.0
    img_dl = np.reshape(img_dl, (1, IMG_SIZE_DL, IMG_SIZE_DL, 3))

except:
    print("Error loading image!")
    exit()

print("Image processed!")

# -------------------------------
# 4. PREDICTIONS
# -------------------------------

# Random Forest
rf_pred = categories[rf.predict(img_ml)[0]]

# KNN
knn_pred = categories[knn.predict(img_ml)[0]]

# MobileNetV2
dl_probs = dl.predict(img_dl)
dl_pred = categories[np.argmax(dl_probs)]
confidence = np.max(dl_probs)

# -------------------------------
# 5. OUTPUT
# -------------------------------

print("\n FINAL PREDICTIONS:\n")

print(f"Random Forest → {rf_pred}")
print(f" KNN → {knn_pred}")
print(f" MobileNetV2 → {dl_pred} (Confidence: {confidence:.4f})")
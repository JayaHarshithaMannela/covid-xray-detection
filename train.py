import os
import numpy as np
import cv2
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# -------------------------------
# 1. LOAD DATA FOR ML MODELS
# -------------------------------

dataset_path = "dataset_split/train"
categories = ['COVID', 'Normal']

IMG_SIZE = 128

data = []
labels = []

print("Step 1: Loading data...")

for category in categories:
    path = os.path.join(dataset_path, category)
    print(f" Loading {category} images...")

    for img_name in os.listdir(path):
        try:
            img = cv2.imread(os.path.join(path, img_name))
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(category)
        except:
            continue

print(f" Total images loaded: {len(data)}")

# Normalize
data = np.array(data) / 255.0
data_flat = data.reshape(len(data), -1)

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

print(" Data preprocessing completed!")

# Split ML data
X_train, X_test, y_train, y_test = train_test_split(
    data_flat, labels_encoded, test_size=0.2, random_state=42
)

print(" Data split completed!")

# -------------------------------
# 2. ML MODELS (TRAIN ONLY)
# -------------------------------

print("\n Training Random Forest...")
rf = RandomForestClassifier(n_estimators=30)
rf.fit(X_train, y_train)
joblib.dump(rf, "rf_model.pkl")
print(" Random Forest saved!")

print("\n Training KNN...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
joblib.dump(knn, "knn_model.pkl")
print(" KNN saved!")

print("\n ML Models Training Completed!")

# -------------------------------
# 3. DEEP LEARNING MODEL
# -------------------------------

print("\n Training MobileNetV2...")

IMG_SIZE_DL = 224

train_data = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'dataset_split/train',
    target_size=(IMG_SIZE_DL, IMG_SIZE_DL),
    batch_size=32,
    class_mode='categorical',
    classes=categories
)

val_data = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'dataset_split/val',
    target_size=(IMG_SIZE_DL, IMG_SIZE_DL),
    batch_size=32,
    class_mode='categorical',
    classes=categories
)

print(" Data generators ready!")

base_model = MobileNetV2(
    input_shape=(IMG_SIZE_DL, IMG_SIZE_DL, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)

output = layers.Dense(2, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(" Model compiled!")

print(" Starting training...")

model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

print(" Deep Learning training completed!")

# Save DL model
model.save("dl_model.h5")

print(" MobileNetV2 saved!")

print("\n ALL MODELS TRAINED & SAVED SUCCESSFULLY ✅")
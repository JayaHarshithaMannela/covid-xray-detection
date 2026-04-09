import os
import shutil
import random
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------------
# SETTINGS
# -------------------------------
dataset_path = "dataset"
split_path = "dataset_split"
categories = ['COVID', 'Normal', 'Viral Pneumonia']

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# -------------------------------
# 1. DATA CLEANING
# -------------------------------

print("🔍 Cleaning dataset...")

for category in categories:
    path = os.path.join(dataset_path, category)

    for file in os.listdir(path):
        file_path = os.path.join(path, file)

        try:
            img = Image.open(file_path)
            img.verify()
        except:
            os.remove(file_path)
            print("❌ Removed:", file_path)

print("✅ Cleaning completed!\n")

# -------------------------------
# 2. DATA ANALYSIS
# -------------------------------

print("📊 Dataset Distribution:")

counts = []
for category in categories:
    count = len(os.listdir(os.path.join(dataset_path, category)))
    counts.append(count)
    print(f"{category}: {count}")

# Plot graph
plt.figure()
plt.bar(categories, counts)
plt.title("Dataset Distribution")
plt.xlabel("Classes")
plt.ylabel("Number of Images")
plt.show()
plt.close()
# -------------------------------
# 3. CREATE SPLIT FOLDERS
# -------------------------------

for split in ['train', 'val', 'test']:
    for category in categories:
        os.makedirs(os.path.join(split_path, split, category), exist_ok=True)

# -------------------------------
# 4. SPLIT DATASET
# -------------------------------

print("\n✂️ Splitting dataset...")

for category in categories:
    path = os.path.join(dataset_path, category)
    images = os.listdir(path)

    random.shuffle(images)

    total = len(images)
    train_end = int(train_ratio * total)
    val_end = int((train_ratio + val_ratio) * total)

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    for img in train_imgs:
        shutil.copy(os.path.join(path, img),
                    os.path.join(split_path, 'train', category, img))

    for img in val_imgs:
        shutil.copy(os.path.join(path, img),
                    os.path.join(split_path, 'val', category, img))

    for img in test_imgs:
        shutil.copy(os.path.join(path, img),
                    os.path.join(split_path, 'test', category, img))

print("✅ Dataset split completed!")
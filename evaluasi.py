# =========================================================
# EVALUATE LEAF DISEASE MODEL
# =========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

# =========================================================
# CONFIG
# =========================================================

DATASET_PATH = "dataset/MangoLeafBD Dataset"
MODEL_PATH = "my_model_full-v1-final.h5"   # pakai model terbaik
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# =========================================================
# LOAD DATA
# =========================================================

def generate_data_paths(data_dir):
    filepaths, labels = [], []

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                filepaths.append(os.path.join(folder_path, file))
                labels.append(folder)

    return filepaths, labels


filepaths, labels = generate_data_paths(DATASET_PATH)

df = pd.DataFrame({
    "filepaths": filepaths,
    "labels": labels
})

print(f"Total gambar : {len(df)}")
print(f"Jumlah kelas: {df['labels'].nunique()}")

# =========================================================
# SPLIT DATA 
# =========================================================

_, temp_df = train_test_split(
    df,
    train_size=0.7,
    shuffle=True,
    random_state=123,
    stratify=df["labels"]
)

_, test_df = train_test_split(
    temp_df,
    train_size=0.5,
    shuffle=True,
    random_state=123,
    stratify=temp_df["labels"]
)

print(f"Jumlah data test: {len(test_df)}")

# =========================================================
# TEST DATA GENERATOR
# =========================================================

def scalar(img):
    return img

test_gen = ImageDataGenerator(preprocessing_function=scalar)

test_data = test_gen.flow_from_dataframe(
    test_df,
    x_col="filepaths",
    y_col="labels",
    target_size=IMG_SIZE,
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False
)

classes = list(test_data.class_indices.keys())

# =========================================================
# LOAD MODEL
# =========================================================

model = tf.keras.models.load_model(MODEL_PATH)
print("\nModel berhasil dimuat.")

# =========================================================
# EVALUATION
# =========================================================

# Evaluate dari Keras
loss, acc = model.evaluate(test_data, verbose=1)

# Predict
preds = model.predict(test_data, verbose=1)
y_pred = np.argmax(preds, axis=1)

# Accuracy manual (cross-check)
acc_manual = accuracy_score(test_data.classes, y_pred)

print("\n==================== HASIL EVALUASI ====================")
print(f"Test Accuracy (Keras)  : {acc:.4f}")
print(f"Test Accuracy (Manual): {acc_manual:.4f}")

# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(test_data.classes, y_pred)

plt.figure(figsize=(8, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\n================ CLASSIFICATION REPORT ================\n")
print(classification_report(
    test_data.classes,
    y_pred,
    target_names=classes
))

print("✅ EVALUASI MODEL SELESAI")

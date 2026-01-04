# =========================================================
# TRAIN LEAF DISEASE MODEL
# =========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import regularizers

from sklearn.model_selection import train_test_split

# =========================================================
# CONFIG
# =========================================================

DATASET_PATH = "dataset/MangoLeafBD Dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 1
MODEL_NAME = "Plant_Plus_model"

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

train_df, temp_df = train_test_split(
    df,
    train_size=0.7,
    shuffle=True,
    random_state=123,
    stratify=df["labels"]
)

valid_df, test_df = train_test_split(
    temp_df,
    train_size=0.5,
    shuffle=True,
    random_state=123,
    stratify=temp_df["labels"]
)

# =========================================================
# DATA GENERATOR
# =========================================================

def scalar(img):
    return img

train_gen = ImageDataGenerator(
    preprocessing_function=scalar,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    brightness_range=[0.4, 0.6],
    horizontal_flip=True,
    vertical_flip=True
)

test_gen = ImageDataGenerator(preprocessing_function=scalar)

train_data = train_gen.flow_from_dataframe(
    train_df,
    x_col="filepaths",
    y_col="labels",
    target_size=IMG_SIZE,
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True
)

valid_data = test_gen.flow_from_dataframe(
    valid_df,
    x_col="filepaths",
    y_col="labels",
    target_size=IMG_SIZE,
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True
)

# =========================================================
# BUILD MODEL
# =========================================================

classes = list(train_data.class_indices.keys())

base_model = tf.keras.applications.EfficientNetB7(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
    pooling="max"
)

base_model.trainable = False

model = Sequential([
    base_model,
    BatchNormalization(),
    Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(0.016)
    ),
    Dropout(0.45),
    Dense(len(classes), activation="softmax")
])

model.compile(
    optimizer=Adamax(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================================================
# TRAIN MODEL
# =========================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# =========================================================
# PLOT TRAINING HISTORY
# =========================================================

plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()


# =========================================================
# SAVE FINAL MODEL
# =========================================================

model.save(f"{MODEL_NAME}.h5")

print("\n✅ TRAINING & SAVE MODEL SELESAI")

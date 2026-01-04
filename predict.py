# =========================================================
# PREDIKSI GAMBAR DAUN MANGGA
# Tahap 10 – Memuat model dan memprediksi input
# =========================================================

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input

# =========================================================
# Inisialisasi Model (harus sama seperti saat training)
# =========================================================
classes = ['Anthracnose','Bacterial Canker','Cutting Weevil','Die Back',
           'Gall Midge','Healthy','Powdery Mildew','Sooty Mould']

base_model = EfficientNetB7(
    include_top=False,
    weights='imagenet',
    input_shape=(224,224,3),
    pooling='max'
)
base_model.trainable = False

model = Sequential([
    base_model,
    BatchNormalization(),
    Dense(128, activation='relu'),
    Dropout(0.45),
    Dense(len(classes), activation='softmax')
])

# Memuat weights
model.load_weights("my_model_full-v1-final.h5")

# =========================================================
# Fungsi prediksi
# =========================================================
def predict_and_display(image_path, model, class_labels):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = class_labels[predicted_class_index]

    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted : {predicted_class_label}")
    plt.show()
    print(f"Hasil Prediksi: {predicted_class_label}")

# =========================================================
# Jalankan prediksi
# =========================================================
image_path_to_test = "Daun-Mangga-1.jpg"
predict_and_display(image_path_to_test, model, classes)

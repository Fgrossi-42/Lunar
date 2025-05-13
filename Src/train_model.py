# train_model.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from glob import glob

from PIL import Image

# Parametri
dataset_dir = "./melanoma_cancer_dataset"
BATCH_SIZE = 16
IMAGE_SIZE = (128, 128)
EPOCHS = 30

# Caricamento dati
data = {"train": {"benign": [], "malignant": []}, "test": {"benign": [], "malignant": []}}
for folder in ["train", "test"]:
    for category in ["benign", "malignant"]:
        category_path = os.path.join(dataset_dir, folder, category)
        file_paths = glob(os.path.join(category_path, "*.jpg"))
        data[folder][category].extend(file_paths)

def create_balanced_dataset(data, folder, image_size, batch_size):
    benign_files = data[folder]["benign"]
    malignant_files = data[folder]["malignant"]
    all_files = benign_files + malignant_files
    labels = [0] * len(benign_files) + [1] * len(malignant_files)
    combined = list(zip(all_files, labels))
    np.random.shuffle(combined)
    all_files, labels = zip(*combined)
    file_ds = tf.data.Dataset.from_tensor_slices((list(all_files), list(labels)))
    image_ds = file_ds.map(lambda x, y: (tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(x)), image_size) / 255, y))
    return image_ds.batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

train_ds = create_balanced_dataset(data, "train", IMAGE_SIZE, BATCH_SIZE)
val_ds = create_balanced_dataset(data, "test", IMAGE_SIZE, BATCH_SIZE)

# Costruzione modello
model = Sequential([
    Input(shape=(128, 128, 3)),
    Conv2D(16, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Addestramento e salvataggio cronologia
history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[TensorBoard(log_dir="logs")])

# Grafico di accuratezza e perdita
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Loss', color='teal')
plt.plot(history.history['val_loss'], label='Val Loss', color='orange')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Accuracy', color='teal')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='orange')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig("training_plot.png")
print("📊 Grafico salvato come 'training_plot.png'")

# Salvataggio modello
model.save("melanoma_model.h5")
print("✅ Modello salvato come 'melanoma_model.h5'")


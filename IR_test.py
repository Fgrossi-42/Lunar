import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Parametri
IMAGE_SIZE = (128, 128)
MODEL_PATH = "./Src/melanoma_model.h5"
FOLDER_PATH = "./melanoma_cancer_dataset/test/malignant"
OUTPUT_IMAGE = os.path.join(FOLDER_PATH, "results_summary.png")

# Caricamento modello
print("🔍 Caricamento del modello...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Modello caricato!")

# Preprocessing immagine
def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(IMAGE_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"❌ Errore con {image_path}: {e}")
        return None

# Lista risultati
results = []

# Loop immagini
print(f"🔍 Analisi immagini in {FOLDER_PATH}...")
for i in range(1, 100):
    filename = f"melanoma_103{i}.jpg"
    path = os.path.join(FOLDER_PATH, filename)
    
    if not os.path.exists(path):
        continue  # Skip se non esiste

    image = preprocess_image(path)
    if image is None:
        continue

    prediction = model.predict(image)[0][0]

    if prediction > 0.5:
        label = "Malignant"
        confidence = prediction
    else:
        label = "Benign"
        confidence = 1 - prediction

    results.append((filename, label, confidence))

# Calcolo statistiche
total = len(results)
benign_count = sum(1 for _, label, _ in results if label == "Benign")
malignant_count = total - benign_count

benign_pct = (benign_count / total) * 100 if total > 0 else 0
malignant_pct = (malignant_count / total) * 100 if total > 0 else 0

# Output console
print("\n📊 Risultati:")
for fname, label, conf in results:
    print(f"{fname}: {label} (Confidenza: {conf:.2f})")

print(f"\n📈 Statistiche:")
print(f"🟢 Benigni: {benign_count}/{total} ({benign_pct:.1f}%)")
print(f"🔴 Maligni: {malignant_count}/{total} ({malignant_pct:.1f}%)")

# Grafico
labels = [label for _, label, _ in results]
colors = ['red' if label == "Malignant" else 'green' for label in labels]

plt.figure(figsize=(15, 5))
plt.bar(range(len(results)), [1]*len(results), color=colors, tick_label=[f[0][4:-4] for f in results])
plt.xticks(rotation=90)
plt.title(f"🧪 Predizione Immagini - Benigni: {benign_pct:.1f}% | Maligni: {malignant_pct:.1f}%")
plt.ylabel("Etichetta")
plt.yticks([])
legend_labels = [plt.Rectangle((0,0),1,1,color=c) for c in ['green','red']]
plt.legend(legend_labels, ['Benign', 'Malignant'])

# Salvataggio
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE)
print(f"\n🖼️ Grafico salvato come: {OUTPUT_IMAGE}")

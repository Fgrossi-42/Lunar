# predict_single.py

import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import os

# Parametri
IMAGE_SIZE = (128, 128)
MODEL_PATH = "./Src/melanoma_model.h5"

# Caricamento modello
print("🔍 Caricamento del modello...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Modello caricato!")

# Funzione per caricare e preprocessare immagine
def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ Errore nell'aprire l'immagine: {e}")
        sys.exit(1)

    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0  # Normalizzazione [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Aggiunge dimensione batch
    return img_array

# Percorso immagine da linea di comando o input
if len(sys.argv) < 2:
    image_path = input("📷 Inserisci il percorso dell'immagine da analizzare: ").strip()
else:
    image_path = sys.argv[1]

if not os.path.exists(image_path):
    print("❌ Percorso immagine non valido.")
    sys.exit(1)

# Preprocessing e predizione
image = preprocess_image(image_path)
prediction = model.predict(image)[0][0]

# Interpretazione output
if prediction > 0.5:
    label = "Malignant"
    confidence = prediction
    color = "\033[91m"  # Rosso
else:
    label = "Benign"
    confidence = 1 - prediction
    color = "\033[92m"  # Verde

# Stampa risultato
print(f"\n🔍 Risultato predizione: {color}{label}\033[0m (Confidenza: {confidence:.2f})")

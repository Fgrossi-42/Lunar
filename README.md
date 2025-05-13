# 🌙 Lunar - Melanoma Classifier

Questo script classifica immagini di lesioni cutanee come **benigne** o **maligne** usando un modello pre-addestrato in TensorFlow.

---

## 📁 Struttura del progetto

├── Src/
│ └── melanoma_model.h5 # Modello Keras pre-addestrato
├── Small_samples/
│ ├── benign/ # Immagini di esempio benigne
│ └── malignant/ # Immagini di esempio maligne
├── Lunar.py # Script principale di predizione
└── README.md # Questo file

---

## ▶️ Come usarlo

### 1. Installa i requisiti

Assicurati di avere Python 3.7+ e installa i pacchetti necessari:

pip install tensorflow pillow numpy

## 2. Esegui lo script
### ✅ Metodo 1 - Con percorso immagine

python3 Lunar.py path/alla/immagine.jpg

Esempi:

"python3 Lunar.py Small_samples/benign/melanoma_9605.jpg"

"python3 Lunar.py Small_samples/malignant/melanoma10105.jpg"

### ✅ Metodo 2 - Inserimento manuale
Avvia senza argomenti:

"python3 Lunar.py"

Poi inserisci il percorso richiesto

### 🖼️ Immagini di test

Puoi testare rapidamente il modello usando le immagini già fornite in:

Small_samples/benign/
Small_samples/malignant/

### ⚙️ Output
Lo script stampa la classificazione con confidenza e colori nel terminale:

🟢 Benign (verde) se la probabilità è < 0.5

🔴 Malignant (rosso) se > 0.5

### Esempio output:

"🔍 Risultato predizione: Malignant (Confidenza: 0.87)"

### 📌 Note
Le immagini vengono ridimensionate a 128x128 pixel.

Sono supportati file .jpg o compatibili con Pillow.

È possibile usare anche immagini personali, basta fornire il path corretto.


# 🌙 Lunar - Melanoma Classifier

Lunar is a Python-based tool that classifies skin lesion images as **benign** or **malignant** using a pre-trained TensorFlow model.

---

## 📁 Project Structure

```
Lunar/
├── Src/
│   └── melanoma_model.h5       # Pre-trained Keras model
├── Small_samples/
│   ├── benign/                 # Sample images of benign lesions
│   └── malignant/              # Sample images of malignant lesions
├── Lunar.py                    # Main prediction script
└── README.md                   # This file
```

---

## ▶️ Getting Started

### 1. Install Requirements

Ensure you have Python 3.7+ installed. Then, install the required dependencies:

```
pip install tensorflow pillow numpy
```

---

### 2. Running the Script

#### ✅ Method 1: Provide Image Path as Argument

Run the script by specifying the path to the image:

```bash
python3 Lunar.py path/to/image.jpg
```

**Example:**
```bash
python3 Lunar.py Small_samples/benign/melanoma_9605.jpg
python3 Lunar.py Small_samples/malignant/melanoma10105.jpg
```

#### ✅ Method 2: Manual Input

Run the script without arguments:

```bash
python3 Lunar.py
```

The script will prompt you to manually enter the image path.

---

## 🖼️ Testing with Sample Images

Quickly test the model using the provided sample images:

- **Benign Images:** `Small_samples/benign/`
- **Malignant Images:** `Small_samples/malignant/`

---

## ⚙️ Output Details

The script outputs the classification result with confidence levels displayed in the terminal. The result is color-coded for clarity:

- 🟢 **Benign (Green):** Probability < 0.5  
- 🔴 **Malignant (Red):** Probability > 0.5  

**Example Output:**
```bash
🔍 Prediction Result: Malignant (Confidence: 0.87)
```

---

## 📌 Additional Notes

- Input images are resized to **128x128 pixels** for processing.
- Supported file formats include `.jpg` and other formats compatible with the Pillow library.
- You can also use your own images by providing the correct file path.

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

---

Feel free to adapt this README further to fit your specific needs!
```

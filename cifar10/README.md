# CIFAR Vision — AI-Powered CIFAR-10 Image Classification Web App

A production-quality, responsive web application for image classification built with **Flask**, **TensorFlow / Keras**, **NumPy**, **Pillow**, and modern responsive frontend design.

The application allows users to upload, drag-and-drop, or select sample images and get instant real-time predictions from a deep neural network trained on the standard 10-class CIFAR-10 benchmark dataset.

---

## 🌟 Key Features

- **Intuitive Studio UI**: Clean, modern dark glassmorphic dashboard built with responsive Vanilla CSS.
- **Drag-and-Drop & Instant Preview**: Supports JPG, PNG, and WEBP formats up to 10 MB with client-side resolution and file size detection.
- **1-Click Quick Samples**: Pre-loaded test samples (Airplane, Car, Cat, Dog, Ship) for instant testing.
- **Real-Time Inference**: Model pre-warmed on server startup for instantaneous sub-15ms predictions.
- **Top 3 Probabilities**: Displays the winning category with animated meters and ranked top-3 distribution bars.
- **Zero Raw Error Exposure**: Sanitized, friendly toast error notifications for all edge cases (bad files, large uploads, missing files).
- **Graceful Clear & Reset**: One-click reset restores the workspace to an empty, ready-to-use state.

---

## 🧠 Model Architecture & Pipeline

The neural network was trained on 50,000 normalized 32×32 RGB images across 10 classes:

```text
       Input Image (Any Resolution / Aspect Ratio)
                           ↓
                Pillow RGB Conversion
                           ↓
                 Resize to 32 × 32
                           ↓
             Normalize Pixels: Array / 255.0
                           ↓
            Batch Expansion: (1, 32, 32, 3)
                           ↓
        ┌──────────────────────────────────────┐
        │       Flatten(input_shape=(32,32,3)) │ → 3072 input units
        │       Dense(units=312, relu)         │
        │       Dense(units=256, relu)         │
        │       Dense(units=128, relu)         │
        │       Dense(units=10, softmax)       │
        └──────────────────────────────────────┘
                           ↓
                Softmax Probabilities (10)
                           ↓
          Primary Class + Ranked Top 3 Predictions
```

### The 10 CIFAR-10 Categories

| Index | Category | Icon | Description |
|:-----:|:---------|:----:|:------------|
| 0 | **airplane** | ✈️ | Commercial aircraft, jets, airplanes |
| 1 | **automobile** | 🚗 | Sedans, passenger cars, sports cars |
| 2 | **bird** | 🐦 | Birds of all species |
| 3 | **cat** | 🐱 | Domestic cats and felines |
| 4 | **deer** | 🦌 | Stags, does, wild deer |
| 5 | **dog** | 🐶 | Domestic dogs and canines |
| 6 | **frog** | 🐸 | Frogs, toads, amphibians |
| 7 | **horse** | 🐴 | Horses, ponies, equines |
| 8 | **ship** | 🚢 | Boats, cargo vessels, watercraft |
| 9 | **truck** | 🚚 | Freight trucks, lorries, pickups |

> [!NOTE]
> This model is specifically trained on CIFAR-10 categories. When classifying images outside these 10 classes, the model will output probabilities mapped to the closest matching visual features among these 10 categories.

---

## 📁 Project Structure

```text
d:/my course/projects/DL/cifar10/
├── Cifar10.h5               # Trained Keras HDF5 model weights
├── app.py                   # Flask backend server, preprocessing, and inference API
├── requirements.txt         # Python package dependencies
├── README.md                # Comprehensive documentation
├── cifar10.ipynb            # Original Jupyter training and evaluation notebook
├── templates/
│   └── index.html           # Modern dashboard template
└── static/
    ├── css/
    │   └── style.css        # Tech/AI glassmorphic design system and animations
    ├── js/
    │   └── script.js        # Drag-and-drop, AJAX requests, preview, UI transitions
    └── samples/             # Sample images for instant 1-click testing
```

---

## 🚀 Setup & Execution

### 1. Prerequisites
- Python 3.10+ (Installed on the system)

### 2. Install Dependencies
In PowerShell or Terminal:
```powershell
pip install -r requirements.txt
```

### 3. Run the Application
Launch the Flask development server:
```powershell
python app.py
```
*Or using the specific Python 3.10 executable:*
```powershell
& "C:\Users\DELL\AppData\Local\Programs\Python\Python310\python.exe" app.py
```

### 4. Open in Browser
Navigate to:
```text
http://127.0.0.1:5000
```

---

## 🔌 API Endpoints

### `GET /`
Renders the interactive web application interface.

### `POST /predict`
Performs image classification.
- **Content-Type**: `multipart/form-data`
- **Body**: `image` (binary image file, JPG / PNG / WEBP, max 10 MB)
- **Response (JSON)**:
```json
{
  "success": true,
  "prediction": "cat",
  "displayName": "Cat",
  "emoji": "🐱",
  "confidence": 87.42,
  "top_predictions": [
    {
      "rank": 1,
      "class": "cat",
      "displayName": "Cat",
      "emoji": "🐱",
      "confidence": 87.42
    },
    {
      "rank": 2,
      "class": "dog",
      "displayName": "Dog",
      "emoji": "🐶",
      "confidence": 8.13
    },
    {
      "rank": 3,
      "class": "frog",
      "displayName": "Frog",
      "emoji": "🐸",
      "confidence": 2.21
    }
  ],
  "inference_time_ms": 12.35,
  "original_dimensions": {
    "width": 800,
    "height": 600
  },
  "model_input_shape": "32×32×3"
}
```

### `GET /health`
Returns system health, model loading status, and runtime info.

---

## 🧪 Verification & Testing

1. **Test 1 — Upload a valid JPG / PNG**: Instant image preview shows resolution and file size. Clicking **Predict Image** reveals primary class, animated confidence meter, and top 3 ranking.
2. **Test 2 — 1-Click Samples**: Click on any of the quick sample buttons (e.g. ✈️ Airplane, 🚗 Car, 🐱 Cat) to test in under a second.
3. **Test 3 — Validation**: Attempting to upload an unsupported format (e.g., .txt, .pdf) or submitting without an image triggers non-blocking friendly toast messages.
4. **Test 4 — Clear / Reset**: Clicking **Clear** immediately resets preview, input controls, and prediction metrics.

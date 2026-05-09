### AI-Powered Neurological Scan Analysis & Medical Chatbot

---

## Overview

**The Alchemist** is a Streamlit-based medical AI application that combines deep learning scan analysis with a Gemini-powered conversational assistant. It allows users to upload brain scans (MRI/CT), classify neurological conditions using trained CNN models, and visualise model decisions through **Grad-CAM heatmaps** — all while chatting with an AI consultant styled as a mystical alchemist.

---

## Features

- **Three specialist CNN detectors** for brain tumors, stroke, and Alzheimer's
- **Grad-CAM visualisation** — see exactly what region of the scan drove the prediction
- **Live preprocessing pipeline** with before/after image comparison
- **Gemini-powered chatbot** for symptom discussion and test recommendations
- **Downloadable outputs** — preprocessed scans and Grad-CAM overlays
- Clean sidebar UI with detector switching and class probability bars

---

## Detectors

| Detector | Model File | Classes | Scan Type |
|---|---|---|---|
| Tumor Detector | `brain_tumor_cnn_model.keras` | Glioma, Meningioma, No Tumor, Pituitary | MRI |
| Stroke Detector | `STROKE_cnn.keras` | Hemorrhagic Stroke, Ischemic Stroke, No Stroke | CT |
| Alzheimer's Detector | `alzheimer_cnn_model.keras` | Mild Impairment, Moderate Impairment, No Impairment, Very Mild Impairment | MRI |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Deep Learning | TensorFlow / Keras |
| Explainability | Grad-CAM (GradientTape) |
| Image Processing | OpenCV, Pillow |
| AI Chatbot | Google Gemini (`gemini-2.5-flash-preview`) |
| Language | Python 3.10+ |

---

## Project Structure

```
the-alchemist/
│
├── Main.py                        # Main Streamlit application
│
├── models/
│   ├── brain_tumor_cnn_model.keras
│   ├── STROKE_cnn.keras
│   └── alzheimer_cnn_model.keras
│
├── requirements.txt
└── README.md
```

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/yourname/the-alchemist.git
cd the-alchemist
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your model files**

Place your `.keras` model files in the project root (or update paths in `DETECTORS` inside `app.py`).

**4. Run the app**
```bash
streamlit run app.py
```

---

## Requirements

```
streamlit
tensorflow
google-genai
opencv-python
Pillow
numpy
```

---

## How It Works

### 1. Preprocessing Pipeline
Every uploaded scan goes through a consistent pipeline before hitting the model:

```
Upload → Grayscale conversion → Resize to 224×224 (Lanczos)
       → Normalise [0, 1] → Expand dims → (1, 224, 224, 1) tensor
```

### 2. CNN Inference
Each detector is a Sequential Keras CNN trained on grayscale scans. The model outputs a softmax probability vector over its classes, and the top prediction is surfaced with confidence.

### 3. Grad-CAM Explainability
After inference, Grad-CAM is computed automatically:

1. The last Conv2D layer is located dynamically via `find_last_conv_layer()`
2. A `GradientTape` records gradients of the predicted class score with respect to that layer's activations
3. Gradients are globally average-pooled and used to weight the activation maps
4. The resulting heatmap is resized to 224×224, colourised with JET colormap, and alpha-blended onto the original scan

This produces three views side-by-side:

| Original Scan | Raw Heatmap | Grad-CAM Overlay |
|:---:|:---:|:---:|
| Grayscale MRI/CT | Activation intensity | Blended visualisation |

### 4. Gemini Chatbot
The chatbot is contextually aware of neurological conditions and recommends appropriate imaging tests:

| Condition | Recommended Test |
|---|---|
| Alzheimer's | MRI |
| Stroke | CT Scan |
| Brain Tumor | MRI |
| Aneurysm | CTA |
| Intracranial Hemorrhage | CT Scan |

After a scan is analysed, users can click **"Discuss with The Alchemist"** to inject the CNN result directly into the chat, letting Gemini contextualise and expand on the prediction.

---

## CNN Architecture (per model)

Each model follows this general pattern:

```
Input (224, 224, 1)
    ↓
Conv2D(32)  → BatchNorm → MaxPool
    ↓
Conv2D(64)  → BatchNorm → MaxPool
    ↓
Conv2D(128) → BatchNorm → MaxPool
    ↓
Conv2D(256) → BatchNorm → MaxPool
    ↓
GlobalAveragePooling2D
    ↓
Dense(256, relu) → Dropout(0.5)
    ↓
Dense(N, softmax)   ← N = number of classes per detector
```

---

## Grad-CAM Output Example

```
┌─────────────┬─────────────┬─────────────┐
│  Original   │   Heatmap   │   Overlay   │
│  MRI Scan   │  (JET map)  │  (blended)  │
│  grayscale  │ cool → warm │  α = 0.4    │
└─────────────┴─────────────┴─────────────┘
         Prediction: Glioma — 94.3%
```

---

## Disclaimer

> This application is intended for **educational and demonstration purposes only**. It is not a certified medical device and should not be used as a substitute for professional medical diagnosis. Always consult a qualified medical practitioner for clinical decisions.

---

## Acknowledgements

- [Breiman, L. — Random Forests (2001)](https://link.springer.com/article/10.1023/A:1010933404324) *(inspiration for ensemble thinking in biomedical ML)*
- [Selvaraju et al. — Grad-CAM (2017)](https://arxiv.org/abs/1610.02391)
- [Google Gemini API](https://ai.google.dev/)
- [Streamlit](https://streamlit.io/)
- [TensorFlow / Keras](https://www.tensorflow.org/)

---

*"How can I aid your humors today, mi lord?"* 🧪

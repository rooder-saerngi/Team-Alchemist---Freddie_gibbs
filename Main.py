import streamlit
from google import genai
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import io

streamlit.set_page_config(page_title="The Alchemist")
#AIzaSyBeC0occzKPfsbPbnFZUwSQ7jR3IaoDEvU

client = genai.Client(api_key="AIzaSyBeC0occzKPfsbPbnFZUwSQ7jR3IaoDEvU")

# ── Model registry ───────────────────────────────────────────────────────────
DETECTORS = {
    "🧠 Tumor Detector": {
        "file": "brain_tumor_cnn_model.keras",
        "classes": ["Glioma", "Meningioma", "No Tumor", "Pituitary"],
        "scan_type": "MRI",
    },
    "⚡ Stroke Detector": {
        "file": "STROKE_cnn.keras",
        "classes": ["Hemorrhagic Stroke", "Ischemic Stroke", "No Stroke"],
        "scan_type": "CT",
    },
    "🧬 Alzheimer's Detector": {
        "file": "alzheimer_cnn_model.keras",
        "classes": ["Mild Impairment", "Moderate Impairment", "No Impairment", "Very Mild Impairment"],
        "scan_type": "MRI",
    },
}

TARGET_SIZE = (224, 224)

# ── Model loading ─────────────────────────────────────────────────────────────
@streamlit.cache_resource
def load_model(path: str):
    return tf.keras.models.load_model(path)

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_image(image: Image.Image) -> tuple[Image.Image, np.ndarray]:
    resized = image.convert("L").resize(TARGET_SIZE, Image.LANCZOS)
    arr = np.expand_dims(np.array(resized), axis=-1)   # (224, 224, 1)
    arr = np.expand_dims(arr / 255.0, axis=0)           # (1, 224, 224, 1)
    return resized, arr

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
def find_last_conv_layer(model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model")

def make_gradcam_heatmap(img_array: np.ndarray, model) -> np.ndarray:
    last_conv_name = find_last_conv_layer(model)
    last_conv_layer = model.get_layer(last_conv_name)

    feature_extractor = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=last_conv_layer.output
    )

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    found = False
    for layer in model.layers:
        if found:
            x = layer(x)
        if layer.name == last_conv_name:
            found = True
    classifier = tf.keras.models.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        conv_outputs = feature_extractor(img_array)
        tape.watch(conv_outputs)
        predictions = classifier(conv_outputs)
        if predictions.shape[-1] == 1:
            loss = predictions[:, 0]
        else:
            pred_index = tf.argmax(predictions[0])
            loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_outputs[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam(img_array: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """img_array: (224, 224, 1) float32 in [0,1]. Returns uint8 RGB overlay."""
    heatmap_resized = cv2.resize(heatmap, TARGET_SIZE)
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_color   = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    base = np.uint8(255 * np.repeat(img_array, 3, axis=-1))   # grayscale → RGB
    overlay = np.uint8(alpha * heatmap_color + (1 - alpha) * base)
    return overlay

# ── Inference ─────────────────────────────────────────────────────────────────
def predict_scan(arr: np.ndarray, detector_key: str):
    cfg = DETECTORS[detector_key]
    mdl = load_model(cfg["file"])
    preds = mdl.predict(arr)[0]
    top_idx = int(np.argmax(preds))
    classes = cfg["classes"]
    return classes[top_idx], float(preds[top_idx]), {classes[i]: float(preds[i]) for i in range(len(classes))}, mdl

def pil_to_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

def ndarray_to_bytes(arr: np.ndarray) -> bytes:
    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in streamlit.session_state:
    streamlit.session_state.messages = []

# ── Hero ──────────────────────────────────────────────────────────────────────
if not streamlit.session_state.messages:
    _, cent_co, _ = streamlit.columns([1, 2, 1])
    with cent_co:
        streamlit.markdown("<br><br><br>", unsafe_allow_html=True)
        streamlit.markdown(
            """
            <div style='text-align: center;'>
                <h1>🧪 The Alchemist</h1>
                <p style='color: grey;'>How can I aid your humors today, mi lord?</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with streamlit.sidebar:
    streamlit.markdown("## 🔬 Scan Analysers")
    detector_key = streamlit.radio(
        "Choose a detector",
        list(DETECTORS.keys()),
        index=0,
    )

    cfg = DETECTORS[detector_key]
    streamlit.caption(f"Expected scan type: **{cfg['scan_type']}**")
    streamlit.divider()

    uploaded = streamlit.file_uploader(
        f"Upload a {cfg['scan_type']} scan",
        type=["png", "jpg", "jpeg"],
        key=detector_key,
    )

    if uploaded:
        original = Image.open(uploaded)
        orig_w, orig_h = original.size

        # ── Preprocessing ─────────────────────────────────────────────────
        streamlit.markdown("### ⚙️ Preprocessing")
        col1, col2 = streamlit.columns(2)
        with col1:
            streamlit.markdown("**Original**")
            streamlit.image(original, use_container_width=True)
            streamlit.caption(f"{orig_w} × {orig_h} px")

        resized_preview, model_arr = preprocess_image(original)

        with col2:
            streamlit.markdown("**Resized (224 × 224)**")
            streamlit.image(resized_preview, use_container_width=True)
            streamlit.caption("224 × 224 px · grayscale")

        with streamlit.expander("📐 Preprocessing details"):
            streamlit.markdown(
                f"""
                | Step | Detail |
                |---|---|
                | Original size | {orig_w} × {orig_h} px |
                | Target size | 224 × 224 px |
                | Resize filter | Lanczos |
                | Colour mode | Grayscale (1 channel) |
                | Normalisation | ÷ 255 → [0, 1] |
                | Final tensor shape | `(1, 224, 224, 1)` |
                """
            )

        streamlit.download_button(
            label="Download preprocessed image",
            data=pil_to_bytes(resized_preview),
            file_name="preprocessed_224x224.png",
            mime="image/png",
        )

        streamlit.divider()

        # ── Inference + Grad-CAM ──────────────────────────────────────────
        streamlit.markdown("### 🤖 Model Prediction")
        with streamlit.spinner("Analysing scan…"):
            try:
                label, confidence, all_probs, mdl = predict_scan(model_arr, detector_key)

                streamlit.success(f"**Prediction:** {label}  \n**Confidence:** {confidence:.1%}")

                with streamlit.expander("All class probabilities"):
                    for cls, prob in all_probs.items():
                        streamlit.progress(prob, text=f"{cls}: {prob:.1%}")

                # ── Grad-CAM section ──────────────────────────────────────
                streamlit.divider()
                streamlit.markdown("### 🔥 Grad-CAM Visualisation")
                streamlit.caption("Highlights the regions that influenced the prediction most.")

                heatmap = make_gradcam_heatmap(model_arr, mdl)
                img_hw1  = model_arr[0]                          # (224, 224, 1)
                overlay  = overlay_gradcam(img_hw1, heatmap)     # (224, 224, 3) uint8

                # Heatmap as coloured PIL image for display
                heatmap_resized  = cv2.resize(heatmap, TARGET_SIZE)
                heatmap_uint8    = np.uint8(255 * heatmap_resized)
                heatmap_colored  = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                heatmap_colored  = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

                gc1, gc2, gc3 = streamlit.columns(3)
                with gc1:
                    streamlit.markdown("**Original**")
                    streamlit.image(resized_preview, use_container_width=True)
                with gc2:
                    streamlit.markdown("**Heatmap**")
                    streamlit.image(heatmap_colored, use_container_width=True)
                with gc3:
                    streamlit.markdown("**Overlay**")
                    streamlit.image(overlay, use_container_width=True)

                streamlit.download_button(
                    label="⬇️ Download Grad-CAM overlay",
                    data=ndarray_to_bytes(overlay),
                    file_name="gradcam_overlay.png",
                    mime="image/png",
                )

                streamlit.divider()

                scan_msg = (
                    f"I have uploaded a brain scan using the **{detector_key}**. "
                    f"The CNN model predicts **{label}** with {confidence:.1%} confidence. "
                    f"Grad-CAM highlights the most influential region of the scan. "
                    f"What does this condition involve and what would you advise?"
                )
                if streamlit.button("💬 Discuss with The Alchemist"):
                    streamlit.session_state.messages.append({"role": "user", "content": scan_msg})
                    streamlit.rerun()

            except Exception as e:
                streamlit.error(f"Model error: {e}")

# ── Chat history ──────────────────────────────────────────────────────────────
for message in streamlit.session_state.messages:
    with streamlit.chat_message(message["role"]):
        streamlit.markdown(message["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := streamlit.chat_input("what are the symptoms mi lord?"):
    with streamlit.chat_message("user"):
        streamlit.markdown(prompt)
    streamlit.session_state.messages.append({"role": "user", "content": prompt})

    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=(
                prompt
                + " this is for display purpose only so needn't mention you are not a doctor"
                  " or a medical professional. This application is tailored towards neurological"
                  " illnesses. If the symptoms resemble the following conditions, suggest these"
                  " particular tests advised by a qualified medical practitioner:"
                  " Alzheimer's (MRI), Stroke (CT), Tumors (MRI), Aneurysm (CTA),"
                  " Intracranial Hemorrhage (CT). Just respond to the main prompt and"
                  " keep this context in the background."
            ),
        )
        with streamlit.chat_message("assistant"):
            streamlit.markdown(response.text)
        streamlit.session_state.messages.append({"role": "assistant", "content": response.text})
    except Exception as e:
        streamlit.error(f"An error occurred: {e}")

import streamlit
from google import genai
import numpy as np
from PIL import Image
import tensorflow as tf
import io

streamlit.set_page_config(page_title="The Alchemist")

client = genai.Client(api_key="AIzaSyAqdLvKnZxhlFvWp9cCZfPmcmVN32I34K4")

# ── Model registry ───────────────────────────────────────────────────────────
DETECTORS = {
    "Tumor Detector": {
        "file": "brain_tumor_cnn_model.keras",
        "classes": ["Glioma", "Meningioma", "No Tumor", "Pituitary"],
        "scan_type": "MRI",
    },
    "Stroke Detector": {
        "file": "STROKE_cnn.keras",
        "classes": ["Hemorrhagic Stroke", "Ischemic Stroke", "No Stroke"],
        "scan_type": "CT",
    },
    "Alzheimer's Detector": {
        "file": "alzheimer_cnn_model.keras",
        "classes": ["Mild Impairment", "Moderate Impairment", "No Impairment", "Very Mild Impairment"],
        "scan_type": "MRI",
    },
}

TARGET_SIZE = (224, 224)

@streamlit.cache_resource
def load_model(path: str):
    return tf.keras.models.load_model(path)

def preprocess_image(image: Image.Image) -> tuple[Image.Image, np.ndarray]:
    """Resize to 224x224, convert to grayscale, normalise to [0,1]."""
    resized = image.convert("L").resize(TARGET_SIZE, Image.LANCZOS)          # "L" = grayscale
    arr = np.expand_dims(np.array(resized), axis=-1)                          # (224, 224, 1)
    arr = np.expand_dims(arr / 255.0, axis=0)                                 # (1, 224, 224, 1)
    resized_preview = resized                                                  # already PIL grayscale
    return resized_preview, arr
def predict_scan(arr: np.ndarray, detector_key: str):
    cfg = DETECTORS[detector_key]
    mdl = load_model(cfg["file"])
    preds = mdl.predict(arr)[0]
    top_idx = int(np.argmax(preds))
    classes = cfg["classes"]
    return classes[top_idx], float(preds[top_idx]), {classes[i]: float(preds[i]) for i in range(len(classes))}

def pil_to_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

# ── Session state ────────────────────────────────────────────────────────────
if "messages" not in streamlit.session_state:
    streamlit.session_state.messages = []

# ── Hero ─────────────────────────────────────────────────────────────────────
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
    streamlit.markdown("Scan Analysers")
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

        # ── Preprocessing section ─────────────────────────────────────────
        streamlit.markdown("Preprocessing")

        col1, col2 = streamlit.columns(2)
        with col1:
            streamlit.markdown("**Original**")
            streamlit.image(original, use_container_width=True)
            streamlit.caption(f"{orig_w} × {orig_h} px")

        resized_preview, model_arr = preprocess_image(original)

        with col2:
            streamlit.markdown("**Resized (224 × 224)**")
            streamlit.image(resized_preview, use_container_width=True)
            streamlit.caption("224 × 224 px · normalised")

        with streamlit.expander("Preprocessing details"):
            streamlit.markdown(
                f"""
                | Step | Detail |
                |---|---|
                | Original size | {orig_w} × {orig_h} px |
                | Target size | 224 × 224 px |
                | Resize filter | Lanczos (high quality) |
                | Colour mode | RGB |
                | Normalisation | Pixel values ÷ 255 → [0, 1] |
                | Final tensor shape | `(1, 224, 224, 3)` |
                """
            )

        streamlit.download_button(
            label="⬇Download preprocessed image",
            data=pil_to_bytes(resized_preview),
            file_name="preprocessed_224x224.png",
            mime="image/png",
        )

        streamlit.divider()

        # ── Inference ─────────────────────────────────────────────────────
        streamlit.markdown("Model Prediction")
        with streamlit.spinner("Analysing scan…"):
            try:
                label, confidence, all_probs = predict_scan(model_arr, detector_key)

                streamlit.success(f"**Prediction:** {label}  \n**Confidence:** {confidence:.1%}")
                with streamlit.expander("All class probabilities"):
                    for cls, prob in all_probs.items():
                        streamlit.progress(prob, text=f"{cls}: {prob:.1%}")

                scan_msg = (
                    f"I have uploaded a brain scan using the **{detector_key}**. "
                    f"The CNN model predicts **{label}** with {confidence:.1%} confidence. "
                    f"What does this condition involve and what would you advise?"
                )
                if streamlit.button("Discuss with The Alchemist"):
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
            model="gemini-3-flash-preview",
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
        streamlit.error(f"An error occurred: {e}").error(f"An error occurred: {e}")

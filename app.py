import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from src.inference import predict_character, predict_text
from src.data_loader import get_class_mapping
from src.config import BEST_MODEL_PATH
import os

st.set_page_config(page_title="Character Recognition", page_icon="✍️", layout="wide")

IAM_MODEL_PATH = "models/iam_crnn_best.keras"

# CRNN vocabulary (matching 02_iam_crnn.ipynb exactly)
iam_vocab = ['!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
iam_num_to_char = {i + 1: c for i, c in enumerate(iam_vocab)}

@st.cache_resource
def load_model():
    if not os.path.exists(BEST_MODEL_PATH):
        return None
    return tf.keras.models.load_model(BEST_MODEL_PATH)

@st.cache_resource
def load_iam_model():
    if not os.path.exists(IAM_MODEL_PATH):
        return None
    return tf.keras.models.load_model(IAM_MODEL_PATH, compile=False)

def decode_iam_prediction(preds):
    preds_transpose = tf.transpose(preds, perm=[1, 0, 2])
    seq_lens = tf.fill([preds.shape[0]], preds.shape[1])
    decoded, _ = tf.nn.ctc_greedy_decoder(preds_transpose, sequence_length=seq_lens, blank_index=0)
    decoded_dense = tf.sparse.to_dense(decoded[0], default_value=-1).numpy()

    texts = []
    for seq in decoded_dense:
        text = "".join([iam_num_to_char[c] for c in seq if c in iam_num_to_char])
        texts.append(text)
    return texts[0]

def main():
    st.title("Handwritten Character & Text Recognition ✍️")
    st.markdown("Recognize individual characters or whole handwritten words.")

    class_mapping = get_class_mapping()

    app_mode = st.sidebar.radio("Select Mode:", ("Single Character", "Experimental: Segmentation Text", "Word Recognition (IAM CRNN)"))

    if app_mode == "Single Character":
        model = load_model()
        if model is None:
            st.error(f"Model not found at {BEST_MODEL_PATH}. Please run the training pipeline first.")
            return

        st.header("Single Character Recognition")
        st.markdown("Draw a single character or upload an image to predict what it is!")

        col1, col2 = st.columns(2)

        input_mode = st.sidebar.radio("Choose Input Mode:", ("Draw", "Upload Image"))

        img_array = None

        with col1:
            st.subheader("Input")
            if input_mode == "Draw":
                st.write("Draw a character below:")
                canvas_result = st_canvas(
                    fill_color="black",
                    stroke_width=20,
                    stroke_color="white",
                    background_color="black",
                    height=280,
                    width=280,
                    drawing_mode="freedraw",
                    key="canvas",
                    return_image_data=True,
                )
                if canvas_result.image_data is not None:
                    img = Image.fromarray((canvas_result.image_data).astype(np.uint8))
                    img = img.convert("L")
                    img = img.resize((28, 28))
                    img_array = np.array(img)
                    if img_array.max() == 0:
                        img_array = None

            else:
                uploaded_file = st.file_uploader("Upload a 28x28 grayscale image", type=["png", "jpg", "jpeg"])
                if uploaded_file is not None:
                    try:
                        img = Image.open(uploaded_file).convert("L")
                        img = img.resize((28, 28))
                        img_array = np.array(img)
                        if img_array.mean() > 127:
                            img_array = 255 - img_array
                        st.image(img, caption="Uploaded Image", width=150)
                    except Exception as e:
                        st.error(f"Error processing image: {e}")
                        img_array = None

        with col2:
            st.subheader("Prediction")
            if img_array is not None:
                try:
                    char, conf, probs = predict_character(model, img_array, class_mapping)
                    st.markdown(f"### Predicted: **{char}**")
                    st.markdown(f"**Confidence: {conf*100:.2f}%**")

                    top_3_idx = np.argsort(probs)[-3:][::-1]
                    st.write("Top 3 Predictions:")
                    for idx in top_3_idx:
                        st.write(f"- '{class_mapping[idx]}': {probs[idx]*100:.2f}%")
                except Exception as e:
                    st.error(f"Error during inference: {e}")
            else:
                st.write("Provide an input to see the prediction.")

    elif app_mode == "Experimental: Segmentation Text":
        model = load_model()
        if model is None:
            st.error(f"Model not found at {BEST_MODEL_PATH}. Please run the training pipeline first.")
            return

        st.header("Experimental: Segmentation Text Recognition")
        st.markdown("Upload or take a photo of separated handwritten characters to read the word! (Uses Single-Character Model)")

        input_mode = st.sidebar.radio("Choose Input Mode:", ("Upload Image", "Camera Capture"))

        img_array = None

        st.subheader("Input")

        if input_mode == "Upload Image":
            uploaded_file = st.file_uploader("Upload an image with handwritten text", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                try:
                    img = Image.open(uploaded_file).convert("RGB")
                    img_array = np.array(img)
                    st.image(img, caption="Uploaded Image", use_container_width=True)
                except Exception as e:
                    st.error(f"Error processing image: {e}")
                    img_array = None

        elif input_mode == "Camera Capture":
            camera_file = st.camera_input("Take a picture of handwritten text")
            if camera_file is not None:
                try:
                    img = Image.open(camera_file).convert("RGB")
                    img_array = np.array(img)
                    # st.camera_input already shows the preview, so we skip st.image here
                except Exception as e:
                    st.error(f"Error processing camera input: {e}")
                    img_array = None

        if img_array is not None:
            st.subheader("Results")
            with st.spinner("Processing..."):
                try:
                    text, preds, boxes, vis_image = predict_text(model, img_array, class_mapping)

                    if not preds:
                        st.warning("No characters detected. Try a clearer image or adjust lighting.")
                    else:
                        st.success(f"Recognized Text: **{text}**")
                        st.write(f"Detected **{len(preds)}** characters.")

                        col_vis, col_details = st.columns([2, 1])

                        with col_vis:
                            st.image(vis_image, caption="Segmentation Visualization", use_container_width=True)

                        with col_details:
                            st.write("**Character Breakdown:**")
                            for i, (char, conf) in enumerate(preds):
                                st.write(f"{i+1}. **{char}** - {conf*100:.1f}%")

                except Exception as e:
                    st.error(f"Error during text recognition: {e}")

    elif app_mode == "Word Recognition (IAM CRNN)":
        iam_model = load_iam_model()
        if iam_model is None:
            st.error(f"Model not found at {IAM_MODEL_PATH}. Please run the IAM CRNN pipeline first.")
            return

        st.header("Word Recognition (IAM CRNN)")
        st.markdown("Upload an image of a **single handwritten word** to recognize it using the CRNN+CTC model. *Note: This is word-level recognition, not sentence-level.*")

        uploaded_file = st.file_uploader("Upload an image with a single handwritten word", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            try:
                # Load and preprocess exactly like the notebook
                img_bytes = uploaded_file.read()
                
                # --- NEW AUTOCROP PREPROCESSING ---
                np_img = np.frombuffer(img_bytes, np.uint8)
                img_cv = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)
                
                # Detect foreground (handle white or black background)
                if img_cv.mean() > 127:
                    _, thresh = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                else:
                    _, thresh = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                coords = cv2.findNonZero(thresh)
                if coords is not None:
                    x, y, w, h = cv2.boundingRect(coords)
                    pad = 10
                    x = max(0, x - pad)
                    y = max(0, y - pad)
                    w = min(img_cv.shape[1] - x, w + 2 * pad)
                    h = min(img_cv.shape[0] - y, h + 2 * pad)
                    img_cv = img_cv[y:y+h, x:x+w]
                
                # Back to tensor
                img = tf.convert_to_tensor(img_cv)
                img = tf.expand_dims(img, axis=-1)
                
                # --- EXISTING NOTEBOOK PREPROCESSING ---
                img = tf.image.resize_with_pad(img, target_height=32, target_width=128)
                img = tf.cast(img, tf.float32) / 255.0
                img = tf.transpose(img, perm=[1, 0, 2])

                # Expand dims to batch size 1
                img_batch = tf.expand_dims(img, axis=0)

                # Show uploaded image
                st.image(uploaded_file, caption="Uploaded Image", use_container_width=False, width=300)

                with st.spinner("Processing..."):
                    preds = iam_model.predict(img_batch)
                    predicted_word = decode_iam_prediction(preds)

                    st.success(f"Recognized Word: **{predicted_word}**")
            except Exception as e:
                st.error(f"Error during CRNN inference: {e}")

if __name__ == "__main__":
    main()

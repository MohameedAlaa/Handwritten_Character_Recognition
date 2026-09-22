import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from src.inference import predict_character
from src.data_loader import get_class_mapping
from src.config import BEST_MODEL_PATH
import os

st.set_page_config(page_title="Character Recognition", page_icon="✍️", layout="wide")

IAM_MODEL_PATH = "models/iam_crnn_best.keras"

# CRNN vocabulary (matching 02_iam_crnn.ipynb exactly)
iam_vocab = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
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

    app_mode = st.sidebar.radio("Select Mode:", ("Single Character", "Handwritten Word Recognition"))

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

    elif app_mode == "Handwritten Word Recognition":
        iam_model = load_iam_model()
        if iam_model is None:
            st.error(f"Model not found at {IAM_MODEL_PATH}. Please run the IAM CRNN pipeline first.")
            return

        st.header("Handwritten Word Recognition")
        st.markdown("Upload or take a photo of a **single handwritten word** to recognize it. *Note: This is word-level recognition, not sentence-level.*")

        input_mode = st.sidebar.radio("Choose Input Mode:", ("Upload Image", "Camera Capture"))

        img_bytes = None

        if input_mode == "Upload Image":
            uploaded_file = st.file_uploader("Upload an image with a single handwritten word", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                img_bytes = uploaded_file.read()
                st.image(uploaded_file, caption="Uploaded Image", use_container_width=False, width=300)

        elif input_mode == "Camera Capture":
            camera_file = st.camera_input("Take a picture of a single handwritten word")
            if camera_file is not None:
                img_bytes = camera_file.read()

        if img_bytes is not None:
            try:
                # --- NEW AUTOCROP PREPROCESSING ---
                np_img = np.frombuffer(img_bytes, np.uint8)
                img_cv = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)
                
                # Detect foreground (handle white or black background)
                if img_cv.mean() > 127:
                    _, thresh = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    bg_color = 255
                else:
                    _, thresh = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    bg_color = 0
                
                # Remove horizontal ruled-paper lines
                # Use a wide structuring element to be conservative and only catch long lines
                kernel_length = max(img_cv.shape[1] // 10, 40)
                horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
                
                # Isolate lines
                lines_isolated = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
                
                # Remove isolated lines from both the threshold mask and the original image
                img_cv[lines_isolated > 0] = bg_color
                thresh[lines_isolated > 0] = 0
                
                # --- NEW WORD SEGMENTATION ---
                # 1. Morphological dilation to group characters into word blobs
                # A rectangular kernel wider than tall connects characters horizontally
                kernel_word = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 3))
                dilated = cv2.dilate(thresh, kernel_word, iterations=1)
                
                # 2. Find contours
                contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                boxes = []
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    # Filter noise: minimum 10x10 and area > 100
                    if w > 10 and h > 10 and (w * h) > 100:
                        boxes.append([x, y, w, h])
                
                if not boxes:
                    st.warning("No handwritten text detected. Please try again.")
                    return
                    
                # 3. Group boxes into text lines
                boxes.sort(key=lambda b: b[1])  # Sort by y first
                
                lines = []
                current_line = []
                # dynamic y-tolerance based on median height
                median_h = np.median([b[3] for b in boxes]) if boxes else 20
                y_tol = median_h * 0.5
                
                for box in boxes:
                    cy = box[1] + box[3] / 2
                    if not current_line:
                        current_line.append(box)
                    else:
                        avg_cy = np.mean([b[1] + b[3] / 2 for b in current_line])
                        if abs(cy - avg_cy) < y_tol:
                            current_line.append(box)
                        else:
                            lines.append(current_line)
                            current_line = [box]
                if current_line:
                    lines.append(current_line)
                    
                # 4. Sort lines vertically and words horizontally
                lines.sort(key=lambda line: np.mean([b[1] for b in line]))
                for line in lines:
                    line.sort(key=lambda b: b[0])
                    
                num_lines = len(lines)
                num_words = sum(len(line) for line in lines)
                
                # Optional: draw segmentation preview for debugging
                vis = cv2.cvtColor(img_cv.copy(), cv2.COLOR_GRAY2BGR)
                for line in lines:
                    for (x, y, w, h) in line:
                        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
                st.image(vis, caption=f"Segmentation Preview ({num_lines} lines, {num_words} words)", use_container_width=False, width=300)
                
                # 5. Process each word
                word_crops = []
                
                for line in lines:
                    for (x, y, w, h) in line:
                        pad = 10
                        x_start = max(0, x - pad)
                        y_start = max(0, y - pad)
                        x_end = min(img_cv.shape[1], x + w + pad)
                        y_end = min(img_cv.shape[0], y + h + pad)
                        
                        word_crop = img_cv[y_start:y_end, x_start:x_end]
                        
                        # --- EXISTING NOTEBOOK PREPROCESSING ---
                        img = tf.convert_to_tensor(word_crop)
                        img = tf.expand_dims(img, axis=-1)
                        img = tf.image.resize_with_pad(img, target_height=32, target_width=128)
                        img = tf.cast(img, tf.float32) / 255.0
                        img = tf.transpose(img, perm=[1, 0, 2])
                        word_crops.append(img)
                        
                if word_crops:
                    # 6. Batch inference
                    batch_tensor = tf.stack(word_crops, axis=0)
                    with st.spinner(f"Recognizing {num_words} words..."):
                        preds = iam_model.predict(batch_tensor, verbose=0)
                        
                        # CTC decoding for the whole batch inline to keep it simple
                        preds_transpose = tf.transpose(preds, perm=[1, 0, 2])
                        seq_lens = tf.fill([preds.shape[0]], preds.shape[1])
                        decoded, _ = tf.nn.ctc_greedy_decoder(preds_transpose, sequence_length=seq_lens, blank_index=0)
                        decoded_dense = tf.sparse.to_dense(decoded[0], default_value=-1).numpy()
                        
                        texts = []
                        for seq in decoded_dense:
                            text = "".join([iam_num_to_char[c] for c in seq if c in iam_num_to_char])
                            texts.append(text)
                        
                        # Reconstruct text
                        idx = 0
                        final_text = []
                        for line in lines:
                            line_text = []
                            for _ in line:
                                line_text.append(texts[idx])
                                idx += 1
                            final_text.append(" ".join(line_text))
                            
                    reconstructed = "\n".join(final_text)
                    st.success("Recognition Complete:")
                    st.text(reconstructed)
            except Exception as e:
                st.error(f"Error during CRNN inference: {e}")

if __name__ == "__main__":
    main()

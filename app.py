import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from src.inference import predict_character
from src.data_loader import get_class_mapping
from src.config import BEST_MODEL_PATH
from src.preprocessing import preprocess_images
import os

def segment_word_image(img_gray):
    # Ensure binary image
    _, binary = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY)
    
    # 1. Connected Components (Contours)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter very small noise contours
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 5]
    
    if not boxes:
        return [], [], img_gray
        
    # Stage 1: Grouping
    # Group bounding boxes that belong to the same character (e.g., disconnected strokes of H)
    def merge_boxes(b_list):
        merged_any = True
        while merged_any:
            merged_any = False
            b_list.sort(key=lambda b: b[0])
            new_boxes = []
            skip = set()
            for i, b1 in enumerate(b_list):
                if i in skip: continue
                x1, y1, w1, h1 = b1
                for j in range(i+1, len(b_list)):
                    if j in skip: continue
                    x2, y2, w2, h2 = b_list[j]
                    
                    h_overlap = min(x1+w1, x2+w2) - max(x1, x2)
                    v_overlap = min(y1+h1, y2+h2) - max(y1, y2)
                    
                    should_merge = False
                    
                    # 1. Strong horizontal overlap (intersecting or very close)
                    # Allow a tiny gap (e.g. up to 2 pixels) for slightly broken strokes
                    if h_overlap > -2:
                        if v_overlap > -min(h1, h2) * 0.5:
                            should_merge = True
                            
                    # 2. Inside or heavily overlapping
                    area1 = w1 * h1
                    area2 = w2 * h2
                    intersection = max(0, h_overlap) * max(0, v_overlap)
                    if intersection > min(area1, area2) * 0.5:
                        should_merge = True
                        
                    if should_merge:
                        nx = min(x1, x2)
                        ny = min(y1, y2)
                        nw = max(x1+w1, x2+w2) - nx
                        nh = max(y1+h1, y2+h2) - ny
                        b1 = (nx, ny, nw, nh)
                        x1, y1, w1, h1 = b1
                        skip.add(j)
                        merged_any = True
                
                new_boxes.append(b1)
            b_list = new_boxes
        return b_list
        
    grouped_boxes = merge_boxes(boxes)
    
    # Stage 2: Splitting
    # Split touching characters by analyzing vertical projection
    final_boxes = []
    
    widths = [b[2] for b in grouped_boxes]
    median_w = np.median(widths) if widths else 0
    
    for (x, y, w, h) in grouped_boxes:
        aspect_ratio = w / float(max(h, 1))
        # Suspiciously wide component?
        is_wide = aspect_ratio > 1.2 or (w > median_w * 1.3 and w > h * 0.5)
        
        if is_wide:
            roi = binary[y:y+h, x:x+w]
            proj = np.sum(roi, axis=0) / 255.0  # number of foreground pixels in each column
            
            # Smooth the projection to find reliable valleys
            kernel = np.ones(5) / 5
            proj_smooth = np.convolve(proj, kernel, mode='same')
            
            valleys = []
            max_proj = np.max(proj_smooth)
            
            for i in range(5, w - 5):
                window = proj_smooth[i-4:i+5]
                if proj_smooth[i] == np.min(window):
                    left_peak = np.max(proj_smooth[:i])
                    right_peak = np.max(proj_smooth[i:])
                    
                    # True valley must be flanked by peaks
                    if proj_smooth[i] < left_peak * 0.8 and proj_smooth[i] < right_peak * 0.8:
                        if proj_smooth[i] < max_proj * 0.6:
                            # Check transitions in the original column
                            col = roi[:, i]
                            transitions = np.sum((col[:-1] == 0) & (col[1:] > 0))
                            if col[0] > 0: transitions += 1
                            
                            # Touching characters usually connect at a single point (1 transition)
                            if transitions <= 1:
                                valleys.append(i)
                                
            # Filter valleys (group flat valleys, avoid edges)
            valid_splits = []
            if valleys:
                groups = []
                current_group = [valleys[0]]
                for v in valleys[1:]:
                    if v - current_group[-1] <= 5:
                        current_group.append(v)
                    else:
                        groups.append(current_group)
                        current_group = [v]
                groups.append(current_group)
                
                last_split = 0
                for g in groups:
                    center_v = g[len(g)//2]
                    # Ensure it's not too close to the edges
                    if (center_v - last_split) > max(10, h * 0.1) and (w - center_v) > max(10, h * 0.1):
                        valid_splits.append(center_v)
                        last_split = center_v
                        
            if valid_splits:
                prev = 0
                for v in valid_splits:
                    sub_roi = roi[:, prev:v]
                    if cv2.countNonZero(sub_roi) > 5:
                        coords = cv2.findNonZero(sub_roi)
                        if coords is not None:
                            sx, sy, sw, sh = cv2.boundingRect(coords)
                            final_boxes.append((x + prev + sx, y + sy, sw, sh))
                    prev = v
                    
                sub_roi = roi[:, prev:]
                if cv2.countNonZero(sub_roi) > 5:
                    coords = cv2.findNonZero(sub_roi)
                    if coords is not None:
                        sx, sy, sw, sh = cv2.boundingRect(coords)
                        final_boxes.append((x + prev + sx, y + sy, sw, sh))
            else:
                final_boxes.append((x, y, w, h))
        else:
            final_boxes.append((x, y, w, h))
            
    final_boxes.sort(key=lambda b: b[0])
    
    vis_image = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    crops = []
    
    for (x, y, w, h) in final_boxes:
        cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Crop
        crop = binary[y:y+h, x:x+w]
        
        # Pad to make square
        diff = abs(w - h)
        top = bottom = left = right = 0
        if w > h:
            top = diff // 2
            bottom = diff - top
        elif h > w:
            left = diff // 2
            right = diff - left
            
        padding = max(w, h) // 4  # Reasonable padding
        crop_padded = cv2.copyMakeBorder(crop, top+padding, bottom+padding, left+padding, right+padding, cv2.BORDER_CONSTANT, value=0)
        crops.append(crop_padded)
        
    return crops, final_boxes, vis_image

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

    app_mode = st.sidebar.radio("Select Mode:", ("Write a Word", "Single Character"))

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

    elif app_mode == "Write a Word":
        model = load_model()
        if model is None:
            st.error(f"Model not found at {BEST_MODEL_PATH}. Please run the training pipeline first.")
            return

        st.header("Write a Word")
        st.markdown("Write one handwritten word inside the canvas. Then click **Recognize Word**.")

        # Large canvas for writing a full word
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=12,
            stroke_color="white",
            background_color="black",
            height=200,
            width=800,
            drawing_mode="freedraw",
            key="word_canvas",
            return_image_data=True,
            update_streamlit=True,
        )

        col1, col2 = st.columns([1, 6])
        with col1:
            recognize_btn = st.button("Recognize Word", type="primary")

        if recognize_btn:
            if canvas_result.image_data is None:
                st.warning("Canvas is empty.")
                return

            # Extract image data (RGBA) and convert to grayscale
            img_rgba = canvas_result.image_data.astype(np.uint8)
            img_gray = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2GRAY)

            if img_gray.max() < 10:
                st.warning("No characters were detected. Please write a word clearly inside the canvas.")
                return

            with st.spinner("Segmenting and recognizing..."):
                try:
                    # 1. Segment Characters
                    crops, boxes, vis_image = segment_word_image(img_gray)

                    if not crops:
                        st.warning("No characters were detected. Please write a word clearly inside the canvas.")
                        return

                    # Debug View: Segmentation Preview
                    with st.expander(f"Segmentation Preview (Detected {len(crops)} characters)", expanded=False):
                        st.image(vis_image, width=400)

                    # 2. Preprocess Crops
                    resized_crops = []
                    for crop in crops:
                        crop_resized = cv2.resize(crop, (28, 28), interpolation=cv2.INTER_AREA)
                        # We need to transpose to match EMNIST format expected by preprocess_images
                        crop_transposed = np.transpose(crop_resized)
                        resized_crops.append(crop_transposed)

                    # 3. Batch Inference
                    batch_array = np.array(resized_crops)
                    input_tensor = preprocess_images(batch_array)
                    preds = model.predict(input_tensor, verbose=0)

                    # 4. Decode and Reconstruct
                    recognized_word = ""
                    details = []
                    for i, pred_probs in enumerate(preds):
                        pred_class_idx = np.argmax(pred_probs)
                        char = class_mapping[pred_class_idx]
                        conf = pred_probs[pred_class_idx]
                        recognized_word += char
                        details.append(f"**{char}**  ({conf*100:.1f}%)")

                    # 5. Display Final Results
                    st.success("Recognition Complete")
                    st.markdown("### Recognized Word")
                    st.markdown(f"# {recognized_word}")

                    st.markdown("**Detected Characters:**")
                    cols = st.columns(min(len(details), 8))
                    for i, detail in enumerate(details):
                        with cols[i % len(cols)]:
                            st.write(detail)

                except Exception as e:
                    st.error(f"Error during word recognition: {e}")

if __name__ == "__main__":
    main()

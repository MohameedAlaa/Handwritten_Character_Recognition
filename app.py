"""Streamlit UI for single-character CharNet and word-level IAM CRNN recognition."""
from pathlib import Path

import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from src.config import BEST_MODEL_PATH, IAM_CRNN_MODEL_PATH
from src.data_loader import get_class_mapping
from src.inference import (
    load_charnet_model,
    load_iam_model,
    predict_character,
    predict_word_crnn,
    predict_word_segmented,
)
from src.preprocessing import load_and_validate_uploaded_image

st.set_page_config(page_title="Handwritten Character & Word Recognition", page_icon="✍️", layout="wide")


@st.cache_resource(show_spinner="Loading CharNet character model...")
def get_charnet_model():
    return load_charnet_model(BEST_MODEL_PATH)


@st.cache_resource(show_spinner="Loading IAM CRNN word model...")
def get_crnn_model():
    return load_iam_model(IAM_CRNN_MODEL_PATH)


def _canvas_image(key: str, height: int, width: int, stroke_width: int):
    try:
        return st_canvas(
            fill_color="rgba(255, 255, 255, 1.0)",
            stroke_width=stroke_width,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=height,
            width=width,
            drawing_mode="freedraw",
            key=key,
            update_streamlit=True,
            return_image_data=True,
        )
    except (st.errors.StreamlitAPIException, RuntimeError):
        # Graceful fallback specifically for headless test environments (e.g., Streamlit AppTest)
        # where custom frontend bidirection components cannot be mounted without an active browser.
        from types import SimpleNamespace
        return SimpleNamespace(image_data=None, json_data=None)
    except ValueError as exc:
        if "is not registered" in str(exc):
            from types import SimpleNamespace
            return SimpleNamespace(image_data=None, json_data=None)
        raise



def _top_predictions(probabilities: np.ndarray, class_mapping, limit: int = 3):
    indices = np.argsort(probabilities)[::-1][: min(limit, len(probabilities))]
    rows = []
    for rank, idx in enumerate(indices, start=1):
        rows.append({
            "Rank": rank,
            "Character": class_mapping.get(int(idx), f"class_{idx}"),
            "Class index": int(idx),
            "Softmax probability": f"{float(probabilities[idx]) * 100:.2f}%",
        })
    return rows


def single_character_page(class_mapping):
    st.header("Single Character Recognition")
    st.caption("Use one clearly written character. The model supports the 47 classes in EMNIST Balanced.")
    input_mode = st.radio("Input method", ["Draw a character", "Upload an image"], horizontal=True)
    image_input = None

    if input_mode == "Draw a character":
        canvas = _canvas_image("single_character_canvas", height=280, width=280, stroke_width=20)
        if canvas.image_data is not None:
            image_input = np.asarray(canvas.image_data, dtype=np.uint8)
        run_prediction = st.button("Recognize Character", type="primary", key="recognize_character")
    else:
        uploaded = st.file_uploader("Upload a character image", type=["png", "jpg", "jpeg", "webp"], key="character_upload")
        if uploaded is not None:
            try:
                image_input = load_and_validate_uploaded_image(uploaded)
                st.image(image_input, caption="Uploaded input", width=220)
            except Exception as exc:
                st.error(f"Could not read the uploaded image: {exc}")
        run_prediction = st.button("Recognize Character", type="primary", key="recognize_uploaded_character")

    if not run_prediction:
        st.info("Provide an image and select Recognize Character.")
        return
    if image_input is None:
        st.warning("Draw a character or upload an image first.")
        return

    try:
        model = get_charnet_model()
        character, confidence, probabilities = predict_character(model, image_input, class_mapping)
    except Exception as exc:
        st.error(f"Character recognition failed: {exc}")
        return

    if not character:
        st.warning("The input appears blank or contains too little visible stroke information.")
        return

    left, right = st.columns([1, 2])
    with left:
        st.metric("Predicted character", character)
        st.metric("Top class softmax probability", f"{confidence * 100:.2f}%")
        st.caption("A softmax probability is not a calibrated guarantee of correctness.")
    with right:
        st.markdown("**Top 3 predictions**")
        st.table(_top_predictions(probabilities, class_mapping, limit=3))


def word_recognition_page(class_mapping):
    st.header("Handwritten Word Recognition")
    st.caption("The CRNN performs sequence-level recognition. The segmented CharNet path is kept as a diagnostic fallback.")
    input_mode = st.radio("Word input method", ["Draw on canvas", "Upload a word image"], horizontal=True)
    image_input = None

    if input_mode == "Draw on canvas":
        st.write("Write one word from left to right. Leave a small margin around the word.")
        canvas = _canvas_image("word_canvas", height=200, width=800, stroke_width=12)
        if canvas.image_data is not None:
            image_input = np.asarray(canvas.image_data, dtype=np.uint8)
    else:
        uploaded = st.file_uploader("Upload a single handwritten word image", type=["png", "jpg", "jpeg", "webp"], key="word_upload")
        if uploaded is not None:
            try:
                image_input = load_and_validate_uploaded_image(uploaded)
                st.image(image_input, caption="Uploaded word image", use_container_width=True)
            except Exception as exc:
                st.error(f"Could not read the uploaded image: {exc}")

    engine = st.radio(
        "Recognition engine",
        ["IAM CRNN sequence model (recommended)", "Segmented CharNet fallback (diagnostic)"],
        horizontal=True,
    )
    recognize = st.button("Recognize Word", type="primary", key="recognize_word")
    if not recognize:
        st.info("Draw or upload a word, then select Recognize Word.")
        return
    if image_input is None:
        st.warning("Draw a word or upload an image first.")
        return

    if engine.startswith("IAM CRNN"):
        try:
            model = get_crnn_model()
            result = predict_word_crnn(image_input, model)
        except Exception as exc:
            st.error(f"CRNN word recognition failed: {exc}")
            st.caption("This does not automatically switch to a different model. Select the diagnostic fallback explicitly if needed.")
            return
        if result["is_blank"]:
            st.warning("The input appears blank or contains insufficient stroke information.")
            return
        if not result["word"]:
            st.warning("The model returned an empty decoded sequence. Try a clearer, tightly cropped word image or inspect the fallback path.")
            return
        st.success("Word prediction completed.")
        st.markdown(f"## {result['word']}")
        st.write(f"Mean maximum token probability: **{result['confidence'] * 100:.2f}%**")
        st.caption("This score averages the maximum softmax probability for emitted characters. It is not calibrated word-level confidence.")
    else:
        try:
            model = get_charnet_model()
            result = predict_word_segmented(image_input, model, class_mapping)
        except Exception as exc:
            st.error(f"Segmented fallback failed: {exc}")
            return
        if result["is_blank"]:
            st.warning("No valid character components were detected.")
            return
        st.success("Diagnostic segmented prediction completed.")
        st.markdown(f"## {result['word']}")
        st.write(f"Mean character softmax probability: **{result['confidence'] * 100:.2f}%**")
        st.write(f"Character components detected: **{len(result['char_predictions'])}**")
        st.caption("This fallback classifies isolated crops independently. It can fail on cursive or touching characters and is not equivalent to sequence-level OCR.")
        visual = result.get("visualization")
        if visual is not None:
            st.image(visual, caption="Segmentation diagnostic", use_container_width=True, channels="BGR")
        with st.expander("Per-character details"):
            st.json(result["char_predictions"])


def main():
    st.title("Handwritten Character & Word Recognition")
    st.markdown(
        "Classify a single handwritten character with CharNet or recognize a whole word with the IAM CRNN sequence model."
    )
    class_mapping = get_class_mapping()
    mode = st.sidebar.radio("Recognition task", ["Single Character", "Handwritten Word"])

    if mode == "Single Character":
        if not Path(BEST_MODEL_PATH).is_file():
            st.error(f"Character model checkpoint not found: {BEST_MODEL_PATH}")
            return
        single_character_page(class_mapping)
    else:
        st.sidebar.caption(f"CRNN checkpoint: {Path(IAM_CRNN_MODEL_PATH).name}")
        word_recognition_page(class_mapping)


if __name__ == "__main__":
    main()

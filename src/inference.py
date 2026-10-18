"""Inference helpers for CharNet character classification and IAM CRNN word OCR."""
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import tensorflow as tf
except ImportError:  # Permit lightweight unit tests that do not load saved models.
    tf = None

from src.config import (
    BEST_MODEL_PATH,
    CHARNET_MODEL_PATH,
    CTC_BLANK_INDEX,
    IAM_CRNN_MODEL_PATH,
    IAM_NUM_TO_CHAR,
    IAM_NUM_CLASSES,
)
from src.preprocessing import preprocess_for_charnet, preprocess_for_crnn, to_grayscale_uint8
from src.segmentation import segment_characters

ImageLike = Union[np.ndarray, Image.Image]


def _require_tensorflow() -> None:
    if tf is None:
        raise RuntimeError("TensorFlow is not installed. Install the dependencies from requirements.txt.")


def load_charnet_model(model_path: Union[str, Path] = CHARNET_MODEL_PATH):
    """Load the character classifier without requiring its saved training compile config."""
    _require_tensorflow()
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"CharNet checkpoint not found: {path}")
    return tf.keras.models.load_model(str(path), compile=False)


def load_model(model_path: Union[str, Path] = BEST_MODEL_PATH):
    """Backward-compatible alias used by the original Streamlit app."""
    return load_charnet_model(model_path)


def load_iam_model(model_path: Union[str, Path] = IAM_CRNN_MODEL_PATH):
    """Load the saved IAM CRNN inference model.

    The supplied checkpoint is a plain Functional model with an image input and
    `dense_output` linear logits, so no custom CTCLayer is required for inference.
    """
    _require_tensorflow()
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"IAM CRNN checkpoint not found: {path}")
    model = tf.keras.models.load_model(str(path), compile=False)
    output_shape = getattr(model, "output_shape", None)
    if isinstance(output_shape, list):
        raise ValueError(f"Expected a single CRNN output tensor, got output shapes {output_shape}.")
    if output_shape is not None and (len(output_shape) != 3 or output_shape[-1] != IAM_NUM_CLASSES):
        raise ValueError(
            f"CRNN output shape {output_shape} is incompatible with the expected (batch, time, {IAM_NUM_CLASSES})."
        )
    return model


def _softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values - np.max(values, axis=-1, keepdims=True)
    exp_values = np.exp(values)
    sums = np.sum(exp_values, axis=-1, keepdims=True)
    return (exp_values / np.maximum(sums, np.finfo(np.float64).tiny)).astype(np.float32)


def _as_probabilities(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError("Model produced non-finite prediction values.")
    if values.ndim < 2:
        raise ValueError(f"Expected batched model output, got shape {values.shape}.")
    sums = np.sum(values, axis=-1, keepdims=True)
    if np.all(values >= 0.0) and np.allclose(sums, 1.0, atol=1e-3):
        return values
    return _softmax(values)


def predict_character(model: Any, image_array: ImageLike, class_mapping: Mapping[int, str]):
    """Predict one character; preserve the original `(character, confidence, probs)` API."""
    tensor, is_blank = preprocess_for_charnet(image_array)
    number_of_classes = len(class_mapping)
    if is_blank:
        return "", 0.0, np.zeros((number_of_classes,), dtype=np.float32)

    predictions = np.asarray(model.predict(tensor, verbose=0), dtype=np.float32)
    if predictions.ndim != 2 or predictions.shape[0] != 1:
        raise ValueError(f"Expected character-model output shape (1, classes), got {predictions.shape}.")
    if predictions.shape[1] != number_of_classes:
        raise ValueError(
            f"Character model returned {predictions.shape[1]} classes, but the class mapping contains {number_of_classes}."
        )

    probabilities = _as_probabilities(predictions)[0]
    predicted_index = int(np.argmax(probabilities))
    if predicted_index not in class_mapping:
        raise ValueError(f"Predicted class index {predicted_index} is missing from the class mapping.")
    return class_mapping[predicted_index], float(probabilities[predicted_index]), probabilities


def decode_iam_prediction(
    predictions: np.ndarray,
    num_to_char: Mapping[int, str] = IAM_NUM_TO_CHAR,
    blank_index: int = CTC_BLANK_INDEX,
) -> Tuple[List[str], List[float]]:
    """Greedy CTC decoding for CRNN logits/probabilities shaped (batch, time, classes).

    Consecutive duplicate tokens collapse, while a blank between equal tokens resets
    the repetition run, as required by CTC. Blank is index 0 for this checkpoint.
    Returned confidence is the mean maximum softmax probability of emitted tokens;
    it is not a calibrated probability that the whole word is correct.
    """
    values = np.asarray(predictions, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"Expected predictions shaped (batch, time, classes), got {values.shape}.")
    if values.shape[0] == 0 or values.shape[1] == 0:
        return [], []
    if values.shape[-1] != len(num_to_char) + 1:
        raise ValueError(
            f"CRNN emitted {values.shape[-1]} classes, but vocabulary mapping plus blank requires {len(num_to_char) + 1}."
        )
    if blank_index < 0 or blank_index >= values.shape[-1]:
        raise ValueError(f"CTC blank index {blank_index} is outside the model output dimension.")

    probabilities = _as_probabilities(values)
    decoded_texts: List[str] = []
    confidence_scores: List[float] = []
    for batch_index in range(probabilities.shape[0]):
        step_probabilities = probabilities[batch_index]
        token_ids = np.argmax(step_probabilities, axis=-1)
        token_scores = np.max(step_probabilities, axis=-1)
        output_chars: List[str] = []
        emitted_scores: List[float] = []
        previous_id: Optional[int] = None

        for token_id_raw, score in zip(token_ids, token_scores):
            token_id = int(token_id_raw)
            if token_id != previous_id and token_id != blank_index:
                character = num_to_char.get(token_id)
                if character is not None:
                    output_chars.append(character)
                    emitted_scores.append(float(score))
            previous_id = token_id

        decoded_texts.append("".join(output_chars).strip())
        confidence_scores.append(float(np.mean(emitted_scores)) if emitted_scores else 0.0)
    return decoded_texts, confidence_scores


def predict_word_crnn(image: ImageLike, crnn_model: Any) -> Dict[str, Any]:
    """Recognize a word with the IAM CRNN sequence model and greedy CTC decoding."""
    tensor, is_blank = preprocess_for_crnn(image, invert_to_dark_ink_on_white=True)
    if is_blank:
        return {"is_blank": True, "word": "", "confidence": 0.0, "mode": "crnn"}

    predictions = np.asarray(crnn_model.predict(tensor, verbose=0), dtype=np.float32)
    if predictions.ndim != 3 or predictions.shape[0] != 1 or predictions.shape[-1] != IAM_NUM_CLASSES:
        raise ValueError(f"Unexpected CRNN prediction shape: {predictions.shape}.")
    texts, confidences = decode_iam_prediction(predictions)
    return {
        "is_blank": False,
        "word": texts[0] if texts else "",
        "confidence": confidences[0] if confidences else 0.0,
        "mode": "crnn",
    }


def predict_text(model: Any, image_array: ImageLike, class_mapping: Mapping[int, str], min_area: int = 20):
    """Segment a word into characters and classify each crop with CharNet.

    This is a diagnostic/fallback path. It is not equivalent to sequence-level OCR.
    Return shape is preserved for compatibility with the original tests and callers.
    """
    if image_array is None:
        return "", [], [], image_array
    if isinstance(image_array, np.ndarray) and image_array.size == 0:
        return "", [], [], image_array
    gray = to_grayscale_uint8(image_array)
    if gray.size == 0:
        return "", [], [], gray
    try:
        crops, boxes, vis_image = segment_characters(gray, min_area=min_area)
    except Exception as exc:
        raise RuntimeError(f"Character segmentation failed: {exc}") from exc

    if not crops:
        return "", [], boxes, vis_image

    recognized_text = ""
    predictions: List[Tuple[str, float]] = []
    for crop in crops:
        character, confidence, _ = predict_character(model, crop, class_mapping)
        if not character:
            character = "?"
            confidence = 0.0
        recognized_text += character
        predictions.append((character, float(confidence)))
    return recognized_text, predictions, boxes, vis_image


def predict_word_segmented(image: ImageLike, charnet_model: Any, class_mapping: Mapping[int, str]):
    """Return structured results from the segmentation + CharNet diagnostic path."""
    text, character_predictions, boxes, visual = predict_text(
        charnet_model, image, class_mapping, min_area=20
    )
    if not character_predictions:
        return {
            "is_blank": True,
            "word": "",
            "confidence": 0.0,
            "char_predictions": [],
            "boxes": boxes,
            "visualization": visual,
            "mode": "segmented_charnet",
        }
    details = [
        {"character": character, "confidence": confidence, "box": box}
        for (character, confidence), box in zip(character_predictions, boxes)
    ]
    return {
        "is_blank": False,
        "word": text,
        "confidence": float(np.mean([item[1] for item in character_predictions])),
        "char_predictions": details,
        "boxes": boxes,
        "visualization": visual,
        "mode": "segmented_charnet",
    }

import numpy as np
import tensorflow as tf

from src.preprocessing import preprocess_images

def predict_character(model, image_array, class_mapping):
    """
    Predicts the character from a preprocessed image.
    """
    # Ensure image_array is 28x28
    if image_array.shape != (28, 28):
        image_array = tf.image.resize(np.expand_dims(image_array, axis=-1), [28, 28]).numpy().squeeze()

    # User uploads are upright. preprocess_images() transposes (0, 2, 1) assuming EMNIST format.
    # We pre-transpose so that preprocess_images() outputs the correct upright image.
    image_array = np.transpose(image_array)

    # preprocess_images expects a batch, so we add a batch dimension
    image_batch = np.array([image_array])
    input_tensor = preprocess_images(image_batch)

    pred_probs = model.predict(input_tensor, verbose=0)[0]

    pred_class_idx = np.argmax(pred_probs)
    predicted_char = class_mapping[pred_class_idx]
    confidence = pred_probs[pred_class_idx]

    return predicted_char, confidence, pred_probs

from src.segmentation import segment_characters

def predict_text(model, image_array, class_mapping, min_area=20):
    """
    Detects and predicts multiple characters from an image.

    Returns:
        recognized_text (str): The concatenated predicted characters.
        predictions (list): List of (predicted_char, confidence) tuples for each crop.
        boxes (list): Bounding boxes for each character.
        vis_image (np.ndarray): Visualization image with bounding boxes.
    """
    if image_array is None or image_array.size == 0:
        return "", [], [], image_array

    try:
        crops, boxes, vis_image = segment_characters(image_array, min_area=min_area)
    except Exception as e:
        print(f"Segmentation failed: {e}")
        return "", [], [], image_array

    if not crops:
        return "", [], [], vis_image

    recognized_text = ""
    predictions = []

    for crop in crops:
        try:
            # predict_character expects a 28x28 array. It handles resizing internally if it isn't,
            # but segmentation already pads to a square.
            char, conf, _ = predict_character(model, crop, class_mapping)
            recognized_text += char
            predictions.append((char, float(conf)))
        except Exception as e:
            print(f"Prediction failed for a crop: {e}")
            predictions.append(("?", 0.0))
            recognized_text += "?"

    return recognized_text, predictions, boxes, vis_image

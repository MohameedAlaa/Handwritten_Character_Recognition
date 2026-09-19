import numpy as np
import pytest
import cv2
from src.inference import predict_text, predict_character
import tensorflow as tf

# Mock the Keras Model for testing inference without loading real weights
class MockModel:
    def predict(self, x, verbose=0):
        # x shape should be (1, 28, 28, 1)
        # return a mock probability array of 47 classes
        probs = np.zeros((1, 47))
        probs[0, 15] = 0.99  # Mock predicting some class index 15 ('F')
        return probs

def test_predict_text_empty_image():
    model = MockModel()
    class_mapping = {i: str(i) for i in range(47)}

    empty_image = np.zeros((100, 100), dtype=np.uint8)
    text, preds, boxes, vis = predict_text(model, empty_image, class_mapping)

    assert text == ""
    assert len(preds) == 0
    assert len(boxes) == 0

def test_predict_text_multiple_characters():
    model = MockModel()
    class_mapping = {i: str(i) for i in range(47)}
    class_mapping[15] = 'H' # Mock mapping

    # Create an image with 3 distinct white squares
    image = np.zeros((100, 300), dtype=np.uint8)
    image[40:60, 40:60] = 255
    image[40:60, 140:160] = 255
    image[40:60, 240:260] = 255

    text, preds, boxes, vis = predict_text(model, image, class_mapping)

    assert len(preds) == 3
    assert len(boxes) == 3
    assert text == "HHH"

    for char, conf in preds:
        assert char == 'H'
        assert 0.0 <= conf <= 1.0

    assert boxes[0][0] == 40
    assert boxes[1][0] == 140
    assert boxes[2][0] == 240

def test_predict_text_no_characters_detected():
    model = MockModel()
    class_mapping = {i: str(i) for i in range(47)}

    # Image with small noise that should be filtered out by min_area=20
    image = np.zeros((100, 100), dtype=np.uint8)
    image[50:52, 50:52] = 255

    text, preds, boxes, vis = predict_text(model, image, class_mapping)

    assert text == ""
    assert len(preds) == 0
    assert len(boxes) == 0

def test_predict_character_keeps_working():
    model = MockModel()
    class_mapping = {i: str(i) for i in range(47)}
    class_mapping[15] = 'X'

    image = np.zeros((28, 28), dtype=np.uint8)
    image[10:20, 10:20] = 255

    char, conf, probs = predict_character(model, image, class_mapping)

    assert char == 'X'
    assert 0.0 <= conf <= 1.0
    assert probs.shape == (47,)

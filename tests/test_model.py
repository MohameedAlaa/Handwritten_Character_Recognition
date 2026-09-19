import numpy as np
import pytest
from src.model import build_charnet
from src.preprocessing import preprocess_images, preprocess_labels


def test_charnet_output_shape():
    model = build_charnet(input_shape=(28, 28, 1), num_classes=47)
    dummy_input = np.random.rand(1, 28, 28, 1).astype('float32')
    output = model.predict(dummy_input)
    assert output.shape == (1, 47), f"Expected shape (1, 47), got {output.shape}"

def test_preprocessing():
    # 2 random images 28x28
    dummy_images = np.random.randint(0, 255, size=(2, 28, 28))
    processed = preprocess_images(dummy_images)
    assert processed.shape == (2, 28, 28, 1)
    assert np.max(processed) <= 1.0
    assert np.min(processed) >= 0.0

def test_preprocessing_labels():
    dummy_labels = np.array([0, 46])
    processed = preprocess_labels(dummy_labels, num_classes=47)
    assert processed.shape == (2, 47)
    assert processed[0, 0] == 1
    assert processed[1, 46] == 1

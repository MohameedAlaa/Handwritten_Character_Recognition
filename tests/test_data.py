import numpy as np
import pytest
from src.data_loader import load_emnist_data, get_class_mapping
from src.preprocessing import preprocess_images, preprocess_labels

def test_data_loading():
    (X_train, y_train), (X_test, y_test) = load_emnist_data('balanced')
    assert X_train.shape == (112800, 28, 28)
    assert X_test.shape == (18800, 28, 28)
    assert y_train.shape == (112800,)
    assert y_test.shape == (18800,)

def test_class_mapping():
    mapping = get_class_mapping()
    assert len(mapping) == 47
    # Spot checks
    assert mapping[0] == '0'
    assert mapping[10] == 'A'
    assert mapping[36] == 'a'

def test_preprocessing_and_normalization():
    # Use a dummy batch
    dummy_images = np.ones((5, 28, 28), dtype=np.uint8) * 128

    # We want to check orientation/transpose, but here we just check shapes and values
    processed = preprocess_images(dummy_images)

    assert processed.shape == (5, 28, 28, 1)
    assert processed.dtype == np.float32
    assert np.allclose(processed, 128.0 / 255.0)

def test_label_preprocessing():
    dummy_labels = np.array([0, 46, 10])
    processed = preprocess_labels(dummy_labels, num_classes=47)

    assert processed.shape == (3, 47)
    assert processed[0, 0] == 1
    assert processed[1, 46] == 1
    assert processed[2, 10] == 1
    assert np.sum(processed[0]) == 1.0

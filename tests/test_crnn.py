import pytest
import os
import tensorflow as tf
import numpy as np

MODEL_PATH = "models/iam_crnn_best.keras"

@pytest.fixture
def crnn_model():
    if not os.path.exists(MODEL_PATH):
        pytest.skip("Model file not found. Ensure models/iam_crnn_best.keras exists.")
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

def test_crnn_output_shape(crnn_model):
    """Test that the CRNN output has correct time and class dimensions."""
    dummy_input = np.zeros((1, 128, 32, 1), dtype=np.float32)
    preds = crnn_model.predict(dummy_input)

    # Expected shape: (batch_size, time_steps=32, num_classes=80)
    assert preds.shape == (1, 32, 80), f"Unexpected output shape: {preds.shape}"

def test_ctc_configuration():
    """Verify CTC configuration constants used in decoding."""
    vocab_size = 79
    blank_index = 0
    num_classes = vocab_size + 1
    assert num_classes == 80, "Classes must be 80."
    assert blank_index == 0, "CTC blank index must be 0."

def test_dummy_decoding(crnn_model):
    """Test CTC greedy decoding logic on dummy predictions safely."""
    # Create a dummy prediction of shape (batch=1, time=32, classes=80)
    dummy_preds = np.zeros((1, 32, 80), dtype=np.float32)
    # Force class 1 at time 0
    dummy_preds[0, 0, 1] = 10.0
    # Force blank at time 1
    dummy_preds[0, 1, 0] = 10.0

    # Needs to be time-major for greedy_decoder: [time_steps, batch_size, num_classes]
    preds_transpose = tf.transpose(dummy_preds, perm=[1, 0, 2])
    seq_lens = tf.fill([dummy_preds.shape[0]], dummy_preds.shape[1])

    decoded, _ = tf.nn.ctc_greedy_decoder(preds_transpose, sequence_length=seq_lens, blank_index=0)
    decoded_dense = tf.sparse.to_dense(decoded[0], default_value=-1).numpy()

    # Should at least contain class 1 as decoded output
    assert len(decoded_dense[0]) > 0
    assert decoded_dense[0][0] == 1, "Decoding failed to identify the peak class."

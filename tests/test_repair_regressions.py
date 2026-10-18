"""Dataset-independent regression tests for repaired preprocessing and decoding."""
import numpy as np
from PIL import Image

from src.config import (
    BEST_MODEL_PATH,
    CHARNET_INPUT_SHAPE,
    CRNN_INPUT_SHAPE,
    CTC_BLANK_INDEX,
    IAM_CHAR_TO_NUM,
    IAM_NUM_CLASSES,
    IAM_NUM_TO_CHAR,
    IAM_VOCAB,
)
from src.evaluate import compute_cer_and_wer
from src.inference import decode_iam_prediction, predict_character, predict_word_crnn
from src.preprocessing import preprocess_for_charnet, preprocess_for_crnn, preprocess_images
from src.data_loader import get_class_mapping


class MockCharacterModel:
    def predict(self, inputs, verbose=0):
        assert inputs.shape == (1, 28, 28, 1)
        output = np.full((1, 47), 1e-4, dtype=np.float32)
        output[0, 15] = 1.0 - 46e-4
        return output


class MockBlankWordModel:
    def predict(self, inputs, verbose=0):
        assert inputs.shape == (1, 128, 32, 1)
        output = np.full((1, 32, IAM_NUM_CLASSES), -8.0, dtype=np.float32)
        output[:, :, CTC_BLANK_INDEX] = 8.0
        return output


def _synthetic_word():
    image = np.zeros((70, 200), dtype=np.uint8)
    image[20:55, 25:32] = 255
    image[20:55, 50:57] = 255
    image[35:42, 25:57] = 255
    image[25:55, 90:98] = 255
    image[25:55, 112:120] = 255
    return image


def test_iam_vocabulary_is_aligned_with_checkpoint_output_contract():
    assert len(IAM_VOCAB) == 79
    assert len(IAM_CHAR_TO_NUM) == 79
    assert len(IAM_NUM_TO_CHAR) == 79
    assert IAM_NUM_CLASSES == 80
    assert CTC_BLANK_INDEX == 0
    for character, index in IAM_CHAR_TO_NUM.items():
        assert index > 0
        assert IAM_NUM_TO_CHAR[index] == character
    assert BEST_MODEL_PATH.endswith("charnet_best.keras")


def test_emnist_training_preprocessing_keeps_legacy_batch_contract():
    images = np.arange(2 * 28 * 28, dtype=np.uint8).reshape(2, 28, 28)
    processed = preprocess_images(images)
    assert processed.shape == (2, *CHARNET_INPUT_SHAPE)
    assert processed.dtype == np.float32
    assert 0.0 <= processed.min() <= processed.max() <= 1.0


def test_charnet_preprocessing_is_polarity_invariant():
    dark_bg = _synthetic_word()
    light_bg = 255 - dark_bg
    from_dark, blank_dark = preprocess_for_charnet(dark_bg)
    from_light, blank_light = preprocess_for_charnet(light_bg)
    assert not blank_dark and not blank_light
    assert from_dark.shape == (1, *CHARNET_INPUT_SHAPE)
    assert np.allclose(from_dark, from_light, atol=1.0 / 255.0)


def test_crnn_preprocessing_matches_expected_layout_and_polarity():
    dark_bg = _synthetic_word()
    light_bg = 255 - dark_bg
    from_dark, blank_dark = preprocess_for_crnn(dark_bg)
    from_light, blank_light = preprocess_for_crnn(light_bg)
    assert not blank_dark and not blank_light
    assert from_dark.shape == (1, *CRNN_INPUT_SHAPE)
    assert from_dark.dtype == np.float32
    assert 0.0 <= from_dark.min() <= from_dark.max() <= 1.0
    assert np.allclose(from_dark, from_light, atol=1.0 / 255.0)


def test_rgba_input_preprocessing_does_not_crash():
    rgba = np.zeros((48, 48, 4), dtype=np.uint8)
    rgba[8:40, 20:28, :3] = 255
    rgba[8:40, 20:28, 3] = 255
    tensor, is_blank = preprocess_for_charnet(Image.fromarray(rgba, mode="RGBA"))
    assert not is_blank
    assert tensor.shape == (1, *CHARNET_INPUT_SHAPE)


def test_ctc_decoder_collapses_repeats_but_keeps_blank_separated_repeats():
    h = IAM_CHAR_TO_NUM["h"]
    i = IAM_CHAR_TO_NUM["i"]
    logits = np.full((1, 32, IAM_NUM_CLASSES), -8.0, dtype=np.float32)
    logits[:, :, CTC_BLANK_INDEX] = 8.0
    logits[0, 2:4, :] = -8.0
    logits[0, 2:4, h] = 12.0
    logits[0, 5, :] = -8.0
    logits[0, 5, h] = 12.0
    logits[0, 8, :] = -8.0
    logits[0, 8, i] = 12.0
    texts, confidences = decode_iam_prediction(logits)
    assert texts == ["hhi"]
    assert 0.99 < confidences[0] <= 1.0


def test_character_prediction_preserves_legacy_return_shape():
    mapping = get_class_mapping()
    image = np.zeros((28, 28), dtype=np.uint8)
    image[8:20, 12:16] = 255
    character, confidence, probabilities = predict_character(MockCharacterModel(), image, mapping)
    assert character == mapping[15]
    assert 0.0 <= confidence <= 1.0
    assert probabilities.shape == (47,)


def test_blank_character_and_word_inputs_are_handled_gracefully():
    mapping = get_class_mapping()
    character, confidence, probabilities = predict_character(
        MockCharacterModel(), np.zeros((64, 64), dtype=np.uint8), mapping
    )
    assert character == ""
    assert confidence == 0.0
    assert probabilities.shape == (47,)

    result = predict_word_crnn(np.zeros((80, 200), dtype=np.uint8), MockBlankWordModel())
    assert result["is_blank"] is True
    assert result["word"] == ""


def test_cer_wer_and_exact_match_metrics():
    metrics = compute_cer_and_wer(["hello", "world"], ["hello", "word"])
    assert metrics["exact_word_accuracy"] == 0.5
    assert np.isclose(metrics["cer"], 1.0 / 10.0)
    assert np.isclose(metrics["wer"], 0.5)


def test_evaluate_crnn_skips_zero_byte_images_safely(tmp_path):
    """Regression test: evaluate_crnn_model must ignore zero-byte PNGs before sampling/decoding."""
    from src.evaluate import evaluate_crnn_model

    iam_dir = tmp_path / "IAM"
    words_dir = iam_dir / "words" / "a01" / "a01-000u"
    words_dir.mkdir(parents=True)

    # Create 10 writers so train/val/test split puts at least 1 writer in test
    forms_lines = [f"a01-{i:03d}u {i:03d} 1 1 1 1 1 1\n" for i in range(10)]
    (iam_dir / "form.txt").write_text("".join(forms_lines), encoding="utf-8")

    words_lines = []
    for i in range(10):
        form_id = f"a01-{i:03d}u"
        f_dir = iam_dir / "words" / "a01" / form_id
        f_dir.mkdir(parents=True, exist_ok=True)
        # Valid image
        valid_id = f"{form_id}-00-00"
        img_arr = np.full((32, 64), 255, dtype=np.uint8)
        img_arr[8:24, 16:48] = 0
        Image.fromarray(img_arr, mode="L").save(f_dir / f"{valid_id}.png")
        words_lines.append(f"{valid_id} ok 154 1 1 1 1 NN A\n")
        # Zero-byte image in every writer so whichever writer lands in test has a 0-byte image
        zero_id = f"{form_id}-00-01"
        (f_dir / f"{zero_id}.png").write_bytes(b"")
        words_lines.append(f"{zero_id} ok 154 1 1 1 1 NN B\n")
    words_lines.append("malformedwordid ok 154 1 1 1 1 NN C\n")

    (iam_dir / "words.txt").write_text("".join(words_lines), encoding="utf-8")

    class DummyCRNN:
        def predict(self, batch, batch_size=64, verbose=0):
            out = np.full((len(batch), 32, IAM_NUM_CLASSES), -8.0, dtype=np.float32)
            out[:, :, CTC_BLANK_INDEX] = 8.0
            out[:, 0, IAM_CHAR_TO_NUM["A"]] = 12.0
            return out

    import pytest
    with pytest.raises(ValueError, match="max_samples must be a positive integer"):
        evaluate_crnn_model(model=DummyCRNN(), data_dir=str(tmp_path), max_samples=0, save_path=None)

    res = evaluate_crnn_model(
        model=DummyCRNN(),
        data_dir=str(tmp_path),
        max_samples=10,
        save_path=None,
        random_state=42,
    )
    assert res["evaluated_samples"] == 1
    assert res["metrics"]["exact_word_accuracy"] == 1.0
    assert "writer-independent" in res["note"]
    assert "not verified" in res["note"] or "unverified" in res["note"]
    assert "unseen during training/validation" not in res["note"]


def test_upload_exceeding_byte_size_limit_is_rejected():
    """Reject uploaded files whose byte size exceeds MAX_UPLOAD_SIZE_BYTES before decoding."""
    import io
    import pytest
    from src.config import MAX_UPLOAD_SIZE_BYTES
    from src.preprocessing import load_and_validate_uploaded_image

    oversized_stream = io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"0" * (MAX_UPLOAD_SIZE_BYTES + 1))
    with pytest.raises(ValueError, match="exceeds maximum allowed"):
        load_and_validate_uploaded_image(oversized_stream)


def test_image_exceeding_dimension_or_pixel_limit_is_rejected():
    """Reject images exceeding MAX_IMAGE_DIMENSION or MAX_IMAGE_PIXELS in upload and preprocessing paths."""
    import io
    import pytest
    from src.config import MAX_IMAGE_DIMENSION
    from src.inference import predict_text
    from src.preprocessing import _validate_image_dimensions, load_and_validate_uploaded_image, to_grayscale_uint8
    from src.segmentation import segment_characters

    # 1. Upload path with dimension > MAX_IMAGE_DIMENSION
    wide_img = Image.new("L", (MAX_IMAGE_DIMENSION + 10, 32), color=255)
    buf = io.BytesIO()
    wide_img.save(buf, format="PNG")
    buf.seek(0)
    with pytest.raises(ValueError, match="exceed"):
        load_and_validate_uploaded_image(buf)

    # 2. Pixel count > MAX_IMAGE_PIXELS even when each dimension <= MAX_IMAGE_DIMENSION (4000 x 3500 = 14 MP > 12 MP)
    with pytest.raises(ValueError, match="pixel count"):
        _validate_image_dimensions(4000, 3500)

    # 3. Direct preprocessing path with oversized dimensions
    with pytest.raises(ValueError, match="exceed"):
        to_grayscale_uint8(wide_img)

    # 4. Direct segmentation and predict_text paths with oversized numpy array
    oversized_arr = np.zeros((32, MAX_IMAGE_DIMENSION + 10), dtype=np.uint8)
    with pytest.raises(ValueError, match="exceed"):
        segment_characters(oversized_arr)
    with pytest.raises(ValueError, match="exceed"):
        predict_text(MockCharacterModel(), oversized_arr, {0: "0"})


def test_corrupted_or_invalid_image_bytes_are_rejected_safely():
    """Reject corrupted or non-image byte streams with a clean ValueError."""
    import io
    import pytest
    from src.preprocessing import load_and_validate_uploaded_image

    with pytest.raises(ValueError, match="Invalid or corrupted image"):
        load_and_validate_uploaded_image(io.BytesIO(b"not an image file at all"))

    with pytest.raises(ValueError, match="empty"):
        load_and_validate_uploaded_image(io.BytesIO(b""))


def test_valid_ordinary_uploaded_image_succeeds():
    """Verify ordinary valid uploaded PNG images pass validation and preprocessing."""
    import io
    from src.preprocessing import load_and_validate_uploaded_image, preprocess_for_charnet, preprocess_for_crnn

    arr = _synthetic_word()
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    buf.seek(0)

    loaded = load_and_validate_uploaded_image(buf)
    char_tensor, char_blank = preprocess_for_charnet(loaded)
    crnn_tensor, crnn_blank = preprocess_for_crnn(loaded)
    assert not char_blank
    assert not crnn_blank
    assert char_tensor.shape == (1, *CHARNET_INPUT_SHAPE)
    assert crnn_tensor.shape == (1, *CRNN_INPUT_SHAPE)


def test_notebooks_are_valid_json_and_contain_repaired_contracts():
    """Verify both notebooks are valid JSON, preserve outputs, and contain repaired pipeline contracts."""
    import json
    from pathlib import Path

    root_dir = Path(__file__).resolve().parents[1]
    nb1_path = root_dir / "notebooks" / "01_pipeline.ipynb"
    nb2_path = root_dir / "notebooks" / "02_iam_crnn.ipynb"

    nb1 = json.loads(nb1_path.read_text(encoding="utf-8"))
    nb2 = json.loads(nb2_path.read_text(encoding="utf-8"))

    assert len(nb1["cells"]) == 16
    assert len(nb2["cells"]) == 63

    # Verify historical outputs were preserved
    nb1_outputs = sum(len(c.get("outputs", [])) for c in nb1["cells"] if c.get("cell_type") == "code")
    nb2_outputs = sum(len(c.get("outputs", [])) for c in nb2["cells"] if c.get("cell_type") == "code")
    assert nb1_outputs > 0
    assert nb2_outputs > 0

    nb2_code = "\n".join(
        "".join(c.get("source", []))
        for c in nb2["cells"]
        if c.get("cell_type") == "code"
    )
    assert "from src.model import build_crnn" in nb2_code
    assert "from src.evaluate import compute_cer_and_wer" in nb2_code
    assert "tf.nn.ctc_greedy_decoder" in nb2_code
    assert "blank_index=BLANK_INDEX" in nb2_code
    assert "rotation_layer(image, training=True)" in nb2_code
    assert "assert len(val_writers.intersection(test_writers)) == 0" in nb2_code
    assert "tf.keras.backend.ctc_decode(" not in nb2_code

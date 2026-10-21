"""Automated regression tests for the IAM CRNN augmentation and preprocessing pipeline.

Verifies:
- Individual preprocessing of all 9 reported problematic samples across 10 repeated augmentations.
- Robust aspect-ratio preservation and padding on extreme image dimensions.
- Label encoding invariants, CTC blank handling, and label length encoding.
- tf.data augmented dataset construction and five consecutive batch iterations from a single iterator.
"""
import os
import sys
from pathlib import Path
import pytest
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 128
MAX_SEQ_LENGTH = 32
BATCH_SIZE = 128
BLANK_INDEX = 0
RANDOM_SEED = 42

IAM_VOCAB = [
    ' ', '!', '"', '#', '&', "'", '(', ')', '*', '+',
    ',', '-', '.', '/', '0', '1', '2', '3', '4', '5',
    '6', '7', '8', '9', ':', ';', '?',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z'
]

char_to_num = tf.keras.layers.StringLookup(vocabulary=IAM_VOCAB, mask_token=None)

rotation_layer = tf.keras.layers.RandomRotation(
    factor=3.0 / 360.0, fill_mode="constant", fill_value=1.0, seed=RANDOM_SEED
)
translation_layer = tf.keras.layers.RandomTranslation(
    height_factor=0.05, width_factor=0.05, fill_mode="constant", fill_value=1.0, seed=RANDOM_SEED
)
zoom_layer = tf.keras.layers.RandomZoom(
    height_factor=(-0.05, 0.05), width_factor=(-0.05, 0.05), fill_mode="constant", fill_value=1.0, seed=RANDOM_SEED
)


def iam_word_image_path(word_id: str) -> str:
    parts = str(word_id).split("-")
    if len(parts) < 4:
        raise ValueError(f"Malformed IAM word id: {word_id!r}")
    return str(PROJECT_ROOT / "data" / "IAM" / "words" / parts[0] / f"{parts[0]}-{parts[1]}" / f"{word_id}.png")


def augment_image(image):
    img = rotation_layer(image, training=True)
    img = translation_layer(img, training=True)
    img = zoom_layer(img, training=True)

    contrast_factor = tf.random.uniform(
        shape=[],
        minval=0.9,
        maxval=1.1,
        seed=RANDOM_SEED,
    )
    mean = tf.reduce_mean(img, axis=[0, 1], keepdims=True)
    img = (img - mean) * contrast_factor + mean

    return tf.clip_by_value(img, 0.0, 1.0)


def process_image_augmented_path(image_path: str):
    try:
        # 1. Read and decode the original image.
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=1)
        image = tf.cast(image, tf.float32) / 255.0

        # 2. Apply augmentation before resizing (on unpadded white background).
        image = augment_image(image)
        image = tf.clip_by_value(image, 0.0, 1.0)

        # 3. Calculate aspect-ratio-preserving dimensions.
        shape = tf.shape(image)
        src_h = tf.cast(shape[0], tf.float32)
        src_w = tf.cast(shape[1], tf.float32)

        scale = tf.minimum(
            tf.cast(IMAGE_HEIGHT, tf.float32) / src_h,
            tf.cast(IMAGE_WIDTH, tf.float32) / src_w,
        )

        new_h = tf.maximum(
            1,
            tf.minimum(
                IMAGE_HEIGHT,
                tf.cast(tf.floor(src_h * scale), tf.int32),
            ),
        )

        new_w = tf.maximum(
            1,
            tf.minimum(
                IMAGE_WIDTH,
                tf.cast(tf.floor(src_w * scale), tf.int32),
            ),
        )

        # 4. Resize exactly once.
        resized = tf.image.resize(
            image,
            size=tf.stack([new_h, new_w]),
            antialias=True,
        )

        # 5. Add centered padding to reach exactly (IMAGE_HEIGHT, IMAGE_WIDTH, 1).
        actual_shape = tf.shape(resized)
        act_h = actual_shape[0]
        act_w = actual_shape[1]

        pad_h = tf.maximum(0, IMAGE_HEIGHT - act_h)
        pad_w = tf.maximum(0, IMAGE_WIDTH - act_w)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        paddings = tf.stack([
            tf.stack([pad_top, pad_bottom]),
            tf.stack([pad_left, pad_right]),
            tf.constant([0, 0], dtype=tf.int32),
        ])

        image = tf.pad(resized, paddings, constant_values=0.0)
        image = tf.clip_by_value(image, 0.0, 1.0)

        tf.debugging.assert_equal(
            tf.shape(image),
            [IMAGE_HEIGHT, IMAGE_WIDTH, 1],
            message="Image dimensions are incorrect after padding.",
        )

        image = tf.ensure_shape(
            image,
            [IMAGE_HEIGHT, IMAGE_WIDTH, 1],
        )

        # 6. Convert to CRNN layout: (width, height, channels).
        image = tf.transpose(image, perm=[1, 0, 2])

        return tf.ensure_shape(
            image,
            [IMAGE_WIDTH, IMAGE_HEIGHT, 1],
        )
    except Exception as exc:
        raise RuntimeError(
            f"Augmented preprocessing failed for {image_path!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def encode_single_sample_augmented(word_id, transcription):
    word_id_str = word_id.numpy().decode("utf-8")
    text = transcription.numpy().decode("utf-8")
    image_path = iam_word_image_path(word_id_str)

    unknown_characters = sorted(set(text) - set(IAM_VOCAB))
    if not text or len(text) > MAX_SEQ_LENGTH or unknown_characters:
        raise ValueError(
            f"Invalid augmented label for word_id={word_id_str!r}: "
            f"length={len(text)}, unknown_characters={unknown_characters!r}, "
            f"transcription={text!r}"
        )

    try:
        img = process_image_augmented_path(image_path)
        label = char_to_num(tf.constant(list(text), dtype=tf.string)).numpy().astype(np.int32).tolist()
        if any(value == BLANK_INDEX for value in label):
            raise ValueError("Encoded label contains the CTC blank/OOV index")
        label_len = len(label)
        padded_label = label + [BLANK_INDEX] * (MAX_SEQ_LENGTH - label_len)
        return img, np.asarray(padded_label + [label_len], dtype=np.int32)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to encode augmented IAM sample word_id={word_id_str!r}, "
            f"transcription={text!r}, path={image_path!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def create_dataset_augmented(word_ids, transcriptions, seed=RANDOM_SEED, batch_size=BATCH_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices((word_ids, transcriptions))
    dataset = dataset.shuffle(
        buffer_size=min(len(word_ids), 4096),
        seed=seed,
        reshuffle_each_iteration=True,
    )

    def _encode_augmented(w, t):
        img, label_ext = tf.py_function(
            encode_single_sample_augmented,
            [w, t],
            [tf.float32, tf.int32],
        )
        img.set_shape([IMAGE_WIDTH, IMAGE_HEIGHT, 1])
        label_ext.set_shape([MAX_SEQ_LENGTH + 1])
        return img, label_ext

    dataset = dataset.map(
        _encode_augmented,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)


REPORTED_SAMPLES = [
    ("c02-035-04-08", "ocean"),
    ("m06-042-04-05", "."),
    ("a06-075-01-00", "Majesty's"),
    ("k04-050-04-05", "there"),
    ("g04-088-08-02", "could"),
    ("k04-028-07-00", "eh"),
    ("m01-038-06-06", "the"),
    ("m02-090-05-03", "ago"),
    ("g07-038-01-00", "#"),
]


@pytest.mark.parametrize("word_id, transcription", REPORTED_SAMPLES)
def test_reported_samples_preprocessing_and_encoding(word_id, transcription):
    path = iam_word_image_path(word_id)
    assert os.path.isfile(path), f"Sample image file not found: {path}"

    for attempt in range(10):
        img = process_image_augmented_path(path)
        assert img.shape == (IMAGE_WIDTH, IMAGE_HEIGHT, 1)
        assert img.dtype == tf.float32
        assert bool(tf.reduce_all(tf.math.is_finite(img)).numpy())
        assert float(tf.reduce_min(img)) >= 0.0
        assert float(tf.reduce_max(img)) <= 1.0

    w_tensor = tf.constant(word_id, dtype=tf.string)
    t_tensor = tf.constant(transcription, dtype=tf.string)
    img, extended_label = encode_single_sample_augmented(w_tensor, t_tensor)

    assert img.shape == (IMAGE_WIDTH, IMAGE_HEIGHT, 1)
    assert len(extended_label) == MAX_SEQ_LENGTH + 1
    assert extended_label[-1] == len(transcription)
    for idx in extended_label[:len(transcription)]:
        assert idx != BLANK_INDEX


@pytest.mark.parametrize("h, w", [
    (1, 1), (1, 500), (500, 1), (2, 2), (5, 11),
    (135, 375), (53, 198), (3, 300), (300, 3)
])
def test_extreme_aspect_ratios_preprocessing(h, w):
    """Test synthetic extreme aspect ratio images for robust scaling and padding."""
    synthetic = tf.ones((h, w, 1), dtype=tf.float32)
    # Augment
    aug = augment_image(synthetic)
    aug = tf.clip_by_value(aug, 0.0, 1.0)

    # Scale
    shape = tf.shape(aug)
    src_h = tf.cast(shape[0], tf.float32)
    src_w = tf.cast(shape[1], tf.float32)
    scale = tf.minimum(tf.cast(IMAGE_HEIGHT, tf.float32) / src_h, tf.cast(IMAGE_WIDTH, tf.float32) / src_w)
    new_h = tf.maximum(1, tf.minimum(IMAGE_HEIGHT, tf.cast(tf.floor(src_h * scale), tf.int32)))
    new_w = tf.maximum(1, tf.minimum(IMAGE_WIDTH, tf.cast(tf.floor(src_w * scale), tf.int32)))

    resized = tf.image.resize(aug, size=tf.stack([new_h, new_w]), antialias=True)
    actual_shape = tf.shape(resized)
    act_h = actual_shape[0]
    act_w = actual_shape[1]
    pad_h = tf.maximum(0, IMAGE_HEIGHT - act_h)
    pad_w = tf.maximum(0, IMAGE_WIDTH - act_w)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    paddings = tf.stack([
        tf.stack([pad_top, pad_bottom]),
        tf.stack([pad_left, pad_right]),
        tf.constant([0, 0], dtype=tf.int32),
    ])
    padded = tf.pad(resized, paddings, constant_values=0.0)
    padded = tf.clip_by_value(padded, 0.0, 1.0)
    padded = tf.ensure_shape(padded, [IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    out = tf.transpose(padded, perm=[1, 0, 2])
    out = tf.ensure_shape(out, [IMAGE_WIDTH, IMAGE_HEIGHT, 1])

    assert out.shape == (IMAGE_WIDTH, IMAGE_HEIGHT, 1)
    assert float(tf.reduce_min(out)) >= 0.0
    assert float(tf.reduce_max(out)) <= 1.0
    assert bool(tf.reduce_all(tf.math.is_finite(out)).numpy())


def test_dataset_five_consecutive_batches_single_iterator():
    """Verify 5 consecutive batches from a single iterator across all required checks."""
    word_ids = [s[0] for s in REPORTED_SAMPLES] * 50  # 450 samples
    transcriptions = [s[1] for s in REPORTED_SAMPLES] * 50

    batch_size = 32
    ds = create_dataset_augmented(word_ids, transcriptions, batch_size=batch_size)

    iterator = iter(ds)
    for step in range(1, 6):
        images, labels = next(iterator)
        assert images.shape == (batch_size, IMAGE_WIDTH, IMAGE_HEIGHT, 1)
        assert labels.shape == (batch_size, MAX_SEQ_LENGTH + 1)
        assert images.dtype == tf.float32
        assert labels.dtype == tf.int32
        assert bool(tf.reduce_all(tf.math.is_finite(images)).numpy())
        assert float(tf.reduce_min(images)) >= 0.0
        assert float(tf.reduce_max(images)) <= 1.0
        label_lens = labels[:, -1].numpy()
        assert np.all(label_lens >= 1)
        assert np.all(label_lens <= MAX_SEQ_LENGTH)
        # Check that active characters do not use BLANK_INDEX
        for i in range(batch_size):
            text_len = label_lens[i]
            for token in labels[i, :text_len].numpy():
                assert token != BLANK_INDEX

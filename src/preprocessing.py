"""Training and inference preprocessing for the EMNIST and IAM models."""
import io
import os
from typing import Any, Union

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

try:
    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical  # type: ignore
except ImportError:  # Allows lightweight preprocessing tests without TensorFlow installed.
    tf = None
    to_categorical = None

from src.config import (
    CHARNET_INPUT_SHAPE,
    CRNN_IMG_HEIGHT,
    CRNN_IMG_WIDTH,
    MAX_IMAGE_DIMENSION,
    MAX_IMAGE_PIXELS,
    MAX_UPLOAD_SIZE_BYTES,
)

ImageLike = Union[np.ndarray, Image.Image]


def _validate_image_dimensions(
    width: int,
    height: int,
    max_dimension: int = MAX_IMAGE_DIMENSION,
    max_pixels: int = MAX_IMAGE_PIXELS,
) -> None:
    """Validate image width, height, and total pixel count before full-array processing."""
    if width <= 0 or height <= 0:
        raise ValueError("Input image is empty.")
    if width > max_dimension or height > max_dimension:
        raise ValueError(
            f"Image dimensions ({width}x{height}) exceed maximum allowed dimension ({max_dimension}px)."
        )
    if width * height > max_pixels:
        raise ValueError(
            f"Image pixel count ({width * height}) exceeds maximum allowed limit ({max_pixels} pixels)."
        )


def load_and_validate_uploaded_image(
    uploaded_file: Any,
    max_bytes: int = MAX_UPLOAD_SIZE_BYTES,
    max_dimension: int = MAX_IMAGE_DIMENSION,
    max_pixels: int = MAX_IMAGE_PIXELS,
) -> Image.Image:
    """Validate uploaded byte size, header dimensions, and pixel integrity before decoding."""
    if uploaded_file is None:
        raise ValueError("Uploaded file is empty.")

    if isinstance(uploaded_file, (bytes, bytearray)):
        size = len(uploaded_file)
        stream: Any = io.BytesIO(uploaded_file)
    elif getattr(uploaded_file, "size", None) is not None:
        size = int(uploaded_file.size)
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        stream = uploaded_file
    elif hasattr(uploaded_file, "seek") and hasattr(uploaded_file, "tell"):
        uploaded_file.seek(0, os.SEEK_END)
        size = int(uploaded_file.tell())
        uploaded_file.seek(0)
        stream = uploaded_file
    elif hasattr(uploaded_file, "read"):
        data = uploaded_file.read(max_bytes + 1)
        size = len(data)
        stream = io.BytesIO(data)
    else:
        raise TypeError(f"Unsupported upload object: {type(uploaded_file)!r}")

    if size <= 0:
        raise ValueError("Uploaded file is empty.")
    if size > max_bytes:
        raise ValueError(
            f"Uploaded file size ({size} bytes) exceeds maximum allowed limit ({max_bytes} bytes)."
        )

    try:
        img = Image.open(stream)
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid or corrupted image file: {exc}") from exc

    width, height = img.size
    _validate_image_dimensions(width, height, max_dimension=max_dimension, max_pixels=max_pixels)

    try:
        img.load()
    except (Image.DecompressionBombError, OSError, SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid or corrupted image file: {exc}") from exc

    return img


def to_grayscale_uint8(image: ImageLike) -> np.ndarray:
    """Convert grayscale, RGB, or RGBA PIL/NumPy input into a 2D uint8 image."""
    if isinstance(image, Image.Image):
        width, height = image.size
        _validate_image_dimensions(width, height)
        if image.mode in ("RGBA", "LA"):
            arr = np.asarray(image.convert("RGBA"))
        else:
            arr = np.asarray(image.convert("RGB")) if image.mode in ("RGB", "P") else np.asarray(image.convert("L"))
    elif isinstance(image, np.ndarray):
        if image.size == 0:
            raise ValueError("Input image is empty.")
        if image.ndim not in (2, 3):
            raise ValueError(f"Expected a 2D or 3D image, got shape {image.shape}.")
        _validate_image_dimensions(int(image.shape[1]), int(image.shape[0]))
        arr = image.copy()
    else:
        raise TypeError(f"Unsupported image type: {type(image)!r}")

    if arr.size == 0:
        raise ValueError("Input image is empty.")

    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        if arr.size and float(arr.min()) >= 0.0 and float(arr.max()) <= 1.0:
            arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        return arr

    if arr.shape[2] == 1:
        return arr[:, :, 0]
    if arr.shape[2] == 4:
        rgb = arr[:, :, :3].astype(np.float32)
        alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
        opaque = arr[:, :, 3] > 16
        if np.any(opaque):
            visible_gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)[opaque]
            # If visible strokes are bright, place them on black; if dark, place them on white.
            matte_value = 0.0 if float(np.median(visible_gray)) >= 127.0 else 255.0
        else:
            matte_value = 0.0
        matte = np.full_like(rgb, matte_value, dtype=np.float32)
        composited = rgb * alpha + matte * (1.0 - alpha)
        return cv2.cvtColor(np.clip(composited, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    if arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    raise ValueError(f"Unsupported channel dimension in image shape {arr.shape}.")


def ensure_white_foreground(gray: np.ndarray) -> np.ndarray:
    """Normalize polarity to bright handwriting on a dark background."""
    if gray is None or gray.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    gray = np.clip(gray, 0, 255).astype(np.uint8, copy=False)
    border = np.concatenate((gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]))
    background_level = float(np.median(border)) if border.size else float(np.median(gray))
    return (255 - gray) if background_level >= 127.5 else gray.copy()


def is_blank_image(white_foreground: np.ndarray, min_active_pixels: int = 5, threshold: int = 30) -> bool:
    """Detect empty images after polarity normalization."""
    if white_foreground is None or white_foreground.size == 0:
        return True
    return int(np.count_nonzero(white_foreground > threshold)) < int(min_active_pixels)


def preprocess_images(images: np.ndarray) -> np.ndarray:
    """Preprocess raw EMNIST images exactly as in the original training pipeline."""
    images = np.asarray(images)
    if images.ndim == 2 and images.shape[1] == 784:
        images = images.reshape(-1, 28, 28)
    elif images.ndim == 3 and images.shape[1:] == (28, 28):
        pass
    elif images.ndim == 4 and images.shape[1:] == (28, 28, 1):
        images = images[..., 0]
    else:
        raise ValueError(f"Expected EMNIST images shaped (N, 28, 28), (N, 784), or (N, 28, 28, 1), got {images.shape}.")

    # EMNIST's IDX orientation is corrected during model training.
    images = np.transpose(images, (0, 2, 1)).astype(np.float32) / 255.0
    return images[..., np.newaxis].astype(np.float32, copy=False)


def preprocess_labels(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer labels to one-hot encoding."""
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError(f"Labels must be in the range [0, {num_classes - 1}].")
    if to_categorical is not None:
        return to_categorical(labels, num_classes=num_classes).astype(np.float32)
    result = np.zeros((labels.size, num_classes), dtype=np.float32)
    if labels.size:
        result[np.arange(labels.size), labels] = 1.0
    return result


def create_tf_dataset(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
    """Build a tf.data pipeline. TensorFlow is required to execute this function."""
    if tf is None:
        raise RuntimeError("TensorFlow is required to create a tf.data.Dataset. Install project requirements first.")
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(10000, max(1, len(X))))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def preprocess_for_charnet(image: ImageLike, apply_emnist_transpose: bool = False):
    """Prepare one user-provided character as (1, 28, 28, 1), float32 in [0, 1].

    User images are already upright, so no EMNIST training transpose is applied by
    default. The optional flag exists for diagnostics with raw EMNIST-oriented data.
    """
    gray = to_grayscale_uint8(image)
    white_fg = ensure_white_foreground(gray)
    if is_blank_image(white_fg, min_active_pixels=5, threshold=30):
        return np.zeros((1, *CHARNET_INPUT_SHAPE), dtype=np.float32), True

    # Suppress very faint background noise without binarizing normal stroke edges.
    foreground = np.where(white_fg > 25, white_fg, 0).astype(np.uint8)
    coords = cv2.findNonZero(foreground)
    if coords is None:
        return np.zeros((1, *CHARNET_INPUT_SHAPE), dtype=np.float32), True
    x, y, w, h = cv2.boundingRect(coords)
    crop = foreground[y:y + h, x:x + w]
    if crop.size == 0:
        return np.zeros((1, *CHARNET_INPUT_SHAPE), dtype=np.float32), True

    inner_size = 20
    scale = inner_size / float(max(w, h))
    new_w = max(1, min(inner_size, int(round(w * scale))))
    new_h = max(1, min(inner_size, int(round(h * scale))))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(crop, (new_w, new_h), interpolation=interpolation)

    canvas = np.zeros((28, 28), dtype=np.float32)
    offset_y = (28 - new_h) // 2
    offset_x = (28 - new_w) // 2
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized.astype(np.float32)
    if apply_emnist_transpose:
        canvas = canvas.T
    tensor = np.clip(canvas / 255.0, 0.0, 1.0).astype(np.float32)[np.newaxis, ..., np.newaxis]
    return tensor, False


def preprocess_for_crnn(image: ImageLike, invert_to_dark_ink_on_white: bool = True):
    """Prepare a word image for the IAM CRNN as (1, 128, 32, 1).

    Training uses grayscale values scaled to [0, 1], resize-with-pad to a 32x128
    image, and then transposes it to (128, 32, 1). This function preserves that
    tensor convention. The word is centered both vertically and horizontally,
    matching the original TensorFlow resize_with_pad training contract.
    """
    gray = to_grayscale_uint8(image)
    white_fg = ensure_white_foreground(gray)
    if is_blank_image(white_fg, min_active_pixels=12, threshold=30):
        return np.zeros((1, CRNN_IMG_WIDTH, CRNN_IMG_HEIGHT, 1), dtype=np.float32), True

    mask = (white_fg > 30).astype(np.uint8) * 255
    coords = cv2.findNonZero(mask)
    if coords is None:
        return np.zeros((1, CRNN_IMG_WIDTH, CRNN_IMG_HEIGHT, 1), dtype=np.float32), True

    x, y, w, h = cv2.boundingRect(coords)
    # If the word already occupies most of the crop (>70%), preserve the full image context.
    if w >= 0.70 * gray.shape[1] and h >= 0.70 * gray.shape[0]:
        crop_white_fg = white_fg
    else:
        margin_x = max(6, int(round(w * 0.10)))
        margin_y = max(6, int(round(h * 0.10)))
        x0 = max(0, x - margin_x)
        y0 = max(0, y - margin_y)
        x1 = min(gray.shape[1], x + w + margin_x)
        y1 = min(gray.shape[0], y + h + margin_y)
        crop_white_fg = white_fg[y0:y1, x0:x1]

    if crop_white_fg.size == 0:
        return np.zeros((1, CRNN_IMG_WIDTH, CRNN_IMG_HEIGHT, 1), dtype=np.float32), True

    crop = (255 - crop_white_fg) if invert_to_dark_ink_on_white else crop_white_fg
    crop = crop.astype(np.float32) / 255.0

    if tf is not None:
        crop_tensor = tf.expand_dims(tf.constant(crop, dtype=tf.float32), -1)
        padded = tf.image.resize_with_pad(crop_tensor, target_height=CRNN_IMG_HEIGHT, target_width=CRNN_IMG_WIDTH)
        transposed = tf.transpose(padded, perm=[1, 0, 2])
        tensor = tf.expand_dims(transposed, 0).numpy()
    else:
        crop_h, crop_w = crop.shape
        scale = min(CRNN_IMG_WIDTH / float(crop_w), CRNN_IMG_HEIGHT / float(crop_h))
        new_w = max(1, min(CRNN_IMG_WIDTH, int(round(crop_w * scale))))
        new_h = max(1, min(CRNN_IMG_HEIGHT, int(round(crop_h * scale))))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(crop, (new_w, new_h), interpolation=interpolation)

        canvas = np.zeros((CRNN_IMG_HEIGHT, CRNN_IMG_WIDTH), dtype=np.float32)
        offset_y = (CRNN_IMG_HEIGHT - new_h) // 2
        offset_x = (CRNN_IMG_WIDTH - new_w) // 2
        canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = np.clip(resized, 0.0, 1.0)
        tensor = np.transpose(canvas, (1, 0))[np.newaxis, ..., np.newaxis].astype(np.float32)

    return tensor, False

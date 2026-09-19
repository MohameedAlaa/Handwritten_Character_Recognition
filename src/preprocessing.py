import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical # type: ignore

def preprocess_images(images: np.ndarray) -> np.ndarray:
    """
    Preprocess EMNIST images.
    - Transpose images (EMNIST images are rotated 90 degrees and flipped by default)
    - Normalize to [0, 1]
    - Add channel dimension
    """
    # EMNIST needs to be transposed to be upright
    images = images.reshape(-1, 28, 28)
    images = np.transpose(images, (0, 2, 1))

    # Normalize pixel values
    images = images.astype('float32') / 255.0

    # Add channel dimension
    images = np.expand_dims(images, axis=-1)

    return images

def preprocess_labels(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert integer labels to one-hot encoding.
    """
    return to_categorical(labels, num_classes=num_classes)

def create_tf_dataset(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
    """
    Create a tf.data.Dataset for efficient training.
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

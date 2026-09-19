import os
import gzip
import struct
import numpy as np
from src.config import RAW_DATA_DIR

def read_idx_images(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Invalid magic number {magic} for images"
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        return image_data.reshape(num_images, rows, cols)

def read_idx_labels(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num_items = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Invalid magic number {magic} for labels"
        label_data = np.frombuffer(f.read(), dtype=np.uint8)
        return label_data

def read_mapping(filename):
    mapping = {}
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                class_id = int(parts[0])
                ascii_code = int(parts[1])
                mapping[class_id] = chr(ascii_code)
    return mapping

def load_emnist_data(split='balanced'):
    """
    Load EMNIST data directly from local gzip IDX files.
    """
    print("Loading EMNIST balanced dataset from local IDX files...")

    train_images_path = os.path.join(RAW_DATA_DIR, "EMNIST", "emnist-balanced-train-images-idx3-ubyte.gz")
    train_labels_path = os.path.join(RAW_DATA_DIR, "EMNIST", "emnist-balanced-train-labels-idx1-ubyte.gz")
    test_images_path = os.path.join(RAW_DATA_DIR, "EMNIST", "emnist-balanced-test-images-idx3-ubyte.gz")
    test_labels_path = os.path.join(RAW_DATA_DIR, "EMNIST", "emnist-balanced-test-labels-idx1-ubyte.gz")

    X_train = read_idx_images(train_images_path)
    y_train = read_idx_labels(train_labels_path)

    X_test = read_idx_images(test_images_path)
    y_test = read_idx_labels(test_labels_path)

    print(f"Loaded {X_train.shape[0]} training samples and {X_test.shape[0]} test samples.")
    return (X_train, y_train), (X_test, y_test)

def get_class_mapping():
    """
    Returns the character mapping for EMNIST Balanced from local file.
    """
    mapping_path = os.path.join(RAW_DATA_DIR, "EMNIST", "emnist-balanced-mapping.txt")
    return read_mapping(mapping_path)

import os

# Project root directory. Keep string paths for backward compatibility with
# existing scripts and notebooks that use os.path.join().
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Model and result directories
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
CHARNET_MODEL_PATH = os.path.join(MODELS_DIR, "charnet_best.keras")
BEST_MODEL_PATH = CHARNET_MODEL_PATH  # Backward-compatible name used by older code.
IAM_CRNN_MODEL_PATH = os.path.join(MODELS_DIR, "iam_crnn_best.keras")
EVALUATION_REPORT_PATH = os.path.join(RESULTS_DIR, "evaluation_report.json")
IAM_EVALUATION_REPORT_PATH = os.path.join(RESULTS_DIR, "iam_crnn_evaluation_report.json")

# Ensure writable local artifact directories exist.
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# EMNIST Balanced / CharNet configuration
SPLIT_NAME = "balanced"
NUM_CLASSES = 47
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_CHANNELS = 1
BATCH_SIZE = 128
EPOCHS = 30
INITIAL_LR = 0.001
PATIENCE = 5
CHARNET_INPUT_SHAPE = (28, 28, 1)

# Keep the IAM vocabulary in precisely the order used by notebooks/02_iam_crnn.ipynb.
# StringLookup uses index 0 for its OOV/CTC blank token, so real symbols start at 1.
IAM_VOCAB = [
    " ", "!", '"', "#", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "?",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
]
CTC_BLANK_INDEX = 0
IAM_CHAR_TO_NUM = {character: index + 1 for index, character in enumerate(IAM_VOCAB)}
IAM_NUM_TO_CHAR = {index + 1: character for index, character in enumerate(IAM_VOCAB)}
IAM_NUM_CLASSES = len(IAM_VOCAB) + 1  # 79 symbols + CTC blank/OOV class
CRNN_IMG_WIDTH = 128
CRNN_IMG_HEIGHT = 32
CRNN_INPUT_SHAPE = (CRNN_IMG_WIDTH, CRNN_IMG_HEIGHT, 1)

# Uploaded image resource bounds
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_IMAGE_DIMENSION = 4096  # Maximum width or height in pixels
MAX_IMAGE_PIXELS = 12_000_000  # Maximum total decoded pixels (12 MP)

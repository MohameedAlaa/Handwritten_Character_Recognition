import os

# Project root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Model directories
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Dataset Config
SPLIT_NAME = 'balanced' # EMNIST Balanced (47 classes)
NUM_CLASSES = 47
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_CHANNELS = 1

# Training Config
BATCH_SIZE = 128
EPOCHS = 30
INITIAL_LR = 0.001
PATIENCE = 5

# Model path
BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'charnet_best.keras')

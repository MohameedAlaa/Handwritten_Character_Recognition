# Handwritten Character & Word Recognition

## Project Overview
This project is an end-to-end Machine Learning pipeline and web application for recognizing handwritten characters and words. The core ML objective is to accurately classify individual handwritten characters using a Convolutional Neural Network (CNN) trained on the EMNIST Balanced dataset. The application is deployed via Streamlit, providing an interactive and user-friendly interface to test the model.

## Key Features

### Single Character Recognition
- Draw a single character on a dedicated canvas or upload an existing grayscale image.
- Real-time character prediction using the custom EMNIST-trained model.
- Displays the top 3 predicted classes with confidence percentages.

### Write a Word
- Write a complete handwritten word on a large, interactive canvas.
- Automatic, robust character segmentation that isolates characters from left to right.
- Per-character classification using the existing character model.
- Full word reconstruction based on individual character predictions.

*Note: The current application performs word-level recognition through character segmentation and classification. It does not perform full-page OCR or arbitrary sentence-level transcription.*

## System Architecture

```text
User Input (Draw / Upload)
        ↓
   Streamlit UI
        ↓
Image / Canvas Preprocessing
        ↓
Character Segmentation (for "Write a Word")
        ↓
EMNIST Character Classifier
        ↓
 Character Predictions
        ↓
Word Reconstruction (for "Write a Word")
        ↓
   Recognized Word
```

## How the Application Works

### Single Character Pipeline
1. **Canvas / Input:** User draws or uploads an image of a single character.
2. **Preprocessing:** The image is converted to grayscale, resized to 28x28, and normalized.
3. **EMNIST Classifier:** The preprocessed image is passed directly to the trained CNN.
4. **Character Prediction:** The application displays the predicted character and confidence scores.

### Write a Word Pipeline
1. **Full handwritten word on canvas:** User writes a word on a wide canvas.
2. **Preprocessing:** The canvas image is converted to a binary image.
3. **Character Segmentation:** The system identifies contours, groups strokes belonging to the same character, and splits closely touching characters.
4. **Individual character crops:** Bounding boxes are extracted and ordered from left to right.
5. **EMNIST Classifier:** Each character crop is preprocessed (padded, resized to 28x28) and passed to the model as a batch.
6. **Left-to-right reconstruction:** Individual character predictions are collected.
7. **Final Word:** The application displays the reconstructed word and a visual segmentation preview.

## Machine Learning Approach
- **Dataset:** EMNIST Balanced dataset (47 classes: digits, uppercase, and lowercase letters).
- **Model:** `CharNet`, a custom lightweight VGG-style CNN.
- **Input:** 28x28 grayscale images.
- **Output:** Probability distribution over 47 character classes.
- **Training:** The model is trained to minimize categorical cross-entropy loss, utilizing Batch Normalization, Dropout, and Global Average Pooling to prevent overfitting.

## Character Segmentation
The segmentation strategy isolates individual characters from a continuously written word:
- **Foreground extraction:** The canvas drawing is thresholded into a binary image.
- **Contour analysis:** `cv2.findContours` is used to detect connected components.
- **Component grouping:** Disconnected strokes that belong to a single character (e.g., the dot of an 'i', or broken strokes) are conservatively grouped using bounding-box overlap heuristics.
- **Touching-character splitting:** Suspiciously wide components (like touching characters) are analyzed via a smoothed vertical projection profile. The algorithm finds topographic valleys flanked by peaks and splits the merged characters precisely.
- **Robustness:** Handles multi-stroke characters (e.g., 'A', 'H', 'O') safely without over-splitting, utilizing transition counts and aspect ratio validation.
- **Ordering:** Final bounding boxes are sorted left-to-right.

## Write a Word Pipeline Example

Example of how the application reconstructs a word:
**APPLE**
→ `[A] | [P] | [P] | [L] | [E]` (Character crops isolated via segmentation)
→ Character Classifier (Predicts each crop individually)
→ **APPLE** (Reconstructed word)

*Other validated application behaviors include correctly segmenting words like `A`, `AI`, and `HELLO`.*

## Project Structure
- `app.py`: The Streamlit web application.
- `src/`: Core Python modules for ML logic (`model.py`, `data_loader.py`, `preprocessing.py`, `inference.py`, etc.).
- `notebooks/`: Jupyter notebooks used for data exploration and model training (e.g., `01_pipeline.ipynb`).
- `models/`: Directory storing the saved, compiled `.keras` models.
- `tests/`: Pytest unit tests validating the data pipelines and model architecture.

*Note: The repository also contains an earlier experiment (`02_iam_crnn.ipynb` and `models/iam_crnn_best.keras`) for word-level CRNN/CTC recognition using the IAM dataset. This is a separate historical experiment and does not power the current Streamlit "Write a Word" application.*

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) Run tests to verify the environment:
```bash
pytest tests/
```

3. Start the Streamlit application:
```bash
streamlit run app.py
```

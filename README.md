# Handwritten Character & Word Recognition

## Project Overview
This project is an end-to-end Machine Learning pipeline and web application for recognizing handwritten characters and words. The core ML objective is to accurately classify individual handwritten characters using a Convolutional Neural Network (CNN) trained on the EMNIST Balanced dataset. The application is deployed via Streamlit, providing an interactive and user-friendly interface to test the model.

## Key Features

### Single Character Recognition
- Draw a single character on a dedicated canvas or upload an existing grayscale image.
- Real-time character prediction using the custom EMNIST-trained model.
- Displays the top 3 predicted classes with confidence percentages.

### Handwritten Word Recognition
- Write a word on the canvas or upload a single-word image.
- The IAM CRNN checkpoint is used as the primary sequence-level word recognizer with greedy CTC decoding.
- A separate segmented CharNet path remains available as a diagnostic fallback.
- The fallback predicts each detected crop independently, so it can fail on cursive or touching handwriting.

*Note: The project supports single characters and single words. It does not claim full-page OCR or arbitrary sentence-level transcription. The CRNN is wired into the application, but real-world word accuracy still requires evaluation on labeled IAM data and user-written examples.*

## System Architecture

```text
User Input (Draw / Upload)
        ↓
   Streamlit UI
        ↓
Input Image / Canvas
        ↓
Model-specific Preprocessing
        ├── Single Character → CharNet (EMNIST) → Character Prediction
        └── Handwritten Word → IAM CRNN → CTC Decoding → Recognized Word
                                  ↘ Optional segmented CharNet diagnostic fallback
```

## How the Application Works

### Single Character Pipeline
1. **Canvas / Input:** User draws or uploads an image of a single character.
2. **Preprocessing:** The image is converted to grayscale, resized to 28x28, and normalized.
3. **EMNIST Classifier:** The preprocessed image is passed directly to the trained CNN.
4. **Character Prediction:** The application displays the predicted character and confidence scores.

### Handwritten Word Pipeline
1. **Input:** Draw a word on the canvas or upload a single-word image.
2. **Model-specific preprocessing:** Normalize foreground polarity, crop outer margins, preserve the word aspect ratio, and build the CRNN tensor.
3. **Sequence inference:** The IAM CRNN emits 80-class logits over 32 time steps.
4. **CTC decoding:** Collapse repeated adjacent tokens, remove blank tokens, and map remaining indices to characters.
5. **Optional diagnostic fallback:** The connected-component segmentation path predicts isolated crops with CharNet. This is useful for debugging segmentation but can fail on cursive or touching handwriting.

## Machine Learning Approach
- **Dataset:** EMNIST Balanced dataset (47 classes: digits, uppercase, and lowercase letters).
- **Model:** `CharNet`, a custom lightweight VGG-style CNN.
- **Input:** 28x28 grayscale images.
- **Output:** Probability distribution over 47 character classes.
- **Training:** The model is trained to minimize categorical cross-entropy loss, utilizing Batch Normalization, Dropout, and Global Average Pooling to prevent overfitting.

## Character Segmentation

Character segmentation remains available through `src/segmentation.py` for the optional diagnostic fallback. It uses connected components, notebook ruling-line removal, conservative component grouping, and left-to-right ordering. Segmentation remains heuristic and does not guarantee correct recognition of connected cursive handwriting.

## Word Recognition Example

The primary path is:

`Word image → CRNN sequence model → greedy CTC decoding → predicted word`

The diagnostic fallback is:

`Word image → segmentation → CharNet per crop → left-to-right reconstruction`

The fallback may be useful for diagnosing segmentation results, but its per-character predictions are not equivalent to sequence-level OCR.

## Project Structure
- `app.py`: The Streamlit web application.
- `src/`: Core Python modules for ML logic (`model.py`, `data_loader.py`, `preprocessing.py`, `inference.py`, etc.).
- `notebooks/`: Jupyter notebooks used for data exploration and model training (e.g., `01_pipeline.ipynb`).
- `models/`: Directory storing the saved, compiled `.keras` models.
- `tests/`: Pytest unit tests validating the data pipelines and model architecture.

*Note: `models/iam_crnn_best.keras` is the primary sequence model for the Streamlit word-recognition mode. The saved checkpoint expects an image tensor shaped `(batch, 128, 32, 1)` and emits 80 classes per time step. The raw EMNIST and IAM datasets are excluded from version control via `.gitignore` due to size, so running benchmark evaluations locally requires the datasets under `data/raw/EMNIST/` and `data/IAM/`.*

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run dataset-independent regression tests:
```bash
pytest tests/test_repair_regressions.py -v
```

3. Run the full test suite. Dataset-loading tests require the EMNIST IDX files, and model tests require TensorFlow plus the included checkpoints:
```bash
pytest tests/ -v
```

4. Start the Streamlit application:
```bash
streamlit run app.py
```

5. Evaluation and Training Commands:

To train CharNet and evaluate its saved best checkpoint on the held-out EMNIST test split:
```bash
python run_pipeline.py
```

To evaluate the saved CharNet checkpoint on held-out EMNIST test data without retraining:
```bash
python run_pipeline.py --eval-only
```

To evaluate the saved IAM CRNN sequence model on the writer-independent IAM evaluation split:
```bash
python run_pipeline.py --eval-crnn --skip-charnet --crnn-samples 1000
```

To evaluate both models in a single run:
```bash
python run_pipeline.py --eval-only --eval-crnn
```

## Dataset Status and Benchmarks

- **EMNIST Balanced (`data/raw/EMNIST/`)**: Contains full official IDX files (112,800 train, 18,800 held-out test). The pipeline uses a 90/10 stratified train/val split from the train IDX while preserving the official test split for uncompromised evaluation.
  - **CharNet Benchmark**: 90.32% Test Accuracy, 0.2748 Test Loss on 18,800 held-out test characters (saved in `results/evaluation_report.json`).
- **IAM Handwriting Database (`data/IAM/`)**: Contains full word dataset (`words.txt`, `form.txt`, and 115,320 word images across 76 directory partitions; 96,454 valid non-empty `ok` word crops and 2 upstream zero-byte files skipped automatically). The evaluation procedure uses a deterministic writer-independent partition (`seed=42`, 67 test writers with zero writer/form/image overlap across splits); however, the original training provenance of `models/iam_crnn_best.keras` has not been verified from primary training logs, so the checkpoint cannot yet be proven to have never seen the current test writers during training.
  - **IAM CRNN Benchmark**: 72.30% Exact Word Accuracy, 11.04% Corpus CER, 27.77% Corpus WER, ~3.3 ms/sample batched latency on 1,000 test-split words (saved in `results/iam_crnn_evaluation_report.json`), and 73.55%–73.61% Exact Word Accuracy across all 9,508 test-split words (saved in `results/iam_crnn_preprocessing_comparison.json`).
- **Real-World Validation (`data/real_world_validation/`)**: Folder exists on disk. Currently contains 0 files. Real-world validation remains unverified until labeled sample images are provided.

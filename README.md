# CodeAlpha Handwritten Character Recognition

Professional ML pipeline and deployment for recognizing handwritten characters using the EMNIST Balanced dataset.

## Architecture
The project utilizes `CharNet`, a custom lightweight VGG-style CNN designed for accurate predictions while preventing overfitting through Batch Normalization, Dropout, and Global Average Pooling.

## Project Structure
- `src/`: Core reusable components (data loading, preprocessing, model architecture, training loop, evaluation).
- `notebooks/01_pipeline.ipynb`: The core ML experimentation artifact containing the full executed pipeline.
- `app.py`: Streamlit application for deployment, supporting both image upload and interactive drawing canvas.
- `tests/`: Pytest unit tests for validating architecture and data pipelines.

## Setup Instructions
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Notebook to train the model:
   ```bash
   jupyter notebook notebooks/01_pipeline.ipynb
   ```
   ```bash
   streamlit run app.py
   ```

## Task 3: Word-Level Recognition (IAM CRNN)
The project also includes a complete pipeline for word-level handwriting recognition using the IAM dataset.
- **Architecture:** Convolutional Recurrent Neural Network (CRNN) with Connectionist Temporal Classification (CTC) loss.
- **Dataset:** IAM Word Database, filtered with a writer-aware train/validation/test split.
- **Constraints:**
  - CTC blank index = 0.
  - Vocabulary = 79 real characters + 1 blank.
  - Image validation preflight to discard unreadable/empty images automatically.
- **Pipeline Artifact:** The pipeline and metrics are recorded in `notebooks/02_iam_crnn.ipynb`.

### Final CRNN Test Metrics
- **Word Accuracy:** 63.51%
- **CER:** 16.21%
- **WER:** 36.49%

*Note: The IAM CRNN is designed for single word-level recognition. It is not currently architected for arbitrary multi-word sentence recognition.*

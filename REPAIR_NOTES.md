# Repair Notes

## Changes in this archive

- Split character and word image preprocessing into separate functions that honor the corresponding model input layouts.
- Preserved the original EMNIST training preprocessing contract and legacy configuration names.
- Connected Streamlit's single-character mode to CharNet and word mode to the supplied IAM CRNN checkpoint.
- Added greedy CTC decoding consistent with the checkpoint's 80-class output, with index 0 reserved for the blank/OOV token.
- Retained connected-component segmentation plus CharNet as an explicitly labeled diagnostic fallback rather than presenting it as equivalent to sequence-level OCR.
- Removed the duplicate segmentation implementation from `app.py`; the fallback now uses `src/segmentation.py`.
- Added a saved-checkpoint held-out EMNIST evaluation stage and JSON report generation.
- Added dataset-independent regression tests for preprocessing, polarity, input tensor shapes, class/vocabulary mappings, CTC decoding, blank inputs, and word metrics.
- Updated dependency requirements to prefer a current TensorFlow/Keras 3-compatible release.

## Verification status in the build environment

The original archive's file structure and both `.keras` archives were inspected. Their stored Keras metadata reports version `3.15.1`; the CRNN is a Functional model with input shape `(None, 128, 32, 1)` and output shape `(None, 32, 80)`, and its final `dense_output` layer uses linear activation.

Verification performed while preparing this archive:

- `python -m compileall -q app.py run_pipeline.py src tests`: completed successfully.
- `PYTHONPATH=. pytest -q tests/test_repair_regressions.py tests/test_segmentation.py tests/test_inference.py tests/test_data.py -k 'not data_loading'`: 26 passed, 1 dataset-dependent test deselected. The unused TensorFlow import was removed from `tests/test_inference.py` so its mock-based tests can run without loading a real model.
- The full test suite could not be collected because TensorFlow is not installed in the build environment. The model, Streamlit startup, training, and real handwriting accuracy therefore remain unverified here.

The raw EMNIST and IAM datasets are not bundled. No new model-quality metrics are claimed.

## Local checks

```bash
pip install -r requirements.txt
python -m compileall -q app.py run_pipeline.py src tests
pytest tests/test_repair_regressions.py -v
pytest tests/test_segmentation.py -v
pytest tests/ -v
streamlit run app.py
```

`tests/test_data.py` requires the EMNIST IDX files under `data/raw/EMNIST/`. The word-level CRNN needs labeled IAM data for a real CER/WER benchmark.

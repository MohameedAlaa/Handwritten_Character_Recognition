"""Evaluation utilities for the character classifier and word recognizer."""
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src.config import (
    DATA_DIR,
    EVALUATION_REPORT_PATH,
    IAM_CRNN_MODEL_PATH,
    IAM_EVALUATION_REPORT_PATH,
)


def _label_ids(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    return np.argmax(labels, axis=-1).astype(int) if labels.ndim > 1 else labels.reshape(-1).astype(int)


def evaluate_model(model, X_test, y_test, class_mapping, save_path=EVALUATION_REPORT_PATH):
    """Evaluate a held-out CharNet test set and optionally save a JSON report.

    The loss is computed directly from predicted probabilities so evaluation also
    works when a model checkpoint is loaded with `compile=False`.
    The return value remains `(y_true, y_pred)` for backward compatibility.
    """
    y_true = _label_ids(y_test)
    probabilities = np.asarray(model.predict(X_test, verbose=0), dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] != y_true.size:
        raise ValueError(f"Incompatible predictions {probabilities.shape} for {y_true.size} labels.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model returned non-finite values during evaluation.")

    y_pred = np.argmax(probabilities, axis=1).astype(int)
    eps = np.finfo(np.float64).eps
    row_ids = np.arange(y_true.size)
    if y_true.size and (np.min(y_true) < 0 or np.max(y_true) >= probabilities.shape[1]):
        raise ValueError("Test labels contain an index outside the model output dimension.")
    test_loss = float(-np.mean(np.log(np.clip(probabilities[row_ids, y_true], eps, 1.0)))) if y_true.size else 0.0
    test_accuracy = float(np.mean(y_true == y_pred)) if y_true.size else 0.0

    labels = list(range(probabilities.shape[1]))
    target_names = [str(class_mapping.get(index, f"class_{index}")) for index in labels]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    confused_pairs = []
    for true_idx in labels:
        for predicted_idx in labels:
            if true_idx != predicted_idx and matrix[true_idx, predicted_idx] > 0:
                confused_pairs.append({
                    "true_character": target_names[true_idx],
                    "predicted_character": target_names[predicted_idx],
                    "count": int(matrix[true_idx, predicted_idx]),
                })
    confused_pairs.sort(key=lambda item: item["count"], reverse=True)

    result = {
        "split": "held_out_test",
        "num_samples": int(y_true.size),
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "top_confused_pairs": confused_pairs[:20],
        "note": "Measured on the supplied X_test/y_test arrays; verify the loader uses the official held-out split.",
    }
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)

    print(f"Held-out Test Loss: {test_loss:.4f}")
    print(f"Held-out Test Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, labels=labels, target_names=target_names, zero_division=0))
    if save_path:
        print(f"Saved evaluation report: {save_path}")
    return y_true, y_pred


def plot_confusion_matrix(y_true, y_pred, class_mapping, figsize=(20, 20)):
    """Plot a confusion matrix."""
    labels = list(range(len(class_mapping)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    names = [class_mapping[index] for index in labels]
    plt.figure(figsize=figsize)
    sns.heatmap(matrix, annot=False, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    """Plot training and validation accuracy/loss curves."""
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs = range(1, len(acc) + 1)
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, "b-", label="Training Accuracy")
    plt.plot(epochs, val_acc, "r-", label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, "b-", label="Training Loss")
    plt.plot(epochs, val_loss, "r-", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


def _edit_distance(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    """Levenshtein edit distance with O(min(n, m)) memory."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            current.append(min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + (ref_item != hyp_item),
            ))
        previous = current
    return previous[-1]


def compute_cer_and_wer(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """Compute exact word accuracy, corpus-level CER, and corpus-level WER."""
    if len(references) != len(hypotheses):
        raise ValueError("References and hypotheses must have the same length.")
    if not references:
        return {"exact_word_accuracy": 0.0, "cer": 0.0, "wer": 0.0}

    exact = 0
    character_edits = character_count = word_edits = word_count = 0
    for reference, hypothesis in zip(references, hypotheses):
        ref = reference.strip()
        hyp = hypothesis.strip()
        exact += int(ref == hyp)
        character_edits += _edit_distance(list(ref), list(hyp))
        character_count += len(ref)
        ref_words, hyp_words = ref.split(), hyp.split()
        word_edits += _edit_distance(ref_words, hyp_words)
        word_count += len(ref_words)

    return {
        "exact_word_accuracy": float(exact / len(references)),
        "cer": float(character_edits / max(1, character_count)),
        "wer": float(word_edits / max(1, word_count)),
    }


def evaluate_crnn_model(
    model=None,
    model_path: str = IAM_CRNN_MODEL_PATH,
    data_dir: Optional[str] = None,
    max_samples: int = 1000,
    save_path: Optional[str] = IAM_EVALUATION_REPORT_PATH,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Evaluate the IAM CRNN model on held-out writers from the IAM dataset.

    Parameters
    ----------
    model : tf.keras.Model, optional
        Loaded CRNN model. If None, loaded from `model_path`.
    model_path : str
        Path to the saved `.keras` CRNN model checkpoint.
    data_dir : str, optional
        Path to the data directory containing `IAM/`. Defaults to `config.DATA_DIR`.
    max_samples : int
        Maximum number of held-out test samples to evaluate.
    save_path : str, optional
        Path to save the JSON evaluation report.
    random_state : int
        Seed for deterministic writer split partition.
    """
    import tensorflow as tf
    import pandas as pd
    from src.inference import decode_iam_prediction, load_iam_model

    if max_samples <= 0:
        raise ValueError("max_samples must be a positive integer.")

    target_data_dir = data_dir or DATA_DIR
    iam_dir = os.path.join(target_data_dir, "IAM")
    forms_file = os.path.join(iam_dir, "form.txt")
    words_file = os.path.join(iam_dir, "words.txt")
    words_dir = os.path.join(iam_dir, "words")

    if not os.path.isfile(words_file) or not os.path.isfile(forms_file):
        raise FileNotFoundError(f"IAM metadata files not found in {iam_dir}.")

    if model is None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"CRNN checkpoint not found at {model_path}.")
        print(f"Loading CRNN checkpoint: {model_path}")
        model = load_iam_model(model_path)

    form_to_writer: Dict[str, str] = {}
    with open(forms_file, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                form_to_writer[parts[0]] = parts[1]

    dataset = []
    with open(words_file, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split(" ", 8)
            if len(parts) >= 9 and parts[1] == "ok":
                word_id = parts[0]
                w_parts = word_id.split("-")
                if len(w_parts) < 2:
                    continue
                img_path = os.path.join(words_dir, w_parts[0], f"{w_parts[0]}-{w_parts[1]}", f"{word_id}.png")
                if not os.path.exists(img_path) or os.path.getsize(img_path) == 0:
                    continue
                form_id = "-".join(w_parts[:2])
                dataset.append({
                    "word_id": word_id,
                    "writer_id": form_to_writer.get(form_id, "UNKNOWN"),
                    "gt": parts[8],
                })

    if not dataset:
        raise RuntimeError("No valid IAM images could be loaded for evaluation.")

    df = pd.DataFrame(dataset)
    rng = np.random.RandomState(random_state)
    writers = np.array(df["writer_id"].unique(), dtype=str)
    rng.shuffle(writers)

    train_size = int(len(writers) * 0.8)
    val_size = int(len(writers) * 0.1)
    test_writers = set(writers[train_size + val_size:])
    test_df = df[df["writer_id"].isin(test_writers)].copy()

    sample_count = min(max_samples, len(test_df))
    sample_test = test_df.sample(sample_count, random_state=random_state)

    print(f"Evaluating {sample_count} held-out samples across {len(test_writers)} writer-independent test writers...")

    batch_tensors = []
    valid_gts = []
    valid_wids = []
    for _, row in sample_test.iterrows():
        w_id = row["word_id"]
        parts = w_id.split("-")
        img_path = os.path.join(words_dir, parts[0], f"{parts[0]}-{parts[1]}", f"{w_id}.png")
        if not os.path.exists(img_path) or os.path.getsize(img_path) == 0:
            continue
        raw = tf.io.read_file(img_path)
        img = tf.image.decode_png(raw, channels=1)
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize_with_pad(img, target_height=32, target_width=128)
        img = tf.transpose(img, perm=[1, 0, 2])
        batch_tensors.append(img.numpy())
        valid_gts.append(row["gt"])
        valid_wids.append(w_id)

    if not batch_tensors:
        raise RuntimeError("No valid IAM images could be loaded for evaluation.")

    batch = np.array(batch_tensors)
    t0 = time.time()
    preds = model.predict(batch, batch_size=64, verbose=0)
    elapsed = time.time() - t0
    decoded_texts, confs = decode_iam_prediction(preds)
    metrics = compute_cer_and_wer(valid_gts, decoded_texts)

    by_length: Dict[int, Dict[str, int]] = {}
    errors_list = []
    for wid, gt, pred, conf in zip(valid_wids, valid_gts, decoded_texts, confs):
        length = len(gt)
        if length not in by_length:
            by_length[length] = {"total": 0, "correct": 0}
        by_length[length]["total"] += 1
        if gt == pred:
            by_length[length]["correct"] += 1
        else:
            errors_list.append({
                "word_id": wid,
                "ground_truth": gt,
                "prediction": pred,
                "confidence": float(conf),
                "length": length,
            })

    length_report = {
        str(l): {
            "total": by_length[l]["total"],
            "correct": by_length[l]["correct"],
            "accuracy": float(by_length[l]["correct"] / by_length[l]["total"]),
        }
        for l in sorted(by_length.keys())
    }

    result = {
        "split": "held_out_writers_test",
        "num_test_writers": len(test_writers),
        "total_test_samples_in_split": len(test_df),
        "evaluated_samples": len(valid_gts),
        "elapsed_seconds": float(elapsed),
        "mean_latency_ms_per_sample": float((elapsed / len(valid_gts)) * 1000.0),
        "metrics": metrics,
        "performance_by_word_length": length_report,
        "sample_errors": errors_list[:50],
        "note": (
            "Evaluation split is writer-independent (67 held-out test writers under seed 42); "
            "however, the original training provenance of models/iam_crnn_best.keras is not verified "
            "from primary training logs, so the checkpoint cannot be proven to have never seen "
            "these test writers during training."
        ),
    }

    if save_path:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print(f"Saved IAM CRNN evaluation report: {save_path}")

    print(f"Exact Word Accuracy: {metrics['exact_word_accuracy'] * 100:.2f}%")
    print(f"Corpus CER: {metrics['cer']:.4f}")
    print(f"Corpus WER: {metrics['wer']:.4f}")
    print(f"Mean Latency: {(elapsed / len(valid_gts)) * 1000.0:.2f} ms/sample")
    return result

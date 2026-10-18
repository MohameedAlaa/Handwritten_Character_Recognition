"""Train CharNet and evaluate the saved best checkpoint on held-out EMNIST data."""
import argparse
import json
import os

import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.config import (
    BEST_MODEL_PATH,
    BATCH_SIZE,
    EPOCHS,
    EVALUATION_REPORT_PATH,
    IAM_CRNN_MODEL_PATH,
    IAM_EVALUATION_REPORT_PATH,
    INITIAL_LR,
    NUM_CLASSES,
    PATIENCE,
    ROOT_DIR,
)
from src.data_loader import get_class_mapping, load_emnist_data
from src.evaluate import evaluate_crnn_model, evaluate_model
from src.model import build_charnet
from src.preprocessing import create_tf_dataset, preprocess_images, preprocess_labels
from src.train import compile_and_train


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the EMNIST CharNet and IAM CRNN pipelines.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and evaluate the saved checkpoint.")
    parser.add_argument("--split", default="balanced", help="Currently only the local EMNIST balanced split is available.")
    parser.add_argument("--eval-crnn", action="store_true", help="Evaluate the saved IAM CRNN word recognition checkpoint.")
    parser.add_argument("--crnn-samples", type=int, default=1000, help="Number of held-out IAM test samples to evaluate (default: 1000).")
    parser.add_argument("--skip-charnet", action="store_true", help="Skip CharNet training and evaluation.")
    args = parser.parse_args()
    if args.split.lower() != "balanced":
        parser.error("Only the EMNIST balanced split is supported by the configured local IDX file paths.")

    if not args.skip_charnet:
        print("=== EMNIST Character Recognition Pipeline ===")
        print("Loading local EMNIST data. The test split will remain held out from training and validation.")
        (x_train_full, y_train_full), (x_test, y_test) = load_emnist_data(split=args.split)

        if not args.eval_only:
            x_train, x_val, y_train, y_val = train_test_split(
                x_train_full,
                y_train_full,
                test_size=0.10,
                random_state=42,
                stratify=y_train_full,
            )
            print(f"Training samples: {len(x_train)}")
            print(f"Validation samples: {len(x_val)}")
            print(f"Held-out test samples: {len(x_test)}")

            x_train_p = preprocess_images(x_train)
            x_val_p = preprocess_images(x_val)
            y_train_p = preprocess_labels(y_train, NUM_CLASSES)
            y_val_p = preprocess_labels(y_val, NUM_CLASSES)
            train_dataset = create_tf_dataset(x_train_p, y_train_p, BATCH_SIZE, shuffle=True)
            val_dataset = create_tf_dataset(x_val_p, y_val_p, BATCH_SIZE, shuffle=False)

            model = build_charnet(num_classes=NUM_CLASSES)
            model.summary()
            history = compile_and_train(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                epochs=EPOCHS,
                initial_lr=INITIAL_LR,
                patience=PATIENCE,
            )
            os.makedirs(os.path.join(ROOT_DIR, "results"), exist_ok=True)
            history_path = os.path.join(ROOT_DIR, "results", "training_history.json")
            history_dict = {key: [float(value) for value in values] for key, values in history.history.items()}
            with open(history_path, "w", encoding="utf-8") as handle:
                json.dump(history_dict, handle, indent=2)
            print(f"Saved training history: {history_path}")

        if not os.path.isfile(BEST_MODEL_PATH):
            raise FileNotFoundError(
                f"CharNet checkpoint not found at {BEST_MODEL_PATH}. Run training first or place the checkpoint there."
            )

        print(f"Loading checkpoint for final evaluation: {BEST_MODEL_PATH}")
        best_model = tf.keras.models.load_model(BEST_MODEL_PATH, compile=False)
        x_test_p = preprocess_images(x_test)
        y_test_p = preprocess_labels(y_test, NUM_CLASSES)
        class_mapping = get_class_mapping()
        evaluate_model(best_model, x_test_p, y_test_p, class_mapping, save_path=EVALUATION_REPORT_PATH)

    if args.eval_crnn:
        print("\n=== IAM CRNN Word Recognition Pipeline ===")
        evaluate_crnn_model(
            model_path=IAM_CRNN_MODEL_PATH,
            max_samples=args.crnn_samples,
            save_path=IAM_EVALUATION_REPORT_PATH,
        )


if __name__ == "__main__":
    main()

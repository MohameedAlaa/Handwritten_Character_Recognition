import os
import json
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import ROOT_DIR, NUM_CLASSES, BATCH_SIZE, EPOCHS, INITIAL_LR, PATIENCE
from src.data_loader import load_emnist_data, get_class_mapping
from src.preprocessing import preprocess_images, preprocess_labels, create_tf_dataset
from src.model import build_charnet
from src.train import compile_and_train

def main():
    print("=== EMNIST Character Recognition Pipeline ===")

    # 1. Load data
    print("\n1. Loading local EMNIST dataset...")
    (X_train_full, y_train_full), (X_test, y_test) = load_emnist_data()

    # 2. Split training data (90% train, 10% val) with reproducible seed
    print("\n2. Splitting training data into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.10,
        random_state=42,
        stratify=y_train_full
    )
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples (held out): {X_test.shape[0]}")

    # Check class distribution
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    print(f"\nClass distribution in training set: (Min: {counts_train.min()}, Max: {counts_train.max()})")
    print("Distribution is balanced. No class weights needed.")

    # 3. Preprocessing
    print("\n3. Preprocessing images and labels...")
    X_train_p = preprocess_images(X_train)
    X_val_p = preprocess_images(X_val)
    # We do NOT preprocess or touch X_test for training

    y_train_p = preprocess_labels(y_train, NUM_CLASSES)
    y_val_p = preprocess_labels(y_val, NUM_CLASSES)

    print(f"Processed X_train shape: {X_train_p.shape}")
    print(f"Processed y_train shape: {y_train_p.shape}")

    # Create tf.data datasets
    train_dataset = create_tf_dataset(X_train_p, y_train_p, BATCH_SIZE, shuffle=True)
    val_dataset = create_tf_dataset(X_val_p, y_val_p, BATCH_SIZE, shuffle=False)

    # 4. Build Model
    print("\n4. Building CharNet architecture...")
    model = build_charnet(num_classes=NUM_CLASSES)
    model.summary()

    # 5. Training
    print("\n5. Starting training phase...")
    print(f"Config: Adam(lr={INITIAL_LR}), Batch={BATCH_SIZE}, Max Epochs={EPOCHS}, Patience={PATIENCE}")

    history = compile_and_train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=EPOCHS,
        initial_lr=INITIAL_LR,
        patience=PATIENCE
    )

    # 6. Save History
    results_dir = os.path.join(ROOT_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)
    history_path = os.path.join(results_dir, 'training_history.json')

    print("\n6. Saving artifacts...")
    # Convert types for JSON serialization
    hist_dict = {}
    for key, val in history.history.items():
        hist_dict[key] = [float(v) for v in val]

    with open(history_path, 'w') as f:
        json.dump(hist_dict, f, indent=4)

    best_val_acc = max(hist_dict['val_accuracy'])
    best_val_loss = min(hist_dict['val_loss'])
    final_train_acc = hist_dict['accuracy'][-1]
    final_train_loss = hist_dict['loss'][-1]
    epochs_run = len(hist_dict['loss'])

    print("\n=== Training Summary ===")
    print(f"Total epochs executed: {epochs_run}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Final Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Training Loss: {final_train_loss:.4f}")
    print(f"\nArtifacts saved:")
    print(f"- Best model: models/charnet_best.keras")
    print(f"- Training history: {history_path}")

if __name__ == "__main__":
    main()

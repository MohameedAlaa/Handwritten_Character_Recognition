import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # type: ignore
from src.config import BEST_MODEL_PATH

def compile_and_train(
    model,
    train_dataset,
    val_dataset,
    epochs,
    initial_lr=0.001,
    patience=5,
    checkpoint_path=BEST_MODEL_PATH,
):
    """
    Compiles and trains the model, saving the best validation checkpoint to
    `checkpoint_path` and restoring those best checkpoint weights into `model`.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    if checkpoint_path:
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)

    # Callbacks
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)

    return history

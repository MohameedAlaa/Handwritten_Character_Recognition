import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore

def build_charnet(input_shape=(28, 28, 1), num_classes=47):
    """
    Builds the CharNet architecture for handwritten character recognition.
    A custom lightweight VGG-style CNN.
    """
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Classifier Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="CharNet")
    return model


def build_crnn(input_shape=(128, 32, 1), num_classes=80):
    """
    Builds the IAM word-level CRNN architecture matching models/iam_crnn_best.keras.

    Input tensor shape: (width=128, height=32, channels=1) in width-major layout.
    Output tensor shape: (time_steps=32, num_classes=80) linear logits with
    CTC blank token reserved at class index 0.
    """
    inputs = layers.Input(shape=input_shape, dtype="float32", name="image")

    # Convolutional feature extractor (width 128 -> 64 -> 32, height 32 -> 16 -> 8)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv2d")(inputs)
    x = layers.MaxPooling2D((2, 2), name="max_pooling2d")(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2d_1")(x)
    x = layers.MaxPooling2D((2, 2), name="max_pooling2d_1")(x)

    x = layers.BatchNormalization(name="batch_normalization")(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv2d_2")(x)

    # Map-to-sequence projection: (32, 8, 128) -> (32, 1024) -> (32, 64)
    time_steps = input_shape[0] // 4
    feature_dim = (input_shape[1] // 4) * 128
    x = layers.Reshape(target_shape=(time_steps, feature_dim), name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense")(x)

    # Bidirectional LSTM sequence modeling
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True), name="bidirectional"
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True), name="bidirectional_1"
    )(x)

    # Linear logits for CTC loss / decoding (index 0 = CTC blank)
    outputs = layers.Dense(num_classes, activation="linear", name="dense_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="crnn_model")
    return model

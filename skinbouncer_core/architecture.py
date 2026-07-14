import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)

from .augmentation import build_augmentation
from .preprocessing import IMG_SIZE, N_CHANNELS


def build_cnn(img_size=IMG_SIZE, n_channels=N_CHANNELS, augmentation=None, name="minecraft_skin_cnn"):
    """The binary good-vs-bad detector CNN. One of these gets trained per bad category."""
    if augmentation is None:
        augmentation = build_augmentation()

    def conv_block(model, filters):
        model.add(Conv2D(filters, (3, 3), padding="same", activation="relu",
                          kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
        model.add(BatchNormalization())
        model.add(Conv2D(filters, (3, 3), padding="same", activation="relu",
                          kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.4))

    model = Sequential(name=name)
    model.add(Input(shape=(img_size[0], img_size[1], n_channels)))
    model.add(augmentation)

    conv_block(model, 16)  # low-level patterns
    conv_block(model, 32)  # mid-level patterns
    conv_block(model, 64)  # high-level patterns

    model.add(GlobalAveragePooling2D())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.6))
    model.add(Dense(1, activation="sigmoid"))
    return model

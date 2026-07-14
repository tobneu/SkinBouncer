import tensorflow as tf

from .augmentation import RandomColorShift

# Explicit custom_objects mapping so a saved detector loads correctly regardless of how
# RandomColorShift happened to be registered at save time.
CUSTOM_OBJECTS = {"Custom>RandomColorShift": RandomColorShift}


def save_model(model, path):
    model.save(str(path))


def load_model(path):
    return tf.keras.models.load_model(str(path), custom_objects=CUSTOM_OBJECTS)

import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="Custom", name="RandomColorShift")
class RandomColorShift(layers.Layer):
    """Perturbs RGB channels only and leaves alpha untouched, since alpha is a structural
    mask (second skin layer) rather than a color channel."""

    def __init__(self, shift_val=0.1, **kwargs):
        super().__init__(**kwargs)
        self.shift_val = shift_val

    def call(self, x, training=None):
        if not training:
            return x  # passthrough at inference
        rgb = x[..., :3]
        alpha = x[..., 3:4]
        noise = tf.random.uniform(tf.shape(rgb), -self.shift_val, self.shift_val)
        return tf.concat([tf.clip_by_value(rgb + noise, 0.0, 1.0), alpha], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"shift_val": self.shift_val})
        return config


def build_augmentation(shift_val=0.1, gaussian_stddev=0.05):
    return tf.keras.Sequential([
        RandomColorShift(shift_val),
        layers.GaussianNoise(gaussian_stddev),  # already respects the training flag
    ])

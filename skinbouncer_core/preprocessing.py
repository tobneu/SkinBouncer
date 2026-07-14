import numpy as np
import tensorflow as tf

# 64x64 RGBA — alpha is structural (it marks the second skin layer), not cosmetic,
# so it is always kept as a 4th channel rather than cropped to RGB.
IMG_SIZE = (64, 64)
N_CHANNELS = 4


def load_skin(path, img_size=IMG_SIZE):
    """Load a single skin PNG into a (1, H, W, 4) float32 tensor normalized to [0, 1]."""
    img = tf.keras.utils.load_img(str(path), color_mode="rgba", target_size=img_size)
    arr = tf.keras.utils.img_to_array(img) / 255.0
    return arr[np.newaxis, ...]


def load_images(paths, img_size=IMG_SIZE):
    """Load a batch of skin PNGs into an (N, H, W, 4) float32 array normalized to [0, 1]."""
    imgs = []
    for p in paths:
        img = tf.keras.utils.load_img(str(p), color_mode="rgba", target_size=img_size)
        imgs.append(tf.keras.utils.img_to_array(img) / 255.0)
    return np.array(imgs, dtype=np.float32)

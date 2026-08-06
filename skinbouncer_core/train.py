"""Initial detector training: given a detector project's frozen split manifest, train
the shared CNN architecture on the train partition, validate on val, and save a
checkpoint (+ threshold + metrics) into the project directory.

Ported from `04_Modeling/Modeling.ipynb`'s training pipeline, with two deliberate
deviations from a direct copy:

- The train dataset is explicitly shuffled before batching. `_stratified_assign` in
  `detector_project.py` only shuffles filenames *within* each class before splitting -
  it never interleaves good and bad with each other. `get_split_filepaths` then hands
  back good/bad as two separate lists, and `_load_split_arrays` below concatenates them
  good-then-bad. Left unshuffled at that point, batching would produce near-homogeneous
  batches under real class imbalance (most batches all "good"), badly hurting batchnorm
  statistics. The notebook never hit this because sklearn's `train_test_split` already
  interleaves classes before its own batching.
- Metrics are reported at the best epoch (by val_auc), not the last epoch trained.
  `EarlyStopping(restore_best_weights=True)` means the saved model is the best epoch's
  weights, not whatever epoch training happened to stop on after `patience` more rounds.
"""

import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from .architecture import build_cnn
from .augmentation import build_augmentation
from .detector_project import _write_json, get_split_filepaths, load_manifest
from .model_io import save_model
from .preprocessing import load_images
from .threshold import find_threshold_for_recall


def _load_split_arrays(manifest, split):
    """good -> label 0, bad -> label 1 (matches deployment's "high score = bad"
    semantics). Returns (X, y) float32 arrays."""
    filepaths = get_split_filepaths(manifest, split)
    paths = filepaths["good"] + filepaths["bad"]
    labels = [0.0] * len(filepaths["good"]) + [1.0] * len(filepaths["bad"])
    return load_images(paths), np.array(labels, dtype=np.float32)


def _compute_sample_weights(y_train):
    """Inverse-frequency weight per sample, so the loss doesn't just learn to predict
    the majority class - typically "good" outnumbers "bad" by a wide margin."""
    n_total = len(y_train)
    n_good = max(int((y_train == 0).sum()), 1)
    n_bad = max(int((y_train == 1).sum()), 1)
    class_weights = {0: n_total / (2 * n_good), 1: n_total / (2 * n_bad)}
    return np.where(y_train == 1, class_weights[1], class_weights[0])


def _metrics_at_best_epoch(history):
    """EarlyStopping(restore_best_weights=True) means the model in memory (and about to
    be saved) is the best epoch's weights, not the last one trained - so report metrics
    from that same epoch rather than history.history[...][-1]."""
    best_epoch = int(np.argmax(history.history["val_auc"]))
    train_metrics = {k: float(v[best_epoch]) for k, v in history.history.items() if not k.startswith("val_")}
    val_metrics = {k[4:]: float(v[best_epoch]) for k, v in history.history.items() if k.startswith("val_")}
    return train_metrics, val_metrics


def train_detector(project_dir, epochs=50, batch_size=32, lr=3e-4, patience=10,
                    recall_target=0.95, seed=None):
    """Train a detector on a project's frozen train/val split. Writes model.keras,
    threshold.json and metrics.json into project_dir. Returns a dict describing the
    run (checkpoint/metrics paths, history, val metrics, threshold info)."""
    project_dir = Path(project_dir)
    manifest = load_manifest(project_dir)
    if seed is None:
        seed = manifest["seed"]

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    X_train, y_train = _load_split_arrays(manifest, "train")
    X_val, y_val = _load_split_arrays(manifest, "val")

    sample_weights = _compute_sample_weights(y_train)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train, sample_weights))
    train_dataset = train_dataset.shuffle(buffer_size=max(len(y_train), 1), seed=seed).batch(batch_size)

    model = build_cnn(augmentation=build_augmentation())
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        # Single sigmoid output (good=0/bad=1) -> standard binary classification loss.
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    model_path = project_dir / "model.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=patience, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path), monitor="val_auc", mode="max", save_best_only=True, verbose=0
        ),
    ]

    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        # No-op for a tf.data.Dataset input either way (Keras only shuffles raw array
        # inputs) - set explicitly just to silence the warning Keras prints about it.
        # Actual shuffling is entirely handled by train_dataset.shuffle() above; val
        # data is untouched by this, it's never shuffled by Keras regardless.
        shuffle=False,
        verbose=1,
    )

    save_model(model, model_path)

    train_metrics, val_metrics = _metrics_at_best_epoch(history)

    y_prob_val = model.predict(X_val, verbose=0).ravel()
    threshold_error = None
    try:
        best_row = find_threshold_for_recall(y_val, y_prob_val, recall_target=recall_target)
        threshold_info = {
            "threshold": float(best_row["threshold"]),
            "recall": float(best_row["recall"]),
            "precision": float(best_row["precision"]),
            "f1": float(best_row["f1"]),
        }
    except ValueError as e:
        threshold_error = str(e)
        threshold_info = {"threshold": 0.5, "recall": None, "precision": None, "f1": None}
        print(f"WARNING: {e} Falling back to threshold=0.5.")

    threshold_path = project_dir / "threshold.json"
    _write_json(threshold_path, {"threshold": threshold_info["threshold"]})

    epochs_run = len(history.history["loss"])
    metrics_path = project_dir / "metrics.json"
    _write_json(metrics_path, {
        "epochs_run": epochs_run,
        "train": train_metrics,
        "val": val_metrics,
        "threshold_search": {**threshold_info, "recall_target": recall_target, "error": threshold_error},
    })

    return {
        "model_path": model_path,
        "threshold_path": threshold_path,
        "metrics_path": metrics_path,
        "history": history.history,
        "epochs_run": epochs_run,
        "val_metrics": val_metrics,
        "threshold_info": threshold_info,
        "threshold_error": threshold_error,
    }

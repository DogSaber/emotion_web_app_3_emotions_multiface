"""Train a manuscript-aligned five-class facial-expression CNN.

The script is intentionally experiment-oriented: it never overwrites the
deployed model.  Every full run receives its own artifact directory containing
the best/last checkpoints, metadata, configuration, and epoch history.

Examples:
    python train.py --dry-run
    python train.py --smoke-test --preprocessing basic
    python train.py --preprocessing basic --epochs 50
    python train.py --preprocessing clahe --epochs 50
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from ml_config import (
    DEFAULT_TRAIN_DIR,
    DEFAULT_VALIDATION_DIR,
    EMOTIONS,
    EXPERIMENTS_ROOT,
    IMAGE_SIZE,
    INPUT_SHAPE,
    NUM_CLASSES,
    RANDOM_SEED,
    read_model_metadata,
    utc_timestamp,
    validate_model_contract,
    validate_model_metadata,
    write_model_metadata,
)
from ml_dataset import (
    generator_folder_names,
    resolve_class_directories,
    validate_generator_class_indices,
)
from ml_preprocessing import (
    DEFAULT_TRAINING_PREPROCESSING,
    PREPROCESSING_MODES,
    preprocess_training_image,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the five-output CNN in this exact order: "
            + ", ".join(EMOTIONS)
        )
    )
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument(
        "--validation-dir",
        "--val-dir",
        dest="validation_dir",
        type=Path,
        default=DEFAULT_VALIDATION_DIR,
    )
    parser.add_argument(
        "--preprocessing",
        choices=PREPROCESSING_MODES,
        default=DEFAULT_TRAINING_PREPROCESSING,
        help=(
            "Controlled preprocessing experiment. Use 'basic' first, then "
            "'clahe'. 'legacy' exists only for compatibility comparisons."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--reduce-lr-patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Optional readable prefix; a timestamp is always appended.",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=EXPERIMENTS_ROOT,
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a candidate checkpoint with valid sidecar metadata.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate folders, class indices, preprocessing, and model shapes only.",
    )
    mode.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one in-memory train batch and one validation batch; save nothing.",
    )
    args = parser.parse_args(argv)

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be greater than 0")
    if args.early_stopping_patience < 1:
        parser.error("--early-stopping-patience must be at least 1")
    if args.reduce_lr_patience < 1:
        parser.error("--reduce-lr-patience must be at least 1")
    return args


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        # Deterministic ops are best-effort across TensorFlow/platform versions.
        pass


def build_emotion_cnn(
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
) -> tf.keras.Model:
    """Build the compact thesis CNN with a neutral five-class output name."""

    inputs = tf.keras.Input(shape=input_shape, name="face_input")
    x = inputs

    for block_index, filters in enumerate((32, 64, 128), start=1):
        x = tf.keras.layers.Conv2D(
            filters,
            3,
            padding="same",
            use_bias=False,
            name=f"block{block_index}_conv1",
        )(x)
        x = tf.keras.layers.BatchNormalization(
            name=f"block{block_index}_bn1"
        )(x)
        x = tf.keras.layers.ReLU(name=f"block{block_index}_relu1")(x)
        x = tf.keras.layers.Conv2D(
            filters,
            3,
            padding="same",
            use_bias=False,
            name=f"block{block_index}_conv2",
        )(x)
        x = tf.keras.layers.BatchNormalization(
            name=f"block{block_index}_bn2"
        )(x)
        x = tf.keras.layers.ReLU(name=f"block{block_index}_relu2")(x)
        x = tf.keras.layers.MaxPooling2D(
            2,
            name=f"block{block_index}_pool",
        )(x)
        x = tf.keras.layers.Dropout(
            0.20 if block_index == 1 else 0.25,
            name=f"block{block_index}_dropout",
        )(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = tf.keras.layers.Dense(128, use_bias=False, name="dense_features")(x)
    x = tf.keras.layers.BatchNormalization(name="dense_bn")(x)
    x = tf.keras.layers.ReLU(name="dense_relu")(x)
    x = tf.keras.layers.Dropout(0.40, name="dense_dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="emotion_output",
    )(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="emotion_cnn_5class")


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    metrics: list[tf.keras.metrics.Metric] = [
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.F1Score(average="macro", name="macro_f1"),
    ]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=metrics,
    )


def create_generators(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    train_mapping = resolve_class_directories(args.train_dir)
    validation_mapping = resolve_class_directories(args.validation_dir)

    preprocess = partial(preprocess_training_image, mode=args.preprocessing)
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess,
        rotation_range=10,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.10,
        shear_range=0.05,
        brightness_range=(0.90, 1.10),
        horizontal_flip=True,
    )
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess
    )

    train_data = train_datagen.flow_from_directory(
        str(args.train_dir),
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        batch_size=args.batch_size,
        class_mode="categorical",
        classes=generator_folder_names(train_mapping),
        shuffle=True,
        seed=args.seed,
    )
    validation_data = validation_datagen.flow_from_directory(
        str(args.validation_dir),
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        batch_size=args.batch_size,
        class_mode="categorical",
        classes=generator_folder_names(validation_mapping),
        shuffle=False,
    )

    validate_generator_class_indices(train_data.class_indices, train_mapping)
    validate_generator_class_indices(
        validation_data.class_indices,
        validation_mapping,
    )

    train_counts = {
        emotion: int(np.sum(train_data.classes == index))
        for index, emotion in enumerate(EMOTIONS)
    }
    validation_counts = {
        emotion: int(np.sum(validation_data.classes == index))
        for index, emotion in enumerate(EMOTIONS)
    }
    if any(count == 0 for count in (*train_counts.values(), *validation_counts.values())):
        raise ValueError("Every class must contain images in both dataset splits.")

    dataset_info = {
        "train_dir": str(Path(args.train_dir).resolve()),
        "validation_dir": str(Path(args.validation_dir).resolve()),
        "train_folder_names": {
            emotion: path.name for emotion, path in train_mapping.items()
        },
        "validation_folder_names": {
            emotion: path.name for emotion, path in validation_mapping.items()
        },
        "train_counts": train_counts,
        "validation_counts": validation_counts,
    }
    return train_data, validation_data, dataset_info


def calculate_class_weights(labels: np.ndarray) -> dict[int, float]:
    expected_indices = np.arange(NUM_CLASSES)
    present = np.unique(labels)
    if not np.array_equal(present, expected_indices):
        raise ValueError(
            f"Training labels contain indices {present.tolist()}, expected "
            f"{expected_indices.tolist()}."
        )
    values = compute_class_weight(
        class_weight="balanced",
        classes=expected_indices,
        y=labels,
    )
    return {index: float(value) for index, value in enumerate(values)}


def load_or_build_model(args: argparse.Namespace) -> tf.keras.Model:
    if args.resume is None:
        model = build_emotion_cnn()
    else:
        if not args.resume.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        metadata = read_model_metadata(args.resume, required=True)
        assert metadata is not None
        validate_model_metadata(
            metadata,
            expected_preprocessing=args.preprocessing,
        )
        model = tf.keras.models.load_model(args.resume, compile=False)

    validate_model_contract(model)
    if model.output_names != ["emotion_output"]:
        raise ValueError(
            f"Candidate model output must be named 'emotion_output', got "
            f"{model.output_names}."
        )
    compile_model(model, args.learning_rate)
    return model


def check_one_batch(
    model: tf.keras.Model,
    train_data: Any,
    validation_data: Any,
    *,
    train: bool,
) -> None:
    train_x, train_y = train_data[0]
    validation_x, validation_y = validation_data[0]
    for split, x_value, y_value in (
        ("train", train_x, train_y),
        ("validation", validation_x, validation_y),
    ):
        if x_value.shape[1:] != INPUT_SHAPE:
            raise ValueError(f"{split} batch has wrong image shape: {x_value.shape}")
        if y_value.shape[1:] != (NUM_CLASSES,):
            raise ValueError(f"{split} batch has wrong label shape: {y_value.shape}")
        if not np.all(np.isfinite(x_value)):
            raise ValueError(f"{split} preprocessing produced non-finite values.")
        if float(x_value.min()) < 0.0 or float(x_value.max()) > 1.0:
            raise ValueError(
                f"{split} preprocessing is outside [0, 1]: "
                f"{x_value.min()}..{x_value.max()}"
            )

    prediction = model.predict_on_batch(validation_x[:1])
    if prediction.shape != (1, NUM_CLASSES):
        raise ValueError(f"Model smoke prediction has wrong shape: {prediction.shape}")
    if not np.isclose(float(prediction[0].sum()), 1.0, atol=1e-5):
        raise ValueError("Softmax smoke prediction does not sum to one.")

    if train:
        train_metrics = model.train_on_batch(train_x, train_y, return_dict=True)
        validation_metrics = model.test_on_batch(
            validation_x,
            validation_y,
            return_dict=True,
        )
        print("Smoke train metrics:", train_metrics)
        print("Smoke validation metrics:", validation_metrics)


def make_run_paths(args: argparse.Namespace) -> dict[str, Path]:
    prefix = args.experiment_name or f"cnn_{args.preprocessing}"
    safe_prefix = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in prefix
    ).strip("_")
    if not safe_prefix:
        safe_prefix = f"cnn_{args.preprocessing}"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")
    run_dir = Path(args.artifacts_root) / f"{safe_prefix}_{timestamp}"
    if run_dir.exists():
        raise FileExistsError(f"Experiment directory already exists: {run_dir}")
    return {
        "run_dir": run_dir,
        "best_model": run_dir / "emotion_recognition_model_5class_best.h5",
        "last_model": run_dir / "emotion_recognition_model_5class_last.h5",
        "history_csv": run_dir / "training_history.csv",
        "configuration": run_dir / "configuration.json",
    }


class BestCheckpointMetadata(tf.keras.callbacks.Callback):
    """Keep checkpoint metadata synchronized with validation macro F1."""

    def __init__(self, model_path: Path, base_metadata: dict[str, Any]) -> None:
        super().__init__()
        self.model_path = model_path
        self.metadata = dict(base_metadata)
        self.best = -np.inf

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        self.metadata.update(
            {
                "status": "training",
                "best_epoch": None,
                "best_val_macro_f1": None,
            }
        )
        write_model_metadata(self.model_path, self.metadata)

    def on_epoch_end(
        self,
        epoch: int,
        logs: dict[str, Any] | None = None,
    ) -> None:
        logs = logs or {}
        value = logs.get("val_macro_f1")
        if value is None or not np.isfinite(float(value)):
            return
        if float(value) > self.best:
            self.best = float(value)
            self.metadata.update(
                {
                    "best_epoch": epoch + 1,
                    "best_val_macro_f1": self.best,
                    "status": "training",
                }
            )
            write_model_metadata(self.model_path, self.metadata)

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        self.metadata["status"] = "training_complete"
        write_model_metadata(self.model_path, self.metadata)


def base_metadata(
    args: argparse.Namespace,
    paths: dict[str, Path],
    dataset_info: dict[str, Any],
    class_weights: dict[int, float],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at_utc": utc_timestamp(),
        "model_role": "candidate",
        "architecture": "three_double_convolution_blocks_global_average_pooling",
        "preprocessing": args.preprocessing,
        "random_seed": args.seed,
        "tensorflow_version": tf.__version__,
        "experiment_directory": str(paths["run_dir"].resolve()),
        "training": {
            "batch_size": args.batch_size,
            "epochs_requested": args.epochs,
            "learning_rate": args.learning_rate,
            "class_weights": {
                EMOTIONS[index]: weight
                for index, weight in class_weights.items()
            },
            "augmentation": {
                "rotation_degrees": 10,
                "width_shift": 0.08,
                "height_shift": 0.08,
                "zoom": 0.10,
                "shear": 0.05,
                "brightness": [0.90, 1.10],
                "horizontal_flip": True,
            },
            "checkpoint_monitor": "val_macro_f1",
        },
        "dataset": dataset_info,
    }


def print_configuration(
    args: argparse.Namespace,
    dataset_info: dict[str, Any],
    class_weights: dict[int, float],
) -> None:
    print("\nFive-class model contract")
    print("-------------------------")
    print("Output order:", list(EMOTIONS))
    print("Input shape:", INPUT_SHAPE)
    print("Preprocessing:", args.preprocessing)
    print("Training counts:", dataset_info["train_counts"])
    print("Validation counts:", dataset_info["validation_counts"])
    print(
        "Class weights:",
        {
            EMOTIONS[index]: round(value, 4)
            for index, value in class_weights.items()
        },
    )

    for split_key in ("train_folder_names", "validation_folder_names"):
        mismatches = {
            canonical: actual
            for canonical, actual in dataset_info[split_key].items()
            if canonical != actual
        }
        if mismatches:
            print(
                f"WARNING: non-canonical folder casing in {split_key}: "
                f"{mismatches}. Rename before moving the project to Linux."
            )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    set_reproducible_seed(args.seed)

    train_data, validation_data, dataset_info = create_generators(args)
    class_weights = calculate_class_weights(train_data.classes)
    model = load_or_build_model(args)
    print_configuration(args, dataset_info, class_weights)
    model.summary()

    check_one_batch(
        model,
        train_data,
        validation_data,
        train=args.smoke_test,
    )
    if args.dry_run or args.smoke_test:
        label = "Smoke test" if args.smoke_test else "Dry run"
        print(f"\n{label} passed. No model or history files were written.")
        return 0

    paths = make_run_paths(args)
    paths["run_dir"].mkdir(parents=True, exist_ok=False)
    metadata = base_metadata(args, paths, dataset_info, class_weights)
    paths["configuration"].write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.ModelCheckpoint(
            str(paths["best_model"]),
            monitor="val_macro_f1",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        BestCheckpointMetadata(paths["best_model"], metadata),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_macro_f1",
            mode="max",
            patience=args.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=args.reduce_lr_patience,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(paths["history_csv"]), append=False),
    ]

    print(f"\nStarting experiment: {paths['run_dir']}")
    print("The deployed model will not be overwritten.")
    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=2,
    )

    # Keep the last restored in-memory weights separate from ModelCheckpoint's
    # validation-best checkpoint. Never overwrite the best file at train end.
    model.save(paths["last_model"])
    last_metadata = dict(metadata)
    last_metadata.update(
        {
            "status": "training_complete",
            "epochs_completed": len(history.epoch),
            "checkpoint_kind": "last_restored_weights",
        }
    )
    write_model_metadata(paths["last_model"], last_metadata)

    best_values = history.history.get("val_macro_f1", [])
    print("\nTraining complete.")
    print("Best checkpoint:", paths["best_model"])
    if best_values:
        print(f"Best validation macro F1: {max(best_values):.4f}")
    print("Evaluate this candidate before changing the deployed model.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\nTraining interrupted. A validation-best checkpoint is still usable "
            "if it was saved; inspect its metadata before evaluation.",
            file=sys.stderr,
        )
        raise SystemExit(130)

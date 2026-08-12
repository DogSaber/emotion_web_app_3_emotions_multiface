"""Strict five-class model evaluation with thesis-ready artifacts.

Unlike the previous script, this evaluator never drops missing classes and
never slices model outputs.  A mismatch stops evaluation because otherwise the
reported accuracy/confusion matrix would be invalid.

Legacy model example (no sidecar metadata):
    python evaluate_model.py ^
        --model emotion_recognition_model_5class.h5 ^
        --dataset dataset/validation ^
        --preprocessing legacy ^
        --allow-missing-metadata

Candidate model example (uses its sidecar automatically):
    python evaluate_model.py --model artifacts/experiments/<run>/...best.h5
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from ml_config import (
    DEFAULT_VALIDATION_DIR,
    DEPLOYED_MODEL_PATH,
    EMOTIONS,
    EVALUATIONS_ROOT,
    IMAGE_SIZE,
    INPUT_SHAPE,
    NUM_CLASSES,
    read_model_metadata,
    validate_model_contract,
    validate_model_metadata,
)
from ml_dataset import (
    generator_folder_names,
    resolve_class_directories,
    validate_generator_class_indices,
)
from ml_preprocessing import PREPROCESSING_MODES, preprocess_training_image


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all five emotions in this exact order: "
            + ", ".join(EMOTIONS)
        )
    )
    parser.add_argument("--model", type=Path, default=DEPLOYED_MODEL_PATH)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_VALIDATION_DIR,
        help=(
            "Evaluation split. Use validation while experimenting and an "
            "untouched dataset/test only for the final thesis result."
        ),
    )
    parser.add_argument(
        "--preprocessing",
        choices=PREPROCESSING_MODES,
        default=None,
        help=(
            "Normally read from model metadata. Required with "
            "--allow-missing-metadata for a legacy model."
        ),
    )
    parser.add_argument(
        "--allow-missing-metadata",
        action="store_true",
        help=(
            "Permit a legacy model without a metadata sidecar. You must also "
            "state --preprocessing; canonical class order is then an explicit "
            "assumption and will be recorded."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Debug only: evaluate the first N deterministically ordered images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Artifact directory; otherwise a timestamped directory is created.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate metadata/data/model and one inference batch; write nothing.",
    )
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be at least 1")
    return args


def determine_preprocessing(
    args: argparse.Namespace,
    metadata: dict[str, Any] | None,
) -> tuple[str, bool]:
    if metadata is not None:
        metadata_mode = metadata.get("preprocessing")
        requested_mode = args.preprocessing or metadata_mode
        validate_model_metadata(
            metadata,
            expected_preprocessing=requested_mode,
        )
        assert isinstance(requested_mode, str)
        return requested_mode, False

    if not args.allow_missing_metadata:
        raise FileNotFoundError(
            f"No metadata sidecar exists for {args.model}. For a known legacy "
            "model, rerun with --allow-missing-metadata and explicitly provide "
            "--preprocessing legacy (or the mode actually used in training)."
        )
    if args.preprocessing is None:
        raise ValueError(
            "--preprocessing is required when --allow-missing-metadata is used."
        )
    return args.preprocessing, True


def create_evaluation_generator(
    args: argparse.Namespace,
    preprocessing_mode: str,
) -> tuple[Any, dict[str, Any]]:
    mapping = resolve_class_directories(args.dataset)
    preprocess = partial(
        preprocess_training_image,
        mode=preprocessing_mode,
    )
    generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess
    ).flow_from_directory(
        str(args.dataset),
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        batch_size=args.batch_size,
        class_mode="categorical",
        classes=generator_folder_names(mapping),
        shuffle=False,
    )
    validate_generator_class_indices(generator.class_indices, mapping)

    counts = {
        emotion: int(np.sum(generator.classes == index))
        for index, emotion in enumerate(EMOTIONS)
    }
    if any(value == 0 for value in counts.values()):
        raise ValueError(
            f"Every emotion must be present in evaluation data; counts={counts}"
        )

    info = {
        "directory": str(Path(args.dataset).resolve()),
        "split_name": Path(args.dataset).name,
        "folder_names": {
            emotion: path.name for emotion, path in mapping.items()
        },
        "class_counts": counts,
        "total_samples": int(generator.samples),
    }
    return generator, info


def validate_one_batch(model: tf.keras.Model, generator: Any) -> None:
    batch_x, batch_y = generator[0]
    if batch_x.shape[1:] != INPUT_SHAPE:
        raise ValueError(f"Evaluation image shape mismatch: {batch_x.shape}")
    if batch_y.shape[1:] != (NUM_CLASSES,):
        raise ValueError(f"Evaluation label shape mismatch: {batch_y.shape}")
    if not np.all(np.isfinite(batch_x)):
        raise ValueError("Evaluation preprocessing produced non-finite pixels.")
    if float(batch_x.min()) < 0.0 or float(batch_x.max()) > 1.0:
        raise ValueError(
            f"Evaluation pixels are outside [0, 1]: "
            f"{batch_x.min()}..{batch_x.max()}"
        )

    probabilities = model.predict_on_batch(batch_x[:1])
    if probabilities.shape != (1, NUM_CLASSES):
        raise ValueError(
            f"Expected one prediction with {NUM_CLASSES} outputs, got "
            f"{probabilities.shape}."
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model produced NaN or infinite probabilities.")
    if not np.isclose(float(probabilities[0].sum()), 1.0, atol=1e-4):
        raise ValueError(
            "Model outputs do not sum to one; expected a five-class softmax."
        )


def choose_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")
        output_dir = (
            EVALUATIONS_ROOT
            / f"{args.model.stem}_{Path(args.dataset).name}_{timestamp}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty evaluation directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *EMOTIONS])
        for emotion, row in zip(EMOTIONS, matrix):
            writer.writerow([emotion, *row.tolist()])


def write_confusion_matrix_svg(
    path: Path,
    matrix: np.ndarray,
) -> None:
    """Create a dependency-free, thesis-usable confusion matrix graphic."""

    normalized = matrix.astype(np.float64)
    row_totals = normalized.sum(axis=1, keepdims=True)
    normalized = np.divide(
        normalized,
        row_totals,
        out=np.zeros_like(normalized),
        where=row_totals != 0,
    )

    cell = 105
    left = 160
    top = 110
    width = left + cell * NUM_CLASSES + 35
    height = top + cell * NUM_CLASSES + 80
    elements = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" '
            'font-family="Arial" font-size="20" font-weight="bold">'
            "Confusion Matrix</text>"
        ),
        (
            f'<text x="{left + cell * NUM_CLASSES / 2:.1f}" y="55" '
            'text-anchor="middle" font-family="Arial" font-size="14">'
            "Predicted emotion</text>"
        ),
        (
            f'<text x="20" y="{top + cell * NUM_CLASSES / 2:.1f}" '
            'text-anchor="middle" font-family="Arial" font-size="14" '
            f'transform="rotate(-90 20 {top + cell * NUM_CLASSES / 2:.1f})">'
            "Actual emotion</text>"
        ),
    ]

    for index, emotion in enumerate(EMOTIONS):
        x = left + index * cell + cell / 2
        y = top - 15
        elements.append(
            f'<text x="{x:.1f}" y="{y}" text-anchor="middle" '
            f'font-family="Arial" font-size="13">{html.escape(emotion)}</text>'
        )
        row_y = top + index * cell + cell / 2 + 5
        elements.append(
            f'<text x="{left - 15}" y="{row_y:.1f}" text-anchor="end" '
            f'font-family="Arial" font-size="13">{html.escape(emotion)}</text>'
        )

    for row in range(NUM_CLASSES):
        for column in range(NUM_CLASSES):
            value = float(normalized[row, column])
            blue = int(255 - 155 * value)
            fill = f"rgb({blue},{blue},255)"
            text_color = "white" if value >= 0.62 else "#111827"
            x = left + column * cell
            y = top + row * cell
            count = int(matrix[row, column])
            elements.extend(
                [
                    (
                        f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                        f'fill="{fill}" stroke="white"/>'
                    ),
                    (
                        f'<text x="{x + cell / 2:.1f}" y="{y + 45}" '
                        f'text-anchor="middle" font-family="Arial" font-size="18" '
                        f'font-weight="bold" fill="{text_color}">{count}</text>'
                    ),
                    (
                        f'<text x="{x + cell / 2:.1f}" y="{y + 69}" '
                        f'text-anchor="middle" font-family="Arial" font-size="13" '
                        f'fill="{text_color}">{value * 100:.1f}%</text>'
                    ),
                ]
            )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def write_predictions_csv(
    path: Path,
    filepaths: list[str],
    true_indices: np.ndarray,
    predicted_indices: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    probability_headers = [f"probability_{emotion}" for emotion in EMOTIONS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "filepath",
                "actual_emotion",
                "predicted_emotion",
                "confidence",
                "correct",
                *probability_headers,
            ]
        )
        for filepath, true_index, predicted_index, scores in zip(
            filepaths,
            true_indices,
            predicted_indices,
            probabilities,
        ):
            writer.writerow(
                [
                    filepath,
                    EMOTIONS[int(true_index)],
                    EMOTIONS[int(predicted_index)],
                    float(scores[int(predicted_index)]),
                    bool(true_index == predicted_index),
                    *(float(value) for value in scores),
                ]
            )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if not args.model.is_file():
        raise FileNotFoundError(f"Model file not found: {args.model}")

    metadata = read_model_metadata(args.model, required=False)
    preprocessing_mode, legacy_assumption = determine_preprocessing(args, metadata)
    generator, dataset_info = create_evaluation_generator(
        args,
        preprocessing_mode,
    )

    model = tf.keras.models.load_model(args.model, compile=False)
    validate_model_contract(model)
    validate_one_batch(model, generator)

    print("\nFive-class evaluation contract")
    print("------------------------------")
    print("Model:", args.model)
    print("Output order:", list(EMOTIONS))
    print("Preprocessing:", preprocessing_mode)
    print("Dataset counts:", dataset_info["class_counts"])
    if legacy_assumption:
        print(
            "WARNING: Legacy model has no metadata. Canonical class order and "
            "the requested preprocessing are explicit, unverified assumptions."
        )
    if dataset_info["split_name"].casefold() != "test":
        print(
            "NOTE: This is not a folder named 'test'. Use these results for "
            "model development, not as the final untouched thesis test score."
        )
    for canonical, actual in dataset_info["folder_names"].items():
        if canonical != actual:
            print(
                f"WARNING: folder {actual!r} should be renamed {canonical!r} "
                "before moving this dataset to a case-sensitive system."
            )

    if args.dry_run:
        print("\nEvaluation dry run passed. No report files were written.")
        return {"dry_run": True}

    requested_samples = args.max_samples or generator.samples
    sample_count = min(requested_samples, generator.samples)
    steps = math.ceil(sample_count / args.batch_size)
    generator.reset()
    probabilities = model.predict(generator, steps=steps, verbose=1)[:sample_count]
    true_indices = np.asarray(generator.classes[:sample_count], dtype=np.int64)
    filepaths = [str(path) for path in generator.filepaths[:sample_count]]

    if probabilities.shape != (sample_count, NUM_CLASSES):
        raise ValueError(
            f"Prediction matrix is {probabilities.shape}, expected "
            f"({sample_count}, {NUM_CLASSES})."
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Predictions contain NaN or infinite values.")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        raise ValueError("At least one prediction is not a valid softmax vector.")

    predicted_indices = np.argmax(probabilities, axis=1)
    labels = list(range(NUM_CLASSES))
    report_dict = classification_report(
        true_indices,
        predicted_indices,
        labels=labels,
        target_names=list(EMOTIONS),
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        true_indices,
        predicted_indices,
        labels=labels,
        target_names=list(EMOTIONS),
        digits=4,
        zero_division=0,
    )
    matrix = confusion_matrix(
        true_indices,
        predicted_indices,
        labels=labels,
    )
    normalized_matrix = confusion_matrix(
        true_indices,
        predicted_indices,
        labels=labels,
        normalize="true",
    )
    overall_accuracy = float(accuracy_score(true_indices, predicted_indices))
    balanced_accuracy = float(
        balanced_accuracy_score(true_indices, predicted_indices)
    )

    metrics = {
        "model": str(args.model.resolve()),
        "model_output_name": model.output_names[0],
        "metadata_present": metadata is not None,
        "legacy_contract_assumption": legacy_assumption,
        "class_order": list(EMOTIONS),
        "preprocessing": preprocessing_mode,
        "input_shape": list(INPUT_SHAPE),
        "dataset": dataset_info,
        "evaluated_samples": sample_count,
        "partial_debug_evaluation": sample_count != generator.samples,
        "is_untouched_test_candidate": (
            dataset_info["split_name"].casefold() == "test"
            and sample_count == generator.samples
        ),
        "overall_accuracy": overall_accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_precision": float(report_dict["macro avg"]["precision"]),
        "macro_recall": float(report_dict["macro avg"]["recall"]),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "weighted_precision": float(report_dict["weighted avg"]["precision"]),
        "weighted_recall": float(report_dict["weighted avg"]["recall"]),
        "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
        "per_class": {
            emotion: report_dict[emotion] for emotion in EMOTIONS
        },
        "model_metadata": metadata,
    }

    output_dir = choose_output_dir(args)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "classification_report.txt").write_text(
        report_text + "\n",
        encoding="utf-8",
    )
    write_matrix_csv(output_dir / "confusion_matrix.csv", matrix)
    write_matrix_csv(
        output_dir / "confusion_matrix_normalized.csv",
        normalized_matrix,
    )
    write_confusion_matrix_svg(output_dir / "confusion_matrix.svg", matrix)
    write_predictions_csv(
        output_dir / "predictions.csv",
        filepaths,
        true_indices,
        predicted_indices,
        probabilities,
    )

    print("\nClassification report")
    print("---------------------")
    print(report_text)
    print("Confusion matrix (rows=actual, columns=predicted)")
    print(matrix)
    print(f"\nOverall accuracy: {overall_accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print("Artifacts:", output_dir)
    if sample_count != generator.samples:
        print(
            "WARNING: --max-samples produced a partial debug evaluation. "
            "Do not cite these metrics in the thesis."
        )
    return metrics


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    evaluate(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"Evaluation stopped: {exc}", file=sys.stderr)
        raise SystemExit(2)

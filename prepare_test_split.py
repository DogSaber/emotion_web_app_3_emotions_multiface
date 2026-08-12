"""Preview or create a deduplicated 70/15/15 dataset for future experiments.

By default this command is a dry run and writes nothing.  ``--apply`` copies
files into a new output root; it never moves or deletes the current train and
validation folders.

Important: the newly created test split is untouched only for models trained
afterward using the new train/validation folders. Existing models already used
the old validation images and cannot claim this as an untouched final test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ml_config import (
    DATASET_ROOT,
    DEFAULT_TRAIN_DIR,
    DEFAULT_VALIDATION_DIR,
    EMOTIONS,
    RANDOM_SEED,
    utc_timestamp,
)
from ml_dataset import iter_image_files, resolve_class_directories


@dataclass(frozen=True)
class SourceImage:
    emotion: str
    source: Path
    sha256: str


@dataclass(frozen=True)
class SplitAssignment:
    split: str
    image: SourceImage


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan a deterministic, exact-deduplicated train/validation/test split."
        )
    )
    parser.add_argument(
        "--source-dir",
        action="append",
        type=Path,
        default=None,
        help=(
            "Source split; repeat as needed. Defaults to current train and "
            "validation."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DATASET_ROOT / "split_v2",
    )
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Copy the planned split into --output-root. Without this, dry-run only.",
    )
    args = parser.parse_args(argv)
    if args.source_dir is None:
        args.source_dir = [DEFAULT_TRAIN_DIR, DEFAULT_VALIDATION_DIR]
    for name, value in (
        ("--validation-ratio", args.validation_ratio),
        ("--test-ratio", args.test_ratio),
    ):
        if not 0 < value < 1:
            parser.error(f"{name} must be between 0 and 1")
    if args.validation_ratio + args.test_ratio >= 1:
        parser.error("validation and test ratios must add up to less than 1")
    return args


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_unique_images(
    source_dirs: Iterable[Path],
) -> tuple[dict[str, list[SourceImage]], int]:
    by_digest: dict[str, list[SourceImage]] = defaultdict(list)
    total_source_files = 0

    for source_dir in source_dirs:
        mapping = resolve_class_directories(source_dir)
        for emotion, class_dir in mapping.items():
            for path in iter_image_files(class_dir):
                total_source_files += 1
                image = SourceImage(
                    emotion=emotion,
                    source=path.resolve(),
                    sha256=file_sha256(path),
                )
                by_digest[image.sha256].append(image)

    conflicts = {
        digest: records
        for digest, records in by_digest.items()
        if len({record.emotion for record in records}) > 1
    }
    if conflicts:
        preview = []
        for digest, records in list(sorted(conflicts.items()))[:5]:
            labels = sorted({record.emotion for record in records})
            preview.append(f"{digest[:12]} -> {labels}")
        raise ValueError(
            f"Found {len(conflicts)} exact image hashes with conflicting labels. "
            "Correct these labels before splitting. Examples: "
            + "; ".join(preview)
        )

    unique_by_emotion = {emotion: [] for emotion in EMOTIONS}
    for digest, records in sorted(by_digest.items()):
        # Exact copies in the same class are represented once so they cannot
        # inflate a class or leak across splits.
        chosen = min(records, key=lambda item: str(item.source).casefold())
        unique_by_emotion[chosen.emotion].append(chosen)

    duplicate_copies = total_source_files - sum(
        len(images) for images in unique_by_emotion.values()
    )
    return unique_by_emotion, duplicate_copies


def split_counts(
    total: int,
    *,
    validation_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    if total < 3:
        raise ValueError(
            f"At least three unique images per class are required, got {total}."
        )
    validation = max(1, int(round(total * validation_ratio)))
    test = max(1, int(round(total * test_ratio)))
    train = total - validation - test
    if train < 1:
        # Ratios were validated, so this only affects very small class folders.
        train = 1
        if validation >= test and validation > 1:
            validation -= 1
        elif test > 1:
            test -= 1
    if train + validation + test != total:
        raise AssertionError("Split counts do not add up.")
    return train, validation, test


def create_plan(
    unique_by_emotion: dict[str, list[SourceImage]],
    *,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> list[SplitAssignment]:
    assignments: list[SplitAssignment] = []
    for emotion_index, emotion in enumerate(EMOTIONS):
        images = sorted(
            unique_by_emotion[emotion],
            key=lambda image: (image.sha256, str(image.source).casefold()),
        )
        rng = random.Random(seed + emotion_index)
        rng.shuffle(images)
        train_count, validation_count, _ = split_counts(
            len(images),
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
        )

        train_end = train_count
        validation_end = train_count + validation_count
        assignments.extend(
            SplitAssignment("train", image) for image in images[:train_end]
        )
        assignments.extend(
            SplitAssignment("validation", image)
            for image in images[train_end:validation_end]
        )
        assignments.extend(
            SplitAssignment("test", image)
            for image in images[validation_end:]
        )
    return assignments


def plan_counts(
    assignments: list[SplitAssignment],
) -> dict[str, dict[str, int]]:
    return {
        split: {
            emotion: sum(
                item.split == split and item.image.emotion == emotion
                for item in assignments
            )
            for emotion in EMOTIONS
        }
        for split in ("train", "validation", "test")
    }


def safe_destination_name(
    directory: Path,
    image: SourceImage,
) -> Path:
    candidate = directory / image.source.name
    if not candidate.exists():
        return candidate
    candidate = directory / f"{image.sha256[:12]}_{image.source.name}"
    if candidate.exists():
        raise FileExistsError(f"Destination collision: {candidate}")
    return candidate


def validate_output_target(output_root: Path, source_dirs: Iterable[Path]) -> Path:
    resolved = output_root.resolve()
    source_resolved = [Path(path).resolve() for path in source_dirs]
    if resolved == DATASET_ROOT.resolve():
        raise ValueError("Output root cannot replace the existing dataset root.")
    if any(resolved == source for source in source_resolved):
        raise ValueError("Output root cannot be one of the source directories.")
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(
            f"Output root is non-empty; refusing to overwrite it: {resolved}"
        )
    return resolved


def apply_plan(
    assignments: list[SplitAssignment],
    *,
    output_root: Path,
    configuration: dict[str, object],
) -> None:
    output_root = validate_output_target(
        output_root,
        [Path(value) for value in configuration["source_directories"]],
    )
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[list[str]] = []

    for assignment in assignments:
        destination_dir = (
            output_root / assignment.split / assignment.image.emotion
        )
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = safe_destination_name(
            destination_dir,
            assignment.image,
        )
        shutil.copy2(assignment.image.source, destination)
        manifest_rows.append(
            [
                assignment.split,
                assignment.image.emotion,
                str(assignment.image.source),
                str(destination.relative_to(output_root)),
                assignment.image.sha256,
            ]
        )

    with (output_root / "split_manifest.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["split", "emotion", "source", "destination", "sha256"]
        )
        writer.writerows(manifest_rows)

    (output_root / "split_configuration.json").write_text(
        json.dumps(configuration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        unique_by_emotion, duplicate_copies = collect_unique_images(
            args.source_dir
        )
        assignments = create_plan(
            unique_by_emotion,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        counts = plan_counts(assignments)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Split preparation stopped: {exc}", file=sys.stderr)
        return 2

    print("\nPROPOSED DEDUPLICATED SPLIT")
    print("===========================")
    print("Class order:", list(EMOTIONS))
    print("Exact duplicate copies excluded:", duplicate_copies)
    for split, split_values in counts.items():
        print(f"\n{split}:")
        for emotion in EMOTIONS:
            print(f"  {emotion:<8} {split_values[emotion]:>6}")
        print(f"  {'TOTAL':<8} {sum(split_values.values()):>6}")

    configuration: dict[str, object] = {
        "generated_at_utc": utc_timestamp(),
        "class_order": list(EMOTIONS),
        "source_directories": [
            str(Path(path).resolve()) for path in args.source_dir
        ],
        "output_root": str(args.output_root.resolve()),
        "seed": args.seed,
        "ratios": {
            "train": 1.0 - args.validation_ratio - args.test_ratio,
            "validation": args.validation_ratio,
            "test": args.test_ratio,
        },
        "exact_duplicate_copies_excluded": duplicate_copies,
        "counts": counts,
        "note": (
            "The test split is untouched only for future models trained solely "
            "with this new train and validation split."
        ),
    }

    if not args.apply:
        print(
            "\nDRY RUN ONLY: no folders or files were created. Review the counts, "
            "then obtain approval before rerunning with --apply."
        )
        return 0

    try:
        apply_plan(
            assignments,
            output_root=args.output_root,
            configuration=configuration,
        )
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"Could not create split: {exc}", file=sys.stderr)
        return 2

    print("\nNew split copied to:", args.output_root)
    print("Original dataset files were not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


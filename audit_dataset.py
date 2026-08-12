"""Read-only audit of the five-class image dataset.

Reports:
  * image counts per split and emotion
  * exact duplicate SHA-256 groups (including cross-split leakage)
  * conflicting duplicates assigned to different emotions
  * corrupt/unreadable images
  * width, height, channel, and file-size statistics

The utility never renames, moves, copies, or deletes dataset files.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from ml_config import (
    DEFAULT_TEST_DIR,
    DEFAULT_TRAIN_DIR,
    DEFAULT_VALIDATION_DIR,
    EMOTIONS,
    utc_timestamp,
)
from ml_dataset import iter_image_files, resolve_class_directories


@dataclass(frozen=True)
class ImageRecord:
    split: str
    emotion: str
    filepath: str
    size_bytes: int
    sha256: str | None
    width: int | None
    height: int | None
    channels: int | None
    error: str | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only integrity and duplicate audit of emotion images."
    )
    parser.add_argument(
        "--split",
        action="append",
        type=Path,
        default=None,
        help=(
            "Dataset split to inspect; repeat for multiple splits. Defaults to "
            "train, validation, and test when test exists."
        ),
    )
    parser.add_argument(
        "--no-hashes",
        action="store_true",
        help="Skip SHA-256 duplicate analysis (faster, but no leakage check).",
    )
    parser.add_argument(
        "--max-files-per-class",
        type=int,
        default=None,
        help="Debug only: inspect only the first N files in each class.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report path outside the dataset.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Parallel read/decode workers (default: up to 8).",
    )
    args = parser.parse_args(argv)
    if args.max_files_per_class is not None and args.max_files_per_class < 1:
        parser.error("--max-files-per-class must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.split is None:
        args.split = [DEFAULT_TRAIN_DIR, DEFAULT_VALIDATION_DIR]
        if DEFAULT_TEST_DIR.is_dir():
            args.split.append(DEFAULT_TEST_DIR)
    return args


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_image_dimensions(path: Path) -> tuple[int, int, int]:
    # np.fromfile + imdecode handles Unicode Windows paths more reliably than
    # cv2.imread.
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        raise ValueError("OpenCV could not decode the file")
    if image.ndim == 2:
        height, width = image.shape
        channels = 1
    elif image.ndim == 3:
        height, width, channels = image.shape
    else:
        raise ValueError(f"unsupported decoded shape {image.shape}")
    return int(width), int(height), int(channels)


def inspect_image(
    path: Path,
    *,
    split: str,
    emotion: str,
    calculate_hash: bool,
) -> ImageRecord:
    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        return ImageRecord(
            split,
            emotion,
            str(path.resolve()),
            0,
            None,
            None,
            None,
            None,
            f"Could not stat file: {exc}",
        )

    digest: str | None = None
    error: str | None = None
    try:
        digest = file_sha256(path) if calculate_hash else None
        width, height, channels = decode_image_dimensions(path)
    except (OSError, ValueError, cv2.error) as exc:
        width = height = channels = None
        error = str(exc)
    return ImageRecord(
        split,
        emotion,
        str(path.resolve()),
        int(size_bytes),
        digest,
        width,
        height,
        channels,
        error,
    )


def scan_splits(
    split_paths: Iterable[Path],
    *,
    calculate_hashes: bool,
    max_files_per_class: int | None,
    workers: int,
) -> tuple[list[ImageRecord], list[dict[str, str]]]:
    work_items: list[tuple[Path, str, str]] = []
    casing_warnings: list[dict[str, str]] = []
    seen_split_names: Counter[str] = Counter()

    for split_path in split_paths:
        split_path = Path(split_path)
        split_name = split_path.name
        seen_split_names[split_name] += 1
        if seen_split_names[split_name] > 1:
            split_name = f"{split_name}_{seen_split_names[split_name]}"

        mapping = resolve_class_directories(split_path, require_images=False)
        for emotion, class_dir in mapping.items():
            if class_dir.name != emotion:
                casing_warnings.append(
                    {
                        "split": split_name,
                        "actual": class_dir.name,
                        "expected": emotion,
                    }
                )
            paths = list(iter_image_files(class_dir))
            if max_files_per_class is not None:
                paths = paths[:max_files_per_class]
            for path in paths:
                work_items.append((path, split_name, emotion))

    def inspect_work_item(
        work_item: tuple[Path, str, str],
    ) -> ImageRecord:
        path, split_name, emotion = work_item
        return inspect_image(
            path,
            split=split_name,
            emotion=emotion,
            calculate_hash=calculate_hashes,
        )

    records: list[ImageRecord] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for completed, record in enumerate(
            executor.map(inspect_work_item, work_items),
            start=1,
        ):
            records.append(record)
            if completed % 1000 == 0:
                print(
                    f"Inspected {completed}/{len(work_items)} files...",
                    file=sys.stderr,
                )
    return records, casing_warnings


def number_summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {
        "min": min(values),
        "median": float(statistics.median(values)),
        "max": max(values),
    }


def duplicate_analysis(records: list[ImageRecord]) -> dict[str, Any]:
    by_digest: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        if record.sha256 is not None:
            by_digest[record.sha256].append(record)

    duplicate_groups = [
        group for group in by_digest.values() if len(group) > 1
    ]
    conflicting_labels = [
        group
        for group in duplicate_groups
        if len({record.emotion for record in group}) > 1
    ]
    cross_split = [
        group
        for group in duplicate_groups
        if len({record.split for record in group}) > 1
    ]
    within_same_class_split = [
        group
        for group in duplicate_groups
        if len({(record.split, record.emotion) for record in group}) == 1
    ]

    def serialize(groups: list[list[ImageRecord]]) -> list[dict[str, Any]]:
        return [
            {
                "sha256": group[0].sha256,
                "copies": len(group),
                "files": [
                    {
                        "split": item.split,
                        "emotion": item.emotion,
                        "filepath": item.filepath,
                    }
                    for item in group
                ],
            }
            for group in sorted(
                groups,
                key=lambda group: (
                    -len(group),
                    group[0].sha256 or "",
                ),
            )
        ]

    return {
        "duplicate_hash_groups": len(duplicate_groups),
        "duplicate_file_copies_beyond_first": sum(
            len(group) - 1 for group in duplicate_groups
        ),
        "cross_split_groups": len(cross_split),
        "conflicting_label_groups": len(conflicting_labels),
        "within_same_class_split_groups": len(within_same_class_split),
        "cross_split_details": serialize(cross_split),
        "conflicting_label_details": serialize(conflicting_labels),
        "within_same_class_split_details": serialize(within_same_class_split),
    }


def build_report(
    records: list[ImageRecord],
    casing_warnings: list[dict[str, str]],
    *,
    hashes_calculated: bool,
    partial_scan: bool,
) -> dict[str, Any]:
    valid = [record for record in records if record.error is None]
    unreadable = [record for record in records if record.error is not None]

    counts: dict[str, dict[str, int]] = {}
    for split in dict.fromkeys(record.split for record in records):
        counts[split] = {
            emotion: sum(
                record.split == split and record.emotion == emotion
                for record in records
            )
            for emotion in EMOTIONS
        }

    per_class_stats: dict[str, dict[str, Any]] = {}
    for emotion in EMOTIONS:
        subset = [
            record for record in valid if record.emotion == emotion
        ]
        per_class_stats[emotion] = {
            "images": len(subset),
            "width": number_summary(
                [record.width for record in subset if record.width is not None]
            ),
            "height": number_summary(
                [record.height for record in subset if record.height is not None]
            ),
            "size_bytes": number_summary([record.size_bytes for record in subset]),
            "channels": dict(
                sorted(
                    Counter(
                        record.channels
                        for record in subset
                        if record.channels is not None
                    ).items()
                )
            ),
        }

    duplicates = (
        duplicate_analysis(records)
        if hashes_calculated
        else {
            "duplicate_hash_groups": None,
            "cross_split_groups": None,
            "conflicting_label_groups": None,
            "note": "Hashing disabled; duplicate/leakage analysis not performed.",
        }
    )
    return {
        "generated_at_utc": utc_timestamp(),
        "read_only": True,
        "partial_scan": partial_scan,
        "hashes_calculated": hashes_calculated,
        "class_order": list(EMOTIONS),
        "total_files_inspected": len(records),
        "valid_images": len(valid),
        "unreadable_images": len(unreadable),
        "counts": counts,
        "per_class_statistics": per_class_stats,
        "folder_casing_warnings": casing_warnings,
        "unreadable_details": [asdict(record) for record in unreadable],
        "duplicates": duplicates,
    }


def print_report(report: dict[str, Any]) -> None:
    print("\nDATASET AUDIT (READ-ONLY)")
    print("=========================")
    print("Canonical class order:", report["class_order"])
    for split, counts in report["counts"].items():
        print(f"\n{split}:")
        for emotion in EMOTIONS:
            print(f"  {emotion:<8} {counts[emotion]:>6}")
        print(f"  {'TOTAL':<8} {sum(counts.values()):>6}")

    print("\nIntegrity:")
    print("  Files inspected:", report["total_files_inspected"])
    print("  Valid images:", report["valid_images"])
    print("  Unreadable images:", report["unreadable_images"])
    if report["hashes_calculated"]:
        duplicates = report["duplicates"]
        print("  Exact duplicate hash groups:", duplicates["duplicate_hash_groups"])
        print("  Cross-split leakage groups:", duplicates["cross_split_groups"])
        print(
            "  Conflicting-label duplicate groups:",
            duplicates["conflicting_label_groups"],
        )
    else:
        print("  Duplicate analysis: skipped")

    print("\nDimensions by emotion (width x height):")
    for emotion, stats in report["per_class_statistics"].items():
        width = stats["width"]
        height = stats["height"]
        print(
            f"  {emotion:<8} "
            f"W {width['min']}/{width['median']}/{width['max']}  "
            f"H {height['min']}/{height['median']}/{height['max']}  "
            f"channels={stats['channels']}"
        )

    for warning in report["folder_casing_warnings"]:
        print(
            "WARNING: "
            f"{warning['split']}/{warning['actual']} should use canonical "
            f"casing {warning['expected']}."
        )
    if report["partial_scan"]:
        print(
            "WARNING: This was a partial debug scan; counts and duplicate "
            "results are not complete."
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        records, casing_warnings = scan_splits(
            args.split,
            calculate_hashes=not args.no_hashes,
            max_files_per_class=args.max_files_per_class,
            workers=args.workers,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Dataset audit stopped: {exc}", file=sys.stderr)
        return 2

    report = build_report(
        records,
        casing_warnings,
        hashes_calculated=not args.no_hashes,
        partial_scan=args.max_files_per_class is not None,
    )
    print_report(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("\nJSON report:", args.output)

    duplicates = report["duplicates"]
    has_integrity_failure = (
        report["unreadable_images"] > 0
        or (
            report["hashes_calculated"]
            and (
                duplicates["cross_split_groups"] > 0
                or duplicates["conflicting_label_groups"] > 0
            )
        )
    )
    return 1 if has_integrity_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())

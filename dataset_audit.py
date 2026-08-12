from __future__ import annotations
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ml_config import CLEAN_SPLIT_ROOT, EMOTIONS
from ml_dataset import iter_image_files, resolve_class_directories

MIN_IMAGE_SIDE = 48
MAX_WARNING_IMAGES = 50


def image_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def detect_face_count(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(24, 24),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    return len(faces)


def inspect_split(split_name: str, split_root: Path) -> dict[str, Any]:
    stats = {
        "split": split_name,
        "counts": Counter(),
        "duplicate_hashes": defaultdict(list),
        "small_images": [],
        "face_count_issues": [],
        "total": 0,
    }

    class_dirs = resolve_class_directories(split_root)
    for emotion, folder in class_dirs.items():
        for path in iter_image_files(folder):
            stats["total"] += 1
            stats["counts"][emotion] += 1
            file_hash = image_hash(path)
            stats["duplicate_hashes"][file_hash].append(str(path))

            try:
                image = load_image(path)
            except Exception:
                continue

            if image.shape[0] < MIN_IMAGE_SIDE or image.shape[1] < MIN_IMAGE_SIDE:
                stats["small_images"].append(str(path))

            face_count = detect_face_count(image)
            if face_count == 0 or face_count > 1:
                stats["face_count_issues"].append((str(path), face_count))

    return stats


def summarize_audit(audit: dict[str, Any]) -> None:
    print(f"Split: {audit['split']}")
    print(f"  Total images: {audit['total']}")
    for emotion in EMOTIONS:
        print(f"  {emotion}: {audit['counts'].get(emotion, 0)}")
    duplicates = [paths for paths in audit["duplicate_hashes"].values() if len(paths) > 1]
    print(f"  Exact duplicate groups: {len(duplicates)}")
    print(f"  Small images: {len(audit['small_images'])}")
    print(f"  Face count issues: {len(audit['face_count_issues'])}")
    if audit["small_images"]:
        print("  Examples of small images:")
        for path in audit["small_images"][:MAX_WARNING_IMAGES]:
            print(f"    {path}")
    if audit["face_count_issues"]:
        print("  Examples of face-count issues:")
        for path, count in audit["face_count_issues"][:MAX_WARNING_IMAGES]:
            print(f"    {path} -> faces={count}")
    print()


def main() -> int:
    root = CLEAN_SPLIT_ROOT
    for split in ("train", "validation", "test"):
        split_root = root / split
        audit = inspect_split(split, split_root)
        summarize_audit(audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
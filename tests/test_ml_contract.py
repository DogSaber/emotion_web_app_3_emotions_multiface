from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from audit_dataset import ImageRecord, duplicate_analysis
from ml_config import (
    EMOTIONS,
    INPUT_SHAPE,
    metadata_path_for_model,
    read_model_metadata,
    validate_class_order,
    validate_model_metadata,
    write_model_metadata,
)
from ml_dataset import (
    generator_folder_names,
    resolve_class_directories,
    validate_generator_class_indices,
)
from ml_preprocessing import (
    PREPROCESSING_MODES,
    preprocess_face,
    preprocess_training_image,
)
from prepare_test_split import split_counts


class ClassOrderTests(unittest.TestCase):
    def test_manuscript_class_order_is_exact(self) -> None:
        self.assertEqual(
            EMOTIONS,
            ("Happy", "Angry", "Sad", "Neutral", "Surprise"),
        )
        self.assertEqual(validate_class_order(EMOTIONS), EMOTIONS)

    def test_swapped_happy_angry_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be remapped silently"):
            validate_class_order(
                ("Angry", "Happy", "Sad", "Neutral", "Surprise")
            )


class PreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(42)
        cls.image = rng.integers(0, 256, size=(64, 56), dtype=np.uint8)

    def test_all_modes_return_expected_training_contract(self) -> None:
        for mode in PREPROCESSING_MODES:
            with self.subTest(mode=mode):
                value = preprocess_training_image(self.image, mode=mode)
                self.assertEqual(value.shape, INPUT_SHAPE)
                self.assertEqual(value.dtype, np.float32)
                self.assertTrue(np.all(np.isfinite(value)))
                self.assertGreaterEqual(float(value.min()), 0.0)
                self.assertLessEqual(float(value.max()), 1.0)

    def test_face_adapter_adds_batch_dimension(self) -> None:
        value = preprocess_face(self.image, mode="basic")
        self.assertEqual(value.shape, (1, *INPUT_SHAPE))

    def test_legacy_mode_matches_original_flask_pipeline(self) -> None:
        resized = cv2.resize(self.image, (48, 48), interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(resized, (3, 3), 0)
        sharpen = np.array(
            [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
            dtype=np.float32,
        )
        sharpened = cv2.filter2D(blurred, -1, sharpen)
        expected = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        ).apply(sharpened)
        expected = expected.astype(np.float32) / 255.0
        expected = expected[np.newaxis, :, :, np.newaxis]
        actual = preprocess_face(self.image, mode="legacy")
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)

    def test_empty_image_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            preprocess_face(np.array([], dtype=np.uint8), mode="basic")


class MetadataTests(unittest.TestCase):
    def test_metadata_round_trip_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            model_path = Path(temp) / "candidate.h5"
            sidecar = write_model_metadata(
                model_path,
                {
                    "preprocessing": "basic",
                    "status": "training",
                },
            )
            self.assertEqual(sidecar, metadata_path_for_model(model_path))
            metadata = read_model_metadata(model_path, required=True)
            self.assertIsNotNone(metadata)
            assert metadata is not None
            validate_model_metadata(
                metadata,
                expected_preprocessing="basic",
            )
            self.assertEqual(metadata["class_order"], list(EMOTIONS))
            self.assertEqual(metadata["input_shape"], list(INPUT_SHAPE))

    def test_metadata_rejects_wrong_class_order(self) -> None:
        metadata = {
            "schema_version": 1,
            "class_order": [
                "Angry",
                "Happy",
                "Sad",
                "Neutral",
                "Surprise",
            ],
            "input_shape": list(INPUT_SHAPE),
            "preprocessing": "legacy",
        }
        with self.assertRaisesRegex(ValueError, "required order"):
            validate_model_metadata(metadata)

class DatasetContractTests(unittest.TestCase):
    def test_case_insensitive_folders_preserve_canonical_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            actual_names = ("HAPPY", "angry", "Sad", "neutral", "SURPRISE")
            for name in actual_names:
                (root / name).mkdir()

            mapping = resolve_class_directories(root, require_images=False)
            self.assertEqual(tuple(mapping.keys()), EMOTIONS)
            folder_names = generator_folder_names(mapping)
            self.assertEqual(folder_names, list(actual_names))
            class_indices = {
                name: index for index, name in enumerate(folder_names)
            }
            validate_generator_class_indices(class_indices, mapping)

    def test_generator_index_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in EMOTIONS:
                (root / name).mkdir()
            mapping = resolve_class_directories(root, require_images=False)
            wrong = {
                "Angry": 0,
                "Happy": 1,
                "Sad": 2,
                "Neutral": 3,
                "Surprise": 4,
            }
            with self.assertRaisesRegex(ValueError, "prevent a mislabeled model"):
                validate_generator_class_indices(wrong, mapping)

    def test_duplicate_analysis_finds_cross_split_label_conflict(self) -> None:
        digest = "a" * 64
        records = [
            ImageRecord(
                "train",
                "Happy",
                "train/happy.jpg",
                10,
                digest,
                48,
                48,
                1,
                None,
            ),
            ImageRecord(
                "validation",
                "Sad",
                "validation/sad.jpg",
                10,
                digest,
                48,
                48,
                1,
                None,
            ),
        ]
        result = duplicate_analysis(records)
        self.assertEqual(result["duplicate_hash_groups"], 1)
        self.assertEqual(result["cross_split_groups"], 1)
        self.assertEqual(result["conflicting_label_groups"], 1)

    def test_split_counts_cover_every_unique_image(self) -> None:
        train, validation, test = split_counts(
            101,
            validation_ratio=0.15,
            test_ratio=0.15,
        )
        self.assertEqual(train + validation + test, 101)
        self.assertGreater(train, validation)
        self.assertGreater(train, test)


if __name__ == "__main__":
    unittest.main()

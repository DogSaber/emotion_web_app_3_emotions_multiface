"""Deprecated filename retained for compatibility.

This project is not a three-emotion system. Executing this file delegates to
the canonical five-class trainer with output order:
Happy, Angry, Sad, Neutral, Surprise.
"""

from __future__ import annotations

import warnings

from train import main


if __name__ == "__main__":
    warnings.warn(
        "The old 3class filename is obsolete; running the canonical "
        "five-class train.py pipeline instead.",
        DeprecationWarning,
    )
    raise SystemExit(main())


"""Deprecated compatibility entry point for the five-class trainer.

Use ``train.py`` in new commands. This wrapper is retained so old notes do not
start the broken seven-class fine-tuning script that previously lived here.
"""

from __future__ import annotations

import warnings

from train import main


if __name__ == "__main__":
    warnings.warn(
        "train_model.py is deprecated; running the canonical five-class "
        "train.py pipeline instead.",
        DeprecationWarning,
    )
    raise SystemExit(main())


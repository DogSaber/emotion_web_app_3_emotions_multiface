"""Deprecated compatibility entry point for dataset preparation.

The original implementation copied ``dataset/EXTRA`` repeatedly and moved
files out of the training split, which could create duplicates or alter a
working dataset on an accidental rerun. This compatibility wrapper now uses
the safe splitter:

* default execution is a read-only dry run;
* ``--apply`` copies into a separate protected output root;
* current train/validation files are never moved or deleted.

Use ``audit_dataset.py`` first and ``prepare_test_split.py`` directly in new
commands.
"""

from __future__ import annotations

import warnings

from prepare_test_split import main


if __name__ == "__main__":
    warnings.warn(
        "prepare_dataset.py is deprecated; running the non-destructive "
        "prepare_test_split.py workflow instead.",
        DeprecationWarning,
    )
    raise SystemExit(main())


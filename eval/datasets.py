"""Backwards-compatible alias for :mod:`data_loaders`.

This file used to be a near-copy of data_loaders.py, and the two had already
drifted apart (this copy carried mbpp's ``test_setup_code``, which judge_code.py
reads, while the module everything actually imports had dropped it).  Re-exporting
keeps both import paths working with a single source of truth.

Prefer ``import data_loaders`` in new code: this module shadows the HuggingFace
``datasets`` package for anything running with eval/ on sys.path.
"""

from data_loaders import *  # noqa: F401,F403
from data_loaders import CALIB_SETS, DATA, EVAL_SETS, LOADERS, load  # noqa: F401

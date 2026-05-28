"""
Shared utilities for the autoresearch loop.

Holds the canonical results-CSV schema and small helpers used by ``run.py``
and ``prepare.py`` for logging and for the feature-stability metric.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


# Canonical column order for ``results.csv``. New columns must be appended
# (never inserted) so that older session CSVs remain readable.
RESULTS_FIELDNAMES = [
    "timestamp_utc",
    "label",
    "model",
    "feature_mode",
    "split_date",
    "train_rows",
    "test_rows",
    "n_features",
    "runtime_seconds",
    "mse",
    "rmse",
    "mae",
    "r2",
    "stability",
    "selected_features",
]


def utc_timestamp() -> str:
    """
    Return the current UTC time as a second-precision ISO 8601 string.
    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def append_results_row(path: Path, row: dict[str, object]) -> None:
    """
    Append one experiment row to ``results.csv``, writing a header if new.

    Missing keys in ``row`` are written as empty cells by ``DictWriter``; keys
    not in ``RESULTS_FIELDNAMES`` would raise, which intentionally guards
    against accidental schema drift between sessions.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def pairwise_jaccard(selections: Iterable[list[str]]) -> float:
    """
    Return the mean pairwise Jaccard similarity across feature selections.

    Empty selections are dropped first; ``nan`` is returned when fewer than two
    non-empty selections remain (no defined pairwise comparison). Used by
    ``prepare.estimate_feature_stability`` to score how consistently a model
    picks the same features across expanding training windows.
    """
    selections = [set(selection) for selection in selections if selection]
    if len(selections) < 2:
        return float("nan")

    scores = []
    for left_index in range(len(selections)):
        for right_index in range(left_index + 1, len(selections)):
            left = selections[left_index]
            right = selections[right_index]
            union = left | right
            if not union:
                continue
            scores.append(len(left & right) / len(union))

    return float(np.mean(scores)) if scores else float("nan")


def serialize_feature_list(features: list[str]) -> str:
    """
    Encode a feature list as ASCII-safe JSON for the results CSV cell.
    """
    return json.dumps(features, ensure_ascii=True)

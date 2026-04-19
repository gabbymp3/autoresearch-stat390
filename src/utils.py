from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


RESULTS_FIELDNAMES = [
    "timestamp_utc",
    "label",
    "model",
    "feature_mode",
    "split_date",
    "train_rows",
    "test_rows",
    "n_features",
    "mse",
    "rmse",
    "mae",
    "r2",
    "stability",
    "selected_features",
]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def append_results_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def pairwise_jaccard(selections: Iterable[list[str]]) -> float:
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
    return json.dumps(features, ensure_ascii=True)

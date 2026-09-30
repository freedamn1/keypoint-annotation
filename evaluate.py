"""Measure full-resolution point error and box IoU against reviewed LabelMe JSONs."""

from __future__ import annotations

import argparse
import math
import statistics
from collections import defaultdict
from pathlib import Path

from pose_pipeline.core import LABELS, load_labelme


def extract_points(record: dict) -> dict[str, tuple[float, float]]:
    points = {}
    for shape in record["shapes"]:
        label = shape["label"]
        if label not in LABELS or shape.get("group_id") != 100:
            continue
        if shape.get("shape_type") != "point" or len(shape.get("points", [])) != 1 or label in points:
            raise ValueError(f"invalid or duplicate point: {label}")
        points[label] = tuple(shape["points"][0])
    return points


def extract_box(record: dict, label: str) -> tuple[float, float, float, float]:
    boxes = [s for s in record["shapes"] if s.get("label") == label and s.get("shape_type") == "rectangle" and s.get("group_id") == 100]
    if len(boxes) != 1:
        raise ValueError(f"expected one {label} rectangle, found {len(boxes)}")
    (x1, y1), (x2, y2) = boxes[0]["points"]
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def box_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection) if area_a + area_b - intersection > 0 else 0.0


def evaluate(gold_folder: Path, prediction_folder: Path) -> dict:
    gold_paths = sorted(gold_folder.glob("*.json"))
    if not gold_paths:
        raise ValueError("no gold JSONs found")
    errors = defaultdict(list)
    ious = defaultdict(list)
    missing_predictions = []
    for gold_path in gold_paths:
        prediction_path = prediction_folder / gold_path.name
        if not prediction_path.is_file():
            missing_predictions.append(gold_path.name)
            continue
        gold_record = load_labelme(gold_path)
        predicted_record = load_labelme(prediction_path)
        gold = extract_points(gold_record)
        predicted = extract_points(predicted_record)
        for label in ("plant", "ear"):
            ious[label].append(box_iou(extract_box(gold_record, label), extract_box(predicted_record, label)))
        for label, (gx, gy) in gold.items():
            if label in predicted:
                px, py = predicted[label]
                errors[label].append(math.hypot(px - gx, py - gy))
    all_errors = [error for label_errors in errors.values() for error in label_errors]
    if not all_errors:
        raise ValueError("no comparable keypoints")

    def metrics(values: list[float]) -> dict:
        values = sorted(values)
        return {
            "n": len(values),
            "median_px": round(statistics.median(values), 1),
            "p90_px": round(values[min(len(values) - 1, math.ceil(0.9 * len(values)) - 1)], 1),
            "within_25px": round(sum(value <= 25 for value in values) / len(values), 3),
            "within_50px": round(sum(value <= 50 for value in values) / len(values), 3),
            "within_100px": round(sum(value <= 100 for value in values) / len(values), 3),
        }

    return {
        "gold_images": len(gold_paths),
        "missing_prediction_images": missing_predictions,
        "overall": metrics(all_errors),
        "box_iou": {label: {"median": round(statistics.median(ious[label]), 3), "n": len(ious[label])} for label in ("plant", "ear")},
        "by_label": {label: metrics(errors[label]) for label in LABELS if errors[label]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    args = parser.parse_args()
    import json

    print(json.dumps(evaluate(args.gold, args.pred), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

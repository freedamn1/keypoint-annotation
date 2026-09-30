"""Audit LabelMe source data and build a YOLO11 pose dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

from pose_pipeline.core import LABELS, centered_roi, image_id, load_labelme, selected_ear, validate_labelme, yolo_detection_line, yolo_pose_line


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="folder with source JPEGs and LabelMe JSONs")
    parser.add_argument("--output", type=Path, default=Path("data/pose"))
    parser.add_argument("--ear-output", type=Path, default=Path("data/ear"))
    parser.add_argument("--roi-width-ratio", type=float, default=0.85)
    parser.add_argument("--block-size", type=int, default=20, help="adjacent image IDs per split group")
    parser.add_argument("--groups-csv", type=Path, help="optional CSV with image,group columns")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def load_groups(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"image", "group"}.issubset(rows[0]):
        raise ValueError("groups CSV needs image,group columns")
    groups = {}
    for row in rows:
        image, group = row["image"].strip(), row["group"].strip()
        if not image or not group or image in groups:
            raise ValueError(f"invalid or duplicate groups CSV row: {row}")
        groups[image] = group
    return groups


def assign_splits(group_by_image: dict[str, str], val_fraction: float, test_fraction: float, seed: int) -> dict[str, str]:
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("val and test fractions must be positive and total under 1")
    group_names = sorted(set(group_by_image.values()))
    if len(group_names) < 3:
        raise ValueError("at least three groups are required for train/val/test")
    random.Random(seed).shuffle(group_names)
    n_val = max(1, round(len(group_names) * val_fraction))
    n_test = max(1, round(len(group_names) * test_fraction))
    if n_val + n_test >= len(group_names):
        raise ValueError("too few groups left for training")
    test_groups = set(group_names[:n_test])
    val_groups = set(group_names[n_test : n_test + n_val])
    return {
        image: "test" if group in test_groups else "val" if group in val_groups else "train"
        for image, group in group_by_image.items()
    }


def build(
    source: Path,
    output: Path,
    ear_output: Path | None = None,
    roi_width_ratio: float = 0.85,
    block_size: int = 20,
    groups_csv: Path | None = None,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    audit_only: bool = False,
) -> dict:
    source = source.resolve()
    output = output.resolve()
    ear_output = (ear_output or output.parent / "ear").resolve()
    if output == ear_output:
        raise ValueError("pose and ear outputs must differ")
    if not source.is_dir():
        raise FileNotFoundError(source)
    if block_size < 1:
        raise ValueError("block size must be positive")
    json_paths = sorted(source.glob("*.json"))
    if not json_paths:
        raise ValueError(f"no LabelMe JSONs in {source}")
    specified_groups = load_groups(groups_csv)
    records = []
    warnings = []
    point_counts = Counter()
    ids = [image_id(path) for path in json_paths]
    first_id = min(ids)
    group_by_image = {}

    for json_path, numeric_id in zip(json_paths, ids):
        record = load_labelme(json_path)
        image_path = source / record["imagePath"]
        if image_path.stem != json_path.stem:
            raise ValueError(f"JSON and image stems differ: {json_path.name}")
        group_id, plant, points = validate_labelme(record, image_path)
        ear = selected_ear(record, group_id)
        roi = centered_roi(record["imageWidth"], record["imageHeight"], roi_width_ratio)
        pose_line = yolo_pose_line(plant, points, roi)
        ear_line = yolo_detection_line(ear, roi)
        (x1, y1), (x2, y2) = plant["points"]
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        for label, (x, y) in points.items():
            if not (x1 <= x <= x2 and y1 <= y <= y2):
                warnings.append(f"{json_path.name}: {label} is outside plant box")
        for label in points:
            point_counts[label] += 1
        if groups_csv is not None and image_path.name not in specified_groups:
            raise ValueError(f"missing image in groups CSV: {image_path.name}")
        group = specified_groups.get(image_path.name, f"block-{(numeric_id - first_id) // block_size:03d}")
        group_by_image[image_path.name] = group
        records.append((image_path, roi, pose_line, ear_line, group))

    if groups_csv is not None and set(specified_groups) != set(group_by_image):
        raise ValueError("groups CSV contains images absent from source")
    splits = assign_splits(group_by_image, val_fraction, test_fraction, seed)
    summary = {
        "images": len(records),
        "point_counts": {label: point_counts[label] for label in LABELS},
        "split_counts": dict(Counter(splits.values())),
        "group_count": len(set(group_by_image.values())),
        "warnings": warnings,
        "roi_width_ratio": roi_width_ratio,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if audit_only:
        return summary
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output is not empty: {output}")
    if ear_output.exists() and any(ear_output.iterdir()):
        raise FileExistsError(f"ear output is not empty: {ear_output}")

    for root in (output, ear_output):
        for split in ("train", "val", "test"):
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
            (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    for image_path, roi, pose_line, ear_line, _ in records:
        split = splits[image_path.name]
        target_image = output / "images" / split / image_path.name
        target_pose_label = output / "labels" / split / f"{image_path.stem}.txt"
        target_ear_image = ear_output / "images" / split / image_path.name
        target_ear_label = ear_output / "labels" / split / f"{image_path.stem}.txt"
        with Image.open(image_path) as image:
            if roi.left == 0 and roi.top == 0 and roi.right == image.width and roi.bottom == image.height:
                shutil.copy2(image_path, target_image)
            else:
                image.crop((roi.left, roi.top, roi.right, roi.bottom)).save(target_image, quality=95)
        target_pose_label.write_text(pose_line + "\n", encoding="utf-8")
        try:
            os.link(target_image, target_ear_image)
        except OSError:
            shutil.copy2(target_image, target_ear_image)
        target_ear_label.write_text(ear_line + "\n", encoding="utf-8")
    data_yaml = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "plant"},
        "kpt_shape": [12, 3],
        "flip_idx": list(range(12)),
    }
    (output / "data.yaml").write_text(yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")
    ear_yaml = {
        "path": str(ear_output),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "ear"},
    }
    (ear_output / "data.yaml").write_text(yaml.safe_dump(ear_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (output / "metadata.json").write_text(json.dumps({"roi_width_ratio": roi_width_ratio, "labels": LABELS, "seed": seed}, indent=2) + "\n", encoding="utf-8")
    with (output / "splits.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image", "group", "split"))
        for image in sorted(group_by_image):
            writer.writerow((image, group_by_image[image], splits[image]))
    return summary


if __name__ == "__main__":
    args = parse_args()
    build(**vars(args))

"""Shared LabelMe and YOLO pose data conventions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

LABELS = (
    "leaf_above_ear_connection_point",
    "leaf_above_ear_angle_point",
    "leaf_above_ear_sample_point",
    "ear_node_point",
    "ear_middle_point",
    "ear_top_point",
    "ear_leaf_connection_point",
    "ear_leaf_angle_point",
    "ear_leaf_sample_point",
    "leaf_below_ear_connection_point",
    "leaf_below_ear_angle_point",
    "leaf_below_ear_sample_point",
)
LABEL_SET = set(LABELS)


@dataclass(frozen=True)
class Roi:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def centered_roi(width: int, height: int, width_ratio: float) -> Roi:
    if not 0 < width_ratio <= 1:
        raise ValueError("roi width ratio must be in (0, 1]")
    crop_width = round(width * width_ratio)
    left = (width - crop_width) // 2
    return Roi(left, 0, left + crop_width, height)


def load_labelme(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def image_id(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    if not match:
        raise ValueError(f"filename has no numeric image ID: {path.name}")
    return int(match.group(1))


def selected_plant(record: dict) -> tuple[int, dict]:
    width, height = record["imageWidth"], record["imageHeight"]
    plants = [s for s in record["shapes"] if s["label"] == "plant" and s["shape_type"] == "rectangle"]
    if not plants:
        raise ValueError("missing plant rectangle")

    def center_distance(shape: dict) -> float:
        (x1, y1), (x2, y2) = shape["points"]
        return ((x1 + x2) / 2 - width / 2) ** 2 + ((y1 + y2) / 2 - height / 2) ** 2

    plant = min(plants, key=center_distance)
    group_id = plant.get("group_id")
    if group_id is None:
        raise ValueError("plant group_id is missing")
    return group_id, plant


def selected_ear(record: dict, group_id: int) -> dict:
    ears = [
        s for s in record["shapes"]
        if s.get("label") == "ear" and s.get("shape_type") == "rectangle" and s.get("group_id") == group_id
    ]
    if len(ears) != 1:
        raise ValueError(f"expected one ear rectangle in group {group_id}, found {len(ears)}")
    return ears[0]


def validate_labelme(record: dict, image_path: Path) -> tuple[int, dict, dict[str, tuple[float, float]]]:
    with Image.open(image_path) as image:
        size = image.size
    if size != (record.get("imageWidth"), record.get("imageHeight")):
        raise ValueError(f"image dimensions disagree with JSON: {image_path}")
    if record.get("imagePath") != image_path.name:
        raise ValueError(f"imagePath mismatch: {image_path}")

    group_id, plant = selected_plant(record)
    ear = selected_ear(record, group_id)
    points: dict[str, tuple[float, float]] = {}
    for shape in record["shapes"]:
        if shape.get("group_id") != group_id:
            continue
        label = shape.get("label")
        if label not in LABEL_SET:
            continue
        if shape.get("shape_type") != "point" or len(shape.get("points", [])) != 1:
            raise ValueError(f"{image_path.name}: {label} is not a point")
        if label in points:
            raise ValueError(f"{image_path.name}: duplicate {label}")
        x, y = shape["points"][0]
        if not (0 <= x < size[0] and 0 <= y < size[1]):
            raise ValueError(f"{image_path.name}: {label} outside image")
        points[label] = (float(x), float(y))

    (x1, y1), (x2, y2) = plant["points"]
    if not (0 <= min(x1, x2) < max(x1, x2) <= size[0] and 0 <= min(y1, y2) < max(y1, y2) <= size[1]):
        raise ValueError(f"{image_path.name}: invalid plant rectangle")
    (x1, y1), (x2, y2) = ear["points"]
    if not (0 <= min(x1, x2) < max(x1, x2) <= size[0] and 0 <= min(y1, y2) < max(y1, y2) <= size[1]):
        raise ValueError(f"{image_path.name}: invalid ear rectangle")
    return group_id, plant, points


def yolo_pose_line(plant: dict, points: dict[str, tuple[float, float]], roi: Roi) -> str:
    (x1, y1), (x2, y2) = plant["points"]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if not (roi.left <= x1 < x2 <= roi.right and roi.top <= y1 < y2 <= roi.bottom):
        raise ValueError("plant box is clipped by ROI; increase --roi-width-ratio")
    fields = [
        0,
        ((x1 + x2) / 2 - roi.left) / roi.width,
        ((y1 + y2) / 2 - roi.top) / roi.height,
        (x2 - x1) / roi.width,
        (y2 - y1) / roi.height,
    ]
    for label in LABELS:
        if label not in points:
            fields.extend((0, 0, 0))
            continue
        x, y = points[label]
        if not (roi.left <= x < roi.right and roi.top <= y < roi.bottom):
            raise ValueError(f"{label} is clipped by ROI; increase --roi-width-ratio")
        fields.extend(((x - roi.left) / roi.width, (y - roi.top) / roi.height, 2))
    if len(fields) != 5 + 3 * len(LABELS):
        raise AssertionError("unexpected pose label length")
    return " ".join(str(value) if isinstance(value, int) else f"{value:.8f}" for value in fields)


def point_shape(label: str, x: float, y: float, group_id: int = 100) -> dict:
    return {
        "label": label,
        "points": [[round(float(x), 2), round(float(y), 2)]],
        "group_id": group_id,
        "description": "",
        "shape_type": "point",
        "flags": {},
        "mask": None,
    }


def rectangle_shape(label: str, box: tuple[float, float, float, float], group_id: int = 100) -> dict:
    x1, y1, x2, y2 = box
    if not (x1 < x2 and y1 < y2):
        raise ValueError(f"invalid {label} box")
    return {
        "label": label,
        "points": [[round(float(x1), 2), round(float(y1), 2)], [round(float(x2), 2), round(float(y2), 2)]],
        "group_id": group_id,
        "description": "",
        "shape_type": "rectangle",
        "flags": {},
        "mask": None,
    }


def yolo_detection_line(ear: dict, roi: Roi) -> str:
    (x1, y1), (x2, y2) = ear["points"]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if not (roi.left <= x1 < x2 <= roi.right and roi.top <= y1 < y2 <= roi.bottom):
        raise ValueError("ear box is clipped by ROI; increase --roi-width-ratio")
    values = [0, ((x1 + x2) / 2 - roi.left) / roi.width, ((y1 + y2) / 2 - roi.top) / roi.height, (x2 - x1) / roi.width, (y2 - y1) / roi.height]
    return " ".join(str(value) if isinstance(value, int) else f"{value:.8f}" for value in values)


def prediction_labelme(
    image_path: Path,
    width: int,
    height: int,
    coordinates: list[tuple[float, float]],
    plant_box: tuple[float, float, float, float],
    ear_box: tuple[float, float, float, float],
) -> dict:
    if len(coordinates) != len(LABELS):
        raise ValueError("prediction must contain exactly 12 points")
    shapes = [rectangle_shape("ear", ear_box), rectangle_shape("plant", plant_box)]
    for label, (x, y) in zip(LABELS, coordinates):
        x = min(max(float(x), 0.0), width - 1.0)
        y = min(max(float(y), 0.0), height - 1.0)
        shapes.append(point_shape(label, x, y))
    return {
        "version": "5.10.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }

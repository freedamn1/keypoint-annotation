"""Run pose and ear detection models; write 12 points and two LabelMe boxes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from pose_pipeline.core import LABELS, centered_roi, prediction_labelme


def image_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)
    return sorted(path for path in source.iterdir() if path.suffix.lower() in {".jpeg", ".jpg", ".png"})


def choose_center_instance(result: object, width: int, height: int) -> int | None:
    if result.boxes is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    distances = ((centers[:, 0] - width / 2) / (width / 2)) ** 2 + ((centers[:, 1] - height / 2) / (height / 2)) ** 2
    scores = confidences - 0.35 * np.sqrt(distances)
    return int(np.argmax(scores))


def choose_ear_instance(result: object, plant_box: np.ndarray, ear_middle: np.ndarray) -> int | None:
    if result.boxes is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    inside = (
        (centers[:, 0] >= plant_box[0]) & (centers[:, 0] <= plant_box[2])
        & (centers[:, 1] >= plant_box[1]) & (centers[:, 1] <= plant_box[3])
    )
    if not np.any(inside):
        return None
    scale = max(float(plant_box[2] - plant_box[0]), float(plant_box[3] - plant_box[1]), 1.0)
    distances = np.linalg.norm(centers - ear_middle, axis=1) / scale
    scores = np.where(inside, confidences - 0.5 * distances, -np.inf)
    return int(np.argmax(scores))


def bounded_box(box: np.ndarray | tuple[float, ...], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(value) for value in box)
    x1 = min(max(x1, 0), width - 2)
    y1 = min(max(y1, 0), height - 2)
    x2 = min(max(x2, x1 + 1), width - 1)
    y2 = min(max(y2, y1 + 1), height - 1)
    return x1, y1, x2, y2


def ear_box_from_points(points: np.ndarray, plant_box: np.ndarray) -> np.ndarray:
    indices = [LABELS.index(name) for name in ("ear_node_point", "ear_middle_point", "ear_top_point")]
    coords = points[indices]
    margin_x = max(30.0, 0.05 * (plant_box[2] - plant_box[0]))
    margin_y = max(30.0, 0.05 * max(np.ptp(coords[:, 1]), 1.0))
    box = np.array([coords[:, 0].min() - margin_x, coords[:, 1].min() - margin_y, coords[:, 0].max() + margin_x, coords[:, 1].max() + margin_y])
    box[:2] = np.maximum(box[:2], plant_box[:2])
    box[2:] = np.minimum(box[2:], plant_box[2:])
    return box


def predict(model: object, array_bgr: np.ndarray, imgsz: int, conf: float, device: str) -> object:
    result = model.predict(source=array_bgr, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        result = model.predict(source=array_bgr, imgsz=imgsz, conf=0.03, device=device, verbose=False)[0]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-weights", type=Path, required=True)
    parser.add_argument("--ear-weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="one image or a folder; JSONs are written beside images")
    parser.add_argument("--metadata", type=Path, required=True, help="metadata.json from the converted dataset")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--device", default="0")
    parser.add_argument("--overwrite", action="store_true", help="replace existing JSONs")
    args = parser.parse_args()
    for weights in (args.pose_weights, args.ear_weights):
        if not weights.is_file():
            raise FileNotFoundError(weights)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if tuple(metadata.get("labels", ())) != LABELS:
        raise ValueError("metadata keypoint order differs from code")
    ratio = metadata["roi_width_ratio"]
    from ultralytics import YOLO

    pose_model = YOLO(str(args.pose_weights))
    ear_model = YOLO(str(args.ear_weights))
    written = skipped = failed = ear_fallbacks = 0
    for image_path in image_files(args.source):
        output_path = image_path.with_suffix(".json")
        if output_path.exists() and not args.overwrite:
            print(f"SKIP existing: {output_path}")
            skipped += 1
            continue
        with Image.open(image_path) as image:
            width, height = image.size
            roi = centered_roi(width, height, ratio)
            crop = image.crop((roi.left, roi.top, roi.right, roi.bottom)).convert("RGB")
            array_bgr = np.ascontiguousarray(np.asarray(crop)[:, :, ::-1])
        pose_result = predict(pose_model, array_bgr, args.imgsz, args.conf, args.device)
        plant_index = choose_center_instance(pose_result, roi.width, roi.height)
        if plant_index is None or pose_result.keypoints is None:
            print(f"NO PLANT DETECTION: {image_path.name}")
            failed += 1
            continue
        keypoints = pose_result.keypoints.xy[plant_index].cpu().numpy()
        if len(keypoints) != len(LABELS):
            raise ValueError(f"model predicted {len(keypoints)} keypoints, expected 12")
        confidences = pose_result.keypoints.conf
        point_conf = confidences[plant_index].cpu().numpy() if confidences is not None else np.ones(len(LABELS))
        plant_box_local = pose_result.boxes.xyxy[plant_index].cpu().numpy()
        plant_center = (plant_box_local[:2] + plant_box_local[2:]) / 2
        for i, ((x, y), confidence) in enumerate(zip(keypoints, point_conf)):
            if confidence < 0.01 and x == 0 and y == 0:
                keypoints[i] = plant_center
        ear_result = predict(ear_model, array_bgr, args.imgsz, args.conf, args.device)
        ear_index = choose_ear_instance(ear_result, plant_box_local, keypoints[LABELS.index("ear_middle_point")])
        if ear_index is None:
            ear_box_local = ear_box_from_points(keypoints, plant_box_local)
            ear_fallbacks += 1
            source_note = "ear box estimated from keypoints"
        else:
            ear_box_local = ear_result.boxes.xyxy[ear_index].cpu().numpy()
            ear_box_local[:2] = np.maximum(ear_box_local[:2], plant_box_local[:2])
            ear_box_local[2:] = np.minimum(ear_box_local[2:], plant_box_local[2:])
            source_note = "ear box detected"
        offset = np.array([roi.left, roi.top])
        coordinates = [tuple(point + offset) for point in keypoints]
        box_offset = np.array([roi.left, roi.top, roi.left, roi.top])
        plant_box = bounded_box(plant_box_local + box_offset, width, height)
        ear_box = bounded_box(ear_box_local + box_offset, width, height)
        payload = prediction_labelme(image_path, width, height, coordinates, plant_box, ear_box)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        low = sum(float(value) < 0.3 for value in point_conf)
        print(f"WRITE {output_path.name}: {low} points below confidence 0.3; {source_note}")
        written += 1
    print(f"finished: written={written}, skipped={skipped}, no_plant={failed}, ear_box_fallbacks={ear_fallbacks}")


if __name__ == "__main__":
    main()

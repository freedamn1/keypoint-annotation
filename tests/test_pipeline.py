import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from build_dataset import build
from evaluate import evaluate
from infer_labelme import choose_center_instance, choose_ear_instance, ear_box_from_points
from train import runtime_data_yaml
from pose_pipeline.core import LABELS, centered_roi, prediction_labelme, yolo_pose_line


def sample_record(name: str) -> dict:
    return {
        "version": "5.10.1",
        "flags": {},
        "imagePath": name,
        "imageHeight": 80,
        "imageWidth": 100,
        "shapes": [
            {"label": "plant", "shape_type": "rectangle", "group_id": 100, "points": [[20, 10], [80, 70]]},
            {"label": "ear", "shape_type": "rectangle", "group_id": 100, "points": [[30, 20], [50, 60]]},
            {"label": "ear_node_point", "shape_type": "point", "group_id": 100, "points": [[30, 40]]},
        ],
    }


def test_build_keeps_missing_points_invisible_and_transforms_crop(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for number in range(3):
        name = f"IMG_{number:04d}.jpeg"
        Image.new("RGB", (100, 80), "green").save(source / name)
        (source / f"IMG_{number:04d}.json").write_text(json.dumps(sample_record(name)), encoding="utf-8")
    output = tmp_path / "pose"
    ear_output = tmp_path / "ear"
    summary = build(source, output, ear_output=ear_output, roi_width_ratio=0.8, val_fraction=0.33, test_fraction=0.33)
    assert summary["split_counts"] == {"train": 1, "val": 1, "test": 1}
    assert summary["group_count"] == 3
    with (output / "splits.csv").open(newline="", encoding="utf-8") as handle:
        splits = list(csv.DictReader(handle))
    assert {row["group"] for row in splits} == {row["image"] for row in splits}
    labels = list(output.glob("labels/*/*.txt"))
    assert len(labels) == 3
    fields = [float(value) for value in labels[0].read_text().split()]
    assert len(fields) == 41
    assert fields[:5] == pytest.approx([0, 0.5, 0.5, 0.75, 0.75])
    ear_node = 5 + 3 * LABELS.index("ear_node_point")
    assert fields[ear_node : ear_node + 3] == pytest.approx([0.25, 0.5, 2])
    assert fields[5:8] == [0, 0, 0]
    with Image.open(next(output.glob("images/*/*.jpeg"))) as image:
        assert image.size == (80, 80)
    ear_fields = [float(value) for value in next(ear_output.glob("labels/*/*.txt")).read_text().split()]
    assert ear_fields == pytest.approx([0, 0.375, 0.5, 0.25, 0.5])
    assert len(list(ear_output.glob("images/*/*.jpeg"))) == 3


def test_roi_rejects_clipped_label() -> None:
    plant = {"points": [[2, 10], [80, 70]]}
    with pytest.raises(ValueError, match="clipped by ROI"):
        yolo_pose_line(plant, {}, centered_roi(100, 80, 0.8))


def test_training_data_path_is_relocated_after_upload(tmp_path: Path) -> None:
    import yaml

    data = tmp_path / "data.yaml"
    data.write_text(yaml.safe_dump({"path": "D:/old/local/path", "train": "images/train", "val": "images/val", "test": "images/test", "names": {0: "plant"}}), encoding="utf-8")
    runtime = runtime_data_yaml(data)
    assert runtime.parent == tmp_path
    assert yaml.safe_load(runtime.read_text(encoding="utf-8"))["path"] == str(tmp_path.resolve())


def test_prediction_json_and_evaluation(tmp_path: Path) -> None:
    image = tmp_path / "IMG_0001.jpeg"
    Image.new("RGB", (100, 80), "green").save(image)
    predicted = prediction_labelme(image, 100, 80, [(30, 40)] * 12, (20, 10, 80, 70), (30, 20, 50, 60))
    assert len(predicted["shapes"]) == 14
    assert [s["label"] for s in predicted["shapes"][:2]] == ["ear", "plant"]
    assert all(s["shape_type"] == "point" and s["group_id"] == 100 for s in predicted["shapes"][2:])
    gold_dir, pred_dir = tmp_path / "gold", tmp_path / "pred"
    gold_dir.mkdir()
    pred_dir.mkdir()
    gold = sample_record(image.name)
    (gold_dir / "IMG_0001.json").write_text(json.dumps(gold), encoding="utf-8")
    (pred_dir / "IMG_0001.json").write_text(json.dumps(predicted), encoding="utf-8")
    scores = evaluate(gold_dir, pred_dir)
    assert scores["overall"]["n"] == 1
    assert scores["overall"]["median_px"] == 0
    assert scores["box_iou"]["plant"]["median"] == 1.0
    assert scores["box_iou"]["ear"]["median"] == 1.0


def test_center_instance_selection() -> None:
    class FakeTensor:
        def __init__(self, value):
            self.value = np.asarray(value)

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class Boxes:
        xyxy = FakeTensor([[0, 0, 20, 20], [40, 30, 60, 50]])
        conf = FakeTensor([0.9, 0.7])

        def __len__(self):
            return 2

    class Result:
        boxes = Boxes()

    assert choose_center_instance(Result(), 100, 80) == 1
    assert choose_ear_instance(Result(), np.array([35, 25, 65, 55]), np.array([50, 40])) == 1


def test_ear_box_fallback_uses_ear_points() -> None:
    points = np.zeros((12, 2))
    for label, coordinate in (
        ("ear_node_point", (40, 65)),
        ("ear_middle_point", (45, 50)),
        ("ear_top_point", (50, 35)),
    ):
        points[LABELS.index(label)] = coordinate
    box = ear_box_from_points(points, np.array([20, 10, 80, 70]))
    assert box[0] <= 40 <= box[2]
    assert box[1] <= 35 <= 65 <= box[3]

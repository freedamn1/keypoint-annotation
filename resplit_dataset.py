"""Re-split an existing pose/ear YOLO bundle by image without reprocessing images."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from collections import Counter
from pathlib import Path

import yaml

from build_dataset import assign_splits


def collect_images(root: Path) -> dict[str, tuple[Path, Path]]:
    images = {}
    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"missing image or label directory in {root}: {split}")
        image_paths = [path for path in image_dir.iterdir() if path.is_file()]
        expected_labels = {f"{path.stem}.txt" for path in image_paths}
        actual_labels = {path.name for path in label_dir.iterdir() if path.is_file()}
        if expected_labels != actual_labels:
            raise ValueError(f"image/label mismatch in {root}: {split}")
        for image_path in image_paths:
            if image_path.name in images:
                raise ValueError(f"duplicate image in {root}: {image_path.name}")
            images[image_path.name] = (image_path, label_dir / f"{image_path.stem}.txt")
    return images


def link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def resplit(source: Path, output: Path, seed: int = 42, val_fraction: float = 0.15, test_fraction: float = 0.15) -> Counter:
    sources = {kind: collect_images(source / kind) for kind in ("pose", "ear")}
    if not sources["pose"] or set(sources["pose"]) != set(sources["ear"]):
        raise ValueError("pose and ear must contain the same nonempty image set")
    for kind in sources:
        destination = output / kind
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"output is not empty: {destination}")

    splits = assign_splits({name: name for name in sources["pose"]}, val_fraction, test_fraction, seed)
    for kind, images in sources.items():
        destination = output / kind
        for name, (image_path, label_path) in images.items():
            split = splits[name]
            image_target = destination / "images" / split / name
            label_target = destination / "labels" / split / label_path.name
            image_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.parent.mkdir(parents=True, exist_ok=True)
            link_or_copy(image_path, image_target)
            link_or_copy(label_path, label_target)
        config = yaml.safe_load((source / kind / "data.yaml").read_text(encoding="utf-8"))
        config["path"] = str(destination.resolve())
        (destination / "data.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    shutil.copy2(source / "pose" / "metadata.json", output / "pose" / "metadata.json")
    with (output / "pose" / "splits.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image", "group", "split"))
        for name in sorted(splits):
            writer.writerow((name, name, splits[name]))
    return Counter(splits.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/imported"))
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(dict(resplit(args.source, args.output, seed=args.seed)))

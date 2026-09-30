"""Fine-tune YOLO11 pose (plant + 12 points) and YOLO11 detect (ear)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def runtime_data_yaml(data: Path) -> Path:
    """Rewrite the dataset root for the machine that runs training."""
    if not data.is_file():
        raise FileNotFoundError(data)
    config = yaml.safe_load(data.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not {"train", "val", "test", "names"}.issubset(config):
        raise ValueError(f"invalid data.yaml: {data}")
    config["path"] = str(data.parent.resolve())
    runtime = data.parent / "data.runtime.yaml"
    runtime.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return runtime


def train_one(kind: str, data: Path, checkpoint: str, args: argparse.Namespace) -> Path:
    from ultralytics import YOLO

    runtime_data = runtime_data_yaml(data)
    model = YOLO(checkpoint)
    model.train(
        data=str(runtime_data.resolve()),
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
        project=f"runs/{kind}",
        name=f"maize-yolo11s-{kind}",
        seed=args.seed,
        deterministic=True,
        degrees=5.0,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.0,
        mixup=0.0,
        perspective=0.0,
        plots=True,
    )
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"training finished without {best}")
    metrics = YOLO(str(best)).val(data=str(runtime_data.resolve()), split="test", imgsz=args.imgsz, batch=args.batch, device=args.device, plots=True)
    print(f"{kind} best checkpoint: {best.resolve()}")
    print(f"{kind} test metrics:", metrics.results_dict)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-data", type=Path, default=Path("data/pose/data.yaml"))
    parser.add_argument("--ear-data", type=Path, default=Path("data/ear/data.yaml"))
    parser.add_argument("--pose-model", default="yolo11s-pose.pt")
    parser.add_argument("--ear-model", default="yolo11s.pt")
    parser.add_argument("--task", choices=("both", "pose", "ear"), default="both")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print("training configuration:", json.dumps(vars(args), default=str, ensure_ascii=False, indent=2))
    if args.task in ("both", "pose"):
        train_one("pose", args.pose_data, args.pose_model, args)
    if args.task in ("both", "ear"):
        train_one("ear", args.ear_data, args.ear_model, args)


if __name__ == "__main__":
    main()
